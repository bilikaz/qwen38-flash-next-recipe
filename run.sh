#!/usr/bin/env bash
# Qwen3.8-Flash-Next (hibrid47: NVFP4 n-gram table demand-paged, fp8 KV) on ONE DGX Spark: download if needed, wait for
# memory, evict stale page cache, serve, wait healthy.
# Everything is configured in recipe.yaml. OpenAI API on :$PORT. ./stop.sh stops, ./view.sh stats.
set -euo pipefail
cd "$(dirname "$0")"
command -v docker >/dev/null || { echo "docker is required"; exit 1; }

# --- tiny recipe.yaml reader (two-level: section -> key: value; strips quotes/comments) ---------
rkey() {  # rkey <section> <key>
  awk -v s="$1" -v k="$2" '
    /^[A-Za-z_]/ { sec=$1; sub(":$","",sec) }
    sec==s && $1==k":" {
      sub(/^[ ]*[^:]*:[ ]*/,""); sub(/[ ]+#.*$/,"")
      gsub(/^["\x27]|["\x27]$/,""); print; exit
    }' recipe.yaml
}
rsection() {  # all key/value lines of a section, "key<TAB>value" (quotes/comments stripped)
  awk -v s="$1" '
    /^[A-Za-z_]/ { sec=$1; sub(":$","",sec); next }
    sec==s && $1 ~ /^[A-Za-z0-9_-]+:$/ || (sec==s && /^[ ]+[A-Za-z0-9_-]+:[ ]/) {
      line=$0; sub(/^[ ]+/,"",line)
      key=line; sub(/:.*/,"",key)
      val=line; sub(/^[^:]*:[ ]*/,"",val); sub(/[ ]+#.*$/,"",val)
      gsub(/^["\x27]|["\x27]$/,"",val)
      if (key != "") print key "\t" val
    }' recipe.yaml
}

IMAGE="$(rkey server image)";       PORT="$(rkey server port)"
HF_REPO="$(rkey server model)"
MODELS_DIR="$(rkey server models_dir)"; CACHE_DIR="$(rkey server cache_dir)"
CPUSET="$(rkey server cpuset)"
NAME="qwen38-flash-next"
mkdir -p "$MODELS_DIR" "$CACHE_DIR"
MODELS_ABS="$(cd "$MODELS_DIR" && pwd)"; CACHE_ABS="$(cd "$CACHE_DIR" && pwd)"
LOCAL_NAME="$(basename "$HF_REPO")"
MODEL_DIR="$MODELS_ABS/$LOCAL_NAME"

# --- weights: ~99G, resumable (rerun on interruption) -------------------------------------------
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "· downloading $HF_REPO -> $MODEL_DIR"
  if command -v hf >/dev/null; then
    hf download "$HF_REPO" --local-dir "$MODEL_DIR"
  else
    # -t gives tqdm a TTY so per-file progress bars actually render; HF_TOKEN passes
    # through if exported (higher rate limits) and is harmless when unset.
    TTY=""; [ -t 1 ] && TTY="-t"
    # the container runs as root — hand the files back to the host user afterwards
    docker run --rm $TTY -e HF_TOKEN -v "$MODELS_ABS:/dl" --entrypoint python3 "$IMAGE" \
      -c "from huggingface_hub import snapshot_download; import subprocess; snapshot_download('$HF_REPO', local_dir='/dl/$LOCAL_NAME'); subprocess.run(['chown', '-R', '$(id -u):$(id -g)', '/dl/$LOCAL_NAME'], check=False)"
  fi
fi

# --- assemble docker env + vllm flags straight from the recipe ----------------------------------
ENVS=()
while IFS=$'\t' read -r k v; do [ -n "$k" ] && ENVS+=(-e "$k=$v"); done < <(rsection env)
FLAGS=()
while IFS=$'\t' read -r k v; do
  case "$v" in
    true)        FLAGS+=("--$k");;
    false|null|"") ;;
    *)           FLAGS+=("--$k" "$v");;
  esac
done < <(rsection vllm)

# kernel page compaction — read-only check (the fix needs root → ./tune-host.sh). On a Spark the GPU's memory is
# ordinary pages; the kernel's proactive compactor migrating them measured as 4-5 s stalls every ~37 s (~10 %).
cp_now=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo "?")
if [ "$cp_now" != 0 ]; then
  echo "  ⚠ vm.compaction_proactiveness is $cp_now (want 0): expect ~10 % lower throughput and periodic 4-5 s stalls"
  echo "    under load. One-time fix, needs sudo, shows what it runs first:  ./tune-host.sh"
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true

# memory gate (unified memory: a serve relaunched seconds after a teardown gets a PHANTOM "CUDA out of memory" — the
# previous container's GPU pages take 30-60 s to come back). The load needs ~100G available; the table then fills the rest.
t=0; while :; do
  avail=$(free -g | awk '/^Mem:/{print $7}')
  [ "${avail:-0}" -ge 100 ] && { echo "  ✓ memory: ${avail}G available"; break; }
  [ "$t" -ge 120 ] && { echo "  ✗ only ${avail}G available after 120 s (need ~100G) — another container on the box? (docker ps)"; exit 1; }
  [ "$t" = 0 ] && echo "  · waiting for memory to come back (${avail}G available, need 100G)…"
  sleep 5; t=$((t+5))
done
# evict our own checkpoint files from the page cache (no root: POSIX_FADV_DONTNEED via dd) — the GPU driver wants pages
# that are FREE, and a 60-70G stale shard cache during the load has stalled it
find "$MODEL_DIR" -type f -name "*.safetensors" -exec dd if={} iflag=nocache count=0 status=none \; 2>/dev/null || true
echo "  · page cache: checkpoint files evicted — MemFree $(awk '/^MemFree/{printf "%d", $2/1048576}' /proc/meminfo)G"
[ "$(free -g | awk '/^Swap:/{print $2}')" -gt 0 ] || echo "  · no swap on this box: fine — the table's rows are re-read from NVMe when the kernel needs the pages"
echo "· starting $NAME  ($IMAGE)  on :$PORT — first boot reaches healthy in ~12 min (weights 11 min), then the table populates (~30 s)"
docker run -d --name "$NAME" --gpus all --ipc=host \
  ${CPUSET:+--cpuset-cpus "$CPUSET"} \
  -p "$PORT:8000" \
  -v "$MODELS_ABS:/models" -v "$CACHE_ABS:/cache" \
  -v "$(readlink -f "$MODEL_DIR"):/models/$LOCAL_NAME" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer-workspace \
  -e VLLM_CACHE_ROOT=/cache/vllm-cache \
  "${ENVS[@]}" \
  --entrypoint vllm "$IMAGE" serve "/models/$LOCAL_NAME" --port 8000 "${FLAGS[@]}" >/dev/null

echo "· streaming engine logs until healthy (Ctrl-C detaches; the container keeps booting)"
docker logs -f "$NAME" 2>&1 &
LOGS=$!
trap 'kill "$LOGS" 2>/dev/null' EXIT INT TERM
for i in $(seq 1 240); do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    kill "$LOGS" 2>/dev/null; wait "$LOGS" 2>/dev/null
    echo
    echo "──────────────────────────────────────────────────────────"
    echo "✓ server booted — OpenAI-compatible API is live"
    echo "    endpoint : http://127.0.0.1:$PORT/v1"
    echo "    monitor  : ./view.sh          (throughput, KV, acceptance)"
    echo "    logs     : docker logs -f $NAME"
    echo "    stop     : ./stop.sh"
    echo "──────────────────────────────────────────────────────────"
    exit 0
  fi
  if ! docker ps -q --filter "name=$NAME" | grep -q .; then
    kill "$LOGS" 2>/dev/null; wait "$LOGS" 2>/dev/null
    echo "✗ container exited — see above"; exit 1
  fi
  sleep 5
done
echo "✗ not healthy after 20 min — still booting? watch: docker logs -f $NAME"; exit 1
