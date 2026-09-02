#!/usr/bin/env bash
# Qwen3.8-Flash-Next (hibrid46, 4.6-bit) on ONE DGX Spark: download if needed, serve, wait healthy.
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

# --- weights: ~91G, resumable (rerun on interruption) -------------------------------------------
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "· downloading $HF_REPO -> $MODEL_DIR"
  if command -v hf >/dev/null; then
    hf download "$HF_REPO" --local-dir "$MODEL_DIR"
  else
    docker run --rm -v "$MODELS_ABS:/dl" --entrypoint python3 "$IMAGE" \
      -c "from huggingface_hub import snapshot_download; snapshot_download('$HF_REPO', local_dir='/dl/$LOCAL_NAME')"
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

docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "· starting $NAME  ($IMAGE)  on :$PORT — first boot reaches healthy in ~15 min"
docker run -d --name "$NAME" --gpus all --ipc=host \
  ${CPUSET:+--cpuset-cpus "$CPUSET"} \
  -p "$PORT:8000" \
  -v "$MODELS_ABS:/models" -v "$CACHE_ABS:/cache" \
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
