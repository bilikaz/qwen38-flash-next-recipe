#!/usr/bin/env bash
# ple.sh — poke the demand-paged n-gram (PLE) table of the RUNNING serve (no restart, no root).
#   ./ple.sh populate   pull the whole 26.9 GiB table into memory now (repeatable; each pass re-touches every page)
#   ./ple.sh mmap       eager steps (prefill/warm-ups) gather via the memory-mapped path
#   ./ple.sh nvme       eager steps gather via direct NVMe reads (no page cache growth)
#   ./ple.sh status     what the serve reports + table pages mapped / free / swap right now
# Mechanism: a watcher thread in the image polls /cache/mbx-ple-{prewarm,mmap,nvme} every 2 s, acts, deletes the file,
# prints a "PLE MMAP:" line. /cache in the container = this kit's cache/ dir on the host.
set -euo pipefail
cd "$(dirname "$0")"
NAME="qwen38-flash-next"
CACHE="$(awk '$1=="cache_dir:"{print $2; exit}' recipe.yaml)"; CACHE="${CACHE:-./cache}"
mem() {
  awk '/^MemFree:/{f=$2}/^MemAvailable:/{a=$2}/^Mapped:/{m=$2}/^SwapTotal:/{st=$2}/^SwapFree:/{sf=$2}
       END{printf "  free %.1f G · available %.1f G · file pages mapped %.1f G (table ≈ this minus ~1 G) · swap used %.1f G\n",
           f/1048576,a/1048576,m/1048576,(st-sf)/1048576}' /proc/meminfo
}
last() { docker logs "$NAME" 2>&1 | grep -E "PLE MMAP: (gather path|populated|populating|heap trim)" | tail -"${1:-3}" | sed 's/^(Worker pid=[0-9]*) //'; }
case "${1:-status}" in
  populate|prewarm) n0=$(docker logs "$NAME" 2>&1 | grep -c "PLE MMAP: populated" || true)
      touch "$CACHE/mbx-ple-prewarm"; echo "· populate requested — waiting for the pass to finish (10–30 s)"; mem
      for i in $(seq 1 60); do sleep 2; n1=$(docker logs "$NAME" 2>&1 | grep -c "PLE MMAP: populated" || true); [ "$n1" -gt "$n0" ] && break; done
      last 1; mem ;;
  mmap|nvme) touch "$CACHE/mbx-ple-$1"; sleep 4; last 1 ;;
  status) echo "· serve: $(docker ps --format '{{.Status}}' --filter name="$NAME")"; last 3; mem ;;
  *) sed -n 2,7p "$0"; exit 2 ;;
esac
