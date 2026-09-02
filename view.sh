#!/usr/bin/env bash
# Live stats: container / health / memory / the engine's own throughput + speculative metrics.
set -euo pipefail
cd "$(dirname "$0")"
NAME="qwen38-flash-next"
PORT="$(awk '$1=="port:"{print $2; exit}' recipe.yaml)"

S="$(docker ps --filter "name=$NAME" --format '{{.Status}}')"
[ -n "$S" ] && echo "container : $S" || { echo "container : not running (./run.sh)"; exit 0; }
H="$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT:-8000}/health" || true)"
echo "health    : HTTP $H  (API http://127.0.0.1:${PORT:-8000}/v1)"
echo "memory    : $(free -g | awk '/Mem:/{print $3"G used / "$7"G available"}')"
echo
echo "── engine (last 3 windows: throughput · running · KV) ──"
docker logs "$NAME" 2>&1 | grep -a "Engine 000" | tail -3 \
  | sed -E 's/.*INFO [0-9: -]+\[loggers.py:[0-9]+\] //'
echo
echo "── speculative decoding (acceptance = tokens per step) ──"
docker logs "$NAME" 2>&1 | grep -a "SpecDecoding" | tail -2 \
  | sed -E 's/.*INFO [0-9: -]+\[metrics.py:[0-9]+\] //'
