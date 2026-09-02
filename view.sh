#!/usr/bin/env bash
# Attach to the live server logs (vLLM's own output: throughput windows, KV usage,
# speculative-decoding metrics). Ctrl-C detaches; the server keeps running.
set -euo pipefail
cd "$(dirname "$0")"
NAME="qwen38-flash-next"
PORT="$(awk '$1=="port:"{print $2; exit}' recipe.yaml)"

S="$(docker ps --filter "name=$NAME" --format '{{.Status}}')"
[ -n "$S" ] || { echo "not running (./run.sh starts it)"; exit 1; }
H="$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT:-8000}/health" || true)"
echo "· $NAME: $S — health HTTP $H — API http://127.0.0.1:${PORT:-8000}/v1   (Ctrl-C detaches)"
exec docker logs -f --tail 30 "$NAME"
