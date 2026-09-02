#!/usr/bin/env bash
# Stop and remove the serve container. Weights and caches stay — ./run.sh brings it back fast.
set -euo pipefail
docker rm -f qwen38-flash-next >/dev/null 2>&1 && echo "✓ stopped" || echo "· not running"
