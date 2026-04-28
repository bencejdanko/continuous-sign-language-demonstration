#!/usr/bin/env bash
# start.sh — Start both servers on the AGX Orin.
# Usage:
#   HF_TOKEN=hf_xxx ./start.sh
#   HF_TOKEN=hf_xxx CAM_INDEX=1 ./start.sh

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

export HF_TOKEN="${HF_TOKEN:?Please set HF_TOKEN}"
export CAM_INDEX="${CAM_INDEX:-0}"
export INFERENCE_PORT="${INFERENCE_PORT:-8001}"

echo "==> Starting inference server on port $INFERENCE_PORT …"
python inference_server.py &
INFER_PID=$!

# Give the inference server time to download + load models
echo "    Waiting for models to load (up to 60 s)…"
for i in $(seq 1 60); do
  if curl -sf "http://localhost:$INFERENCE_PORT/health" > /dev/null 2>&1; then
    echo "    Inference server ready ✓"
    break
  fi
  sleep 1
done

echo "==> Starting demo app on port 8000 …"
python app.py &
APP_PID=$!

echo ""
echo "  Demo:            http://$(hostname -I | awk '{print $1}'):8000"
echo "  Inference health: http://localhost:$INFERENCE_PORT/health"
echo ""
echo "  Press Ctrl-C to stop both servers."

trap "kill $INFER_PID $APP_PID 2>/dev/null; echo 'Stopped.'" EXIT INT TERM
wait
