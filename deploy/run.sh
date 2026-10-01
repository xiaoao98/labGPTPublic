#!/usr/bin/env bash
# Start the service in the foreground. For a long-running one see nohup in DEPLOY.md.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)

[ -f labgpt.env ] || { echo "labgpt.env is missing; run ./install.sh first" >&2; exit 1; }
source ./labgpt.env

# The model and index travel with the bundle, and the host has no route to huggingface.co,
# so the offline flags are set here rather than left to be forgotten.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LABGPT_EMBED_MODEL="$HERE/models/bge-small-en-v1.5"
export LABGPT_INDEX_DIR="$HERE/index"
export LABGPT_LOG_PATH="${LABGPT_LOG_PATH:-$HERE/qa.jsonl}"
export PYTHONPATH="$HERE/app"

PORT="${LABGPT_PORT:-8090}"
echo "starting on 0.0.0.0:$PORT, logging questions to $LABGPT_LOG_PATH"
exec ./venv/bin/uvicorn serve:app --app-dir app --host 0.0.0.0 --port "$PORT"
