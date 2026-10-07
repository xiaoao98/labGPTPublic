#!/usr/bin/env bash
# Install LabGPT on a host with no internet access and no root.
#
# Everything needed is in this directory: an interpreter, the wheels, the code, the
# embedding model and the index. Nothing is fetched. The system python is not used, it is
# 3.9 on RHEL 9 and this needs 3.10 or newer.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)

echo "==> unpacking the interpreter"
rm -rf python
tar xzf python-linux-x86_64.tar.gz     # extracts to ./python
./python/bin/python3 --version

echo "==> creating the virtual environment"
rm -rf venv
./python/bin/python3 -m venv venv

echo "==> installing from the bundled wheels, with no network"
# --no-index is the point: if a wheel were missing this fails here rather than silently
# reaching for pypi, which on this host would hang instead of erroring.
./venv/bin/pip install --quiet --no-index --find-links wheels \
    torch pyyaml numpy sentence-transformers transformers rich fastapi uvicorn

echo "==> checking it imports without touching the network"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$HERE/app" \
  ./venv/bin/python -c "import torch, sentence_transformers, fastapi, uvicorn; print('    ok')"

if [ ! -f labgpt.env ]; then
  cat > labgpt.env <<'ENVEOF'
# Filled in by hand on the server. Not readable by anyone else: chmod 600 this file.
export LABGPT_API_TOKEN=
export AZURE_OPENAI_GATEWAY=
export AZURE_OPENAI_TEAM_ID=
export AZURE_OPENAI_MODEL_ID=
export AZURE_OPENAI_API_VERSION=
export APIM_OPENAI_SUBSCRIPTION_KEY=
ENVEOF
  chmod 600 labgpt.env
  echo "==> wrote labgpt.env; fill it in before starting"
fi

echo
echo "installed. Next:"
echo "  1. edit $HERE/labgpt.env and fill in the token and the Azure settings"
echo "  2. $HERE/run.sh"
