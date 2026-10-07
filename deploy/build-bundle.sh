#!/usr/bin/env bash
# Build the offline bundle that deploys LabGPT to a host with no internet and no root.
#
# Run this on a machine that does have internet. It produces one tarball holding an
# interpreter, every wheel, the application, the embedding model and an index, which is
# everything the target needs, because the target can download nothing.
#
# WHY EACH PIECE IS HERE, since every one of them was a failure first:
#
#   A bundled interpreter, because RHEL 9 ships Python 3.9 and this needs 3.10 or newer.
#   python-build-standalone is relocatable, so it runs from wherever it is unpacked.
#
#   CPU-only torch, from the pytorch cpu index. The default wheels carry CUDA and turn a
#   285 MB download into something over 2 GB, for a GPU the server does not have.
#
#   Wheels for manylinux x86_64 rather than for this machine. Built on a Mac, these are
#   the wrong architecture by default and the failure appears only on the server.
#
#   The embedding model as files. The host cannot reach huggingface.co, and the index
#   records which model built it: a different one retrieves nonsense rather than failing.
#
# Verified by installing the result in a registry.access.redhat.com/ubi9/ubi container.

set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)

MODEL_DIR="${LABGPT_EMBED_MODEL:-$REPO/bge-small-en-v1.5}"
INDEX_DIR="${LABGPT_INDEX_DIR:-$REPO/.index}"
OUT="${1:-$HOME/Desktop/labgpt-bundle.tar.gz}"
PYTHON_VERSION=3.11
STAGE=$(mktemp -d)/labgpt-bundle
trap 'rm -rf "$(dirname "$STAGE")"' EXIT

[ -d "$MODEL_DIR" ] || { echo "no embedding model at $MODEL_DIR; set LABGPT_EMBED_MODEL" >&2; exit 1; }
[ -d "$INDEX_DIR" ] || { echo "no index at $INDEX_DIR; build one with labrag.cli index" >&2; exit 1; }

mkdir -p "$STAGE"/{wheels,app,models}

echo "==> wheels for linux x86_64, python $PYTHON_VERSION, cpu torch"
# --only-binary=:all: refuses source distributions, which would be built for the wrong
# platform here and could not be built at all on the offline target.
python3 -m pip download --quiet --dest "$STAGE/wheels" \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 \
    --platform manylinux2014_x86_64 --platform any \
    --python-version "$PYTHON_VERSION" --implementation cp --only-binary=:all: \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    "torch>=2.2" "pyyaml>=6.0" "numpy>=1.24" "sentence-transformers>=3.0" \
    "transformers>=4.44" "rich>=13.0" "fastapi>=0.110" "uvicorn>=0.29"

echo "==> interpreter"
URL=$(curl -sS https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest \
      | python3 -c "
import json, sys
for asset in json.load(sys.stdin)['assets']:
    name = asset['name']
    if 'cpython-$PYTHON_VERSION' in name and name.endswith('x86_64-unknown-linux-gnu-install_only.tar.gz'):
        print(asset['browser_download_url'])
        break
")
[ -n "$URL" ] || { echo "no standalone python $PYTHON_VERSION build found" >&2; exit 1; }
curl -sSL -o "$STAGE/python-linux-x86_64.tar.gz" "$URL"

echo "==> application"
cp -R labrag serve.py demo.py labgpt_config.py labgpt_metrics.py web pyproject.toml "$STAGE/app/"
find "$STAGE/app" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

echo "==> embedding model"
# Only what sentence-transformers loads. The published repository also carries an onnx
# copy and a pytorch_model.bin duplicate of the same weights, which triple the size.
mkdir -p "$STAGE/models/$(basename "$MODEL_DIR")"
for f in config.json config_sentence_transformers.json modules.json sentence_bert_config.json \
         special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt model.safetensors; do
    cp "$MODEL_DIR/$f" "$STAGE/models/$(basename "$MODEL_DIR")/" 2>/dev/null || true
done
cp -R "$MODEL_DIR/1_Pooling" "$STAGE/models/$(basename "$MODEL_DIR")/"

echo "==> index"
cp -R "$INDEX_DIR" "$STAGE/index"

cp deploy/install.sh deploy/run.sh deploy/DEPLOY.md "$STAGE/"
chmod +x "$STAGE/install.sh" "$STAGE/run.sh"

echo "==> packing"
tar czf "$OUT" -C "$(dirname "$STAGE")" "$(basename "$STAGE")"
echo
echo "built $OUT ($(du -h "$OUT" | cut -f1))"
echo "  the index and the model are in it, so this file carries the corpus: treat it as"
echo "  institutional data, and send it to the server rather than to a shared drive."
