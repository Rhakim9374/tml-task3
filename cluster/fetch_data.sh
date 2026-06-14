#!/usr/bin/env bash
# Download train.npz (~127 MB) from HuggingFace into data/. Idempotent: skips the
# file if it already exists. Run once on the login node after cloning.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

mkdir -p data checkpoints runlogs

URL="https://huggingface.co/datasets/SprintML/tml26_task3/resolve/main/train.npz"
DST="data/train.npz"

if [ -s "$DST" ]; then
    echo "already present: $DST"
else
    echo "==> downloading train.npz (~127 MB)"
    wget -q --tries=5 --continue "$URL" -O "$DST" \
        || { rm -f "$DST"; echo "FAILED: $URL" >&2; exit 1; }
    echo "  got $DST"
fi

echo "FETCH OK"
