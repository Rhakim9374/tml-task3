#!/usr/bin/env bash
# Invoked by cluster/train.sub inside the pytorch docker image. Installs deps,
# then runs one adversarial-training job. Any extra args are forwarded to
# scripts.train (e.g. --arch resnet34 --method pgd --epochs 80).
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

pip install --quiet -r requirements.txt

python -m scripts.train --data data/train.npz "$@"

echo "TRAIN OK"
