#!/usr/bin/env bash
# Fetch PathMNIST (MedMNIST v2) -> data/pathmnist.npz. Run on the LOGIN node (it
# has internet; the shared filesystem makes data/ visible to the GPU jobs).
#
# Provides the 90k TRAIN superset (supersedes our 50k) + a clean 10k VAL split for
# the external-data experiments. We only ever use train/val, never the test split.
# Using this REQUIRES that external data is permitted by the task rules -- see
# plans/IF_external_data_allowed.txt (leakage policy) and confirm with course staff.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "$SCRIPT_DIR/.." && pwd -P)"
mkdir -p data

if [ -f data/pathmnist.npz ]; then
  echo "data/pathmnist.npz already present -- skipping download"
else
  # The medmnist package knows the current mirror; download straight into data/.
  pip install --user --quiet medmnist
  python - <<'PY'
from medmnist import PathMNIST
PathMNIST(split="train", download=True, root="data")   # writes data/pathmnist.npz
print("fetched -> data/pathmnist.npz")
PY
fi

# Sanity-check shapes, splits, and label range.
python - <<'PY'
import numpy as np
d = np.load("data/pathmnist.npz")
print("keys:", sorted(d.keys()))
print("train:", d["train_images"].shape, " val:", d["val_images"].shape, " test:", d["test_images"].shape)
print("labels:", int(d["train_labels"].min()), "-", int(d["train_labels"].max()),
      "(expected 0-8)")
PY
