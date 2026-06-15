#!/usr/bin/env bash
# Fetch PathMNIST (MedMNIST v2) -> data/pathmnist.npz. Run on the LOGIN node (it
# has internet; the shared filesystem makes data/ visible to the GPU jobs).
#
# Provides the 90k TRAIN superset (supersedes our 50k) + a clean 10k VAL split for
# the external-data experiments. We only ever use train/val, never the test split.
# Using this REQUIRES that external data is permitted by the task rules -- see
# plans/IF_external_data_allowed.txt (leakage policy) and confirm with course staff.
#
# Downloads the npz directly (no pip; the login node's Python is externally managed).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "$SCRIPT_DIR/.." && pwd -P)"
mkdir -p data

URL="https://zenodo.org/records/10519652/files/pathmnist.npz?download=1"

if [ -f data/pathmnist.npz ]; then
  echo "data/pathmnist.npz already present -- skipping download"
elif command -v curl >/dev/null 2>&1; then
  echo "downloading via curl..."
  curl -fL -o data/pathmnist.npz "$URL"
elif command -v wget >/dev/null 2>&1; then
  echo "downloading via wget..."
  wget -O data/pathmnist.npz "$URL"
else
  # Last resort: let medmnist fetch it, inside a throwaway venv (no system pollution).
  echo "no curl/wget; falling back to a temporary venv + medmnist..."
  python3 -m venv /tmp/mvenv
  /tmp/mvenv/bin/pip install -q medmnist
  /tmp/mvenv/bin/python -c "from medmnist import PathMNIST; PathMNIST(split='train', download=True, root='data')"
fi

# Sanity-check the archive. Prefer numpy (full shapes + label range); fall back to
# stdlib zipfile so this still works on the login node, whose python3 has no numpy.
python3 - <<'PY'
try:
    import numpy as np
    d = np.load("data/pathmnist.npz")
    print("keys:", sorted(d.keys()))
    print("train:", d["train_images"].shape, " val:", d["val_images"].shape, " test:", d["test_images"].shape)
    print("labels:", int(d["train_labels"].min()), "-", int(d["train_labels"].max()), "(expected 0-8)")
except ImportError:
    import zipfile
    names = set(zipfile.ZipFile("data/pathmnist.npz").namelist())
    print("members:", sorted(names))
    expected = {f"{s}_{k}.npy" for s in ("train", "val", "test") for k in ("images", "labels")}
    missing = expected - names
    print("all splits present" if not missing else f"MISSING: {sorted(missing)}")
PY
echo "OK -> data/pathmnist.npz"
