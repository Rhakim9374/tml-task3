#!/usr/bin/env bash
# Regularization sweep for the robustness task.
#
# Fixed: resnet50 + TRADES (beta=6) + grad clipping (always on). EMA is always
# on; we only vary its decay. We sweep the four generalization knobs ONE AT A
# TIME around a strong baseline (coordinate search, not a full grid) so each
# knob's effect is interpretable and the job count stays small enough to run in
# parallel and finish well before the deadline. A full 3^4 grid would be 81
# runs; this is 9.
#
# After all jobs finish, rank them with:
#   ~/.tml-venv/bin/python -m scripts.collect_sweep --glob "checkpoints/sweep_*.pt" --arch resnet50
#
# Override epoch count for a faster ranking pass, e.g.: EPOCHS=40 bash cluster/launch_sweep.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

EPOCHS="${EPOCHS:-50}"
COMMON="--arch resnet50 --method trades --beta 6.0 --epochs ${EPOCHS} --grad-clip 1.0"

# name | knob args (varied dimension in **bold** conceptually).
# Baseline: ema 0.999, wd 5e-4, dropout 0.1, label-smoothing 0.1.
CONFIGS=(
  "baseline|--ema-decay 0.999  --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1"
  "ema9995|--ema-decay 0.9995 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1"
  "ema998|--ema-decay 0.998  --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1"
  "wd1e3|--ema-decay 0.999  --weight-decay 1e-3 --dropout 0.1 --label-smoothing 0.1"
  "wd2e4|--ema-decay 0.999  --weight-decay 2e-4 --dropout 0.1 --label-smoothing 0.1"
  "drop0|--ema-decay 0.999  --weight-decay 5e-4 --dropout 0.0 --label-smoothing 0.1"
  "drop2|--ema-decay 0.999  --weight-decay 5e-4 --dropout 0.2 --label-smoothing 0.1"
  "ls0|--ema-decay 0.999  --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.0"
  "ls2|--ema-decay 0.999  --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.2"
)

mkdir -p checkpoints runlogs
echo "launching ${#CONFIGS[@]} jobs (resnet50, ${EPOCHS} epochs each)"
for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"
  extra="${entry#*|}"
  out="checkpoints/sweep_${name}.pt"
  echo "  submit ${name} -> ${out}"
  condor_submit cluster/train.sub \
    -append "args=${COMMON} ${extra} --out ${out}" \
    -append "tag=sweep_${name}"
done
echo "all submitted. watch with: condor_q   |   rank with: scripts.collect_sweep"
