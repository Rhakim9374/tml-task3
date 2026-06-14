#!/usr/bin/env bash
# Parallel sweep for the robustness task -- 20 independent 1-GPU jobs, designed
# to saturate ~20 cluster GPUs at once (each is a separate condor_submit, so the
# scheduler spreads them across machines/GPUs; wall-clock ~= one job).
#
# Coordinate search around a strong baseline plus a direct SGD-vs-SAM showdown.
# Baseline: resnet50, TRADES, SGD, beta=6, ema=0.999, wd=5e-4, dropout=0.1,
#           label-smoothing=0.1, cutout=0, grad-clip=5.
#
# After all jobs finish, rank with:
#   ~/.tml-venv/bin/python -m scripts.collect_sweep --glob "checkpoints/sweep_*.pt" --arch resnet50
#
# Override epochs for a faster pass: EPOCHS=40 bash cluster/launch_sweep.sh
# (SAM jobs cost ~2x per epoch, so they finish later -- that's fine, they run in
# parallel and collect_sweep just reads whatever checkpoints exist.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

EPOCHS="${EPOCHS:-50}"
BASE="--arch resnet50 --epochs ${EPOCHS} --grad-clip 5.0"

# name | full knob args (each config sets every knob explicitly for clarity)
CONFIGS=(
  # --- baseline + direct SGD vs SAM ---
  "baseline|--method trades --optimizer sgd  --beta 6.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 0"
  "sam|--method trades --optimizer sam  --beta 6.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 0"
  # --- objective: pgd-at and mart (vs trades baseline) ---
  "pgd|--method pgd    --optimizer sgd  --beta 6.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 0"
  "mart|--method mart   --optimizer sgd  --beta 6.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 0"
  # --- TRADES beta (clean/robust dial) ---
  "beta3|--method trades --optimizer sgd  --beta 3.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 0"
  "beta9|--method trades --optimizer sgd  --beta 9.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 0"
  # --- EMA decay ---
  "ema9995|--method trades --optimizer sgd  --beta 6.0 --ema-decay 0.9995 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 0"
  "ema998|--method trades --optimizer sgd  --beta 6.0 --ema-decay 0.998 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 0"
  # --- weight decay ---
  "wd1e3|--method trades --optimizer sgd  --beta 6.0 --ema-decay 0.999 --weight-decay 1e-3 --dropout 0.1 --label-smoothing 0.1 --cutout 0"
  "wd2e4|--method trades --optimizer sgd  --beta 6.0 --ema-decay 0.999 --weight-decay 2e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 0"
  # --- dropout ---
  "drop0|--method trades --optimizer sgd  --beta 6.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.0 --label-smoothing 0.1 --cutout 0"
  "drop2|--method trades --optimizer sgd  --beta 6.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.2 --label-smoothing 0.1 --cutout 0"
  # --- label smoothing ---
  "ls0|--method trades --optimizer sgd  --beta 6.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.0 --cutout 0"
  "ls2|--method trades --optimizer sgd  --beta 6.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.2 --cutout 0"
  # --- Cutout augmentation (pairs with EMA) ---
  "cutout8|--method trades --optimizer sgd  --beta 6.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 8"
  "cutout16|--method trades --optimizer sgd  --beta 6.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 16"
  # --- SAM combined with the most promising add-ons ---
  "sam_cutout8|--method trades --optimizer sam  --beta 6.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 8"
  "sam_beta9|--method trades --optimizer sam  --beta 9.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 0"
  "sam_mart|--method mart   --optimizer sam  --beta 6.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 0"
  "sam_pgd|--method pgd    --optimizer sam  --beta 6.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --cutout 0"
)

mkdir -p checkpoints runlogs
echo "launching ${#CONFIGS[@]} jobs (resnet50, ${EPOCHS} epochs; 1 GPU each)"
for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"
  extra="${entry#*|}"
  out="checkpoints/sweep_${name}.pt"
  echo "  submit ${name} -> ${out}"
  condor_submit cluster/train.sub \
    -append "args=${BASE} ${extra} --out ${out}" \
    -append "tag=sweep_${name}"
done
echo "all submitted. watch: condor_q   |   rank: scripts.collect_sweep"
