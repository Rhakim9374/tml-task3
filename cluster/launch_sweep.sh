#!/usr/bin/env bash
# The final sweep: the winning PGD-AT recipe across 3 seeds + a small-dropout
# variant. launch_best.sh runs the single best config; this is how we found it --
# once the recipe is fixed, the seed is the main remaining lever. All runs train
# on the provided 50k and validate on a fixed 10% held-out split.
#
# Recipe: resnet50 PGD-AT (eps 8/255, 7 inner steps), 300 epochs, SGD, cosine LR
# 0.05 -> 0 after a 5-ep warmup, weight decay 1e-3, EMA 0.999, grad clipping,
# always-on D4 augmentation. Each run also runs a strong CE+DLR eval every 10
# epochs, saving a best-by-true-robustness <out>.strong.pt next to <out>.pt.
#
# External-data ablation (negative result -- 90k PathMNIST gave no gain): add
# `--extra-data data/pathmnist.npz --extra-train` to BASE (see fetch_pathmnist.sh).
#
# Rank after:
#   python -m scripts.analyze_trends   --glob "runlogs/sweep_*.out"
#   python -m scripts.rank_robust      --glob "checkpoints/sweep_*.pt" --arch resnet50
#   python -m scripts.eval_on_provided --glob "checkpoints/sweep_*.pt" --arch resnet50
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

EPOCHS="${EPOCHS:-300}"
BASE="--arch resnet50 --epochs ${EPOCHS} --method pgd --optimizer sgd --weight-decay 1e-3 \
--ema-decay 0.999 --label-smoothing 0 --grad-clip 5.0 --warmup 5 --lr 0.05 --steps 7 \
--strong-eval-every 10"

# name | knob args (all steps-7 PGD-AT; dropout variant shares seed 1 with pgd_s1).
CONFIGS=(
  "pgd_s1|--seed 1"
  "pgd_s2|--seed 2"
  "pgd_s3|--seed 3"
  "pgd_drop05|--seed 1 --dropout 0.05"
)

mkdir -p checkpoints runlogs
echo "launching ${#CONFIGS[@]} jobs (resnet50, ${EPOCHS} epochs, cosine 0.05->0; 1 GPU each)"
for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"
  extra="${entry#*|}"
  out="checkpoints/sweep_${name}.pt"
  echo "  submit ${name} -> ${out}"
  condor_submit cluster/train.sub \
    -append "args=${BASE} ${extra} --out ${out}" \
    -append "tag=sweep_${name}"
done
echo "all submitted. watch: condor_q   |   rank: scripts.rank_robust"
