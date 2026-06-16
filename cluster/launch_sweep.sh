#!/usr/bin/env bash
# Batch 5 (FINAL): seeds + a small-dropout variant of the winning recipe, longer
# and with a gentler LR schedule.
#
# Settled: plain PGD-AT (resnet50, wd 1e-3, EMA 0.999) is the recipe; steps 7 was
# the best clean/robust balance in batch 4; external PathMNIST data gave NO gain
# (capacity-bound, not data-bound) -> internal 50k only. Best to date: pgd200_s7.
#
# This batch keeps that recipe and changes only the schedule + explores the seed
# lottery (the one remaining lever) plus a gentle dropout:
#   - 300 epochs (a bit longer)
#   - peak LR 0.05 (lower than the previous 0.1) and NO min-lr floor (cosine -> 0):
#     a gentler, fully-annealed schedule for a more converged final model.
#   - 3 seeds (variance on the winner) + 1 small-dropout (0.05) variant at seed 1.
#
# Trains on the full provided 50k; validates leakage-free on the PathMNIST VAL
# split (disjoint from the 50k) so selection stays honest. Each run also does the
# strong CE+DLR eval every 10 epochs (-> best-by-true-robustness <out>.strong.pt).
#
# Rank after (MUST pass --extra-data so val is the clean PathMNIST split):
#   python -m scripts.analyze_trends  --glob "runlogs/sweep_*.out"
#   python -m scripts.rank_robust     --glob "checkpoints/sweep_*.pt" --arch resnet50 --extra-data data/pathmnist.npz
#   python -m scripts.eval_on_provided --glob "checkpoints/sweep_*.pt" --arch resnet50   # test-preprocessing reference
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

EPOCHS="${EPOCHS:-300}"
# Peak LR 0.05 (down from 0.1); no --min-lr => cosine anneals fully to 0.
BASE="--arch resnet50 --epochs ${EPOCHS} --method pgd --optimizer sgd --weight-decay 1e-3 \
--ema-decay 0.999 --label-smoothing 0 --grad-clip 5.0 --warmup 5 --lr 0.05 --steps 7 \
--strong-eval-every 10 --extra-data data/pathmnist.npz"

# name | knob args (all steps-7 PGD-AT on the full 50k).
CONFIGS=(
  "pgd250_s1|--seed 1"
  "pgd250_s2|--seed 2"
  "pgd250_s3|--seed 3"
  "pgd250_drop05|--seed 1 --dropout 0.05"
)

mkdir -p checkpoints runlogs
echo "launching ${#CONFIGS[@]} jobs (resnet50, ${EPOCHS} epochs, peak-lr 0.05, cosine->0; 1 GPU each)"
for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"
  extra="${entry#*|}"
  out="checkpoints/sweep_${name}.pt"
  echo "  submit ${name} -> ${out}"
  condor_submit cluster/train.sub \
    -append "args=${BASE} ${extra} --out ${out}" \
    -append "tag=sweep_${name}"
done
echo "all submitted. watch: condor_q   |   rank: scripts.rank_robust --extra-data data/pathmnist.npz"
