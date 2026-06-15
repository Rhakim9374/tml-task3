#!/usr/bin/env bash
# Batch 4: long-LR-floor trend + PGD-step balance + external data (6 jobs, <10 GPUs).
#
# Settled so far: plain PGD-AT (resnet50, wd 1e-3, EMA 0.999) is the recipe; every
# robust-overfit lever (AWP/piecewise/high-WD/dropout) was neutral-to-negative
# because the runs PLATEAU, not overfit. Best to date: pgd_s1 -> leaderboard 0.6092.
# Data-limited ceiling looks ~0.61-0.62; the only lever big enough for the 0.05 gap
# to the top is more in-distribution data.
#
# So this batch is a clean 2x3 grid -- data {provided-50k, PathMNIST-90k} x PGD
# training steps {7, 10, 15} -- all sharing ONE schedule change:
#   - 200 epochs with a cosine FLOOR of min-lr 0.005: the LR never anneals to 0, so
#     we can read whether robust acc is truly plateaued or still climbing.
#   - steps 7 vs 10 vs 15: the only untested clean/robust *balance* knob (fewer
#     steps -> higher clean, our binding side; more -> more robust, lower clean).
#   - half the grid trains on the PathMNIST TRAIN superset (90k, supersedes our 50k).
#
# LEAKAGE-SAFE: every run validates on the PathMNIST VAL split (disjoint from both
# the 50k and the 90k superset, and from the hidden test) -- so internal/external
# numbers are comparable and uncontaminated. NEVER touches the test split. Requires
# external data to be rules-permitted (see plans/IF_external_data_allowed.txt).
#
# Prereq:  bash cluster/fetch_pathmnist.sh        (login node, once)
# Rank after (MUST pass --extra-data so external models are scored on the clean val):
#   python -m scripts.analyze_trends --glob "runlogs/sweep_*.out"
#   python -m scripts.rank_robust --glob "checkpoints/sweep_*.pt" --arch resnet50 --extra-data data/pathmnist.npz
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

EPOCHS="${EPOCHS:-200}"
BASE="--arch resnet50 --epochs ${EPOCHS} --method pgd --optimizer sgd --weight-decay 1e-3 \
--ema-decay 0.999 --dropout 0 --label-smoothing 0 --grad-clip 5.0 --warmup 5 \
--min-lr 0.005 --strong-eval-every 10 --extra-data data/pathmnist.npz"

# name | knob args.  Internal trains on the provided 50k; ext_* adds --extra-train (90k).
CONFIGS=(
  "pgd200_s7|--steps 7"
  "pgd200_s10|--steps 10"
  "pgd200_s15|--steps 15"
  "ext_pgd200_s7|--steps 7 --extra-train"
  "ext_pgd200_s10|--steps 10 --extra-train"
  "ext_pgd200_s15|--steps 15 --extra-train"
)

mkdir -p checkpoints runlogs
echo "launching ${#CONFIGS[@]} jobs (resnet50, ${EPOCHS} epochs, min-lr 0.005; 1 GPU each)"
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
