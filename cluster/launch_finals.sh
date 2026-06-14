#!/usr/bin/env bash
# Finals wave: longer-trained (default 100-epoch) runs of the a-priori strongest
# configs, launched IN PARALLEL with the sweep to use surplus GPUs (>20 total).
#
# Rationale: the EPOCHS=40 sweep gives guaranteed anneal-completed, decision-ready
# models inside the 7-10h window. These finals are upside -- if they anneal-finish
# in time they beat the 40-epoch models; if not, their periodic best-checkpoints
# are still readable. Distinct final_*.pt prefix so collect_sweep can rank them
# separately or together.
#
# Rank with:
#   ~/.tml-venv/bin/python -m scripts.collect_sweep --glob "checkpoints/final_*.pt" --arch resnet50
#
# Override epochs: EPOCHS=80 bash cluster/launch_finals.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

EPOCHS="${EPOCHS:-100}"
BASE="--arch resnet50 --epochs ${EPOCHS} --grad-clip 5.0 --ema-decay 0.999 --weight-decay 5e-4 --dropout 0.1 --label-smoothing 0.1 --eval-every 5"

# Strongest a-priori candidates: TRADES SGD/SAM at two betas, MART, and a
# no-cutout hedge (in case Cutout hurts on this unknown dataset).
CONFIGS=(
  "sgd_b6|--method trades --optimizer sgd --beta 6.0 --cutout 8"
  "sam_b6|--method trades --optimizer sam --beta 6.0 --cutout 8"
  "sgd_b9|--method trades --optimizer sgd --beta 9.0 --cutout 8"
  "sam_b9|--method trades --optimizer sam --beta 9.0 --cutout 8"
  "mart_sgd|--method mart   --optimizer sgd --beta 6.0 --cutout 8"
  "sgd_b6_nocut|--method trades --optimizer sgd --beta 6.0 --cutout 0"
)

mkdir -p checkpoints runlogs
echo "launching ${#CONFIGS[@]} finals (resnet50, ${EPOCHS} epochs; 1 GPU each)"
for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"
  extra="${entry#*|}"
  out="checkpoints/final_${name}.pt"
  echo "  submit ${name} -> ${out}"
  condor_submit cluster/train.sub \
    -append "args=${BASE} ${extra} --out ${out}" \
    -append "tag=final_${name}"
done
echo "all finals submitted."
