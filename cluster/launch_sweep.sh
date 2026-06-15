#!/usr/bin/env bash
# Focused 100-epoch batch (batch 2), pruned from the 50-epoch sweep results.
#
# Findings carried in: everything was undertrained at 50 epochs (-> 100 here);
# dropout HURTS (-> 0); SAM gave no benefit at 2x cost (-> dropped); cutout lowered
# the score (-> dropped, D4 rotation is always-on instead); label smoothing was
# masking-suspect (-> 0); higher weight decay (1e-3) beat 5e-4 on robustness (a
# stronger lever than beta). This batch combines the winning knobs and adds the
# untested high-upside levers: AdvProp dual-BN and H&E stain jitter.
#
# Common base: resnet50, SGD, dropout 0, label-smoothing 0, D4 rotation (default),
# grad-clip 5, EMA 0.999 (off under dual-bn). Each run ALSO does a strong CE+DLR
# (AutoAttack-style) eval every 10 epochs and writes a best-by-true-robustness
# checkpoint <out>.strong.pt -- so finished runs already carry honest robustness.
#
# Rank after:  python -m scripts.analyze_trends --glob "runlogs/sweep_*.out"
#              python -m scripts.collect_sweep   --glob "checkpoints/*.pt" --arch resnet50
#              python -m scripts.autoattack_eval --glob "checkpoints/*.strong.pt" --arch resnet50
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

EPOCHS="${EPOCHS:-100}"
BASE="--arch resnet50 --epochs ${EPOCHS} --optimizer sgd --dropout 0 --label-smoothing 0 --grad-clip 5.0 --warmup 5 --strong-eval-every 10"

# name | knob args (method / beta / weight-decay / dual-bn / jitter / ema)
CONFIGS=(
  # --- frontier: method x beta, winning knobs (wd 1e-3, EMA on) ---
  "pgd|--method pgd    --weight-decay 1e-3 --ema-decay 0.999"
  "trades_b6|--method trades --beta 6.0  --weight-decay 1e-3 --ema-decay 0.999"
  "trades_b9|--method trades --beta 9.0  --weight-decay 1e-3 --ema-decay 0.999"
  "trades_b12|--method trades --beta 12.0 --weight-decay 1e-3 --ema-decay 0.999"
  "mart|--method mart   --weight-decay 1e-3 --ema-decay 0.999"
  # --- weight-decay lever (it beat beta for robustness) ---
  "trades_b6_wd2e3|--method trades --beta 6.0 --weight-decay 2e-3 --ema-decay 0.999"
  "trades_b6_wd5e4|--method trades --beta 6.0 --weight-decay 5e-4 --ema-decay 0.999"
  "pgd_wd5e4|--method pgd    --weight-decay 5e-4 --ema-decay 0.999"
  "mart_wd2e3|--method mart   --weight-decay 2e-3 --ema-decay 0.999"
  # --- AdvProp dual-BN (highest-upside untested lever; EMA off) ---
  "dbn_trades_b6|--method trades --beta 6.0 --weight-decay 1e-3 --dual-bn --ema-decay 0"
  "dbn_trades_b9|--method trades --beta 9.0 --weight-decay 1e-3 --dual-bn --ema-decay 0"
  "dbn_mart|--method mart   --weight-decay 1e-3 --dual-bn --ema-decay 0"
  # --- H&E stain jitter ---
  "cj_trades_b6|--method trades --beta 6.0 --weight-decay 1e-3 --color-jitter 0.1 --ema-decay 0.999"
  "dbn_cj_b6|--method trades --beta 6.0 --weight-decay 1e-3 --color-jitter 0.1 --dual-bn --ema-decay 0"
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
