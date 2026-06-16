#!/usr/bin/env bash
# Reproduce our best leaderboard model (public score 0.6273): PGD-AT resnet50.
#
# Recipe (the winning configuration from our sweeps):
#   - PGD adversarial training (Madry et al.), L-inf eps = 8/255, 7 inner steps
#   - resnet50, trained from scratch on the provided 50k (no external data)
#   - 300 epochs, SGD + momentum, cosine LR from peak 0.05 -> 0 after 5-ep warmup
#   - weight decay 1e-3, EMA 0.999, dropout 0.05, gradient clipping
#   - always-on D4 augmentation (flip + 90-deg rotation), suited to the
#     orientation-invariant histopathology images
#
# Trains on the provided 50k only; the best-by-score checkpoint is selected on a
# fixed 10% held-out validation split and saved to checkpoints/best.pt -- already
# in submission format (a plain resnet50 state dict, fc -> 9). One GPU job.
#
# Direct (cluster-agnostic) equivalent of the command submitted below:
#   python -m scripts.train --data data/train.npz --arch resnet50 --method pgd \
#     --epochs 300 --lr 0.05 --warmup 5 --weight-decay 1e-3 --ema-decay 0.999 \
#     --steps 7 --dropout 0.05 --grad-clip 5.0 --seed 1 --out checkpoints/best.pt
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(cd "$SCRIPT_DIR/.." && pwd -P)"

ARGS="--arch resnet50 --method pgd --epochs 300 --optimizer sgd --lr 0.05 --warmup 5 \
--weight-decay 1e-3 --ema-decay 0.999 --steps 7 --dropout 0.05 --grad-clip 5.0 \
--seed 1 --strong-eval-every 10 --out checkpoints/best.pt"

condor_submit cluster/train.sub -append "args=${ARGS}" -append "tag=best"
echo "submitted. best state dict -> checkpoints/best.pt (+ checkpoints/best.strong.pt)"
