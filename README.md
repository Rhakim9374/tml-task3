# TML26 Task 3 — Adversarial Robustness

Train a torchvision ResNet (resnet18/34/50) robust to L-inf adversarial attacks.
Score = `0.5 * clean_accuracy + 0.5 * robust_accuracy`; clean accuracy must exceed
50% or the submission is rejected.

This README explains **how to recreate our best leaderboard result** (public score
**0.6273**): a `resnet50` trained with PGD adversarial training on the provided 50k.

## Best model — the recipe

PGD adversarial training (Madry et al.), L-inf `eps = 8/255`, `resnet50` from
scratch on the provided 50k:

| | |
|---|---|
| architecture | `resnet50` (stock torchvision, `fc` → 9 classes, inputs in [0,1]) |
| objective | PGD-AT, `eps = 8/255`, `alpha = 2/255`, 7 inner steps |
| schedule | 300 epochs, SGD + momentum, cosine LR `0.05 → 0` after 5-epoch warmup |
| regularization | weight decay `1e-3`, EMA `0.999`, dropout `0.05`, grad-clip `5.0` |
| augmentation | always-on D4 (random flip + 90° rotation) — the images are orientation-invariant histopathology |
| selection | best-by-score checkpoint on a fixed 10% held-out validation split |

Trained on the provided 50k only (no external data, no pretrained weights).

## Recreate it

```bash
git clone <this-repo-url> && cd tml-task3
bash cluster/fetch_data.sh                 # downloads data/train.npz (~127 MB)
```

Train (one GPU; the best state dict is written to `checkpoints/best.pt`):

```bash
# cluster-agnostic (any machine with a GPU):
python -m scripts.train --data data/train.npz --arch resnet50 --method pgd \
  --epochs 300 --lr 0.05 --warmup 5 --weight-decay 1e-3 --ema-decay 0.999 \
  --steps 7 --dropout 0.05 --grad-clip 5.0 --seed 1 --out checkpoints/best.pt

# or, on the HTCondor cluster:
bash cluster/launch_best.sh
```

Verify it is submission-ready (runs the server's `(1,3,32,32) → (1,9)` shape
assertions plus local clean/robust accuracy):

```bash
python -m scripts.evaluate --ckpt checkpoints/best.pt --arch resnet50
```

Submit (`.pt` state dict + architecture name; one submission per group per hour):

```bash
export TML_API_KEY=<your key>
python -m scripts.submit --file checkpoints/best.pt --model-name resnet50
```

`checkpoints/best.pt` is already in submission format — a plain `resnet50` state
dict — so the server loads it directly into its stock model.

## Repository layout

```
src/
  model.py        stock torchvision ResNet, fc -> 9, optional functional dropout
  data.py         npz loader, val split, on-GPU D4/crop/jitter/cutout augmentation
  attacks.py      L-inf PGD (Madry), [0,1] pixel space
  train.py        PGD-AT / TRADES / MART objectives + epoch loop (SGD/AdamW/SAM)
  eval.py         clean accuracy, PGD-20 robust accuracy, unified score
  robust_eval.py  strong multi-restart CE+DLR PGD (dependency-free AutoAttack proxy)
  ema.py          EMA weight averaging      awp.py     adversarial weight perturbation
  sam.py          sharpness-aware minimizer dualbn.py  AdvProp dual BatchNorm
scripts/
  train.py            CLI: adversarial training with periodic eval + best-checkpoint save
  evaluate.py         CLI: clean/robust/score + submission shape check for one checkpoint
  rank_robust.py      CLI: rank checkpoints by worst-case of PGD-20 and the strong attack
  analyze_trends.py   CLI: per-run eval trajectories (both attacks) from training logs
  eval_on_provided.py CLI: score checkpoints on the provided 50k (test-preprocessing reference)
  submit.py           CLI: POST a .pt state dict to the leaderboard (key from TML_API_KEY)
cluster/
  fetch_data.sh / fetch_pathmnist.sh   download the provided 50k / PathMNIST (ablation)
  train.sub / run_train.sh             HTCondor template + in-container entrypoint
  launch_best.sh                       reproduce the best model (this recipe)
  launch_sweep.sh                      the seed + dropout sweep that found it
  interactive.sub                      interactive GPU job for analysis/submission
```

`src/train.py` exposes every objective and technique we explored (PGD-AT, TRADES,
MART, SAM, AdvProp dual-BN, AWP, EMA, eps-warmup, an external-data option) behind
CLI flags, so each experiment in the report is reproducible by setting flags.
