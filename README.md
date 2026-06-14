# TML26 Task 3 — Adversarial Robustness

Train a torchvision ResNet (resnet18/34/50) that is robust to L-inf adversarial
attacks. Final score = `0.5 * clean_accuracy + 0.5 * robust_accuracy`; clean
accuracy must exceed 50% or the submission is rejected.

> This README will be replaced with the exact recipe that recreates our best
> leaderboard result once we have one. The steps below are the current workflow.

## Setup (cluster login node)

```bash
ssh <atml_teamXXX>@conduit2.hpc.uni-saarland.de
git clone <this-repo-url> code && cd code

python3 -m venv ~/.tml-venv
~/.tml-venv/bin/pip install -r requirements.txt

bash cluster/fetch_data.sh          # downloads data/train.npz (~127 MB)
```

## Train (GPU job)

Fixed choices: **resnet50** (capacity helps robustness), **TRADES**, EMA weight
averaging and gradient clipping always on. A single run:

```bash
condor_submit cluster/train.sub \
    -append "args=--arch resnet50 --method trades --beta 6.0 --epochs 60 \
                  --grad-clip 1.0 --ema-decay 0.999 --weight-decay 5e-4 \
                  --dropout 0.1 --label-smoothing 0.1 \
                  --out checkpoints/resnet50_trades.pt" \
    -append "tag=trades_b6"
```

`scripts/train.py` saves the best-by-unified-score state dict (the **EMA**
weights), evaluating clean and PGD-20 robust accuracy on a fixed 10% validation
split every few epochs.

## Sweep + finals (≈26 parallel GPU jobs)

Two waves launched together, each job a separate `condor_submit` (1 GPU), so
HTCondor spreads them across the cluster; wall-clock ≈ one job per wave.

**Wave 1 — sweep (20 jobs, `EPOCHS=40`).** Coordinate search around a strong
baseline + a direct SGD-vs-SAM comparison: optimizer (SGD/SAM), objective
(TRADES/PGD-AT/MART), TRADES beta, EMA decay, weight decay, dropout, label
smoothing, Cutout. At 40 epochs the cosine schedule **anneal-completes inside a
7–10 h window**, so these are decision-ready *and* submission-ready.

**Wave 2 — finals (6 jobs, `EPOCHS=100`).** Longer runs of the a-priori
strongest configs, as parallel upside on surplus GPUs.

```bash
EPOCHS=40  bash cluster/launch_sweep.sh    # checkpoints/sweep_*.pt
EPOCHS=100 bash cluster/launch_finals.sh   # checkpoints/final_*.pt
condor_q                                    # confirm Running, not Idle; note epoch time
```

Each job saves its **best-so-far** checkpoint every 5 epochs, so you can rank at
any time — even mid-flight — without waiting for jobs to finish:

```bash
~/.tml-venv/bin/python -m scripts.collect_sweep \
    --glob "checkpoints/sweep_*.pt" --arch resnet50      # or "checkpoints/*.pt"
```

> Timing: check the first epoch's printed wall-clock right after launch. If it's
> far from ~1–3 min/epoch, `condor_rm` and relaunch with an `EPOCHS` that
> anneal-finishes in your window (cosine LR must reach 0 for best robustness).

## Evaluate locally before submitting

```bash
~/.tml-venv/bin/python -m scripts.evaluate \
    --ckpt checkpoints/resnet18_trades.pt --arch resnet18 --data data/train.npz
```

This also runs the `(1,3,32,32) -> (1,9)` shape assertion the server checks.

## Submit

```bash
export TML_API_KEY=<your key>
~/.tml-venv/bin/python -m scripts.submit \
    --file checkpoints/resnet18_trades.pt --model-name resnet18
```

One submission per group every 60 minutes.

## Layout

```
src/
  model.py     make_model(arch): vanilla torchvision ResNet, fc -> 9 classes
  data.py      train.npz loader, fixed val split, on-GPU crop/flip augmentation
  attacks.py   L-inf FGSM / PGD (pixel space [0,1])
  train.py     PGD-AT and TRADES training objectives + epoch loop
  eval.py      clean accuracy, PGD-20 robust accuracy, unified score
scripts/
  train.py     CLI: adversarial training with periodic eval + best-checkpoint save
  evaluate.py  CLI: clean/robust/score for a saved checkpoint (+ shape check)
  submit.py    CLI: POST a .pt state dict to the leaderboard (key from TML_API_KEY)
cluster/
  fetch_data.sh  download train.npz from HuggingFace
  train.sub      HTCondor submit template for one training run
  run_train.sh   in-container entrypoint: pip install + scripts.train
```
