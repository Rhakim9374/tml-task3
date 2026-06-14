# TML26 Task 3 — Adversarial Robustness

Train a torchvision ResNet (resnet18/34/50) that is robust to L-inf adversarial
attacks. Final score = `0.5 * clean_accuracy + 0.5 * robust_accuracy`; clean
accuracy must exceed 50% or the submission is rejected.

> This README will be replaced with the exact recipe that recreates our best
> leaderboard result once we have one. The steps below are the current workflow.

## Setup (login node — download only, no compute)

```bash
ssh <atml_teamXXX>@conduit2.hpc.uni-saarland.de
git clone <this-repo-url> code && cd code
bash cluster/fetch_data.sh          # downloads data/train.npz (~127 MB)
```

GPU jobs install `requirements.txt` themselves inside the Docker image, so the
login node needs no virtualenv. All compute (training and analysis) runs in
HTCondor jobs.

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

`scripts/train.py` saves the best-by-unified-score state dict (the **EMA**
weights) every 5 epochs, evaluating clean and PGD-20 robust accuracy on a fixed
10% validation split.

> Timing: check the first epoch's printed wall-clock right after launch. If it's
> far from ~1–3 min/epoch, `condor_rm` and relaunch with an `EPOCHS` that
> anneal-finishes in your window (cosine LR must reach 0 for best robustness).

## Analysis (interactive GPU job — not the login node)

Ranking and evaluation run PGD attacks, so do them on a GPU worker:

```bash
condor_submit -i cluster/interactive.sub      # opens a shell on a GPU node
# inside the container:
cd ~/code && pip install -r requirements.txt

# rank all checkpoints by validation unified score (works mid-flight):
python -m scripts.collect_sweep --glob "checkpoints/*.pt" --arch resnet50

# evaluate one checkpoint (also runs the server's (1,3,32,32)->(1,9) shape check):
python -m scripts.evaluate --ckpt checkpoints/sweep_baseline.pt --arch resnet50
```

## Submit

From the interactive job (it only needs `requests`; one submission per group
every 60 minutes):

```bash
export TML_API_KEY=<your key>
python -m scripts.submit --file checkpoints/<best>.pt --model-name resnet50
```

## Layout

```
src/
  model.py     make_model(arch): vanilla torchvision ResNet, fc -> 9 classes
  data.py      train.npz loader, fixed val split, on-GPU crop/flip/cutout aug
  attacks.py   L-inf PGD (pixel space [0,1])
  train.py     PGD-AT / TRADES / MART objectives + epoch loop (SGD/AdamW/SAM)
  eval.py      clean accuracy, PGD-20 robust accuracy, unified score
  ema.py       EMA weight averaging
  sam.py       Sharpness-Aware Minimization optimizer
scripts/
  train.py         CLI: adversarial training with periodic eval + best-checkpoint save
  evaluate.py      CLI: clean/robust/score for a checkpoint (+ shape check)
  collect_sweep.py CLI: rank checkpoints by validation unified score
  submit.py        CLI: POST a .pt state dict to the leaderboard (key from TML_API_KEY)
cluster/
  fetch_data.sh    download train.npz from HuggingFace
  train.sub        HTCondor submit template for one training run
  run_train.sh     in-container entrypoint: pip install + scripts.train
  launch_sweep.sh  submit the 20-job sweep
  launch_finals.sh submit the 6-job finals wave
  interactive.sub  interactive GPU job for analysis/submission
```
