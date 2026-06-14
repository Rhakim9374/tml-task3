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

```bash
condor_submit cluster/train.sub \
    -append "args=--arch resnet18 --method trades --beta 6.0 --epochs 60 \
                  --out checkpoints/resnet18_trades.pt" \
    -append "tag=trades_b6"
```

`scripts/train.py` saves the best-by-unified-score state dict, evaluating clean
and PGD-20 robust accuracy on a fixed 10% validation split every few epochs.

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
