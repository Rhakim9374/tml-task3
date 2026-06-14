"""Evaluate a saved state dict: clean accuracy + PGD robust accuracy + score.

    python -m scripts.evaluate --ckpt checkpoints/model.pt --arch resnet18 --data data/train.npz

Also runs the exact (1,3,32,32) -> (1,9) shape assertions the server checks, so a
green run here means the checkpoint is submission-ready.
"""

import argparse

import torch

from src.data import get_datasets, make_loader
from src.eval import evaluate_clean, evaluate_robust, unified_score
from src.model import make_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint's clean/robust accuracy")
    p.add_argument("--ckpt", required=True, help="path to a state-dict .pt file")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--data", default="data/train.npz")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--alpha", type=float, default=2 / 255)
    p.add_argument("--steps", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = make_model(args.arch).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Submission-format sanity check (mirrors the server's assertions).
    with torch.no_grad():
        out = model(torch.randn(1, 3, 32, 32, device=device))
    assert out.shape == (1, 9), f"output shape {tuple(out.shape)} != (1, 9)"
    print("shape check OK: (1,3,32,32) -> (1,9)")

    _, val_ds = get_datasets(args.data, val_frac=args.val_frac, seed=args.seed)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)

    clean = evaluate_clean(model, val_loader, device)
    robust = evaluate_robust(model, val_loader, device, eps=args.eps, alpha=args.alpha, steps=args.steps)
    print(f"clean ={clean:.4f}")
    print(f"robust={robust:.4f}  (PGD-{args.steps}, eps={args.eps:.4f})")
    print(f"score ={unified_score(clean, robust):.4f}")
    if clean <= 0.50:
        print("WARNING: clean accuracy <= 0.50 -> server would REJECT this submission")


if __name__ == "__main__":
    main()
