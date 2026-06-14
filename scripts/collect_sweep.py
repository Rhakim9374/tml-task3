"""Rank sweep checkpoints by unified score on the fixed validation split.

Loads every checkpoint matching ``--glob``, evaluates clean + PGD robust
accuracy on the same held-out split used during training, and prints a table
sorted by unified score (best first) so we can pick the config to scale up and
submit.

    python -m scripts.collect_sweep --glob "checkpoints/sweep_*.pt" --arch resnet50
"""

import argparse
import glob
import os

import torch

from src.data import get_datasets, make_loader
from src.eval import evaluate_clean, evaluate_robust, unified_score
from src.model import make_model


def parse_args():
    p = argparse.ArgumentParser(description="Rank sweep checkpoints by unified score")
    p.add_argument("--glob", default="checkpoints/sweep_*.pt", help="checkpoint glob")
    p.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
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

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no checkpoints match {args.glob!r}")

    _, val_ds = get_datasets(args.data, val_frac=args.val_frac, seed=args.seed)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)
    model = make_model(args.arch).to(device)

    rows = []
    for path in paths:
        model.load_state_dict(torch.load(path, map_location=device), strict=True)
        model.eval()
        clean = evaluate_clean(model, val_loader, device)
        robust = evaluate_robust(model, val_loader, device, eps=args.eps, alpha=args.alpha, steps=args.steps)
        score = unified_score(clean, robust)
        rows.append((os.path.basename(path), clean, robust, score))
        print(f"  scored {os.path.basename(path)}: clean={clean:.4f} robust={robust:.4f} score={score:.4f}", flush=True)

    rows.sort(key=lambda r: r[3], reverse=True)
    print("\n=== ranked by unified score (PGD-{} eval, eps={:.4f}) ===".format(args.steps, args.eps))
    print(f"{'checkpoint':<28}{'clean':>9}{'robust':>9}{'score':>9}")
    for name, clean, robust, score in rows:
        flag = "  <- REJECT (clean<=.50)" if clean <= 0.50 else ""
        print(f"{name:<28}{clean:>9.4f}{robust:>9.4f}{score:>9.4f}{flag}")
    print(f"\nbest: {rows[0][0]}  (score={rows[0][3]:.4f})")


if __name__ == "__main__":
    main()
