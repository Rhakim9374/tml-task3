"""Rank checkpoints by TRUE robustness (strong attack), not inflated PGD.

`scripts.collect_sweep` ranks fast with PGD-20, which can overestimate robustness
under gradient masking (TRADES + label smoothing are prone to it). Use this on
the top sweep candidates to pick the genuinely robust one before submitting.

    # dependency-free strong eval (CE+DLR multi-restart PGD) on a val subset:
    python -m scripts.autoattack_eval --glob "checkpoints/*.pt" --arch resnet50 --n-samples 1000

    # full official AutoAttack on one checkpoint (pip install git+.../auto-attack):
    python -m scripts.autoattack_eval --ckpt checkpoints/final.pt --arch resnet50 --autoattack

A large gap between collect_sweep's PGD robust acc and this strong robust acc is a
gradient-masking red flag -- prefer the model with the higher *strong* score.
"""

import argparse
import glob
import os

import torch

from src.data import get_datasets, make_loader
from src.eval import evaluate_clean, unified_score
from src.model import make_model
from src.robust_eval import autoattack_accuracy, strong_robust_accuracy


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate true robustness of checkpoints")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="single checkpoint")
    g.add_argument("--glob", help="glob of checkpoints to rank")
    p.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--data", default="data/train.npz")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--n-samples", type=int, default=1000,
                   help="cap val samples (strong eval is expensive); 0 = all")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--autoattack", action="store_true", help="use official AutoAttack")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_ds = get_datasets(args.data, val_frac=args.val_frac, seed=args.seed)
    if args.n_samples and args.n_samples < len(val_ds):
        val_ds = torch.utils.data.Subset(val_ds, list(range(args.n_samples)))
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)
    model = make_model(args.arch).to(device)

    paths = [args.ckpt] if args.ckpt else sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit("no checkpoints found")

    rows = []
    for path in paths:
        try:
            model.load_state_dict(torch.load(path, map_location=device), strict=True)
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {type(e).__name__}", flush=True)
            continue
        model.eval()
        clean = evaluate_clean(model, val_loader, device)

        if args.autoattack:
            xs = torch.cat([x for x, _ in val_loader]).to(device)
            ys = torch.cat([y for _, y in val_loader]).to(device)
            robust = autoattack_accuracy(model, xs, ys, eps=args.eps, bs=args.batch_size)
        else:
            robust = strong_robust_accuracy(
                model, val_loader, device, eps=args.eps, steps=args.steps, restarts=args.restarts
            )
        score = unified_score(clean, robust)
        rows.append((os.path.basename(path), clean, robust, score))
        print(f"  {os.path.basename(path)}: clean={clean:.4f} robust={robust:.4f} score={score:.4f}", flush=True)

    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"\n=== ranked by TRUE robust score (n={len(val_ds)}, eps={args.eps:.4f}) ===")
    print(f"{'checkpoint':<28}{'clean':>9}{'robust':>9}{'score':>9}")
    for name, clean, robust, score in rows:
        flag = "  <- REJECT (clean<=.50)" if clean <= 0.50 else ""
        print(f"{name:<28}{clean:>9.4f}{robust:>9.4f}{score:>9.4f}{flag}")
    if rows:
        print(f"\nbest: {rows[0][0]}  (score={rows[0][3]:.4f})")


if __name__ == "__main__":
    main()
