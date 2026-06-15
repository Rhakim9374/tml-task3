"""Evaluate swept checkpoints on the PROVIDED 50k -- the task's own distribution.

A reference cross-check to ``rank_robust --extra-data`` (which scores on the clean,
leakage-free PathMNIST val resized with OUR bilinear). The provided 50k carries the
assignment's exact preprocessing -- the closest proxy we have for the hidden test's
preprocessing -- so this is the most faithful clean/robust read on the task's own
pixels. Reports both attacks (PGD-20 and strong CE+DLR), same as rank_robust.

CAVEAT -- contamination: batch-4 models trained on the full 50k (internal) or the
90k PathMNIST superset that contains it (external), so they have SEEN these images.
CLEAN accuracy here is therefore an optimistic upper bound, NOT a generalization
estimate. Robust accuracy is still informative (robustness is not conferred by clean
memorization), and a large clean(provided) - clean(pathmnist-val) gap quantifies how
much is memorization / preprocessing shift. Read this ALONGSIDE rank_robust
--extra-data and the leaderboard, never instead of them.

    python -m scripts.eval_on_provided --glob "checkpoints/sweep_*.pt" --arch resnet50
    python -m scripts.eval_on_provided --glob "checkpoints/sweep_*.pt" --arch resnet50 --n-samples 0  # all 50k
"""

import argparse
import glob
import os

import torch
from torch.utils.data import Subset, TensorDataset

from src.data import load_npz, make_loader
from src.eval import evaluate_clean, evaluate_robust, unified_score
from src.model import make_model
from src.robust_eval import strong_robust_accuracy


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate checkpoints on the provided 50k (both attacks)")
    p.add_argument("--glob", default="checkpoints/sweep_*.pt", help="checkpoint glob")
    p.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--data", default="data/train.npz", help="the provided 50k npz")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--alpha", type=float, default=2 / 255)
    p.add_argument("--pgd-steps", type=int, default=20, help="steps for the PGD-20 (leaderboard-proxy) attack")
    p.add_argument("--strong-steps", type=int, default=50, help="steps for the strong CE+DLR attack")
    p.add_argument("--strong-restarts", type=int, default=2)
    p.add_argument("--n-samples", type=int, default=2000,
                   help="cap eval samples (adversarial eval is expensive over many ckpts); 0 = all 50k")
    p.add_argument("--sort", default="min", choices=["min", "mean", "pgd", "strong", "clean"],
                   help="ranking key (default min = worst-case across the two attacks)")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no checkpoints match {args.glob!r}")

    imgs, labels = load_npz(args.data)
    ds = TensorDataset(imgs, labels)
    if args.n_samples and args.n_samples < len(ds):
        ds = Subset(ds, list(range(args.n_samples)))
    loader = make_loader(ds, args.batch_size, shuffle=False)
    model = make_model(args.arch).to(device)

    rows = []
    for path in paths:
        name = os.path.basename(path)
        try:  # jobs may still be mid-write
            model.load_state_dict(torch.load(path, map_location=device), strict=True)
        except Exception as e:
            print(f"  skip {name}: not readable yet ({type(e).__name__})", flush=True)
            continue
        model.eval()
        clean = evaluate_clean(model, loader, device)
        pgd_rob = evaluate_robust(model, loader, device, eps=args.eps, alpha=args.alpha, steps=args.pgd_steps)
        strong_rob = strong_robust_accuracy(
            model, loader, device, eps=args.eps, steps=args.strong_steps, restarts=args.strong_restarts)
        pgd_score, strong_score = unified_score(clean, pgd_rob), unified_score(clean, strong_rob)
        worst, avg = min(pgd_score, strong_score), 0.5 * (pgd_score + strong_score)
        rows.append((name, clean, pgd_rob, strong_rob, pgd_score, strong_score, pgd_score - strong_score, worst, avg))
        print(f"  scored {name}: clean={clean:.4f} pgd={pgd_rob:.4f} strong={strong_rob:.4f} "
              f"pgd_score={pgd_score:.4f} strong_score={strong_score:.4f} gap={pgd_score - strong_score:+.4f}",
              flush=True)

    if not rows:
        raise SystemExit("no readable checkpoints yet -- wait for the first evals to be saved")

    key = {"min": 7, "mean": 8, "pgd": 4, "strong": 5, "clean": 1}[args.sort]
    rows.sort(key=lambda r: r[key], reverse=True)
    n = len(ds)
    print(f"\n=== PROVIDED 50k (contaminated for batch-4 -- clean is an upper bound), "
          f"ranked by {args.sort} (PGD-{args.pgd_steps} vs strong CE+DLR x{args.strong_restarts}, "
          f"n={n}, eps={args.eps:.4f}) ===")
    print(f"{'checkpoint':<30}{'clean':>8}{'pgdRob':>8}{'strRob':>8}{'pgdScr':>8}{'strScr':>8}{'gap':>8}{'min':>8}")
    for name, clean, pr, sr, ps, ss, gap, worst, avg in rows:
        flags = "  REJECT(clean<=.50)" if clean <= 0.50 else ("  MASKED?" if gap > 0.04 else "")
        print(f"{name:<30}{clean:>8.4f}{pr:>8.4f}{sr:>8.4f}{ps:>8.4f}{ss:>8.4f}{gap:>+8.4f}{worst:>8.4f}{flags}")
    best = rows[0]
    print(f"\nbest by {args.sort}: {best[0]}  (min={best[7]:.4f}  pgd_score={best[4]:.4f}  strong_score={best[5]:.4f})")
    print("reminder: clean is contaminated for batch-4 (trained on this data). Cross-read with "
          "`rank_robust --extra-data data/pathmnist.npz` (clean val) and the leaderboard.")


if __name__ == "__main__":
    main()
