"""Train a robust ResNet via PGD-AT or TRADES, with periodic clean/robust eval.

Run from the repo root, e.g.:
    python -m scripts.train --data data/train.npz --arch resnet18 \
        --method trades --beta 6.0 --epochs 60 --out checkpoints/resnet18_trades.pt

Saves the best-by-unified-score state dict to ``--out`` (a plain .pt state dict,
exactly the submission format). Always validates on a fixed held-out split so the
reported numbers track the leaderboard's clean/robust tradeoff.
"""

import argparse
import os

import torch

from src.data import get_datasets, make_loader
from src.ema import EMA
from src.eval import evaluate_clean, evaluate_robust, unified_score
from src.model import make_model
from src.train import train_epoch


def parse_args():
    p = argparse.ArgumentParser(description="Adversarial training for the robustness task")
    p.add_argument("--data", default="data/train.npz", help="path to train.npz")
    p.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--out", default="checkpoints/model.pt", help="where to save best state dict")

    p.add_argument("--method", default="trades", choices=["pgd", "trades"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4,
                   help="L2 weight decay (~5e-4 is near-optimal for AT; >1e-3 tends to hurt robustness)")
    p.add_argument("--beta", type=float, default=6.0, help="TRADES robustness weight")

    # Generalization knobs.
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dropout prob before fc (0 disables; try 0.1-0.2)")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="clip global grad norm (on by default; 0 disables; raise to ~5.0 if it suppresses learning)")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="label smoothing on the outer loss (0 disables; keep <=0.1)")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay for weight averaging (0 disables; saved checkpoint uses EMA weights)")

    # Threat model (L-inf, pixel space [0,1]). 8/255 is the CIFAR standard.
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--alpha", type=float, default=2 / 255, help="PGD step size")
    p.add_argument("--steps", type=int, default=10, help="inner PGD steps for training")
    p.add_argument("--eval-steps", type=int, default=20, help="PGD steps for robust eval")

    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  args={vars(args)}", flush=True)

    train_ds, val_ds = get_datasets(args.data, val_frac=args.val_frac, seed=args.seed)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers)
    print(f"train={len(train_ds)}  val={len(val_ds)}", flush=True)

    model = make_model(args.arch, dropout=args.dropout).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Weight averaging: maintain EMA weights and evaluate/submit those (they
    # generalize better and resist robust overfitting). The eval model holds
    # whichever weights we score — EMA if enabled, else the live model's.
    ema = EMA(model, args.ema_decay) if args.ema_decay and args.ema_decay > 0 else None
    eval_model = make_model(args.arch).to(device) if ema is not None else model

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            method=args.method, eps=args.eps, alpha=args.alpha,
            steps=args.steps, beta=args.beta,
            grad_clip=args.grad_clip, label_smoothing=args.label_smoothing, ema=ema,
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"[epoch {epoch:3d}] loss={loss:.4f} train_acc={train_acc:.4f} lr={lr:.4f}", flush=True)

        is_last = epoch == args.epochs
        if epoch % args.eval_every == 0 or is_last:
            if ema is not None:
                ema.copy_to(eval_model)
            clean = evaluate_clean(eval_model, val_loader, device)
            robust = evaluate_robust(
                eval_model, val_loader, device, eps=args.eps, alpha=args.alpha, steps=args.eval_steps
            )
            score = unified_score(clean, robust)
            print(
                f"    val: clean={clean:.4f} robust={robust:.4f} score={score:.4f}"
                f"  (best={best_score:.4f})",
                flush=True,
            )
            if score > best_score and clean > 0.50:  # respect the >50% clean gate
                best_score = score
                torch.save(eval_model.state_dict(), args.out)
                print(f"    saved new best (score={score:.4f}) -> {args.out}", flush=True)

    print(f"done. best unified score={best_score:.4f}  saved at {args.out}", flush=True)


if __name__ == "__main__":
    main()
