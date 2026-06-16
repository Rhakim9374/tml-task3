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
from torch.utils.data import Subset

from src.awp import AWP
from src.data import get_datasets, make_loader
from src.dualbn import convert_to_dual_bn, extract_branch_state_dict
from src.ema import EMA
from src.eval import evaluate_clean, evaluate_robust, unified_score
from src.model import make_model
from src.robust_eval import strong_robust_accuracy
from src.sam import SAM
from src.train import train_epoch


def parse_args():
    p = argparse.ArgumentParser(description="Adversarial training for the robustness task")
    p.add_argument("--data", default="data/train.npz", help="path to train.npz")
    p.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--out", default="checkpoints/model.pt", help="where to save best state dict")

    p.add_argument("--method", default="trades", choices=["pgd", "trades", "mart"])
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw", "sam"],
                   help="sgd/sam use momentum; sam wraps sgd (2x cost); adamw needs a ~100x smaller lr")
    p.add_argument("--rho", type=float, default=0.05, help="SAM neighborhood size")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=None,
                   help="peak LR; default 0.1 for sgd/sam, 1e-3 for adamw")
    p.add_argument("--warmup", type=int, default=5,
                   help="linear LR warmup epochs (resnet50 from scratch needs this to avoid NaN)")
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "piecewise"],
                   help="post-warmup LR schedule; piecewise = x0.1 at 50%/75% (Rice robust-overfit recipe)")
    p.add_argument("--min-lr", type=float, default=0.0,
                   help="cosine LR floor (eta_min); >0 keeps late-training LR alive so plateau vs still-climbing is visible")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4,
                   help="L2 weight decay (~5e-4 is near-optimal for AT; >1e-3 tends to hurt robustness)")
    p.add_argument("--beta", type=float, default=6.0, help="TRADES robustness weight")

    # Generalization knobs.
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dropout prob before fc (0 disables; try 0.1-0.2)")
    p.add_argument("--cutout", type=int, default=0,
                   help="Cutout square size in px (0 disables; try 8-16; pairs with EMA)")
    p.add_argument("--color-jitter", type=float, default=0.0,
                   help="colour jitter strength (0 disables; try 0.1-0.2)")
    p.add_argument("--dual-bn", action="store_true",
                   help="AdvProp dual BatchNorm (clean/adv branches, shared weights); "
                        "use with trades/mart; disables EMA")
    p.add_argument("--init-from", default=None,
                   help="warm-start: load this state dict as the model init (weights only; optimizer/EMA/"
                        "schedule start fresh -- a warm restart, not a true resume)")
    p.add_argument("--grad-clip", type=float, default=5.0,
                   help="clip global grad norm (on by default; 0 disables)")
    p.add_argument("--awp-gamma", type=float, default=0.0,
                   help="AWP weight-perturbation size (0 disables; try 0.005-0.01; PGD-AT/single-BN only)")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="label smoothing on the outer loss (0 disables; keep <=0.1)")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay for weight averaging (0 disables; saved checkpoint uses EMA weights)")

    # Threat model (L-inf, pixel space [0,1]). 8/255 is the CIFAR standard.
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--eps-warmup-epochs", type=int, default=0,
                   help="linearly ramp the training attack eps 0->eps over the first N epochs (0 disables)")
    p.add_argument("--alpha", type=float, default=2 / 255, help="PGD step size")
    p.add_argument("--steps", type=int, default=10, help="inner PGD steps for training")
    p.add_argument("--eval-steps", type=int, default=20, help="PGD steps for robust eval")

    # Strong (AutoAttack-style CE+DLR) eval -- also save best-by-TRUE-robustness.
    p.add_argument("--strong-eval-every", type=int, default=0,
                   help="run strong CE+DLR eval every N epochs (0 disables); also writes <out>.strong.pt")
    p.add_argument("--strong-eval-n", type=int, default=1000, help="val subset size for strong eval")
    p.add_argument("--strong-steps", type=int, default=50)
    p.add_argument("--strong-restarts", type=int, default=1)

    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def build_optimizer(args, params):
    """Construct SGD / AdamW / SAM(SGD) from the parsed args."""
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sam":
        return SAM(params, torch.optim.SGD, rho=args.rho,
                   lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    raise ValueError(f"unknown optimizer {args.optimizer!r}")


@torch.no_grad()
def recompute_bn(model, loader, device, max_batches=50):
    """Recompute BatchNorm running stats for averaged (EMA) weights.

    Averaging weights breaks BN: the running mean/var carried in the EMA shadow
    do not correspond to the averaged weights (BN stats are nonlinear in the
    weights), which makes the EMA model collapse to constant outputs. We reset
    the stats and re-estimate them with a forward pass over clean training images
    (SWA-style), matching the clean distribution the model is evaluated on.
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    saved_momentum = {}
    for bn in bns:
        bn.reset_running_stats()
        saved_momentum[bn] = bn.momentum
        bn.momentum = None  # cumulative moving average over the passes
    model.train()
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        model(x.to(device, non_blocking=True))
    for bn in bns:
        bn.momentum = saved_momentum[bn]
    model.eval()


def main():
    args = parse_args()
    if args.lr is None:  # per-optimizer default LR
        args.lr = 1e-3 if args.optimizer == "adamw" else 0.1
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True  # fixed input size -> free conv autotuning speedup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  args={vars(args)}", flush=True)

    train_ds, val_ds = get_datasets(args.data, val_frac=args.val_frac, seed=args.seed)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers)
    print(f"train={len(train_ds)}  val={len(val_ds)}", flush=True)

    model = make_model(args.arch, dropout=args.dropout)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu"), strict=True)
        print(f"warm-start: loaded weights from {args.init_from}", flush=True)
    if args.dual_bn:
        convert_to_dual_bn(model)  # adds the adv BN branch (clean branch == original)
    model = model.to(device)
    optimizer = build_optimizer(args, model.parameters())
    # Linear warmup then the chosen post-warmup schedule. Warmup is essential for
    # resnet50 from scratch: a high peak LR on epoch 1 otherwise diverges to NaN.
    # Cosine anneals smoothly to 0; piecewise holds the peak LR then drops x0.1 at
    # 50%/75% -- the Rice et al. recipe, whose best robust checkpoint lands just
    # after the first drop (best-by-score saving captures it as the early stop).
    warmup_epochs = min(args.warmup, max(0, args.epochs - 1))
    post = args.epochs - warmup_epochs
    if args.lr_schedule == "piecewise":
        main_sched = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(0.5 * post), int(0.75 * post)], gamma=0.1)
    else:
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=post, eta_min=args.min_lr)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, main_sched], milestones=[warmup_epochs])
    else:
        scheduler = main_sched

    # Weight averaging. We evaluate BOTH the live and EMA models each time and keep
    # whichever scores higher. The EMA is started only AFTER warmup, snapshotted
    # from the post-warmup weights: averaging in the chaotic random-init/warmup
    # weights lands in a dead region of weight space (degenerate, constant-output
    # model), so we anchor the average to already-sensible weights.
    # Dual-BN evaluates its extracted clean/adv branches instead of live/EMA, so
    # EMA is disabled there (one BN-averaging mechanism at a time).
    use_ema = bool(args.ema_decay and args.ema_decay > 0) and not args.dual_bn
    eval_model = make_model(args.arch).to(device) if (use_ema or args.dual_bn) else None
    ema = None

    # Adversarial Weight Perturbation: flattens the weight-loss landscape to curb
    # robust overfitting. Started after warmup (early weights are too unstable to
    # ascend meaningfully); the CE ascent assumes single-BN PGD-AT.
    if args.awp_gamma and args.awp_gamma > 0 and (args.dual_bn or args.optimizer == "sam"):
        raise SystemExit("--awp-gamma is only wired for single-BN SGD/AdamW (not dual-bn/sam)")
    awp = AWP(model, gamma=args.awp_gamma) if (args.awp_gamma and args.awp_gamma > 0) else None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_score = -1.0

    # Strong (CE+DLR, AutoAttack-style) eval: keep a SEPARATE best-by-true-
    # robustness checkpoint so we never select a gradient-masked model.
    root, ext = os.path.splitext(args.out)
    strong_out = root + ".strong" + ext
    best_strong = -1.0
    strong_loader = None
    if args.strong_eval_every > 0:
        n = min(args.strong_eval_n, len(val_ds))
        strong_loader = make_loader(Subset(val_ds, list(range(n))), args.batch_size, num_workers=args.workers)

    def eval_and_maybe_save(name, m, do_strong=False):
        nonlocal best_score, best_strong
        clean = evaluate_clean(m, val_loader, device)
        robust = evaluate_robust(
            m, val_loader, device, eps=args.eps, alpha=args.alpha, steps=args.eval_steps
        )
        score = unified_score(clean, robust)
        marker = ""
        if clean > 0.50 and score > best_score:  # respect the >50% clean gate
            best_score = score
            torch.save(m.state_dict(), args.out)
            marker = "  *saved*"
        line = f"    val[{name}]: clean={clean:.4f} robust={robust:.4f} score={score:.4f}{marker}"
        if do_strong:
            s_rob = strong_robust_accuracy(
                m, strong_loader, device, eps=args.eps,
                steps=args.strong_steps, restarts=args.strong_restarts,
            )
            s_score = unified_score(clean, s_rob)
            smark = ""
            if clean > 0.50 and s_score > best_strong:
                best_strong = s_score
                torch.save(m.state_dict(), strong_out)
                smark = "  *saved-strong*"
            line += f"  strong_robust={s_rob:.4f} strong_score={s_score:.4f}{smark}"
        print(line, flush=True)

    for epoch in range(1, args.epochs + 1):
        # Optional eps ramp: weak attacks early stabilize training and lift the
        # clean/robust balance; alpha scales with eps so the step ratio is fixed.
        if args.eps_warmup_epochs and epoch <= args.eps_warmup_epochs:
            frac = epoch / args.eps_warmup_epochs
            eps_t, alpha_t = args.eps * frac, args.alpha * frac
        else:
            eps_t, alpha_t = args.eps, args.alpha
        loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            method=args.method, eps=eps_t, alpha=alpha_t,
            steps=args.steps, beta=args.beta, cutout=args.cutout, jitter=args.color_jitter,
            grad_clip=args.grad_clip, label_smoothing=args.label_smoothing, ema=ema,
            awp=(awp if epoch > warmup_epochs else None),
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"[epoch {epoch:3d}] loss={loss:.4f} train_acc={train_acc:.4f} lr={lr:.4f}", flush=True)

        # Begin EMA accumulation once warmup is done (anchored to current weights).
        if use_ema and ema is None and epoch >= warmup_epochs:
            ema = EMA(model, args.ema_decay)

        is_last = epoch == args.epochs
        if epoch % args.eval_every == 0 or is_last:
            do_strong = args.strong_eval_every > 0 and (epoch % args.strong_eval_every == 0 or is_last)
            if args.dual_bn:
                # Extract each BN branch into a stock model and keep the better.
                # Clean branch: re-estimate BN on clean data (matches clean test).
                # Adv branch: keep its adversarial stats (consistent with its affine).
                for branch in ("clean", "adv"):
                    eval_model.load_state_dict(extract_branch_state_dict(model.state_dict(), branch))
                    if branch == "clean":
                        recompute_bn(eval_model, train_loader, device)
                    eval_and_maybe_save(f"dbn-{branch}", eval_model, do_strong=do_strong)
            else:
                eval_and_maybe_save("live", model, do_strong=do_strong)
                if ema is not None:
                    ema.copy_to(eval_model)
                    recompute_bn(eval_model, train_loader, device)  # fix BN stats for averaged weights
                    eval_and_maybe_save("ema", eval_model, do_strong=do_strong)
            model.train()  # evaluate_* left the model in eval mode

    print(f"done. best score={best_score:.4f} ({args.out})", flush=True)
    if args.strong_eval_every > 0:
        print(f"      best strong score={best_strong:.4f} ({strong_out})", flush=True)


if __name__ == "__main__":
    main()
