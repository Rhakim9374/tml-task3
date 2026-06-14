"""Adversarial training objectives: PGD-AT (Madry) and TRADES.

Both are L-inf, eps in [0,1] pixel space. The unified score weights clean and
robust accuracy equally, so TRADES is offered alongside vanilla PGD-AT: its beta
knob trades clean accuracy for robustness and lets us tune toward the best
combined score.

References:
  Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018.
  Zhang et al., "Theoretically Principled Trade-off between Robustness and Accuracy", ICML 2019.
"""

import torch
import torch.nn.functional as F

from .attacks import pgd_linf
from .data import augment


def pgd_at_loss(model, x, y, eps, alpha, steps, label_smoothing=0.0):
    """Standard PGD adversarial training: cross-entropy on the adversarial input.

    Label smoothing is applied only to this outer classification loss, never to
    the inner attack (which keeps a sharp, un-smoothed CE so it still generates
    strong adversaries).
    """
    x_adv = pgd_linf(model, x, y, eps=eps, alpha=alpha, steps=steps, random_start=True)
    logits_adv = model(x_adv)
    loss = F.cross_entropy(logits_adv, y, label_smoothing=label_smoothing)
    return loss, logits_adv


def trades_loss(model, x, y, eps, alpha, steps, beta, label_smoothing=0.0):
    """TRADES: CE on clean + beta * KL(p(x_adv) || p(x)).

    The inner maximization perturbs x to maximize the KL divergence from the
    clean prediction (a label-free robustness surrogate), keeping clean accuracy
    higher than vanilla PGD-AT at the same eps.
    """
    model.eval()  # keep BN stats stable during the inner attack
    logits_clean = model(x)
    p_clean = F.softmax(logits_clean.detach(), dim=1)

    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        log_p_adv = F.log_softmax(model(x_adv), dim=1)
        kl = F.kl_div(log_p_adv, p_clean, reduction="batchmean")
        (grad,) = torch.autograd.grad(kl, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)

    model.train()
    logits_clean = model(x)
    log_p_adv = F.log_softmax(model(x_adv.detach()), dim=1)
    p_clean = F.softmax(logits_clean, dim=1)
    loss_natural = F.cross_entropy(logits_clean, y, label_smoothing=label_smoothing)
    loss_robust = F.kl_div(log_p_adv, p_clean, reduction="batchmean")
    loss = loss_natural + beta * loss_robust
    return loss, logits_clean


def train_epoch(
    model, loader, optimizer, device, *, method, eps, alpha, steps, beta,
    use_aug=True, grad_clip=0.0, label_smoothing=0.0, ema=None,
):
    """Run one training epoch. Returns (mean_loss, train_accuracy_on_used_logits).

    ``grad_clip`` > 0 clips the global grad norm before each step; ``ema`` (an
    :class:`src.ema.EMA`) is updated after every optimizer step.
    """
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if use_aug:
            x = augment(x)

        optimizer.zero_grad(set_to_none=True)
        if method == "trades":
            loss, logits = trades_loss(model, x, y, eps, alpha, steps, beta, label_smoothing)
        elif method == "pgd":
            loss, logits = pgd_at_loss(model, x, y, eps, alpha, steps, label_smoothing)
        else:
            raise ValueError(f"unknown method {method!r} (expected 'pgd' or 'trades')")

        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, total_correct / total
