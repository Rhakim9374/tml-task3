"""Adversarial training objectives (PGD-AT, TRADES) and the epoch loop.

Both objectives are L-inf, eps in [0,1] pixel space. The unified score weights
clean and robust accuracy equally, so TRADES is offered alongside PGD-AT: its
beta knob trades clean for robust accuracy.

The update supports SGD, AdamW, and SAM. For each batch we generate the
adversarial inputs ONCE at the current weights, then evaluate the outer loss via
a re-runnable closure -- SGD/AdamW call it once; SAM calls it twice (at ``w`` and
at the ascended point ``w + e(w)``) for its sharpness-aware step.

References:
  Madry et al., ICLR 2018 (PGD-AT).
  Zhang et al., ICML 2019 (TRADES).
  Foret et al., ICLR 2021 (SAM).
"""

import torch
import torch.nn.functional as F

from .attacks import pgd_linf
from .data import augment
from .sam import SAM, disable_running_stats, enable_running_stats


def trades_attack(model, x, eps, alpha, steps):
    """Inner TRADES maximization: perturb x to maximize KL(p(x_adv) || p(x))."""
    was_training = model.training
    model.eval()  # keep BN stats fixed during the attack
    p_clean = F.softmax(model(x).detach(), dim=1)

    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        log_p_adv = F.log_softmax(model(x_adv), dim=1)
        kl = F.kl_div(log_p_adv, p_clean, reduction="batchmean")
        (grad,) = torch.autograd.grad(kl, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)

    if was_training:
        model.train()
    return x_adv.detach()


def make_adv(model, x, y, *, method, eps, alpha, steps):
    """Generate the adversarial batch for one step at the current weights.

    PGD-AT and MART use a CE-based PGD attack; TRADES uses its KL-based attack.
    """
    if method in ("pgd", "mart"):
        return pgd_linf(model, x, y, eps=eps, alpha=alpha, steps=steps, random_start=True)
    if method == "trades":
        return trades_attack(model, x, eps, alpha, steps)
    raise ValueError(f"unknown method {method!r} (expected 'pgd', 'trades', or 'mart')")


def loss_closure(model, x, y, x_adv, *, method, beta, label_smoothing):
    """Return a no-arg closure -> (loss, logits) on the fixed ``x_adv`` batch.

    Label smoothing is applied only to the outer classification term, never to
    the inner attack, so adversaries stay sharp.
    """
    if method == "pgd":
        def closure():
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            return loss, logits
        return closure

    if method == "mart":
        # MART (Wang et al. 2020): a misclassification-aware variant. The CE term
        # is boosted by the margin to the most-confident wrong class, and the KL
        # regularizer is up-weighted on examples the clean model is unsure about.
        def closure():
            logits_clean = model(x)
            logits_adv = model(x_adv)
            adv_probs = F.softmax(logits_adv, dim=1)
            top2 = torch.argsort(adv_probs, dim=1)[:, -2:]
            new_y = torch.where(top2[:, -1] == y, top2[:, -2], top2[:, -1])
            loss_adv = F.cross_entropy(logits_adv, y, label_smoothing=label_smoothing) \
                + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
            nat_probs = F.softmax(logits_clean, dim=1)
            true_probs = nat_probs.gather(1, y.unsqueeze(1)).squeeze(1)
            kl = F.kl_div(torch.log(adv_probs + 1e-12), nat_probs, reduction="none").sum(1)
            loss_robust = (kl * (1.0000001 - true_probs)).mean()
            loss = loss_adv + beta * loss_robust
            return loss, logits_adv
        return closure

    # TRADES: CE(clean) + beta * KL(p(x_adv) || p(x)), both terms differentiable.
    def closure():
        logits_clean = model(x)
        log_p_adv = F.log_softmax(model(x_adv), dim=1)
        p_clean = F.softmax(logits_clean, dim=1)
        loss = F.cross_entropy(logits_clean, y, label_smoothing=label_smoothing) \
            + beta * F.kl_div(log_p_adv, p_clean, reduction="batchmean")
        return loss, logits_clean
    return closure


def train_epoch(
    model, loader, optimizer, device, *, method, eps, alpha, steps, beta,
    use_aug=True, cutout=0, grad_clip=0.0, label_smoothing=0.0, ema=None,
):
    """Run one training epoch. Returns (mean_loss, train_accuracy_on_used_logits).

    Works with SGD/AdamW (single step) and SAM (two-step). ``grad_clip`` > 0 clips
    the update gradient; ``ema`` is updated after every optimizer step; ``cutout``
    > 0 erases a random square per sample during augmentation.
    """
    model.train()
    is_sam = isinstance(optimizer, SAM)
    total_loss, total_correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if use_aug:
            x = augment(x, cutout_size=cutout)

        x_adv = make_adv(model, x, y, method=method, eps=eps, alpha=alpha, steps=steps)
        closure = loss_closure(model, x, y, x_adv, method=method, beta=beta, label_smoothing=label_smoothing)

        if is_sam:
            enable_running_stats(model)
            loss, logits = closure()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            disable_running_stats(model)  # don't update BN twice per step
            loss2, _ = closure()
            loss2.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss, logits = closure()
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
