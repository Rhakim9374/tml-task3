"""Reliable robustness evaluation, stronger than vanilla PGD-CE.

Vanilla PGD-CE overestimates robustness under gradient masking (common with
TRADES + label smoothing). This runs multi-restart PGD with both cross-entropy
and the scale-invariant DLR loss -- the white-box core of AutoAttack's APGD --
and counts an example robust only if it survives every attack. The CE-vs-DLR and
multi-restart structure exposes most masking that plain PGD misses, so it serves
as our dependency-free AutoAttack proxy for honest model selection.
"""

import torch
import torch.nn.functional as F


def dlr_loss(logits, y):
    """Difference-of-Logits-Ratio loss (Croce & Hein); higher = more misclassified."""
    z, idx = logits.sort(dim=1, descending=True)
    z_y = logits.gather(1, y[:, None]).squeeze(1)
    top1_is_y = idx[:, 0] == y
    z_other = torch.where(top1_is_y, z[:, 1], z[:, 0])  # best non-true logit
    return -(z_y - z_other) / (z[:, 0] - z[:, 2] + 1e-12)


def _pgd(model, x, y, eps, alpha, steps, loss_name):
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y) if loss_name == "ce" else dlr_loss(logits, y).mean()
        (grad,) = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)
    return x_adv.detach()


@torch.no_grad()
def _hits(model, x, y):
    return model(x).argmax(1) == y


def strong_robust_accuracy(model, loader, device, *, eps=8 / 255, steps=50, restarts=2):
    """Worst-case accuracy over multi-restart CE-PGD and DLR-PGD.

    An example counts as robust only if it is classified correctly clean AND
    survives every attack/restart. ``steps``=50 with 2 restarts of each loss is a
    solid reliability check; raise for a stricter audit.
    """
    model.eval()
    alpha = eps / 4.0
    total, robust = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        surv = _hits(model, x, y).clone()  # must be clean-correct to begin
        for loss_name in ("ce", "dlr"):
            for _ in range(restarts):
                if not surv.any():
                    break
                x_adv = _pgd(model, x, y, eps, alpha, steps, loss_name)
                surv &= _hits(model, x_adv, y)
        robust += surv.sum().item()
        total += y.size(0)
    return robust / total
