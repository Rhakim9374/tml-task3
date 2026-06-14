"""L-infinity PGD attack (Madry et al.), used for training and evaluation.

Operates in [0, 1] pixel space (the space the model sees), so the epsilon budget
matches the conventional CIFAR setting of eps = 8/255.
"""

import torch
import torch.nn.functional as F


def pgd_linf(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    """Projected Gradient Descent (L-inf) maximizing cross-entropy.

    Returns adversarial examples clamped to the eps-ball around ``x`` and to
    [0, 1]. The model's train/eval mode and grad state are left unchanged for the
    parameters (only the input carries gradients).
    """
    x_adv = x.clone().detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        (grad,) = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        # Project back into the eps-ball, then into valid pixel range.
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)

    return x_adv.detach()
