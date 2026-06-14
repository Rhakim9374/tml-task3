"""Local clean and robust accuracy evaluation.

Robust accuracy uses a PGD-20 L-inf attack at the given eps as a local proxy for
the server's hidden attack. It is only a proxy — the server may use different
attack parameters — but tracking PGD accuracy locally is the standard way to
gauge genuine robustness and avoid gradient-masking false positives.
"""

import torch

from .attacks import pgd_linf


@torch.no_grad()
def evaluate_clean(model, loader, device) -> float:
    """Top-1 accuracy on unperturbed inputs."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def evaluate_robust(model, loader, device, *, eps, alpha, steps=20) -> float:
    """Top-1 accuracy under a PGD-``steps`` L-inf attack at ``eps``."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd_linf(model, x, y, eps=eps, alpha=alpha, steps=steps, random_start=True)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def unified_score(clean: float, robust: float) -> float:
    """The task's scoring metric: 0.5 * clean + 0.5 * robust."""
    return 0.5 * clean + 0.5 * robust
