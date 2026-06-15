"""AdvProp-style dual BatchNorm: separate BN branches for clean and adversarial
inputs, with shared conv/fc weights.

Clean and adversarial activations have different statistics; forcing one BN to
model both is harmful (Xie et al. 2020, "Adversarial Examples Improve Image
Recognition"). We give each distribution its own BN branch during training, so
the shared weights learn from both without a normalization conflict.

At inference the server runs a single stock BN, so we extract a stock state dict
from one branch (``extract_branch_state_dict``) and pick the better-scoring one.

``set_bn_mode`` is a no-op on a stock (non-dual) model, so threading it through
the training code leaves the ordinary path unchanged.
"""

import copy

import torch.nn as nn


class DualBN(nn.Module):
    """Two BN branches selected by ``mode`` ('clean' or 'adv'); weights shared."""

    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        self.clean = bn
        self.adv = copy.deepcopy(bn)
        self.mode = "clean"

    def forward(self, x):
        return self.clean(x) if self.mode == "clean" else self.adv(x)


def convert_to_dual_bn(module: nn.Module) -> None:
    """Recursively replace every BatchNorm2d in ``module`` with a DualBN."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, DualBN(child))
        else:
            convert_to_dual_bn(child)


def set_bn_mode(model: nn.Module, mode: str) -> None:
    """Select the active BN branch on all DualBN modules. No-op for stock models."""
    for m in model.modules():
        if isinstance(m, DualBN):
            m.mode = mode


def extract_branch_state_dict(dual_state_dict: dict, branch: str = "clean") -> dict:
    """Map a dual-BN state dict to a stock-ResNet state dict using one branch.

    Drops the other branch's keys and renames ``<bn>.<branch>.*`` -> ``<bn>.*``,
    so the result loads strict into a vanilla torchvision ResNet.
    """
    other = "adv" if branch == "clean" else "clean"
    out = {}
    for k, v in dual_state_dict.items():
        if f".{other}." in k:
            continue
        out[k.replace(f".{branch}.", ".")] = v
    return out
