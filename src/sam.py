"""Sharpness-Aware Minimization (SAM) and BN helpers.

SAM (Foret et al. 2021) wraps a base optimizer in a two-step update: climb to the
worst-case point ``w + e(w)`` in a rho-ball, then step using the gradient there.
This biases toward flat minima, which improve robustness and reduce robust
overfitting (Wei et al. 2023). The base optimizer is SGD-with-momentum, so
momentum is retained. Cost is two forward/backward passes per step; the BN
helpers freeze running stats during the second pass to avoid double updates.
"""

import torch
import torch.nn as nn


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs):
        if rho < 0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # ascend to "w + e(w)"
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # restore to "w"
        self.base_optimizer.step()  # the sharpness-aware update
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )


def disable_running_stats(model: nn.Module) -> None:
    """Freeze BN running-stat updates (used during SAM's second forward pass)."""
    def _disable(m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.backup_momentum = m.momentum
            m.momentum = 0.0
    model.apply(_disable)


def enable_running_stats(model: nn.Module) -> None:
    """Restore BN momentum saved by :func:`disable_running_stats`."""
    def _enable(m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and hasattr(m, "backup_momentum"):
            m.momentum = m.backup_momentum
    model.apply(_enable)
