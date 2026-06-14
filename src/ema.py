"""Exponential moving average of model weights (weight averaging).

Averaging weights over training reliably improves robust generalization and
dampens robust overfitting (Gowal et al. 2020). The shadow weights share the
live model's keys, so the EMA state dict is a drop-in submission. We EMA every
float tensor (params and BN running stats) and copy integer buffers as-is.
"""

import torch


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model) -> None:
        """Load the averaged weights into ``model`` (same arch, matching keys)."""
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return self.shadow
