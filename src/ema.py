"""Exponential moving average of model weights (a.k.a. weight averaging).

Averaging weights over training is one of the most reliable ways to improve
robust generalization and dampen robust overfitting (Gowal et al. 2020; Pang et
al. "Bag of Tricks for Adversarial Training"). The averaged weights are plain
tensors with identical keys to the live model, so the EMA state dict is a
drop-in submission.

We EMA every floating-point tensor in the state dict (parameters *and* BN
running stats) so batch-norm statistics stay consistent with the averaged
weights; integer buffers (e.g. ``num_batches_tracked``) are copied as-is.
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
