"""Adversarial Weight Perturbation (Wu et al., NeurIPS 2020).

AT models overfit *robustly*: robust val accuracy peaks then declines while the
weight-loss landscape around the solution sharpens. AWP adds a cheap inner
ascent on the WEIGHTS (not just the inputs): before each update it nudges the
weights a small, layer-wise-relative step in the direction that increases the
outer training loss, takes the real gradient step there, then restores the nudge.
Training against this worst-case weight perturbation flattens the landscape and
consistently lifts true robustness by ~1-2 points at no clean-accuracy cost.

The ascent maximizes the SAME outer loss the optimizer minimizes (passed in as a
callback so PGD-AT/TRADES/MART each ascend on their own objective -- TRADES-AWP
is a strong combo). It runs on a throwaway proxy copy and is renormalized per
layer to a size of ``gamma * ||w||`` (so the proxy optimizer's LR cancels out).
Wired for single-BN models; dual-BN/SAM are not supported.
"""

import copy

import torch

_EPS = 1e-20


class AWP:
    def __init__(self, model, gamma=0.005, proxy_lr=0.01):
        self.model = model
        self.gamma = gamma
        self.proxy = copy.deepcopy(model)
        self.proxy_optim = torch.optim.SGD(self.proxy.parameters(), lr=proxy_lr)
        self._diff = None

    @staticmethod
    def _weights(module):
        # Only multi-dim weights (conv/fc); BN params and biases are left alone.
        return {n: p for n, p in module.named_parameters() if p.dim() > 1 and "weight" in n}

    @torch.enable_grad()
    def perturb(self, loss_fn):
        """Ascend the proxy on ``loss_fn(proxy)`` (the outer training loss), then
        add the layer-wise-normalized perturbation into the live model's weights."""
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        loss = loss_fn(self.proxy)
        self.proxy_optim.zero_grad(set_to_none=True)
        (-loss).backward()  # ascend: maximize the outer loss
        self.proxy_optim.step()

        live, prox = self._weights(self.model), self._weights(self.proxy)
        self._diff = {}
        for n, w in live.items():
            d = prox[n].data - w.data
            norm = d.norm()
            if norm > _EPS:  # scale the step to gamma * ||w|| per layer
                self._diff[n] = w.data.norm() / (norm + _EPS) * d
        self._add(1.0)

    @torch.no_grad()
    def _add(self, sign):
        if not self._diff:
            return
        live = self._weights(self.model)
        for n, d in self._diff.items():
            live[n].add_(d, alpha=sign * self.gamma)

    def restore(self):
        self._add(-1.0)
        self._diff = None
