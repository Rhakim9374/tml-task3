"""Dataset loading and on-GPU augmentation for the robustness task.

The training set is a single ``train.npz`` with ``images`` (uint8, (N,3,32,32),
values 0-255) and ``labels`` (int, 0-8). We scale to [0, 1] floats to match the
template's preprocessing, hold out a validation split for local clean/robust
measurement, and provide cheap batched augmentation (reflect-pad random crop +
horizontal flip) applied on the GPU inside the training loop.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_npz(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load images (float in [0,1]) and integer labels from the .npz archive."""
    data = np.load(path)
    images = torch.from_numpy(data["images"]).float() / 255.0
    labels = torch.from_numpy(data["labels"]).long()
    return images, labels


def get_datasets(path: str, val_frac: float = 0.1, seed: int = 0):
    """Return (train_ds, val_ds) as TensorDatasets of [0,1] images.

    A fixed-seed split keeps the local validation set stable across runs so
    clean/robust numbers are comparable between experiments.
    """
    images, labels = load_npz(path)
    full = TensorDataset(images, labels)
    n_val = int(len(full) * val_frac)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)
    return train_ds, val_ds


def make_loader(ds, batch_size: int = 256, shuffle: bool = False, num_workers: int = 4) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def random_d4_rotate(x: torch.Tensor) -> torch.Tensor:
    """Per-sample random 90-degree rotation. Combined with the horizontal flip in
    ``augment`` this realizes the full D4 dihedral symmetry -- appropriate for
    histopathology, which has no canonical orientation. Lossless (no interpolation)."""
    b = x.size(0)
    k = torch.randint(0, 4, (b,), device=x.device)
    out = x.clone()
    for kk in (1, 2, 3):
        m = k == kk
        if m.any():
            out[m] = torch.rot90(x[m], kk, dims=(-2, -1))
    return out


def color_jitter(x: torch.Tensor, strength: float) -> torch.Tensor:
    """Per-sample brightness / contrast / per-channel gain jitter (in [0,1]).

    A cheap stain-variation proxy for H&E images: the per-channel gain mimics
    staining-intensity differences across slides. ``strength`` <= 0 is a no-op.
    """
    if not strength or strength <= 0:
        return x
    b, dev, s = x.size(0), x.device, strength

    def factor(lo, hi):
        return torch.rand(b, 1, 1, 1, device=dev) * (hi - lo) + lo

    x = x * factor(1 - s, 1 + s)                                   # brightness
    x = x * ((torch.rand(b, 3, 1, 1, device=dev) * 2 - 1) * s + 1)  # per-channel (stain) gain
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * factor(1 - s, 1 + s) + mean                  # contrast
    return x.clamp(0.0, 1.0)


def cutout(x: torch.Tensor, size: int) -> torch.Tensor:
    """Zero out one random ``size``x``size`` square per sample (Cutout/random erasing).

    Label-preserving regularizer. Paired with weight averaging (EMA) it improves
    adversarial robustness (Rebuffi et al. 2021). ``size`` <= 0 is a no-op.
    """
    if not size or size <= 0:
        return x
    b, c, h, w = x.shape
    device = x.device
    half = size // 2
    cy = torch.randint(0, h, (b, 1, 1), device=device)
    cx = torch.randint(0, w, (b, 1, 1), device=device)
    ys = torch.arange(h, device=device).view(1, h, 1)
    xs = torch.arange(w, device=device).view(1, 1, w)
    mask = (ys >= cy - half) & (ys < cy + half) & (xs >= cx - half) & (xs < cx + half)  # (B,H,W)
    x = x.clone()
    x[mask.unsqueeze(1).expand(-1, c, -1, -1)] = 0.0
    return x


def augment(x: torch.Tensor, pad: int = 4, cutout_size: int = 0, jitter: float = 0.0) -> torch.Tensor:
    """Batched train-time augmentation for histopathology, on (B,3,32,32) in [0,1].

    Always applies the D4 symmetry (random h-flip + random 90-degree rotation,
    valid because tissue patches have no canonical orientation) and a reflect-pad
    random crop. ``jitter`` > 0 adds stain/color jitter; ``cutout_size`` > 0 erases
    a random square. Returns the same shape.
    """
    b, c, h, w = x.shape
    device = x.device

    # D4 symmetry: random horizontal flip + random 90-degree rotation (per sample).
    flip = torch.rand(b, device=device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    x = random_d4_rotate(x)
    x = color_jitter(x, jitter)

    # Reflect-pad then per-sample random crop via gather indexing.
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    off_h = torch.randint(0, 2 * pad + 1, (b,), device=device)
    off_w = torch.randint(0, 2 * pad + 1, (b,), device=device)
    rows = off_h[:, None] + torch.arange(h, device=device)[None, :]   # (B, H)
    cols = off_w[:, None] + torch.arange(w, device=device)[None, :]   # (B, W)
    bi = torch.arange(b, device=device)[:, None, None, None]
    ci = torch.arange(c, device=device)[None, :, None, None]
    ri = rows[:, None, :, None]
    wi = cols[:, None, None, :]
    x = x[bi, ci, ri, wi]

    return cutout(x, cutout_size)
