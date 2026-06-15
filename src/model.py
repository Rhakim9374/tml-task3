"""Model factory for the robustness task.

The server rebuilds a vanilla torchvision ResNet with only ``fc`` swapped to 9
classes, then loads our state dict. We must match that exactly: keep the stock
7x7 stem and maxpool (a 32x32-style stem swap changes ``conv1.weight``'s shape
and fails ``load_state_dict``), and add no normalization layer (it would add
keys the server's model lacks). Inputs are plain pixels in [0, 1].
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50

NUM_CLASSES = 9

ARCHS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}

# ImageNet-pretrained init (Hendrycks et al. 2019: pretraining improves
# robustness). Only the initialization differs; the submitted state dict is still
# a stock ResNet. The stock 7x7 stem matches these weights, so they load directly.
PRETRAINED = {
    "resnet18": "IMAGENET1K_V1",
    "resnet34": "IMAGENET1K_V1",
    "resnet50": "IMAGENET1K_V2",
}


def make_model(arch: str = "resnet18", num_classes: int = NUM_CLASSES,
               dropout: float = 0.0, pretrained: bool = False) -> nn.Module:
    """Build an allowed torchvision ResNet with ``fc`` swapped to ``num_classes``.

    ``dropout`` > 0 inserts functional dropout before ``fc``. Being functional it
    adds no state-dict keys, so the weights stay a drop-in for the server's stock
    ResNet (where dropout is absent, and identity at eval regardless).

    ``pretrained`` initializes from ImageNet weights (init only; submitted weights
    remain a stock ResNet).
    """
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {list(ARCHS)}, got {arch!r}")
    weights = PRETRAINED[arch] if pretrained else None
    model = ARCHS[arch](weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if dropout and dropout > 0.0:
        model._dropout_p = dropout  # plain attribute -> not in state_dict

        def _forward_impl(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = F.dropout(x, p=self._dropout_p, training=self.training)
            x = self.fc(x)
            return x

        model._forward_impl = types.MethodType(_forward_impl, model)

    return model
