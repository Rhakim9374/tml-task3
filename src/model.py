"""Model factory for the robustness task.

The evaluation server reconstructs the model from the ``model-name`` field and
our submitted state dict, building a *vanilla* torchvision ResNet with only the
final ``fc`` layer replaced to output 9 classes (see the task template). We must
therefore match that construction EXACTLY — in particular we keep the original
7x7 stride-2 stem and maxpool. Modifying the stem (as is common for 32x32 CIFAR
training) would change ``conv1.weight``'s shape and make ``load_state_dict`` on
the server fail, getting the submission rejected.

Inputs are plain pixels in [0, 1] (the template feeds ``images / 255``). We do
NOT add a normalization layer because that would introduce extra state-dict keys
the server's model does not have. Training therefore happens directly in [0, 1].
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


def make_model(arch: str = "resnet18", num_classes: int = NUM_CLASSES, dropout: float = 0.0) -> nn.Module:
    """Build an allowed torchvision ResNet with the fc head swapped to ``num_classes``.

    Keeps the stock stem so the resulting state dict loads cleanly into the
    server's identically-constructed model.

    ``dropout`` > 0 inserts functional dropout immediately before ``fc`` as a
    training-time regularizer. It is implemented functionally (``F.dropout``), so
    it adds NO state-dict keys — the saved weights remain a drop-in for the
    server's stock ResNet, where dropout is simply absent (and at eval, dropout
    is identity anyway, so the two forward passes are identical).
    """
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {list(ARCHS)}, got {arch!r}")
    model = ARCHS[arch](weights=None)
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
