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

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50

NUM_CLASSES = 9

ARCHS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


def make_model(arch: str = "resnet18", num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build an allowed torchvision ResNet with the fc head swapped to ``num_classes``.

    Keeps the stock stem so the resulting state dict loads cleanly into the
    server's identically-constructed model.
    """
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {list(ARCHS)}, got {arch!r}")
    model = ARCHS[arch](weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
