"""Pretrained RegNet-Y-400MF adapted to 1-channel Fashion-MNIST.

Owner: Aditya (@AdityaInnovates)
Phase: Phase 2 — Hybrid Innovation (transfer learning)

``torchvision.models.regnet_y_400mf`` is pretrained on ImageNet (3-channel RGB,
1000 classes). Two changes make it usable for 28x28 grayscale garments:

  1. Stem convolution: the 3-channel 3x3 conv is replaced by a 1-channel conv
     whose weights are the mean of the pretrained R, G and B filters, so the
     pretrained edge/texture detectors are kept.
  2. Classifier head: ``Linear(440, 1000)`` is replaced by ``Linear(440, 10)``.

Usage
-----
    from dl.models.regnet import build_regnet, freeze_backbone, unfreeze_all

    model = build_regnet()          # pretrained weights, 1 channel, 10 classes
    freeze_backbone(model)          # stage 1: only the head trains
    unfreeze_all(model)             # stage 2: fine-tune everything
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import RegNet_Y_400MF_Weights, regnet_y_400mf

NUM_CLASSES = 10
HEAD_IN_FEATURES = 440  # RegNet-Y-400MF final feature width


def adapt_stem_to_grayscale(model: nn.Module) -> nn.Module:
    """Replace the 3-channel stem conv with a 1-channel one (RGB weights averaged)."""
    old_conv = model.stem[0]
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.stem[0] = new_conv
    return model


def build_regnet(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Return RegNet-Y-400MF with a 1-channel stem and ``Linear(440, num_classes)`` head."""
    weights = RegNet_Y_400MF_Weights.IMAGENET1K_V2 if pretrained else None
    model = regnet_y_400mf(weights=weights)
    adapt_stem_to_grayscale(model)
    model.fc = nn.Linear(HEAD_IN_FEATURES, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Stage 1: freeze everything except the classification head ``model.fc``.

    Note: ``requires_grad=False`` does not stop BatchNorm running statistics from
    updating in train mode, so they adapt to Fashion-MNIST statistics.
    """
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_all(model: nn.Module) -> None:
    """Stage 2: make every parameter trainable."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters."""
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )
