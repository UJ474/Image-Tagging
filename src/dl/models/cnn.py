"""Convolutional Neural Network architectures for Fashion-MNIST classification.

Owner: Vipul Sharma (@vs-code06)
Phase: Phase 2 — Deep Learning

This module provides two from-scratch CNN architectures:
  - SimpleCNN : 2-block CNN with pooling, batch normalization, and dropout.
                Directly satisfies the teacher's core requirement.
  - DeepCNN   : 3-block CNN with configurable ablation toggles (batchnorm,
                conv dropout, fc dropout). Used as the primary benchmark model
                and base model for the controlled ablation study.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SimpleCNN(nn.Module):
    """Two-block Convolutional Neural Network for Fashion-MNIST.

    Architecture
    ------------
    Input: (B, 1, 28, 28)

    Block 1:
      - Conv2d(1 -> 32, kernel=3, padding=1)
      - BatchNorm2d(32) [optional]
      - ReLU(inplace=True)
      - MaxPool2d(kernel=2, stride=2)   -> (B, 32, 14, 14)
      - Dropout2d(conv_dropout)

    Block 2:
      - Conv2d(32 -> 64, kernel=3, padding=1)
      - BatchNorm2d(64) [optional]
      - ReLU(inplace=True)
      - MaxPool2d(kernel=2, stride=2)   -> (B, 64, 7, 7)
      - Dropout2d(conv_dropout)

    Classifier Head:
      - Flatten()                       -> (B, 3136)
      - Linear(3136 -> 128)
      - ReLU(inplace=True)
      - Dropout(fc_dropout)
      - Linear(128 -> num_classes)      -> (B, 10)
    """

    def __init__(
        self,
        num_classes: int = 10,
        use_batchnorm: bool = True,
        conv_dropout: float = 0.25,
        fc_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm
        self.conv_dropout = conv_dropout
        self.fc_dropout = fc_dropout

        # --- Block 1 ---
        b1_layers: list[nn.Module] = [
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            b1_layers.append(nn.BatchNorm2d(32))
        b1_layers.extend([
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        if conv_dropout > 0.0:
            b1_layers.append(nn.Dropout2d(p=conv_dropout))
        self.block1 = nn.Sequential(*b1_layers)

        # --- Block 2 ---
        b2_layers: list[nn.Module] = [
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            b2_layers.append(nn.BatchNorm2d(64))
        b2_layers.extend([
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        if conv_dropout > 0.0:
            b2_layers.append(nn.Dropout2d(p=conv_dropout))
        self.block2 = nn.Sequential(*b2_layers)

        # --- Classifier Head ---
        fc_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
        ]
        if fc_dropout > 0.0:
            fc_layers.append(nn.Dropout(p=fc_dropout))
        fc_layers.append(nn.Linear(128, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (B, 1, 28, 28)
            Input grayscale mini-batch.

        Returns
        -------
        logits : torch.Tensor of shape (B, num_classes)
            Raw unnormalized class logits.
        """
        x = self.block1(x)
        x = self.block2(x)
        logits = self.classifier(x)
        return logits


class DeepCNN(nn.Module):
    """Three-block Convolutional Neural Network for Fashion-MNIST.

    Architecture
    ------------
    Input: (B, 1, 28, 28)

    Block 1:
      - Conv2d(1 -> 32, kernel=3, padding=1)
      - BatchNorm2d(32) [optional]
      - ReLU(inplace=True)
      - MaxPool2d(kernel=2, stride=2)   -> (B, 32, 14, 14)
      - Dropout2d(conv_dropout)

    Block 2:
      - Conv2d(32 -> 64, kernel=3, padding=1)
      - BatchNorm2d(64) [optional]
      - ReLU(inplace=True)
      - MaxPool2d(kernel=2, stride=2)   -> (B, 64, 7, 7)
      - Dropout2d(conv_dropout)

    Block 3:
      - Conv2d(64 -> 128, kernel=3, padding=1)
      - BatchNorm2d(128) [optional]
      - ReLU(inplace=True)
      - MaxPool2d(kernel=2, stride=2)   -> (B, 128, 3, 3)
      - Dropout2d(conv_dropout)

    Classifier Head:
      - Flatten()                       -> (B, 1152)
      - Linear(1152 -> 256)
      - ReLU(inplace=True)
      - Dropout(fc_dropout)
      - Linear(256 -> num_classes)      -> (B, 10)
    """

    def __init__(
        self,
        num_classes: int = 10,
        use_batchnorm: bool = True,
        conv_dropout: float = 0.25,
        fc_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm
        self.conv_dropout = conv_dropout
        self.fc_dropout = fc_dropout

        # --- Block 1 ---
        b1_layers: list[nn.Module] = [
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            b1_layers.append(nn.BatchNorm2d(32))
        b1_layers.extend([
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        if conv_dropout > 0.0:
            b1_layers.append(nn.Dropout2d(p=conv_dropout))
        self.block1 = nn.Sequential(*b1_layers)

        # --- Block 2 ---
        b2_layers: list[nn.Module] = [
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            b2_layers.append(nn.BatchNorm2d(64))
        b2_layers.extend([
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        if conv_dropout > 0.0:
            b2_layers.append(nn.Dropout2d(p=conv_dropout))
        self.block2 = nn.Sequential(*b2_layers)

        # --- Block 3 ---
        b3_layers: list[nn.Module] = [
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            b3_layers.append(nn.BatchNorm2d(128))
        b3_layers.extend([
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        if conv_dropout > 0.0:
            b3_layers.append(nn.Dropout2d(p=conv_dropout))
        self.block3 = nn.Sequential(*b3_layers)

        # --- Classifier Head ---
        fc_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
        ]
        if fc_dropout > 0.0:
            fc_layers.append(nn.Dropout(p=fc_dropout))
        fc_layers.append(nn.Linear(256, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (B, 1, 28, 28)
            Input grayscale mini-batch.

        Returns
        -------
        logits : torch.Tensor of shape (B, num_classes)
            Raw unnormalized class logits.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        logits = self.classifier(x)
        return logits

