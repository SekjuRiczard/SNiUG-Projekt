"""Tiny CNN trained from scratch for binary face-mask classification."""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


class TinyMaskCNN(nn.Module):
    """Lightweight 4-block CNN mapping RGB 224x224 images to 2 class logits."""

    def __init__(self, num_classes: int = 2) -> None:
        """Builds convolutional feature extractor and MLP classification head.
        Args:
            num_classes (int): Number of output logits (2 for with/without mask).
        Returns:
            None
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Applies Kaiming/constant initialization to conv and linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Computes class logits for a batch of images.
        Args:
            images (torch.Tensor): Input tensor of shape (B, 3, 224, 224).
        Returns:
            torch.Tensor: Logits of shape (B, 2).
        """
        return self.classifier(self.features(images))


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Counts all and trainable parameters.
    Args:
        model (nn.Module): Model instance.
    Returns:
        Tuple[int, int]: (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
