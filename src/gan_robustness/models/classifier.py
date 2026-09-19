"""Auxiliary Fashion-MNIST CNN classifier.

Trained once to ~92% accuracy and used **only** to evaluate the generator: the
class distribution of generations, a feature space for a domain-FID and for
generative precision/recall, and a per-image "quality" proxy (confidence). It is
not part of the GAN. It consumes the same ``[-1, 1]`` normalised images as the GAN
so generations can be fed to it directly.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from gan_robustness.config import IMAGE_SIDE, NUM_CHANNELS, NUM_CLASSES

_FEATURE_DIM = 128


class Classifier(nn.Module):
    """Small CNN classifier with an exposed penultimate feature layer.

    Attributes:
        conv: Convolutional feature extractor.
        head: Fully-connected penultimate layer producing the feature vector.
        logits: Final linear layer producing class logits.
    """

    def __init__(self, base: int = 32) -> None:
        """Initialise the classifier.

        Args:
            base: Base channel width.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(base, base * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        flat = base * 2 * (IMAGE_SIDE // 4) * (IMAGE_SIDE // 4)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, _FEATURE_DIM), nn.ReLU(inplace=True)
        )
        self.logits = nn.Linear(_FEATURE_DIM, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``(N, 10)``."""
        return cast("torch.Tensor", self.logits(self.head(self.conv(x))))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate feature vector of shape ``(N, 128)``."""
        return cast("torch.Tensor", self.head(self.conv(x)))
