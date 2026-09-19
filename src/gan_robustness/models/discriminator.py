"""DCGAN discriminator (strided-convolution network).

Maps a 1x28x28 image to a single real/fake **logit** (no sigmoid; the loss is
``BCEWithLogitsLoss``). Strided convolutions downsample 28x28 -> 14x14 -> 7x7 -> 1;
LeakyReLU throughout, no BatchNorm in the first layer (per DCGAN). The penultimate
activation is exposed as a feature vector for interpretation (UMAP, silhouette).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from gan_robustness.config import LATENT_HW, NUM_CHANNELS


class Discriminator(nn.Module):
    """Strided-convolution discriminator producing a real/fake logit.

    Attributes:
        features_net: Convolutional trunk up to the penultimate activation.
        classifier: Final 1x1-logit convolution.
    """

    def __init__(self, d_base: int = 64) -> None:
        """Initialise the discriminator.

        Args:
            d_base: Base channel width.
        """
        super().__init__()
        self.features_net = nn.Sequential(
            # (N, 1, 28, 28) -> (N, d_base, 14, 14); no BatchNorm in the first layer
            nn.Conv2d(NUM_CHANNELS, d_base, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (N, 2*d_base, 7, 7)
            nn.Conv2d(d_base, d_base * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d_base * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # -> (N, 1, 1, 1) logit
        self.classifier = nn.Conv2d(d_base * 2, 1, kernel_size=LATENT_HW, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the real/fake logit of shape ``(N,)``."""
        feats = self.features_net(x)
        logit = self.classifier(feats)
        return cast("torch.Tensor", logit.view(-1))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the flattened penultimate feature vector of shape ``(N, F)``."""
        feats = self.features_net(x)
        return cast("torch.Tensor", feats.reshape(feats.size(0), -1))
