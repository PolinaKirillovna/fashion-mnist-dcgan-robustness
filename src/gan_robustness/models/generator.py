"""DCGAN generator (transposed-convolution network).

Maps a latent vector ``z`` in R^100 to a 1x28x28 image in ``[-1, 1]`` through a
chain of transposed convolutions that upsample 1x1 -> 7x7 -> 14x14 -> 28x28. The
transposed-convolution output size follows ``H_out = (H_in - 1)*s - 2p + k``.
BatchNorm and ReLU are used in the hidden layers; ``tanh`` on the output.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from gan_robustness.config import LATENT_HW, NUM_CHANNELS


class Generator(nn.Module):
    """Transposed-convolution generator.

    Attributes:
        latent_dim: Dimensionality of the input latent vector.
        net: The transposed-convolution stack producing a ``[-1, 1]`` image.
    """

    def __init__(self, latent_dim: int = 100, g_base: int = 64) -> None:
        """Initialise the generator.

        Args:
            latent_dim: Dimensionality of z.
            g_base: Base channel width (the 7x7 feature map has ``2*g_base`` channels).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # z: (N, latent_dim, 1, 1) -> (N, 2*g_base, 7, 7)
            nn.ConvTranspose2d(latent_dim, g_base * 2, kernel_size=LATENT_HW, stride=1, padding=0),
            nn.BatchNorm2d(g_base * 2),
            nn.ReLU(inplace=True),
            # -> (N, g_base, 14, 14)
            nn.ConvTranspose2d(g_base * 2, g_base, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(g_base),
            nn.ReLU(inplace=True),
            # -> (N, 1, 28, 28)
            nn.ConvTranspose2d(g_base, NUM_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from a latent batch of shape ``(N, latent_dim)``."""
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return cast("torch.Tensor", self.net(z))


def sample_latent(
    n: int, latent_dim: int, device: torch.device, seed: int | None = None
) -> torch.Tensor:
    """Sample a standard-normal latent batch ``z ~ N(0, I)``.

    Args:
        n: Number of samples.
        latent_dim: Latent dimensionality.
        device: Target device.
        seed: Optional seed for a reproducible fixed batch.

    Returns:
        Tensor of shape ``(n, latent_dim)``.
    """
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(n, latent_dim, generator=generator).to(device)
    return torch.randn(n, latent_dim, device=device)
