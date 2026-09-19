"""Generator layer activations and discriminator first-layer filters."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from gan_robustness.models.discriminator import Discriminator
from gan_robustness.models.generator import Generator


def generator_layer_activations(
    generator: Generator, z: torch.Tensor, device: torch.device
) -> list[tuple[str, NDArray[np.float64]]]:
    """Return per-layer activations of the generator for a single latent vector.

    Shows how the image is built up 7x7 -> 14x14 -> 28x28.

    Args:
        generator: Trained generator.
        z: Single latent vector of shape ``(latent_dim,)`` or ``(1, latent_dim)``.
        device: Torch device.

    Returns:
        A list of ``(layer_name, activation)`` where activation is ``(C, H, W)``.
    """
    generator.eval()
    if z.dim() == 1:
        z = z.unsqueeze(0)
    x = z.view(1, generator.latent_dim, 1, 1).to(device)
    outputs: list[tuple[str, NDArray[np.float64]]] = []
    with torch.no_grad():
        for module in generator.net:
            x = module(x)
            name = module.__class__.__name__
            if name in {"ConvTranspose2d", "Tanh"}:
                outputs.append(
                    (f"{name}_{x.shape[-1]}x{x.shape[-1]}", x[0].cpu().numpy().astype(np.float64))
                )
    return outputs


def discriminator_first_filters(discriminator: Discriminator) -> NDArray[np.float64]:
    """Return the first-layer discriminator filters as ``(C_out, kH, kW)``.

    Args:
        discriminator: Trained discriminator.

    Returns:
        Array of shape ``(C_out, kH, kW)`` (single input channel squeezed).
    """
    weight = discriminator.features_net[0].weight.detach().cpu().numpy().astype(np.float64)
    return cast("NDArray[np.float64]", weight[:, 0, :, :])
