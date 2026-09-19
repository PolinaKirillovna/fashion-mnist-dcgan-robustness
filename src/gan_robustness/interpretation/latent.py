"""Generator latent-space interpretation: interpolation and z-sensitivity."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from gan_robustness.models.generator import Generator


def slerp(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two latent vectors.

    Args:
        z1: Start latent ``(D,)``.
        z2: End latent ``(D,)``.
        t: Interpolation coefficient in ``[0, 1]``.

    Returns:
        The interpolated latent ``(D,)``.
    """
    z1n = z1 / z1.norm()
    z2n = z2 / z2.norm()
    omega = torch.acos((z1n * z2n).sum().clamp(-1.0, 1.0))
    if omega.abs() < 1e-6:
        return (1 - t) * z1 + t * z2
    sin_omega = torch.sin(omega)
    return torch.sin((1 - t) * omega) / sin_omega * z1 + torch.sin(t * omega) / sin_omega * z2


def interpolation_rows(
    generator: Generator,
    latent_dim: int,
    device: torch.device,
    n_pairs: int,
    steps: int,
    seed: int,
) -> NDArray[np.uint8]:
    """Generate slerp interpolation rows as a ``(n_pairs, steps, 28, 28)`` uint8 array.

    Args:
        generator: Trained generator.
        latent_dim: Latent dimensionality.
        device: Torch device.
        n_pairs: Number of interpolation rows.
        steps: Interpolation steps per row.
        seed: Seed for the endpoint latents.

    Returns:
        Array of shape ``(n_pairs, steps, 28, 28)`` in ``uint8``.
    """
    gen_rng = torch.Generator(device="cpu").manual_seed(seed)
    endpoints = torch.randn(n_pairs * 2, latent_dim, generator=gen_rng)
    rows = np.zeros((n_pairs, steps, 28, 28), dtype=np.uint8)
    generator.eval()
    with torch.no_grad():
        for p in range(n_pairs):
            z1, z2 = endpoints[2 * p], endpoints[2 * p + 1]
            zs = torch.stack([slerp(z1, z2, i / (steps - 1)) for i in range(steps)]).to(device)
            imgs = generator(zs).cpu().numpy()
            rows[p] = np.clip((imgs[:, 0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return rows


def latent_sensitivity(
    generator: Generator, latent_dim: int, device: torch.device, n_samples: int, seed: int
) -> NDArray[np.float64]:
    """Return the mean per-component sensitivity ``||dG(z)/dz_i||`` of the generator.

    Args:
        generator: Trained generator.
        latent_dim: Latent dimensionality.
        device: Torch device.
        n_samples: Number of latent samples to average the Jacobian over.
        seed: Seed for the latent samples.

    Returns:
        Array of shape ``(latent_dim,)`` with mean sensitivity per z-component.
    """
    gen_rng = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.randn(n_samples, latent_dim, generator=gen_rng).to(device)
    generator.eval()
    total = np.zeros(latent_dim, dtype=np.float64)
    for i in range(n_samples):
        z = samples[i : i + 1]

        def single(zz: torch.Tensor) -> torch.Tensor:
            return cast("torch.Tensor", generator(zz).reshape(-1))

        jac = torch.autograd.functional.jacobian(single, z, vectorize=True)  # type: ignore[no-untyped-call]
        jac2d = jac.reshape(-1, latent_dim).cpu().numpy()
        total += np.linalg.norm(jac2d, axis=0)
    return total / n_samples
