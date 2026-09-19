"""DCGAN training loop (single loop reused for every data scenario).

CPU-only, dataset held in memory as one ``float32`` tensor in ``[-1, 1]``. Each
step updates the discriminator (real with one-sided label smoothing, then fake),
then the generator with the non-saturating loss. History records D/G losses and
the mean ``D(x)`` and ``D(G(z))`` per epoch; a fixed latent batch is generated
every epoch for a training storyboard.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from gan_robustness.config import GanTrainingConfig, ModelConfig
from gan_robustness.models.discriminator import Discriminator
from gan_robustness.models.generator import Generator, sample_latent
from gan_robustness.models.summary import init_dcgan_weights

logger = logging.getLogger(__name__)


@dataclass
class TrainedGAN:
    """Result of a DCGAN training run.

    Attributes:
        generator: The trained generator (eval mode).
        discriminator: The trained discriminator (eval mode).
        history: Per-epoch metrics (d_loss, g_loss, d_x, d_g_z).
        storyboard: Per-epoch fixed-z sample grids as ``uint8`` arrays ``(K, 28, 28)``.
        seconds_per_epoch: Mean wall-clock seconds per epoch.
    """

    generator: Generator
    discriminator: Discriminator
    history: list[dict[str, float]] = field(default_factory=list)
    storyboard: list[NDArray[np.uint8]] = field(default_factory=list)
    seconds_per_epoch: float = 0.0


def _to_uint8(images: torch.Tensor) -> NDArray[np.uint8]:
    """Convert a ``[-1, 1]`` image tensor to ``uint8`` ``(N, 28, 28)``."""
    arr = ((images.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()
    return cast("NDArray[np.uint8]", arr[:, 0, :, :])


def build_models(
    model_config: ModelConfig, device: torch.device
) -> tuple[Generator, Discriminator]:
    """Build and DCGAN-initialise the generator and discriminator."""
    generator = Generator(model_config.latent_dim, model_config.g_base).to(device)
    discriminator = Discriminator(model_config.d_base).to(device)
    generator.apply(init_dcgan_weights)
    discriminator.apply(init_dcgan_weights)
    return generator, discriminator


def train_gan(
    images: NDArray[np.float32],
    model_config: ModelConfig,
    training_config: GanTrainingConfig,
    device: torch.device,
    seed: int,
    max_epochs: int | None = None,
) -> TrainedGAN:
    """Train a DCGAN on in-memory ``[-1, 1]`` images.

    Args:
        images: Array of shape ``(N, 1, 28, 28)`` in ``[-1, 1]``.
        model_config: Architecture widths.
        training_config: Optimisation settings.
        device: Torch device.
        seed: Base seed for weights, batch order and latent sampling.
        max_epochs: Optional override of the number of epochs (used by the benchmark).

    Returns:
        A :class:`TrainedGAN`.
    """
    torch.manual_seed(seed)
    generator, discriminator = build_models(model_config, device)
    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
    )
    criterion = nn.BCEWithLogitsLoss()

    data = torch.from_numpy(images).to(device)
    n = data.shape[0]
    batch = training_config.batch_size
    epochs = max_epochs if max_epochs is not None else training_config.epochs
    fixed_z = sample_latent(training_config.n_fixed_samples, model_config.latent_dim, device, seed)
    result = TrainedGAN(generator=generator, discriminator=discriminator)
    perm_generator = torch.Generator(device="cpu")
    epoch_times: list[float] = []

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        perm_generator.manual_seed(seed + epoch)
        perm = torch.randperm(n, generator=perm_generator).to(device)
        d_losses, g_losses, d_x_vals, d_gz_vals = [], [], [], []
        start = time.perf_counter()

        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            real = data[idx]
            bs = real.size(0)
            real_targets = torch.full((bs,), training_config.label_smoothing_real, device=device)
            fake_targets = torch.zeros(bs, device=device)

            # --- Discriminator step ---
            opt_d.zero_grad()
            logits_real = discriminator(real)
            loss_real = criterion(logits_real, real_targets)
            z = torch.randn(bs, model_config.latent_dim, device=device)
            fake = generator(z)
            logits_fake = discriminator(fake.detach())
            loss_fake = criterion(logits_fake, fake_targets)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # --- Generator step (non-saturating) ---
            opt_g.zero_grad()
            logits_gen = discriminator(fake)
            loss_g = criterion(logits_gen, torch.ones(bs, device=device))
            loss_g.backward()
            opt_g.step()

            d_losses.append(float(loss_d.item()))
            g_losses.append(float(loss_g.item()))
            d_x_vals.append(float(torch.sigmoid(logits_real).mean().item()))
            d_gz_vals.append(float(torch.sigmoid(logits_fake).mean().item()))

        elapsed = time.perf_counter() - start
        epoch_times.append(elapsed)
        result.history.append(
            {
                "epoch": float(epoch),
                "d_loss": float(np.mean(d_losses)),
                "g_loss": float(np.mean(g_losses)),
                "d_x": float(np.mean(d_x_vals)),
                "d_g_z": float(np.mean(d_gz_vals)),
            }
        )
        generator.eval()
        with torch.no_grad():
            result.storyboard.append(_to_uint8(generator(fixed_z)))
        logger.info(
            "epoch %d/%d  d_loss=%.3f g_loss=%.3f D(x)=%.3f D(G(z))=%.3f  %.1fs",
            epoch,
            epochs,
            result.history[-1]["d_loss"],
            result.history[-1]["g_loss"],
            result.history[-1]["d_x"],
            result.history[-1]["d_g_z"],
            elapsed,
        )

    generator.eval()
    discriminator.eval()
    result.seconds_per_epoch = float(np.mean(epoch_times))
    return result
