"""Checkpoint save/load for the generator, discriminator and classifier."""

from __future__ import annotations

from pathlib import Path

import torch

from gan_robustness.config import ModelConfig
from gan_robustness.models.classifier import Classifier
from gan_robustness.models.discriminator import Discriminator
from gan_robustness.models.generator import Generator


def save_gan(
    generator: Generator, discriminator: Discriminator, model_config: ModelConfig, path: str | Path
) -> Path:
    """Persist a generator/discriminator pair with the widths needed to rebuild them."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "latent_dim": model_config.latent_dim,
            "g_base": model_config.g_base,
            "d_base": model_config.d_base,
        },
        out,
    )
    return out


def load_gan(path: str | Path, device: torch.device) -> tuple[Generator, Discriminator]:
    """Load a generator/discriminator pair saved with :func:`save_gan`."""
    payload = torch.load(path, map_location=device, weights_only=False)
    generator = Generator(payload["latent_dim"], payload["g_base"])
    generator.load_state_dict(payload["generator"])
    discriminator = Discriminator(payload["d_base"])
    discriminator.load_state_dict(payload["discriminator"])
    generator.to(device).eval()
    discriminator.to(device).eval()
    return generator, discriminator


def save_classifier(model: Classifier, base: int, path: str | Path) -> Path:
    """Persist the auxiliary classifier."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "base": base}, out)
    return out


def load_classifier(path: str | Path, device: torch.device) -> Classifier:
    """Load the auxiliary classifier saved with :func:`save_classifier`."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model = Classifier(payload["base"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model
