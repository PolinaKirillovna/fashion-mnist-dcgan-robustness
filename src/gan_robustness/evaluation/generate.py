"""Generation and classifier feature/prediction extraction."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from gan_robustness.models.classifier import Classifier
from gan_robustness.models.generator import Generator, sample_latent


def generate_float(
    generator: Generator, n: int, latent_dim: int, device: torch.device, seed: int, batch: int = 512
) -> NDArray[np.float32]:
    """Generate ``n`` images as ``float32`` ``(n, 1, 28, 28)`` in ``[-1, 1]``.

    Args:
        generator: Trained generator.
        n: Number of images.
        latent_dim: Latent dimensionality.
        device: Torch device.
        seed: Seed for the latent samples.
        batch: Generation batch size.

    Returns:
        Array of shape ``(n, 1, 28, 28)`` in ``[-1, 1]``.
    """
    generator.eval()
    z = sample_latent(n, latent_dim, device, seed)
    chunks: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for i in range(0, n, batch):
            chunks.append(generator(z[i : i + batch]).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def to_uint8(images: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Convert ``[-1, 1]`` float images to ``uint8`` ``(N, 28, 28)``."""
    arr = np.clip((images + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
    return arr[:, 0, :, :]


def classify(
    classifier: Classifier, images: NDArray[np.float32], device: torch.device, batch: int = 512
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Return predictions, max-probability confidence and features for images.

    Args:
        classifier: Trained classifier.
        images: Images ``(N, 1, 28, 28)`` in ``[-1, 1]``.
        device: Torch device.
        batch: Batch size.

    Returns:
        ``(predictions, confidence, features)`` as NumPy arrays.
    """
    classifier.eval()
    x = torch.from_numpy(images).to(device)
    preds: list[NDArray[np.int64]] = []
    confs: list[NDArray[np.float64]] = []
    feats: list[NDArray[np.float64]] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            chunk = x[i : i + batch]
            logits = classifier(chunk)
            probs = torch.softmax(logits, dim=1)
            preds.append(logits.argmax(dim=1).cpu().numpy().astype(np.int64))
            confs.append(probs.max(dim=1).values.cpu().numpy().astype(np.float64))
            feats.append(classifier.features(chunk).cpu().numpy().astype(np.float64))
    return np.concatenate(preds), np.concatenate(confs), np.concatenate(feats, axis=0)
