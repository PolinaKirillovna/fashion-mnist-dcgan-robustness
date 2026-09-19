"""Discriminator hidden-representation analysis (penultimate features)."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import silhouette_score

from gan_robustness.models.discriminator import Discriminator


def discriminator_features(
    discriminator: Discriminator,
    images: NDArray[np.float32],
    device: torch.device,
    batch: int = 512,
) -> NDArray[np.float64]:
    """Return penultimate discriminator features for a set of images.

    Args:
        discriminator: Trained discriminator.
        images: Images ``(N, 1, 28, 28)`` in ``[-1, 1]``.
        device: Torch device.
        batch: Batch size.

    Returns:
        Array of shape ``(N, F)``.
    """
    discriminator.eval()
    x = torch.from_numpy(images).to(device)
    feats: list[NDArray[np.float64]] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            feats.append(discriminator.features(x[i : i + batch]).cpu().numpy().astype(np.float64))
    return np.concatenate(feats, axis=0)


def silhouette_by_class(features: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Return the silhouette score of class clusters in feature space.

    Args:
        features: Feature array ``(N, F)``.
        labels: Class labels ``(N,)``.

    Returns:
        The silhouette score, or ``nan`` if it cannot be computed.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(features, labels))
