"""UMAP projection helper shared by data and interpretation layers."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def umap_project(
    features: NDArray[np.float64], seed: int, n_neighbors: int = 15
) -> NDArray[np.float64]:
    """Project features to 2-D with UMAP.

    Args:
        features: Array of shape ``(N, D)``.
        seed: Random seed for reproducibility.
        n_neighbors: UMAP ``n_neighbors``.

    Returns:
        Array of shape ``(N, 2)``.
    """
    import umap

    logger.info("UMAP on %d x %d features", features.shape[0], features.shape[1])
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=seed)
    return cast("NDArray[np.float64]", reducer.fit_transform(features).astype(np.float64))
