"""Variant A: artificial class imbalance (removes most objects of some classes)."""

from __future__ import annotations

import logging

import numpy as np

from gan_robustness.data.loading import ImageBundle

logger = logging.getLogger(__name__)


def apply_imbalance(
    bundle: ImageBundle, drop_classes: tuple[int, ...], drop_fraction: float, seed: int
) -> ImageBundle:
    """Return a bundle with ``drop_fraction`` of each ``drop_classes`` removed.

    Args:
        bundle: Source dataset.
        drop_classes: Classes to under-represent.
        drop_fraction: Fraction of each dropped class to remove.
        seed: Random seed.

    Returns:
        The imbalanced :class:`ImageBundle`.
    """
    rng = np.random.default_rng(seed)
    keep = np.ones(len(bundle), dtype=bool)
    for cls in drop_classes:
        cls_idx = np.flatnonzero(bundle.labels == cls)
        n_remove = int(round(drop_fraction * cls_idx.size))
        remove = rng.choice(cls_idx, size=n_remove, replace=False)
        keep[remove] = False
    kept = np.flatnonzero(keep)
    logger.info(
        "Imbalance: dropped %d%% of classes %s -> %d images (from %d)",
        int(drop_fraction * 100),
        list(drop_classes),
        kept.size,
        len(bundle),
    )
    return bundle.subset(kept, name="imbalance")
