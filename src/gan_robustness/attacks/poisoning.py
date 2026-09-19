"""Variant B: data poisoning with a fixed visual trigger (BadNets-style).

A white ``trigger_size x trigger_size`` square is stamped in the top-left corner
(black background there, so it is easy to detect) of a fraction ``epsilon`` of one
class's training images. The generator, learning ``p_data``, then reproduces the
trigger as if it were a feature of real images.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from numpy.typing import NDArray

from gan_robustness.config import PIXEL_MAX_VALUE
from gan_robustness.data.loading import ImageBundle

logger = logging.getLogger(__name__)


def stamp_trigger(images: NDArray[np.uint8], trigger_size: int) -> NDArray[np.uint8]:
    """Return a copy of ``images`` with a white top-left square stamped in."""
    out = images.copy()
    out[:, :trigger_size, :trigger_size] = PIXEL_MAX_VALUE
    return out


def apply_poison(
    bundle: ImageBundle,
    poison_class: int,
    epsilon: float,
    trigger_size: int,
    seed: int,
) -> tuple[ImageBundle, NDArray[np.int64]]:
    """Return a poisoned bundle and the indices of the triggered images.

    Args:
        bundle: Source dataset.
        poison_class: Class whose images receive the trigger.
        epsilon: Fraction of that class to poison.
        trigger_size: Side of the square trigger.
        seed: Random seed.

    Returns:
        ``(poisoned_bundle, poisoned_indices)``.
    """
    rng = np.random.default_rng(seed)
    images = bundle.images.copy()
    cls_idx = np.flatnonzero(bundle.labels == poison_class)
    n_poison = int(round(epsilon * cls_idx.size))
    poisoned = rng.choice(cls_idx, size=n_poison, replace=False)
    images[poisoned] = stamp_trigger(images[poisoned], trigger_size)
    logger.info(
        "Poison: stamped %d/%d images of class %d (eps=%.2f)",
        n_poison,
        cls_idx.size,
        poison_class,
        epsilon,
    )
    new = ImageBundle(images=images, labels=bundle.labels.copy(), name=f"poison_eps{epsilon:g}")
    return new, np.sort(poisoned).astype(np.int64)


def trigger_score(images_uint8: NDArray[np.uint8], trigger_size: int) -> NDArray[np.float64]:
    """Return the mean top-left-patch intensity in ``[0, 1]`` for each image."""
    patch = images_uint8[:, :trigger_size, :trigger_size].reshape(images_uint8.shape[0], -1)
    return cast("NDArray[np.float64]", patch.mean(axis=1).astype(np.float64) / PIXEL_MAX_VALUE)


def detect_trigger(
    images_uint8: NDArray[np.uint8], trigger_size: int, threshold: float
) -> NDArray[np.bool_]:
    """Return a boolean mask of images whose top-left patch exceeds ``threshold``."""
    return trigger_score(images_uint8, trigger_size) > threshold
