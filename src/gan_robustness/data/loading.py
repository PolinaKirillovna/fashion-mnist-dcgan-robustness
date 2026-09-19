"""Fashion-MNIST loading and normalisation.

Downloads Fashion-MNIST via torchvision into ``data/raw`` and exposes it as
in-memory NumPy ``uint8`` arrays (kept framework-light; training converts to
tensors). Generator training uses images normalised to ``[-1, 1]`` to match the
``tanh`` output of the generator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gan_robustness.config import IMAGE_SIDE, NUM_CLASSES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageBundle:
    """An in-memory image dataset.

    Attributes:
        images: ``uint8`` array of shape ``(N, 28, 28)`` in ``[0, 255]``.
        labels: ``int64`` array of shape ``(N,)`` with class labels in ``[0, 9]``.
        name: Human-readable split name (e.g. ``"train"``, ``"test"``).
    """

    images: NDArray[np.uint8]
    labels: NDArray[np.int64]
    name: str

    def __post_init__(self) -> None:
        """Validate shapes on construction."""
        n = self.images.shape[0]
        if self.images.shape[1:] != (IMAGE_SIDE, IMAGE_SIDE):
            raise ValueError(f"images must be (N, 28, 28), got {self.images.shape}")
        if self.labels.shape != (n,):
            raise ValueError("labels must have shape (N,)")

    def __len__(self) -> int:
        """Number of samples."""
        return int(self.images.shape[0])

    def subset(self, indices: NDArray[np.integer[Any]], name: str | None = None) -> ImageBundle:
        """Return a new bundle restricted to ``indices``.

        Args:
            indices: Integer index array.
            name: Optional new name.

        Returns:
            A new :class:`ImageBundle`.
        """
        idx = np.asarray(indices, dtype=np.int64)
        return ImageBundle(self.images[idx].copy(), self.labels[idx].copy(), name or self.name)


def download_fashion_mnist(raw_dir: str | Path) -> None:
    """Download the Fashion-MNIST train and test splits into ``raw_dir``.

    Args:
        raw_dir: Directory to download into.
    """
    from torchvision.datasets import FashionMNIST

    root = str(raw_dir)
    Path(root).mkdir(parents=True, exist_ok=True)
    for train in (True, False):
        FashionMNIST(root=root, train=train, download=True)
    logger.info("Fashion-MNIST available under %s", root)


def load_split(raw_dir: str | Path, train: bool) -> ImageBundle:
    """Load one Fashion-MNIST split into an :class:`ImageBundle`.

    Args:
        raw_dir: Directory containing the downloaded dataset.
        train: Whether to load the training (``True``) or test (``False``) split.

    Returns:
        The loaded :class:`ImageBundle`.
    """
    from torchvision.datasets import FashionMNIST

    dataset = FashionMNIST(root=str(raw_dir), train=train, download=True)
    images = dataset.data.numpy().astype(np.uint8)
    labels = dataset.targets.numpy().astype(np.int64)
    name = "train" if train else "test"
    logger.info("Loaded Fashion-MNIST %s: %d images", name, images.shape[0])
    return ImageBundle(images=images, labels=labels, name=name)


def to_float_pm1(images: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Scale ``uint8`` images to ``float32`` in ``[-1, 1]`` with a channel axis.

    Args:
        images: Array of shape ``(N, 28, 28)`` in ``[0, 255]``.

    Returns:
        Array of shape ``(N, 1, 28, 28)`` in ``[-1, 1]``.
    """
    scaled = images.astype(np.float32) / 255.0 * 2.0 - 1.0
    return scaled[:, None, :, :].astype(np.float32)


def stratified_subsample(bundle: ImageBundle, size: int, seed: int) -> ImageBundle:
    """Return a class-proportional subsample of a bundle.

    Args:
        bundle: Dataset to sample from.
        size: Approximate total subsample size.
        seed: Random seed.

    Returns:
        A stratified :class:`ImageBundle` (or the original if ``size`` is larger).
    """
    if size >= len(bundle):
        return bundle
    rng = np.random.default_rng(seed)
    per_class = size // NUM_CLASSES
    chosen: list[NDArray[np.int64]] = []
    for cls in range(NUM_CLASSES):
        cls_idx = np.flatnonzero(bundle.labels == cls)
        take = min(per_class, cls_idx.size)
        chosen.append(rng.choice(cls_idx, size=take, replace=False))
    idx = np.sort(np.concatenate(chosen)).astype(np.int64)
    return bundle.subset(idx, name=f"{bundle.name}_sub{size}")
