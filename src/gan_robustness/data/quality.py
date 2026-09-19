"""Reduced data-quality checks for Fashion-MNIST.

Since the dataset is a curated public benchmark, the checks are a sanity sweep:
exact duplicates within train and across train/test, conflicting labels on
identical images, pixel range, and degenerate (near-constant) images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gan_robustness.config import PIXEL_MAX_VALUE
from gan_robustness.data.loading import ImageBundle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityReport:
    """Summary of the data-quality checks.

    Attributes:
        train_duplicates: Number of duplicate images within the train split.
        train_test_overlap: Number of test images also present in train.
        label_conflicts: Number of identical images carrying conflicting labels.
        out_of_range: Number of images with pixel values outside ``[0, 255]``.
        degenerate: Number of near-constant (degenerate) images.
    """

    train_duplicates: int
    train_test_overlap: int
    label_conflicts: int
    out_of_range: int
    degenerate: int

    def to_frame(self) -> pd.DataFrame:
        """Return the report as a two-column dataframe (check, count)."""
        return pd.DataFrame(
            {
                "check": [
                    "train_duplicates",
                    "train_test_overlap",
                    "label_conflicts",
                    "out_of_range",
                    "degenerate",
                ],
                "count": [
                    self.train_duplicates,
                    self.train_test_overlap,
                    self.label_conflicts,
                    self.out_of_range,
                    self.degenerate,
                ],
            }
        )


def _image_keys(images: NDArray[np.uint8]) -> NDArray[np.void]:
    """Return a 1-D array of per-image byte keys for exact matching."""
    arr = np.ascontiguousarray(images.reshape(images.shape[0], -1))
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))).ravel()


def count_duplicate_images(images: NDArray[np.uint8]) -> int:
    """Return the number of duplicate images (extra copies beyond the first)."""
    keys = _image_keys(images)
    return int(len(keys) - len(np.unique(keys)))


def count_overlap(source: NDArray[np.uint8], other: NDArray[np.uint8]) -> int:
    """Return how many images in ``other`` also appear in ``source``."""
    source_keys = set(_image_keys(source).tolist())
    other_keys = _image_keys(other)
    return int(sum(1 for k in other_keys.tolist() if k in source_keys))


def count_label_conflicts(images: NDArray[np.uint8], labels: NDArray[np.int64]) -> int:
    """Return the number of rows whose image appears with more than one label."""
    keys = _image_keys(images)
    frame = pd.DataFrame({"key": keys, "label": labels})
    distinct = frame.groupby("key")["label"].nunique()
    conflict_keys = distinct[distinct > 1].index
    return int(frame["key"].isin(conflict_keys).sum())


def count_degenerate(images: NDArray[np.uint8], std_threshold: float = 1.0) -> int:
    """Return the number of near-constant images (pixel std below threshold)."""
    flat = images.reshape(images.shape[0], -1).astype(np.float64)
    return int((flat.std(axis=1) < std_threshold).sum())


def check_quality(train: ImageBundle, test: ImageBundle) -> QualityReport:
    """Run all quality checks and return a :class:`QualityReport`.

    Args:
        train: Training split.
        test: Test split.

    Returns:
        The populated report.
    """
    out_of_range = int(((train.images < 0) | (train.images > PIXEL_MAX_VALUE)).any())
    report = QualityReport(
        train_duplicates=count_duplicate_images(train.images),
        train_test_overlap=count_overlap(train.images, test.images),
        label_conflicts=count_label_conflicts(train.images, train.labels),
        out_of_range=out_of_range,
        degenerate=count_degenerate(train.images),
    )
    logger.info("Quality report: %s", report)
    return report
