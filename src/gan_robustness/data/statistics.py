"""Per-class distribution and brightness statistics (pure functions)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gan_robustness.config import CLASS_NAMES, NUM_CLASSES


def class_counts(labels: NDArray[np.int64], num_classes: int = NUM_CLASSES) -> NDArray[np.int64]:
    """Return per-class sample counts."""
    return np.bincount(labels, minlength=num_classes).astype(np.int64)


def class_distribution_frame(labels: NDArray[np.int64]) -> pd.DataFrame:
    """Return a dataframe of per-class counts, proportions and class names."""
    counts = class_counts(labels)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "class": np.arange(NUM_CLASSES),
            "class_name": list(CLASS_NAMES),
            "count": counts,
            "proportion": counts / total if total else np.zeros(NUM_CLASSES),
        }
    )


def brightness_by_class(images: NDArray[np.uint8], labels: NDArray[np.int64]) -> pd.DataFrame:
    """Return mean/std pixel intensity and active-fraction per class.

    Args:
        images: Array of shape ``(N, 28, 28)`` in ``[0, 255]``.
        labels: Class labels.

    Returns:
        A dataframe indexed by class with brightness statistics.
    """
    flat = images.reshape(images.shape[0], -1).astype(np.float64)
    mean_intensity = flat.mean(axis=1)
    active_fraction = (flat > 0).mean(axis=1)
    frame = pd.DataFrame(
        {"class": labels, "mean_intensity": mean_intensity, "active_fraction": active_fraction}
    )
    grouped = frame.groupby("class").agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "class",
        "mean_intensity_mean",
        "mean_intensity_std",
        "active_fraction_mean",
        "active_fraction_std",
    ]
    grouped["class_name"] = [CLASS_NAMES[c] for c in grouped["class"]]
    return grouped


def per_image_brightness(images: NDArray[np.uint8], labels: NDArray[np.int64]) -> pd.DataFrame:
    """Return a per-image dataframe of class and mean intensity (for box plots)."""
    flat = images.reshape(images.shape[0], -1).astype(np.float64)
    return pd.DataFrame({"class": labels, "mean_intensity": flat.mean(axis=1)})
