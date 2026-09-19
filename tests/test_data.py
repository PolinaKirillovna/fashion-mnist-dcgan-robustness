"""Tests for data loading helpers and quality checks (synthetic)."""

from __future__ import annotations

import numpy as np

from gan_robustness.config import IMAGE_SIDE
from gan_robustness.data.loading import ImageBundle, stratified_subsample, to_float_pm1
from gan_robustness.data.quality import (
    count_degenerate,
    count_duplicate_images,
    count_label_conflicts,
    count_overlap,
)
from gan_robustness.data.statistics import class_counts, class_distribution_frame


def _img(v: int) -> np.ndarray:
    return np.full((IMAGE_SIDE, IMAGE_SIDE), v, dtype=np.uint8)


def _bundle(images: list[np.ndarray], labels: list[int]) -> ImageBundle:
    return ImageBundle(np.stack(images), np.array(labels, dtype=np.int64), "syn")


def test_to_float_pm1_range_and_shape() -> None:
    real = np.stack([_img(0), _img(255)])
    out = to_float_pm1(real)
    assert out.shape == (2, 1, IMAGE_SIDE, IMAGE_SIDE)
    assert out.min() == -1.0 and out.max() == 1.0


def test_duplicate_and_overlap_counts() -> None:
    a, b, c = _img(10), _img(20), _img(30)
    train = _bundle([a, a, b], [0, 0, 1])  # one duplicate
    test = _bundle([a, c], [0, 2])  # a overlaps train
    assert count_duplicate_images(train.images) == 1
    assert count_overlap(train.images, test.images) == 1


def test_label_conflicts() -> None:
    a = _img(10)
    conflict = _bundle([a, a], [0, 1])  # same image, different labels
    same = _bundle([a, a], [0, 0])
    assert count_label_conflicts(conflict.images, conflict.labels) == 2
    assert count_label_conflicts(same.images, same.labels) == 0


def test_degenerate_detection() -> None:
    const = _img(0)
    rng = np.random.default_rng(0)
    varied = rng.integers(0, 256, size=(IMAGE_SIDE, IMAGE_SIDE), dtype=np.uint8)
    bundle = _bundle([const, varied], [0, 1])
    assert count_degenerate(bundle.images) == 1


def test_stratified_subsample_balance() -> None:
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(200, IMAGE_SIDE, IMAGE_SIDE), dtype=np.uint8)
    labels = np.repeat(np.arange(10), 20).astype(np.int64)
    bundle = ImageBundle(images, labels, "syn")
    sub = stratified_subsample(bundle, 50, seed=1)
    counts = class_counts(sub.labels)
    assert set(counts.tolist()) == {5}


def test_class_distribution_frame() -> None:
    labels = np.repeat(np.arange(10), 3).astype(np.int64)
    frame = class_distribution_frame(labels)
    assert list(frame["count"]) == [3] * 10
    assert frame["class_name"].iloc[0] == "T-shirt/top"
