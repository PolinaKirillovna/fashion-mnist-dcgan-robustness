"""Tests for the data-corruption attacks (synthetic)."""

from __future__ import annotations

import numpy as np

from gan_robustness.attacks.imbalance import apply_imbalance
from gan_robustness.attacks.poisoning import (
    apply_poison,
    detect_trigger,
    stamp_trigger,
    trigger_score,
)
from gan_robustness.config import IMAGE_SIDE
from gan_robustness.data.loading import ImageBundle


def _bundle(n_per_class: int = 20) -> ImageBundle:
    rng = np.random.default_rng(0)
    images = rng.integers(0, 60, size=(n_per_class * 10, IMAGE_SIDE, IMAGE_SIDE), dtype=np.uint8)
    labels = np.repeat(np.arange(10), n_per_class).astype(np.int64)
    return ImageBundle(images, labels, "syn")


def test_imbalance_reduces_target_classes() -> None:
    bundle = _bundle(20)
    out = apply_imbalance(bundle, drop_classes=(5, 8), drop_fraction=0.9, seed=1)
    counts = np.bincount(out.labels, minlength=10)
    assert counts[5] == 2 and counts[8] == 2  # 10% of 20
    assert counts[0] == 20  # untouched


def test_poison_stamps_and_detector_finds_trigger() -> None:
    bundle = _bundle(20)
    poisoned, idx = apply_poison(bundle, poison_class=0, epsilon=0.5, trigger_size=4, seed=1)
    assert idx.size == 10  # 50% of 20
    # stamped images score ~1.0 in the top-left patch
    scores = trigger_score(poisoned.images[idx], trigger_size=4)
    assert (scores > 0.9).all()
    detected = detect_trigger(poisoned.images, trigger_size=4, threshold=0.5)
    assert int(detected.sum()) == 10


def test_stamp_trigger_is_white_corner() -> None:
    imgs = np.zeros((2, IMAGE_SIDE, IMAGE_SIDE), dtype=np.uint8)
    out = stamp_trigger(imgs, trigger_size=4)
    assert (out[:, :4, :4] == 255).all()
    assert (out[:, 4:, 4:] == 0).all()
