"""Tests for generative metrics (synthetic)."""

from __future__ import annotations

import numpy as np

from gan_robustness.evaluation.metrics import (
    class_proportions,
    count_dropped_classes,
    frechet_distance,
    js_divergence,
    mean_pairwise_distance,
    precision_recall,
)


def test_js_divergence_bounds() -> None:
    p = np.full(10, 0.1)
    assert js_divergence(p.copy(), p.copy()) < 1e-9
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(js_divergence(a.copy(), b.copy()) - 1.0) < 1e-6  # disjoint -> 1 bit


def test_class_proportions_and_dropped() -> None:
    labels = np.array([0, 0, 0, 1, 2], dtype=np.int64)
    props = class_proportions(labels)
    assert abs(props[0] - 0.6) < 1e-9
    assert count_dropped_classes(props, threshold=0.05) == 7  # classes 3..9 absent


def test_frechet_distance_zero_for_identical() -> None:
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(200, 16))
    assert frechet_distance(feats, feats.copy()) < 1e-3


def test_precision_recall_high_for_identical() -> None:
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(300, 8))
    precision, recall = precision_recall(feats, feats.copy(), k=3)
    assert precision > 0.9 and recall > 0.9


def test_precision_recall_low_for_disjoint() -> None:
    rng = np.random.default_rng(2)
    real = rng.normal(loc=0.0, size=(300, 8))
    gen = rng.normal(loc=20.0, size=(300, 8))  # far away
    precision, recall = precision_recall(real, gen, k=3)
    assert precision < 0.1 and recall < 0.1


def test_mean_pairwise_distance_positive() -> None:
    rng = np.random.default_rng(3)
    feats = rng.normal(size=(50, 8))
    assert mean_pairwise_distance(feats) > 0.0
