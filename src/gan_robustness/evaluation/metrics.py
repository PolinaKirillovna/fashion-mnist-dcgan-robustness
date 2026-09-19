"""Generative-model metrics.

All metrics operate in the auxiliary-classifier feature space (penultimate layer)
or in pixel space, so they are honest "domain-FID" style measures rather than
Inception-FID. See ``docs/metrics.md`` for definitions and caveats.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from gan_robustness.config import NUM_CLASSES

logger = logging.getLogger(__name__)


def class_proportions(
    labels: NDArray[np.int64], num_classes: int = NUM_CLASSES
) -> NDArray[np.float64]:
    """Return the class-proportion vector of ``labels``."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


def js_divergence(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Return the Jensen-Shannon divergence (bits) between two distributions."""
    eps = 1e-12
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def _kl(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def count_dropped_classes(proportions: NDArray[np.float64], threshold: float = 0.02) -> int:
    """Return how many classes fall below ``threshold`` of the generations."""
    return int((proportions < threshold).sum())


def frechet_distance(feat_real: NDArray[np.float64], feat_gen: NDArray[np.float64]) -> float:
    """Return the Frechet distance between two feature sets (domain-FID).

    Args:
        feat_real: Real-image features ``(N, D)``.
        feat_gen: Generated-image features ``(M, D)``.

    Returns:
        The Frechet distance, or ``inf`` if it cannot be computed.
    """
    if feat_real.shape[0] < 2 or feat_gen.shape[0] < 2:
        return float("inf")
    mu1, mu2 = feat_real.mean(axis=0), feat_gen.mean(axis=0)
    cov1 = np.cov(feat_real, rowvar=False)
    cov2 = np.cov(feat_gen, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * covmean))


def domain_fid_per_class(
    feat_real: NDArray[np.float64],
    labels_real: NDArray[np.int64],
    feat_gen: NDArray[np.float64],
    labels_gen: NDArray[np.int64],
    min_count: int = 20,
) -> dict[int, float]:
    """Return per-class Frechet distance (NaN where a class is too rare)."""
    result: dict[int, float] = {}
    for cls in range(NUM_CLASSES):
        real_c = feat_real[labels_real == cls]
        gen_c = feat_gen[labels_gen == cls]
        if real_c.shape[0] < min_count or gen_c.shape[0] < min_count:
            result[cls] = float("nan")
        else:
            result[cls] = frechet_distance(real_c, gen_c)
    return result


def _knn_radii(features: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Return the distance to the k-th nearest neighbour within ``features``."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(features)
    distances, _ = nn.kneighbors(features)
    return cast("NDArray[np.float64]", distances[:, k].astype(np.float64))


def _fraction_in_manifold(
    query: NDArray[np.float64], reference: NDArray[np.float64], radii: NDArray[np.float64]
) -> float:
    """Return the fraction of query points inside the reference hypersphere union."""
    inside = np.zeros(query.shape[0], dtype=bool)
    block = 1000
    for i in range(0, query.shape[0], block):
        dists = cdist(query[i : i + block], reference)
        inside[i : i + block] = (dists <= radii[None, :]).any(axis=1)
    return float(inside.mean())


def precision_recall(
    feat_real: NDArray[np.float64], feat_gen: NDArray[np.float64], k: int = 3
) -> tuple[float, float]:
    """Return improved precision/recall (Kynkaanniemi et al., 2019).

    Precision = fraction of generated features inside the real manifold
    (realism); recall = fraction of real features inside the generated manifold
    (mode coverage).

    Args:
        feat_real: Real-image features ``(N, D)``.
        feat_gen: Generated-image features ``(M, D)``.
        k: Neighbourhood size defining the manifold.

    Returns:
        ``(precision, recall)``.
    """
    radii_real = _knn_radii(feat_real, k)
    radii_gen = _knn_radii(feat_gen, k)
    precision = _fraction_in_manifold(feat_gen, feat_real, radii_real)
    recall = _fraction_in_manifold(feat_real, feat_gen, radii_gen)
    return precision, recall


def nearest_neighbour_distances(
    query_pixels: NDArray[np.float32], train_pixels: NDArray[np.float32]
) -> NDArray[np.float64]:
    """Return the L2 distance from each query image to its nearest train image."""
    nn = NearestNeighbors(n_neighbors=1).fit(train_pixels)
    distances, _ = nn.kneighbors(query_pixels)
    return cast("NDArray[np.float64]", distances[:, 0].astype(np.float64))


def mean_pairwise_distance(features: NDArray[np.float64], sample: int = 1000) -> float:
    """Return the mean pairwise distance among a sample (diversity proxy)."""
    n = min(sample, features.shape[0])
    sub = features[:n]
    dists = cdist(sub, sub)
    iu = np.triu_indices(n, k=1)
    return float(dists[iu].mean()) if iu[0].size else 0.0
