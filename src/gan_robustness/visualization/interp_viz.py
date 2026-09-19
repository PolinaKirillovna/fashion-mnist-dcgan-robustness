"""Figures for interpretation (Part 5)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from gan_robustness.config import IMAGE_SIDE
from gan_robustness.visualization.style import save_figure


def plot_interpolations(rows: NDArray[np.uint8], title: str, out: str | Path) -> Path:
    """Plot slerp interpolation rows (``(P, steps, 28, 28)``)."""
    p, steps = rows.shape[0], rows.shape[1]
    fig, axes = plt.subplots(p, steps, figsize=(steps, p), squeeze=False)
    for i in range(p):
        for j in range(steps):
            axes[i][j].imshow(rows[i, j], cmap="gray")
            axes[i][j].axis("off")
    fig.suptitle(title)
    return save_figure(fig, out)


def plot_latent_sensitivity(
    sensitivities: dict[str, NDArray[np.float64]], title: str, out: str | Path
) -> Path:
    """Plot sorted per-component latent sensitivity for one or more models."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, sens in sensitivities.items():
        ax.plot(np.sort(sens)[::-1], marker=".", ms=3, label=label)
    ax.set_xlabel("Компонента z (по убыванию чувствительности)")
    ax.set_ylabel(r"$\|\partial G(z)/\partial z_i\|$")
    ax.set_title(title)
    ax.legend()
    return save_figure(fig, out)


def plot_feature_umap(
    coords: NDArray[np.float64],
    groups: list[str],
    group_order: list[str],
    title: str,
    out: str | Path,
) -> Path:
    """Scatter a 2-D feature projection coloured by group."""
    fig, ax = plt.subplots(figsize=(7, 6))
    groups_arr = np.array(groups)
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(group_order):
        mask = groups_arr == name
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.5, color=cmap(i), label=name)
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    return save_figure(fig, out)


def plot_generator_activations(
    activations: list[tuple[str, NDArray[np.float64]]], title: str, out: str | Path
) -> Path:
    """Plot the mean feature map at each generator stage (7x7 -> 14x14 -> 28x28)."""
    n = len(activations)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6), squeeze=False)
    for ax, (name, act) in zip(axes[0], activations, strict=True):
        ax.imshow(act.mean(axis=0), cmap="viridis")
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    return save_figure(fig, out)


def plot_filters(filters: NDArray[np.float64], title: str, out: str | Path, ncols: int = 8) -> Path:
    """Plot first-layer discriminator filters (``(C, k, k)``)."""
    n = filters.shape[0]
    nrows = int(np.ceil(n / ncols))
    vmax = float(np.abs(filters).max()) or 1.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows), squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        ax.axis("off")
        if i < n:
            ax.imshow(filters[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.suptitle(title)
    return save_figure(fig, out)


def plot_sensitivity_maps(
    image: NDArray[np.float32],
    maps: dict[str, NDArray[np.float64]],
    title: str,
    out: str | Path,
) -> Path:
    """Plot the image and its saliency/Grad-CAM/occlusion maps side by side."""
    ncols = 1 + len(maps)
    fig, axes = plt.subplots(1, ncols, figsize=(2.4 * ncols, 2.6), squeeze=False)
    axes[0][0].imshow(image.reshape(IMAGE_SIDE, IMAGE_SIDE), cmap="gray")
    axes[0][0].set_title("изображение", fontsize=9)
    axes[0][0].axis("off")
    for ax, (name, m) in zip(axes[0][1:], maps.items(), strict=True):
        ax.imshow(m, cmap="hot")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    return save_figure(fig, out)


def plot_weight_histograms(
    arrays: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]], title: str, out: str | Path
) -> Path:
    """Overlay weight and bias histograms for one or more models."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, (weights, biases) in arrays.items():
        axes[0].hist(weights, bins=80, alpha=0.5, density=True, label=label)
        axes[1].hist(biases, bins=60, alpha=0.5, density=True, label=label)
    axes[0].set_title("Веса")
    axes[0].set_xlabel("Значение")
    axes[0].legend()
    axes[1].set_title("Смещения")
    axes[1].set_xlabel("Значение")
    axes[1].legend()
    fig.suptitle(title)
    return save_figure(fig, out)
