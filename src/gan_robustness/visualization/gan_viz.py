"""Figures for GAN training and evaluation (Parts 3-4)."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

from gan_robustness.config import CLASS_NAMES, IMAGE_SIDE, NUM_CLASSES
from gan_robustness.visualization.style import save_figure


def tile_grid(images: NDArray[np.uint8], pad: int = 1) -> NDArray[np.uint8]:
    """Tile ``(K, 28, 28)`` images into a single square-ish grid image."""
    k = images.shape[0]
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    canvas = np.zeros(
        (rows * (IMAGE_SIDE + pad) + pad, cols * (IMAGE_SIDE + pad) + pad), dtype=np.uint8
    )
    for i in range(k):
        r, c = divmod(i, cols)
        y = pad + r * (IMAGE_SIDE + pad)
        x = pad + c * (IMAGE_SIDE + pad)
        canvas[y : y + IMAGE_SIDE, x : x + IMAGE_SIDE] = images[i]
    return canvas


def plot_sample_grid(images: NDArray[np.uint8], title: str, out: str | Path) -> Path:
    """Plot a grid of generated ``uint8`` images."""
    grid = tile_grid(images)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="gray")
    ax.axis("off")
    ax.set_title(title)
    return save_figure(fig, out)


def plot_loss_curves(history: list[dict[str, float]], title: str, out: str | Path) -> Path:
    """Plot D/G loss and D(x)/D(G(z)) curves over epochs."""
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame["epoch"], frame["d_loss"], label="D", marker="o", ms=3)
    axes[0].plot(frame["epoch"], frame["g_loss"], label="G", marker="s", ms=3)
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Функция потерь")
    axes[0].legend()
    axes[0].set_title("Потери D и G")
    axes[1].plot(frame["epoch"], frame["d_x"], label="D(x)", marker="o", ms=3)
    axes[1].plot(frame["epoch"], frame["d_g_z"], label="D(G(z))", marker="s", ms=3)
    axes[1].axhline(0.5, color="gray", ls="--", lw=0.8)
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Средняя вероятность")
    axes[1].legend()
    axes[1].set_title("D(x) и D(G(z))")
    fig.suptitle(title)
    return save_figure(fig, out)


def save_storyboard_gif(storyboard: list[NDArray[np.uint8]], out: str | Path, fps: int = 2) -> Path:
    """Save the fixed-z training storyboard as an animated GIF."""
    frames = [tile_grid(frame) for frame in storyboard]
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(outp, frames, duration=1.0 / fps)  # type: ignore[arg-type]
    return outp


def plot_storyboard_strip(
    storyboard: list[NDArray[np.uint8]], epochs: list[int], title: str, out: str | Path
) -> Path:
    """Plot selected storyboard frames side by side (static figure for the report)."""
    fig, axes = plt.subplots(1, len(epochs), figsize=(2.2 * len(epochs), 2.4), squeeze=False)
    for ax, ep in zip(axes[0], epochs, strict=True):
        ax.imshow(tile_grid(storyboard[ep - 1]), cmap="gray")
        ax.axis("off")
        ax.set_title(f"эпоха {ep}", fontsize=9)
    fig.suptitle(title)
    return save_figure(fig, out)


def plot_class_distribution_compare(
    train_props: NDArray[np.float64], gen_props: NDArray[np.float64], title: str, out: str | Path
) -> Path:
    """Compare train vs generated class proportions."""
    frame = pd.DataFrame(
        {
            "class": list(range(NUM_CLASSES)) * 2,
            "Набор": ["train"] * NUM_CLASSES + ["генерации"] * NUM_CLASSES,
            "Доля": np.concatenate([train_props, gen_props]),
        }
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=frame, x="class", y="Доля", hue="Набор", ax=ax)
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Класс")
    ax.set_title(title)
    return save_figure(fig, out)


def plot_memorization(
    gen_distances: NDArray[np.float64], test_distances: NDArray[np.float64], out: str | Path
) -> Path:
    """Overlay histograms of gen->train and test->train nearest-neighbour distances."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(test_distances, bins=30, alpha=0.6, label="test -> train", density=True)
    ax.hist(gen_distances, bins=30, alpha=0.6, label="генерации -> train", density=True)
    ax.set_xlabel("L2-расстояние до ближайшего train-изображения (пиксели)")
    ax.set_ylabel("Плотность")
    ax.set_title("Проверка запоминания: расстояния до ближайшего соседа")
    ax.legend()
    return save_figure(fig, out)


def plot_examples(
    images: NDArray[np.uint8], captions: list[str], title: str, out: str | Path, ncols: int = 8
) -> Path:
    """Plot a labelled gallery of generated images."""
    n = len(images)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.7), squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i], cmap="gray")
            ax.set_title(captions[i], fontsize=7)
    fig.suptitle(title)
    return save_figure(fig, out)
