"""Figures for the data section (Part 1)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

from gan_robustness.config import CLASS_NAMES, IMAGE_SIDE, NUM_CLASSES
from gan_robustness.visualization.style import save_figure


def plot_class_examples(
    images: NDArray[np.uint8], labels: NDArray[np.int64], out: str | Path, per_class: int = 8
) -> Path:
    """Plot a grid of example images, one row per class (with class names)."""
    fig, axes = plt.subplots(
        NUM_CLASSES, per_class, figsize=(per_class, NUM_CLASSES), squeeze=False
    )
    rng = np.random.default_rng(0)
    for cls in range(NUM_CLASSES):
        cls_idx = np.flatnonzero(labels == cls)
        picks = rng.choice(cls_idx, size=min(per_class, cls_idx.size), replace=False)
        for col in range(per_class):
            ax = axes[cls][col]
            ax.axis("off")
            if col < picks.size:
                ax.imshow(images[picks[col]], cmap="gray")
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[cls], rotation=0, ha="right", va="center", fontsize=7)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle("Примеры изображений по классам Fashion-MNIST")
    return save_figure(fig, out)


def plot_class_histogram(labels: NDArray[np.int64], title: str, out: str | Path) -> Path:
    """Plot a bar chart of the number of objects per class."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=np.arange(NUM_CLASSES), y=counts, ax=ax, color="steelblue")
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Класс")
    ax.set_ylabel("Количество объектов")
    ax.set_title(title)
    return save_figure(fig, out)


def plot_brightness_boxplot(per_image: pd.DataFrame, out: str | Path) -> Path:
    """Plot per-class box plots of mean image intensity."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=per_image, x="class", y="mean_intensity", ax=ax, color="lightsteelblue")
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Класс")
    ax.set_ylabel("Средняя интенсивность пикселей")
    ax.set_title("Распределение яркости изображений по классам")
    return save_figure(fig, out)


def plot_umap(
    coords: NDArray[np.float64], labels: NDArray[np.int64], title: str, out: str | Path
) -> Path:
    """Plot a 2-D UMAP projection coloured by class."""
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=5, alpha=0.6)
    handles, _ = scatter.legend_elements()
    ax.legend(handles, CLASS_NAMES, title="Класс", fontsize=6, loc="best")
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    return save_figure(fig, out)


def plot_image_grid(
    images: NDArray[np.float32] | NDArray[np.uint8], title: str, out: str | Path, ncols: int = 8
) -> Path:
    """Plot a square-ish grid of images (values in [-1,1] float or uint8)."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows), squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        ax.axis("off")
        if i < n:
            img = images[i]
            if img.ndim == 3:
                img = img[0]
            ax.imshow(img.reshape(IMAGE_SIDE, IMAGE_SIDE), cmap="gray")
    fig.suptitle(title)
    return save_figure(fig, out)
