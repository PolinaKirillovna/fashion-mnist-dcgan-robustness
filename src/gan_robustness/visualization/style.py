"""Shared plotting style and helpers (headless backend, fixed DPI, Russian captions)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

FIGURE_DPI = 200


def apply_style() -> None:
    """Apply the shared seaborn/matplotlib style."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = FIGURE_DPI
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["font.size"] = 10


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Save a figure at the report DPI and close it.

    Args:
        fig: Matplotlib figure.
        path: Destination path (parent directories are created).

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out
