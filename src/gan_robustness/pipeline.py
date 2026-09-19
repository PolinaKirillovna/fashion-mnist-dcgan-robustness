"""High-level pipeline orchestration.

Wires the data/model/training/evaluation/interpretation layers into the stages
invoked by the CLI. Owns side effects (downloads, checkpoints, tables, figures);
the underlying layers stay pure. Stages are implemented incrementally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gan_robustness.config import Config
from gan_robustness.data.loading import (
    download_fashion_mnist,
    load_split,
    stratified_subsample,
)
from gan_robustness.data.quality import check_quality
from gan_robustness.data.statistics import (
    brightness_by_class,
    class_distribution_frame,
    per_image_brightness,
)
from gan_robustness.projection import umap_project
from gan_robustness.reproducibility import configure_cpu, seed_everything
from gan_robustness.visualization import data_viz as dviz
from gan_robustness.visualization.style import apply_style

logger = logging.getLogger(__name__)


def _write_table(frame: pd.DataFrame, tables_dir: Path, name: str) -> None:
    """Write a dataframe to ``tables_dir/name`` as CSV."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables_dir / name, index=False)
    logger.info("Wrote table %s (%d rows)", name, len(frame))


def run_data_report(config: Config) -> None:
    """Stage 1: download Fashion-MNIST and write the data-quality report.

    Args:
        config: Loaded experiment configuration.
    """
    seed_everything(config.seed)
    configure_cpu(config.num_threads)
    apply_style()
    paths = config.paths
    tables, figures = paths.tables_dir, paths.figures_dir

    download_fashion_mnist(paths.raw_dir)
    train = load_split(paths.raw_dir, train=True)
    test = load_split(paths.raw_dir, train=False)

    # Tables.
    _write_table(class_distribution_frame(train.labels), tables, "part1_class_distribution.csv")
    _write_table(brightness_by_class(train.images, train.labels), tables, "part1_brightness.csv")
    report = check_quality(train, test)
    _write_table(report.to_frame(), tables, "part1_quality.csv")

    # Figures.
    dviz.plot_class_examples(train.images, train.labels, figures / "part1_examples.png")
    dviz.plot_class_histogram(
        train.labels,
        "Распределение классов Fashion-MNIST (train)",
        figures / "part1_class_histogram.png",
    )
    dviz.plot_brightness_boxplot(
        per_image_brightness(train.images, train.labels), figures / "part1_brightness.png"
    )

    sub = stratified_subsample(train, config.evaluation.umap_points, config.seed)
    coords = umap_project(sub.images.reshape(len(sub), -1).astype(np.float64), config.seed)
    dviz.plot_umap(
        coords, sub.labels, "UMAP сырых пикселей (подвыборка)", figures / "part1_umap_pixels.png"
    )

    summary = {
        "n_train": len(train),
        "n_test": len(test),
        "class_counts_train": np.bincount(train.labels, minlength=10).tolist(),
        "quality": {
            "train_duplicates": report.train_duplicates,
            "train_test_overlap": report.train_test_overlap,
            "label_conflicts": report.label_conflicts,
            "out_of_range": report.out_of_range,
            "degenerate": report.degenerate,
        },
    }
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "part1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Part 1 summary: %s", summary)


def run_benchmark(config: Config) -> None:
    """Stage 3 prep: measure one DCGAN epoch on CPU to guide width selection."""
    raise NotImplementedError("Implemented in stage 3 (training).")


def run_classifier(config: Config) -> None:
    """Stage 3: train the auxiliary Fashion-MNIST classifier."""
    raise NotImplementedError("Implemented in stage 3 (training).")


def run_scenario(config: Config, scenario: str) -> None:
    """Stage 3/4: train and evaluate a DCGAN under the given scenario."""
    raise NotImplementedError("Implemented in stages 3-4 (training and attacks).")


def run_interpretation(config: Config) -> None:
    """Stage 5: run interpretation across the trained models."""
    raise NotImplementedError("Implemented in stage 5 (interpretation).")


def run_all(config: Config) -> None:
    """Run the full pipeline end to end."""
    raise NotImplementedError("Implemented across stages 3-5.")
