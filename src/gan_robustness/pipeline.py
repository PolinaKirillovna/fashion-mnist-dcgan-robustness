"""High-level pipeline orchestration.

Wires the data/model/training/evaluation/interpretation layers into the stages
invoked by the CLI. Owns side effects (downloads, checkpoints, tables, figures);
the underlying layers stay pure. Stages are implemented incrementally.
"""

from __future__ import annotations

import logging

from gan_robustness.config import Config

logger = logging.getLogger(__name__)


def run_data_report(config: Config) -> None:
    """Stage 1: download Fashion-MNIST and write the data-quality report."""
    raise NotImplementedError("Implemented in stage 1 (data & quality report).")


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
