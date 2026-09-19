"""Command-line entry points.

Usage: ``python -m gan_robustness <command> [--config path] [--scenario name]``.

The CLI is the outermost layer: it wires configuration to pipeline stages and
owns user-facing output. Stages are implemented in their respective layers.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from gan_robustness.config import Config, load_config

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure root logging for CLI runs."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_download(config: Config, scenario: str) -> int:
    """Download Fashion-MNIST and write the data-quality report."""
    from gan_robustness.pipeline import run_data_report

    run_data_report(config)
    return 0


def cmd_bench(config: Config, scenario: str) -> int:
    """Measure one DCGAN epoch on CPU to guide width selection."""
    from gan_robustness.pipeline import run_benchmark

    run_benchmark(config)
    return 0


def cmd_classifier(config: Config, scenario: str) -> int:
    """Train the auxiliary Fashion-MNIST classifier."""
    from gan_robustness.pipeline import run_classifier

    run_classifier(config)
    return 0


def cmd_train(config: Config, scenario: str) -> int:
    """Train and evaluate a DCGAN under the given scenario."""
    from gan_robustness.pipeline import run_scenario

    run_scenario(config, scenario)
    return 0


def cmd_interpret(config: Config, scenario: str) -> int:
    """Run interpretation across the trained models."""
    from gan_robustness.pipeline import run_interpretation

    run_interpretation(config)
    return 0


def cmd_all(config: Config, scenario: str) -> int:
    """Run the full pipeline: classifier, all scenarios, interpretation."""
    from gan_robustness.pipeline import run_all

    run_all(config)
    return 0


_COMMANDS = {
    "download": cmd_download,
    "bench": cmd_bench,
    "classifier": cmd_classifier,
    "train": cmd_train,
    "interpret": cmd_interpret,
    "all": cmd_all,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="gan-robustness",
        description="DCGAN on Fashion-MNIST: training, robustness and interpretation.",
    )
    parser.add_argument("command", choices=sorted(_COMMANDS), help="Pipeline command.")
    parser.add_argument("--config", default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--scenario",
        default="baseline",
        choices=["baseline", "imbalance", "poison"],
        help="Data scenario for the `train` command.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    config = load_config(args.config)
    return _COMMANDS[args.command](config, args.scenario)


if __name__ == "__main__":
    sys.exit(main())
