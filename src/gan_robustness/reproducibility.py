"""Reproducibility and CPU-thread helpers."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for deterministic runs.

    Args:
        seed: The global seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def configure_cpu(num_threads: int) -> None:
    """Configure the torch CPU thread count.

    Args:
        num_threads: Desired thread count; ``0`` or negative leaves the default.
    """
    import torch

    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info("Torch CPU threads: %d", torch.get_num_threads())
