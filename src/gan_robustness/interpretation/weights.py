"""Weight and bias collection for histogram interpretation (methodichka LR3)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from torch import nn


def weight_bias_arrays(model: nn.Module) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return flattened weights and biases of all conv/linear layers of a model.

    Args:
        model: The model to inspect.

    Returns:
        ``(weights, biases)`` as 1-D NumPy arrays.
    """
    weights: list[NDArray[np.float64]] = []
    biases: list[NDArray[np.float64]] = []
    for name, param in model.named_parameters():
        values = param.detach().cpu().numpy().ravel().astype(np.float64)
        if name.endswith("bias"):
            biases.append(values)
        else:
            weights.append(values)
    weight_arr = np.concatenate(weights) if weights else np.zeros(0)
    bias_arr = np.concatenate(biases) if biases else np.zeros(0)
    return weight_arr, bias_arr
