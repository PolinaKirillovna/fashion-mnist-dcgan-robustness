"""Model weight initialisation, parameter counting and layer summaries."""

from __future__ import annotations

import pandas as pd
from torch import nn


def init_dcgan_weights(module: nn.Module) -> None:
    """Apply the DCGAN weight initialisation in place.

    Convolutional weights ~ N(0, 0.02); BatchNorm weights ~ N(1, 0.02), bias 0.
    Apply via ``model.apply(init_dcgan_weights)``.

    Args:
        module: A module to initialise.
    """
    name = module.__class__.__name__
    if "Conv" in name and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summarize_model(model: nn.Module, name: str) -> pd.DataFrame:
    """Return a per-layer parameter summary of parameter-owning submodules.

    Args:
        model: The model to summarise.
        name: A name for the model (added as a column).

    Returns:
        A dataframe with columns ``model``, ``layer``, ``type`` and ``parameters``,
        plus a final ``TOTAL`` row.
    """
    rows: list[dict[str, object]] = []
    for layer_name, module in model.named_modules():
        own = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        if own > 0:
            rows.append(
                {
                    "model": name,
                    "layer": layer_name,
                    "type": module.__class__.__name__,
                    "parameters": int(own),
                }
            )
    rows.append(
        {"model": name, "layer": "TOTAL", "type": "", "parameters": count_parameters(model)}
    )
    return pd.DataFrame(rows)
