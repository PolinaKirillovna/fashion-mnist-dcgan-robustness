"""Tests for configuration loading and coercion."""

from __future__ import annotations

from pathlib import Path

import pytest

from gan_robustness.config import Config, load_config


def test_default_config() -> None:
    config = Config()
    assert config.seed == 42
    assert config.model.latent_dim == 100
    assert config.imbalance.drop_classes == (5, 8, 9)
    assert config.paths.raw_dir == Path("data/raw")


def test_load_config_from_yaml(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    text = """
seed: 7
model:
  g_base: 32
data:
  subsample_size: 30000
poison:
  epsilons: [0.1, 0.3]
"""
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    config = load_config(path)
    assert config.seed == 7
    assert config.model.g_base == 32
    assert config.data.subsample_size == 30000
    assert config.poison.epsilons == (0.1, 0.3)
    # Unspecified fields keep defaults.
    assert config.gan.epochs == 18


def test_load_config_none_returns_defaults() -> None:
    assert load_config(None).seed == 42


def test_load_config_rejects_unknown_key(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    path = tmp_path / "c.yaml"
    path.write_text("bogus: 1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Unknown config key"):
        load_config(path)
