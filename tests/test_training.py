"""Fast training-loop smoke tests (1 epoch on tiny synthetic data)."""

from __future__ import annotations

import numpy as np
import torch

from gan_robustness.config import GanTrainingConfig, ModelConfig
from gan_robustness.evaluation.generate import generate_float, to_uint8
from gan_robustness.training.gan_trainer import train_gan


def test_train_gan_one_epoch_shapes() -> None:
    images = np.random.default_rng(0).uniform(-1, 1, size=(64, 1, 28, 28)).astype(np.float32)
    mc = ModelConfig(latent_dim=32, g_base=16, d_base=16)
    gc = GanTrainingConfig(epochs=1, batch_size=32, n_fixed_samples=9)
    result = train_gan(images, mc, gc, torch.device("cpu"), seed=0)
    assert len(result.history) == 1
    assert set(result.history[0]) == {"epoch", "d_loss", "g_loss", "d_x", "d_g_z"}
    assert len(result.storyboard) == 1
    assert result.storyboard[0].shape == (9, 28, 28)


def test_generate_float_and_uint8() -> None:
    mc = ModelConfig(latent_dim=32, g_base=16, d_base=16)
    gc = GanTrainingConfig(epochs=1, batch_size=32, n_fixed_samples=4)
    images = np.random.default_rng(1).uniform(-1, 1, size=(64, 1, 28, 28)).astype(np.float32)
    result = train_gan(images, mc, gc, torch.device("cpu"), seed=0)
    gen = generate_float(result.generator, 10, 32, torch.device("cpu"), seed=0)
    assert gen.shape == (10, 1, 28, 28)
    assert gen.min() >= -1.0 and gen.max() <= 1.0
    u8 = to_uint8(gen)
    assert u8.shape == (10, 28, 28) and u8.dtype == np.uint8
