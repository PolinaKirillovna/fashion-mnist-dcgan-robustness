"""Tests for the DCGAN generator/discriminator and the auxiliary classifier."""

from __future__ import annotations

import torch

from gan_robustness.config import IMAGE_SIDE, NUM_CLASSES
from gan_robustness.models.classifier import Classifier
from gan_robustness.models.discriminator import Discriminator
from gan_robustness.models.generator import Generator, sample_latent
from gan_robustness.models.summary import count_parameters, init_dcgan_weights, summarize_model


def test_generator_output_shape_and_range() -> None:
    g = Generator(latent_dim=100, g_base=32)
    z = sample_latent(5, 100, torch.device("cpu"), seed=0)
    out = g(z)
    assert out.shape == (5, 1, IMAGE_SIDE, IMAGE_SIDE)
    assert out.min() >= -1.0 and out.max() <= 1.0  # tanh output


def test_discriminator_logit_and_features() -> None:
    d = Discriminator(d_base=32)
    x = torch.randn(5, 1, IMAGE_SIDE, IMAGE_SIDE)
    assert d(x).shape == (5,)
    feats = d.features(x)
    assert feats.ndim == 2 and feats.shape[0] == 5


def test_classifier_shapes() -> None:
    c = Classifier(base=16)
    x = torch.randn(4, 1, IMAGE_SIDE, IMAGE_SIDE)
    assert c(x).shape == (4, NUM_CLASSES)
    assert c.features(x).shape == (4, 128)


def test_sample_latent_is_reproducible() -> None:
    z1 = sample_latent(8, 100, torch.device("cpu"), seed=123)
    z2 = sample_latent(8, 100, torch.device("cpu"), seed=123)
    assert torch.allclose(z1, z2)


def test_summarize_model_totals() -> None:
    g = Generator(latent_dim=100, g_base=64)
    frame = summarize_model(g, "G")
    total_row = frame[frame["layer"] == "TOTAL"]["parameters"].iloc[0]
    assert int(total_row) == count_parameters(g)
    assert count_parameters(g) > 500_000  # order of ~1M for g_base=64


def test_init_dcgan_weights_runs() -> None:
    g = Generator(latent_dim=100, g_base=16)
    g.apply(init_dcgan_weights)  # should not raise
    assert count_parameters(g) > 0
