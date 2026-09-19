"""Auxiliary-classifier training loop (CPU, in-memory)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from gan_robustness.config import ClassifierConfig, ModelConfig
from gan_robustness.models.classifier import Classifier

logger = logging.getLogger(__name__)


@dataclass
class TrainedClassifier:
    """Result of classifier training.

    Attributes:
        model: The trained classifier (eval mode).
        test_accuracy: Accuracy on the test split.
    """

    model: Classifier
    test_accuracy: float


def train_classifier(
    train_images: NDArray[np.float32],
    train_labels: NDArray[np.int64],
    test_images: NDArray[np.float32],
    test_labels: NDArray[np.int64],
    model_config: ModelConfig,
    training_config: ClassifierConfig,
    device: torch.device,
    seed: int,
) -> TrainedClassifier:
    """Train the auxiliary classifier and evaluate it on the test split.

    Args:
        train_images: Training images ``(N, 1, 28, 28)`` in ``[-1, 1]``.
        train_labels: Training labels.
        test_images: Test images ``(M, 1, 28, 28)`` in ``[-1, 1]``.
        test_labels: Test labels.
        model_config: Architecture widths.
        training_config: Optimisation settings.
        device: Torch device.
        seed: Random seed.

    Returns:
        A :class:`TrainedClassifier`.
    """
    torch.manual_seed(seed)
    model = Classifier(model_config.classifier_base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    x_train = torch.from_numpy(train_images).to(device)
    y_train = torch.from_numpy(train_labels).to(device)
    n = x_train.shape[0]
    batch = training_config.batch_size
    perm_generator = torch.Generator(device="cpu")

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        perm_generator.manual_seed(seed + epoch)
        perm = torch.randperm(n, generator=perm_generator).to(device)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            optimizer.zero_grad()
            loss = criterion(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
        logger.info("classifier epoch %d/%d", epoch, training_config.epochs)

    accuracy = evaluate_accuracy(model, test_images, test_labels, device, batch)
    logger.info("Classifier test accuracy: %.4f", accuracy)
    model.eval()
    return TrainedClassifier(model=model, test_accuracy=accuracy)


def evaluate_accuracy(
    model: Classifier,
    images: NDArray[np.float32],
    labels: NDArray[np.int64],
    device: torch.device,
    batch: int = 512,
) -> float:
    """Return classification accuracy on ``(images, labels)``."""
    model.eval()
    x = torch.from_numpy(images).to(device)
    preds: list[NDArray[np.int64]] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            logits = model(x[i : i + batch])
            preds.append(logits.argmax(dim=1).cpu().numpy().astype(np.int64))
    y_pred = np.concatenate(preds)
    return float((y_pred == labels).mean())
