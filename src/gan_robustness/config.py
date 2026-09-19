"""Experiment configuration.

Frozen dataclasses describing every tunable parameter, plus a YAML loader. No
module reads magic numbers directly; they all flow through here so experiments
are reproducible and self-documenting.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, cast

import yaml

# Fixed by the Fashion-MNIST problem itself.
IMAGE_SIDE: int = 28
NUM_CHANNELS: int = 1
NUM_CLASSES: int = 10
PIXEL_MAX_VALUE: int = 255
CLASS_NAMES: tuple[str, ...] = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations.

    Attributes:
        raw_dir: Directory for the downloaded Fashion-MNIST (not tracked in git).
        processed_dir: Output directory for processed tensors.
        artifacts_dir: Output directory for checkpoints and samples.
        figures_dir: Output directory for report figures.
        tables_dir: Output directory for report tables.
    """

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    figures_dir: Path = Path("reports/figures")
    tables_dir: Path = Path("reports/tables")


@dataclass(frozen=True)
class DataConfig:
    """Dataset options.

    Attributes:
        subsample_size: If set, use a stratified subsample of the training set
            (kept class-proportional) to bound CPU epoch time; ``None`` = full 60k.
    """

    subsample_size: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    """DCGAN and classifier architecture widths.

    Attributes:
        latent_dim: Dimensionality of the latent vector z.
        g_base: Base channel width of the generator (7x7 layer has ``2*g_base``).
        d_base: Base channel width of the discriminator.
        classifier_base: Base channel width of the auxiliary classifier.
    """

    latent_dim: int = 100
    g_base: int = 64
    d_base: int = 64
    classifier_base: int = 32


@dataclass(frozen=True)
class GanTrainingConfig:
    """DCGAN optimisation settings.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        beta1: Adam beta1 (0.5 per DCGAN).
        beta2: Adam beta2.
        label_smoothing_real: Target label for real samples (one-sided smoothing).
        n_fixed_samples: Number of fixed z vectors tracked across epochs.
    """

    epochs: int = 18
    batch_size: int = 256
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    label_smoothing_real: float = 0.9
    n_fixed_samples: int = 64


@dataclass(frozen=True)
class ClassifierConfig:
    """Auxiliary classifier training settings.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: AdamW learning rate.
    """

    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class EvalConfig:
    """Generator-evaluation settings.

    Attributes:
        num_generated: Number of generated images used for metrics.
        precision_recall_k: k for the k-NN manifold precision/recall.
        umap_points: Points per group in UMAP projections.
        memorization_sample: Number of generations checked for nearest-neighbour
            memorisation against the training set.
        low_confidence_threshold: Classifier confidence below which a generation
            is counted as low-confidence.
        dropped_class_threshold: Generated-class share below which a class counts
            as "dropped".
    """

    num_generated: int = 10_000
    precision_recall_k: int = 3
    umap_points: int = 3_000
    memorization_sample: int = 500
    low_confidence_threshold: float = 0.6
    dropped_class_threshold: float = 0.02


@dataclass(frozen=True)
class ImbalanceConfig:
    """Variant A (artificial class imbalance) settings.

    Attributes:
        drop_classes: Classes whose objects are largely removed from training.
        drop_fraction: Fraction of each dropped class removed.
    """

    drop_classes: tuple[int, ...] = (5, 8, 9)  # Sandal, Bag, Ankle boot
    drop_fraction: float = 0.95


@dataclass(frozen=True)
class PoisonConfig:
    """Variant B (data poisoning with a visual trigger) settings.

    Attributes:
        poison_class: Class whose images receive the trigger.
        epsilons: Poisoning fractions to sweep.
        trigger_size: Side of the square trigger in pixels.
        trigger_value: Trigger pixel value on the [-1, 1] scale (1.0 = white).
        detection_threshold: Mean-patch-value threshold for the trigger detector.
    """

    poison_class: int = 0  # T-shirt/top
    epsilons: tuple[float, ...] = (0.05, 0.20)
    trigger_size: int = 4
    trigger_value: float = 1.0
    detection_threshold: float = 0.5


@dataclass(frozen=True)
class Config:
    """Top-level experiment configuration.

    Attributes:
        seed: Global random seed.
        num_threads: Torch CPU thread count (0 = leave default).
        paths: Filesystem locations.
        data: Dataset options.
        model: Architecture widths.
        gan: DCGAN training settings.
        classifier: Classifier training settings.
        evaluation: Evaluation settings.
        imbalance: Variant A settings.
        poison: Variant B settings.
    """

    seed: int = 42
    num_threads: int = 0
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    gan: GanTrainingConfig = field(default_factory=GanTrainingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    poison: PoisonConfig = field(default_factory=PoisonConfig)


_CONFIG_TYPES: dict[str, type] = {
    "PathsConfig": PathsConfig,
    "DataConfig": DataConfig,
    "ModelConfig": ModelConfig,
    "GanTrainingConfig": GanTrainingConfig,
    "ClassifierConfig": ClassifierConfig,
    "EvalConfig": EvalConfig,
    "ImbalanceConfig": ImbalanceConfig,
    "PoisonConfig": PoisonConfig,
}


def _coerce[T](cls: type[T], data: Any) -> T:
    """Recursively build a (possibly nested) dataclass from a mapping."""
    if not is_dataclass(cls):
        return cast("T", data)
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            raise TypeError(f"Unknown config key '{key}' for {cls.__name__}")
        kwargs[key] = _resolve_value(known[key].type, value)
    return cls(**kwargs)


def _resolve_value(field_type: Any, value: Any) -> Any:
    """Coerce a single field value based on its declared type."""
    type_name = field_type if isinstance(field_type, str) else getattr(field_type, "__name__", "")
    if type_name.endswith("Config") and isinstance(value, dict):
        return _coerce(_CONFIG_TYPES[type_name], value)
    if type_name == "Path":
        return Path(value)
    if isinstance(value, list):
        return tuple(value)
    return value


def load_config(path: str | Path | None = None) -> Config:
    """Load a :class:`Config` from a YAML file, falling back to defaults.

    Args:
        path: Path to a YAML config file. If ``None``, returns default config.

    Returns:
        A fully populated :class:`Config`.
    """
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _coerce(Config, raw)
