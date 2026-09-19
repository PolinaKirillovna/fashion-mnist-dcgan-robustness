"""High-level pipeline orchestration.

Wires the data/model/training/evaluation/interpretation layers into the stages
invoked by the CLI. Owns side effects (downloads, checkpoints, tables, figures);
the underlying layers stay pure. Stages are implemented incrementally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from gan_robustness.config import Config
from gan_robustness.data.loading import (
    ImageBundle,
    download_fashion_mnist,
    load_split,
    stratified_subsample,
    to_float_pm1,
)
from gan_robustness.data.quality import check_quality
from gan_robustness.data.statistics import (
    brightness_by_class,
    class_distribution_frame,
    per_image_brightness,
)
from gan_robustness.projection import umap_project
from gan_robustness.reproducibility import configure_cpu, seed_everything
from gan_robustness.visualization import data_viz as dviz
from gan_robustness.visualization.style import apply_style

logger = logging.getLogger(__name__)


def _write_table(frame: pd.DataFrame, tables_dir: Path, name: str) -> None:
    """Write a dataframe to ``tables_dir/name`` as CSV."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables_dir / name, index=False)
    logger.info("Wrote table %s (%d rows)", name, len(frame))


def run_data_report(config: Config) -> None:
    """Stage 1: download Fashion-MNIST and write the data-quality report.

    Args:
        config: Loaded experiment configuration.
    """
    seed_everything(config.seed)
    configure_cpu(config.num_threads)
    apply_style()
    paths = config.paths
    tables, figures = paths.tables_dir, paths.figures_dir

    download_fashion_mnist(paths.raw_dir)
    train = load_split(paths.raw_dir, train=True)
    test = load_split(paths.raw_dir, train=False)

    # Tables.
    _write_table(class_distribution_frame(train.labels), tables, "part1_class_distribution.csv")
    _write_table(brightness_by_class(train.images, train.labels), tables, "part1_brightness.csv")
    report = check_quality(train, test)
    _write_table(report.to_frame(), tables, "part1_quality.csv")

    # Figures.
    dviz.plot_class_examples(train.images, train.labels, figures / "part1_examples.png")
    dviz.plot_class_histogram(
        train.labels,
        "Распределение классов Fashion-MNIST (train)",
        figures / "part1_class_histogram.png",
    )
    dviz.plot_brightness_boxplot(
        per_image_brightness(train.images, train.labels), figures / "part1_brightness.png"
    )

    sub = stratified_subsample(train, config.evaluation.umap_points, config.seed)
    coords = umap_project(sub.images.reshape(len(sub), -1).astype(np.float64), config.seed)
    dviz.plot_umap(
        coords, sub.labels, "UMAP сырых пикселей (подвыборка)", figures / "part1_umap_pixels.png"
    )

    summary = {
        "n_train": len(train),
        "n_test": len(test),
        "class_counts_train": np.bincount(train.labels, minlength=10).tolist(),
        "quality": {
            "train_duplicates": report.train_duplicates,
            "train_test_overlap": report.train_test_overlap,
            "label_conflicts": report.label_conflicts,
            "out_of_range": report.out_of_range,
            "degenerate": report.degenerate,
        },
    }
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "part1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Part 1 summary: %s", summary)


def _resolve_device() -> torch.device:
    """Return the CPU device (Intel-Mac has no CUDA/MPS for this stack)."""
    return torch.device("cpu")


def _train_images(config: Config) -> NDArray[np.float32]:
    """Load the (optionally subsampled) training split as ``[-1, 1]`` images."""
    train = load_split(config.paths.raw_dir, train=True)
    if config.data.subsample_size is not None:
        train = stratified_subsample(train, config.data.subsample_size, config.seed)
    return to_float_pm1(train.images)


def run_benchmark(config: Config) -> None:
    """Measure one DCGAN epoch on CPU and project the full-training budget.

    Args:
        config: Loaded experiment configuration.
    """
    seed_everything(config.seed)
    configure_cpu(config.num_threads)
    from gan_robustness.training.gan_trainer import train_gan

    images = _train_images(config)
    logger.info("Benchmark on %d images, batch %d", images.shape[0], config.gan.batch_size)
    result = train_gan(
        images, config.model, config.gan, _resolve_device(), config.seed, max_epochs=1
    )
    per_epoch = result.seconds_per_epoch
    logger.info(
        "Epoch time: %.1f s | %d epochs -> %.1f min | 4 GANs -> %.1f min",
        per_epoch,
        config.gan.epochs,
        per_epoch * config.gan.epochs / 60,
        per_epoch * config.gan.epochs * 4 / 60,
    )


def _models_dir(config: Config) -> Path:
    """Return (creating) the directory for saved model checkpoints."""
    d = config.paths.artifacts_dir / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_classifier(config: Config) -> None:
    """Train the auxiliary classifier and persist it.

    Args:
        config: Loaded experiment configuration.
    """
    seed_everything(config.seed)
    configure_cpu(config.num_threads)
    from gan_robustness.training.checkpoint import save_classifier
    from gan_robustness.training.classifier_trainer import train_classifier

    train = load_split(config.paths.raw_dir, train=True)
    test = load_split(config.paths.raw_dir, train=False)
    result = train_classifier(
        to_float_pm1(train.images),
        train.labels,
        to_float_pm1(test.images),
        test.labels,
        config.model,
        config.classifier,
        _resolve_device(),
        config.seed,
    )
    save_classifier(
        result.model, config.model.classifier_base, _models_dir(config) / "classifier.pt"
    )
    (config.paths.tables_dir).mkdir(parents=True, exist_ok=True)
    (config.paths.tables_dir / "classifier.json").write_text(
        json.dumps({"test_accuracy": result.test_accuracy}, indent=2), encoding="utf-8"
    )
    logger.info("Classifier saved (test accuracy %.4f)", result.test_accuracy)


def _scenario_training_bundle(
    config: Config, scenario: str, epsilon: float | None
) -> tuple[ImageBundle, float | None]:
    """Return the (possibly corrupted) training bundle for a scenario."""
    from gan_robustness.attacks.imbalance import apply_imbalance
    from gan_robustness.attacks.poisoning import apply_poison

    train = load_split(config.paths.raw_dir, train=True)
    if config.data.subsample_size is not None:
        train = stratified_subsample(train, config.data.subsample_size, config.seed)
    if scenario == "baseline":
        return train, None
    if scenario == "imbalance":
        return apply_imbalance(
            train, config.imbalance.drop_classes, config.imbalance.drop_fraction, config.seed
        ), None
    if scenario == "poison":
        assert epsilon is not None
        bundle, _ = apply_poison(
            train, config.poison.poison_class, epsilon, config.poison.trigger_size, config.seed
        )
        return bundle, epsilon
    raise ValueError(f"Unknown scenario '{scenario}'")


def run_scenario(config: Config, scenario: str) -> None:
    """Train and evaluate a DCGAN under a data scenario.

    For ``poison`` this trains one GAN per configured epsilon.

    Args:
        config: Loaded experiment configuration.
        scenario: One of ``baseline``, ``imbalance``, ``poison``.
    """
    seed_everything(config.seed)
    configure_cpu(config.num_threads)
    epsilons: list[float | None]
    epsilons = list(config.poison.epsilons) if scenario == "poison" else [None]
    for eps in epsilons:
        _train_and_evaluate_scenario(config, scenario, eps)


def _scenario_tag(scenario: str, epsilon: float | None) -> str:
    """Return a filesystem tag for a scenario/epsilon."""
    return scenario if epsilon is None else f"{scenario}_eps{epsilon:g}"


def _load_or_train_classifier(config: Config) -> object:
    """Return the auxiliary classifier, training and saving it if absent."""
    from gan_robustness.training.checkpoint import load_classifier

    path = _models_dir(config) / "classifier.pt"
    if not path.exists():
        run_classifier(config)
    return load_classifier(path, _resolve_device())


def _train_and_evaluate_scenario(
    config: Config, scenario: str, epsilon: float | None
) -> dict[str, object]:
    """Train one DCGAN for a scenario and compute its evaluation metrics + figures."""
    from gan_robustness.evaluation import metrics as gm
    from gan_robustness.evaluation.generate import classify, generate_float, to_uint8
    from gan_robustness.training.checkpoint import save_gan
    from gan_robustness.training.gan_trainer import train_gan
    from gan_robustness.visualization import gan_viz as gviz

    device = _resolve_device()
    tag = _scenario_tag(scenario, epsilon)
    figures, tables = config.paths.figures_dir, config.paths.tables_dir
    classifier = _load_or_train_classifier(config)

    bundle, _ = _scenario_training_bundle(config, scenario, epsilon)
    images = to_float_pm1(bundle.images)
    logger.info("=== Training scenario '%s' on %d images ===", tag, images.shape[0])
    trained = train_gan(images, config.model, config.gan, device, config.seed)
    save_gan(
        trained.generator, trained.discriminator, config.model, _models_dir(config) / f"{tag}.pt"
    )

    # References (test) and generations.
    test = load_split(config.paths.raw_dir, train=False)
    test_float = to_float_pm1(test.images)
    _, _, real_feats = classify(classifier, test_float, device)  # type: ignore[arg-type]
    gen_float = generate_float(
        trained.generator,
        config.evaluation.num_generated,
        config.model.latent_dim,
        device,
        config.seed,
    )
    gen_uint8 = to_uint8(gen_float)
    gen_preds, gen_conf, gen_feats = classify(classifier, gen_float, device)  # type: ignore[arg-type]

    # Metrics.
    train_props = gm.class_proportions(test.labels)  # balanced test = target distribution
    gen_props = gm.class_proportions(gen_preds)
    js = gm.js_divergence(train_props.copy(), gen_props.copy())
    dropped = gm.count_dropped_classes(gen_props, config.evaluation.dropped_class_threshold)
    fid = gm.frechet_distance(real_feats, gen_feats)
    precision, recall = gm.precision_recall(
        real_feats, gen_feats, config.evaluation.precision_recall_k
    )
    per_class_fid = gm.domain_fid_per_class(real_feats, test.labels, gen_feats, gen_preds)
    diversity = gm.mean_pairwise_distance(gen_feats)
    low_conf = float((gen_conf < config.evaluation.low_confidence_threshold).mean())

    # Memorisation (pixel space).
    train_pix = (bundle.images.reshape(len(bundle), -1).astype(np.float32)) / 255.0
    sample = config.evaluation.memorization_sample
    gen_pix = gen_uint8[:sample].reshape(min(sample, len(gen_uint8)), -1).astype(np.float32) / 255.0
    test_pix = (
        test.images[:sample].reshape(min(sample, len(test.images)), -1).astype(np.float32) / 255.0
    )
    gen_nn = gm.nearest_neighbour_distances(gen_pix, train_pix)
    test_nn = gm.nearest_neighbour_distances(test_pix, train_pix)

    row: dict[str, object] = {
        "scenario": tag,
        "domain_fid": round(fid, 3),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "class_js_divergence": round(js, 4),
        "dropped_classes": dropped,
        "mean_confidence": round(float(gen_conf.mean()), 4),
        "low_confidence_fraction": round(low_conf, 4),
        "diversity": round(diversity, 3),
        "gen_nn_median": round(float(np.median(gen_nn)), 3),
        "test_nn_median": round(float(np.median(test_nn)), 3),
        "seconds_per_epoch": round(trained.seconds_per_epoch, 1),
    }
    if scenario == "poison":
        from gan_robustness.attacks.poisoning import detect_trigger

        rate = float(
            detect_trigger(
                gen_uint8, config.poison.trigger_size, config.poison.detection_threshold
            ).mean()
        )
        row["trigger_rate"] = round(rate, 4)
        row["epsilon"] = epsilon

    _append_metric_row(config, row)
    (tables / f"part3_perclass_fid_{tag}.csv").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"class": list(per_class_fid), "domain_fid": list(per_class_fid.values())}).to_csv(
        tables / f"part3_perclass_fid_{tag}.csv", index=False
    )

    # Figures.
    gviz.plot_sample_grid(
        gen_uint8[:64], f"Генерации DCGAN ({tag})", figures / f"part3_samples_{tag}.png"
    )
    gviz.plot_loss_curves(
        trained.history, f"Обучение DCGAN ({tag})", figures / f"part3_loss_{tag}.png"
    )
    gviz.save_storyboard_gif(trained.storyboard, figures / f"part3_storyboard_{tag}.gif")
    epochs = sorted({1, len(trained.storyboard) // 2, len(trained.storyboard)})
    gviz.plot_storyboard_strip(
        trained.storyboard,
        epochs,
        f"Раскадровка обучения ({tag})",
        figures / f"part3_storyboard_{tag}.png",
    )
    gviz.plot_class_distribution_compare(
        train_props,
        gen_props,
        f"Классы: train vs генерации ({tag})",
        figures / f"part3_classdist_{tag}.png",
    )
    gviz.plot_memorization(gen_nn, test_nn, figures / f"part3_memorization_{tag}.png")
    _plot_good_bad(gen_uint8, gen_preds, gen_conf, tag, figures)
    logger.info("Scenario '%s' metrics: %s", tag, row)
    return row


def _plot_good_bad(
    gen_uint8: NDArray[np.uint8],
    gen_preds: NDArray[np.int64],
    gen_conf: NDArray[np.float64],
    tag: str,
    figures: Path,
) -> None:
    """Plot high-confidence and low-confidence generation galleries."""
    from gan_robustness.config import CLASS_NAMES
    from gan_robustness.visualization import gan_viz as gviz

    order = np.argsort(gen_conf)
    low = order[:8]
    high = order[-8:]
    gviz.plot_examples(
        gen_uint8[high],
        [f"{CLASS_NAMES[gen_preds[i]]}\n{gen_conf[i]:.2f}" for i in high],
        f"Удачные генерации ({tag})",
        figures / f"part3_good_{tag}.png",
    )
    gviz.plot_examples(
        gen_uint8[low],
        [f"{CLASS_NAMES[gen_preds[i]]}\n{gen_conf[i]:.2f}" for i in low],
        f"Неудачные/низкоуверенные генерации ({tag})",
        figures / f"part3_bad_{tag}.png",
    )


def _append_metric_row(config: Config, row: dict[str, object]) -> None:
    """Append (or replace) a scenario row in the combined metrics summary CSV."""
    path = config.paths.tables_dir / "metrics_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(existing) and "scenario" in existing.columns:
        existing = pd.DataFrame(existing[existing["scenario"] != row["scenario"]])
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    combined.to_csv(path, index=False)
    logger.info("Updated metrics_summary.csv (%d scenarios)", len(combined))


def run_interpretation(config: Config) -> None:
    """Stage 5: run interpretation across the trained models."""
    raise NotImplementedError("Implemented in stage 5 (interpretation).")


def run_all(config: Config) -> None:
    """Run the full pipeline: classifier, all scenarios, interpretation.

    Args:
        config: Loaded experiment configuration.
    """
    run_classifier(config)
    for scenario in ("baseline", "imbalance", "poison"):
        run_scenario(config, scenario)
    run_interpretation(config)
