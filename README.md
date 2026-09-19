# fashion-mnist-dcgan-robustness

[![CI](https://github.com/PolinaKirillovna/fashion-mnist-dcgan-robustness/actions/workflows/ci.yml/badge.svg)](https://github.com/PolinaKirillovna/fashion-mnist-dcgan-robustness/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An unconditional **DCGAN** on **Fashion-MNIST**, studied not for a headline metric
but to explain *what structure of the data the architecture uses* — mathematically,
in its hidden layers, and in its generations and failures. The project evaluates the
generator, stresses it with controlled data corruption (class imbalance and
poisoning), and interprets both networks.

> Educational project for the ITMO course *AI Systems Validation* (laboratory
> practicum, 2024). Report and notebook are in Russian; code and docs are in English.

## What is done

Work is organised in stages (see `PLAN.md`):

- **Task & data** — Fashion-MNIST description, reduced quality check, input representation.
- **Architecture math** — the minimax game, optimal discriminator, transposed-convolution
  generator, DCGAN inductive bias.
- **Training & evaluation** — baseline DCGAN with generator metrics (domain-FID,
  precision/recall, class-distribution JS divergence, memorisation check).
- **Robustness** — Variant A (artificial class imbalance → mode dropping) and Variant B
  (poisoning with a visual trigger → the generator reproduces the artefact).
- **Interpretation** — latent-space traversals, discriminator feature space, activation
  maps, saliency/Grad-CAM/occlusion, weight histograms.

### Key results

DCGAN (generator 545k params, discriminator 80k), 18 epochs, 30k-image subsample
(CPU budget); auxiliary classifier accuracy 0.90. Metrics on 10 000 generations vs test:

| Scenario | domain-FID | precision | recall | class JS | dropped | trigger rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 39.5 | 0.44 | 0.57 | 0.014 | 0 | — |
| imbalance (drop 95% of 3 classes) | 116.9 | 0.49 | 0.35 | 0.079 | 1 | — |
| poison ε=5% | 38.1 | 0.43 | 0.58 | 0.009 | 0 | 0.000 |
| poison ε=20% | 41.7 | 0.43 | 0.57 | 0.008 | 0 | 0.021 |

**Findings.** Class imbalance causes **mode dropping** — recall collapses (0.57 → 0.35)
and per-class FID of dropped classes rises 1.5–2.5×, while precision (realism) holds,
so overall "quality" hides the failure. Poisoning makes the generator **reproduce the
trigger** as a feature of "real" data, ε-dependent (0% at 5%, 2.1% at 20%); the trigger
forms a separate cluster in the discriminator's feature space. No memorisation is found.

Selected figures:

| | |
|---|---|
| ![Samples](reports/figures/part3_samples_baseline.png) | ![Class dist](reports/figures/part3_classdist_imbalance.png) |
| ![Interpolations](reports/figures/part5_interp_baseline.png) | ![Trigger cluster](reports/figures/part5_dfeatures_poison_umap.png) |

## Repository structure

```
src/gan_robustness/   # config, data, models, training, evaluation, interpretation, attacks, visualization, cli
configs/              # YAML configs (base, smoke)
tests/                # fast unit tests
notebooks/            # final executed notebook
reports/              # report.md, model_passport.md, figures/, tables/
docs/                 # theory, metrics, attacks
data/                 # raw/ (not tracked), processed/
```

## Installation

```bash
conda env create -f environment.yml   # or reuse an existing python 3.12 env
conda activate ai-validation
pip install -e ".[dev]"
make data                             # download Fashion-MNIST into data/raw/
```

## Reproduce

```bash
make lint test        # quality gate
make bench            # measure one DCGAN epoch on CPU (guides width)
make all              # classifier -> baseline -> imbalance -> poison -> interpret
make smoke            # fast end-to-end check (tiny subsample, 1 epoch)
```

Training is CPU-only (Intel Mac). On Intel macOS the last available PyTorch wheel is
2.2.2, which requires `numpy<2`; the whole stack is pinned accordingly in `pyproject.toml`.

## Report and notebook

- Report: [`reports/report.md`](reports/report.md); model passport:
  [`reports/model_passport.md`](reports/model_passport.md)
- Notebook: [`notebooks/lab2_dcgan_robustness.ipynb`](notebooks/lab2_dcgan_robustness.ipynb)

## Source

Task based on: Попов И.Ю., Бучаев А.Я., Есипов Д.А. *Валидация систем искусственного
интеллекта: Лабораторный практикум.* — СПб: Университет ИТМО, 2024. — 36 с.
