# Makefile for fashion-mnist-dcgan-robustness.
# Thin wrappers over the package CLI (python -m gan_robustness ...).

PYTHON ?= python
PKG := gan_robustness
CONFIG ?= configs/base.yaml

.PHONY: help install data lint format typecheck test smoke bench \
        classifier baseline imbalance poison interpret report notebook all clean

help:
	@echo "Targets:"
	@echo "  install    pip install -e '.[dev]'"
	@echo "  data       download Fashion-MNIST into data/raw/"
	@echo "  lint       ruff check + ruff format --check"
	@echo "  format     ruff format + ruff check --fix"
	@echo "  typecheck  mypy on src"
	@echo "  test       pytest"
	@echo "  bench      measure one DCGAN epoch time on CPU"
	@echo "  smoke      fast end-to-end run (subsample, 1 epoch)"
	@echo "  classifier train the auxiliary Fashion-MNIST classifier"
	@echo "  baseline   train the baseline DCGAN and evaluate it"
	@echo "  imbalance  Variant A: class-imbalance experiment"
	@echo "  poison     Variant B: data-poisoning (trigger) experiment"
	@echo "  interpret  interpretation across models"
	@echo "  all        classifier -> baseline -> imbalance -> poison -> interpret"
	@echo "  report     build reports/report.docx via pandoc"
	@echo "  notebook   execute notebooks/lab2_dcgan_robustness.ipynb"

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m $(PKG) download --config $(CONFIG)

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src

test:
	pytest

bench:
	$(PYTHON) -m $(PKG) bench --config $(CONFIG)

smoke:
	$(PYTHON) -m $(PKG) all --config configs/smoke.yaml

classifier:
	$(PYTHON) -m $(PKG) classifier --config $(CONFIG)

baseline:
	$(PYTHON) -m $(PKG) train --config $(CONFIG) --scenario baseline

imbalance:
	$(PYTHON) -m $(PKG) train --config $(CONFIG) --scenario imbalance

poison:
	$(PYTHON) -m $(PKG) train --config $(CONFIG) --scenario poison

interpret:
	$(PYTHON) -m $(PKG) interpret --config $(CONFIG)

all:
	$(PYTHON) -m $(PKG) all --config $(CONFIG)

report:
	pandoc reports/report.md \
		--metadata lang=ru \
		--lua-filter=reports/pagebreak.lua \
		--reference-doc=reports/reference.docx \
		--resource-path=reports \
		-o reports/report.docx

notebook:
	jupyter nbconvert --to notebook --execute --inplace \
		notebooks/lab2_dcgan_robustness.ipynb

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
