PY ?= python

.PHONY: help install dev lint format type test docker clean

help:
	@echo "targets: install dev lint format type test docker clean"

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev,export]" && $(PY) -m pre_commit install

lint:
	$(PY) -m ruff check . && $(PY) -m isort --check-only . && $(PY) -m black --check .

format:
	$(PY) -m isort . && $(PY) -m black .

type:
	$(PY) -m mypy

test:
	$(PY) -m pytest

docker:
	docker build -t piadaptwin:0.1.0 .

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache *.egg-info build dist runs checkpoints
