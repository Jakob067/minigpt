PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: install lint test train-tiny train-small generate ablate
install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -e '.[dev]'
lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
test:
	$(BIN)/pytest -q
train-tiny:
	$(BIN)/minigpt-train --config configs/tiny.yaml
train-small:
	$(BIN)/minigpt-train --config configs/small.yaml
generate:
	$(BIN)/minigpt-generate --checkpoint runs/tiny/best.pt --prompt 'ROMEO:'
ablate:
	$(BIN)/minigpt-ablate --config configs/ablation.yaml
