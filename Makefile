.PHONY: setup quality test

PYTHON := uv run python
RUFF := uv run ruff
PYTEST := uv run pytest

setup:
    uv sync
    uv run pre-commit install

quality:
    $(RUFF) check . --fix
    $(RUFF) format .

test:
    $(PYTEST) tests/
