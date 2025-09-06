# Repository Guidelines

This repository implements a Python chess engine. We use `uv` for environment and dependency management and `pytest` for tests.

## Project Structure & Module Organization
- `src/`: Python source (e.g., `dataloader.py`). Add new modules in `snake_case`; keep files focused and cohesive.
- `tests/`: Test suite and fixtures. `tests/data/`: sample assets used by tests.
- `main.py`: Entry point for quick experiments and manual checks.
- `pyproject.toml`: Project metadata, dependencies, and tooling configuration.

## Build, Test, and Development Commands
- `uv sync`: Create/update the virtual environment and install dependencies.
- `uv run pytest -q`: Run the full test suite.
- `uv run pytest tests/test_dataloader_lc0.py -q`: Run a specific test file (example).
- `uv run python main.py`: Execute the example entry point.

## Coding Style & Naming Conventions
- **Python**: ≥3.13; 4‑space indentation; prefer type hints for new code.
- **Names**: modules `snake_case.py`; classes `CamelCase`; functions `snake_case`.
- **Style**: keep imports tidy; follow PEP 8; keep lines reasonable. Ruff/Black may be added later—format proactively and consistently.

## Testing Guidelines
- **Framework**: `pytest`; place tests under `tests/` with files named `test_*.py`.
- **Focus**: add unit tests for move generation, evaluation, and I/O paths.
- **Data**: use `tests/data/` for assets; keep fixtures deterministic and small.
- **Run**: `uv run pytest -q` (or `uv run test -q` if available).

## Commit & Pull Request Guidelines
- **Commits**: imperative subject (e.g., "Add move generator"); brief body explaining rationale.
- **PRs**: link issues, describe changes and risk, include logs/screenshots when helpful.
- **Checks**: ensure `uv run pytest -q` passes locally before requesting review.

## Security & Configuration Tips
- Avoid committing secrets or large datasets; load credentials from environment variables.
- Prefer small, shareable assets under `tests/data/`.
- Reproduce environments with `uv sync`; keep dependencies minimal and scoped to needs.
