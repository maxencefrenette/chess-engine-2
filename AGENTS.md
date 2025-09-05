# Repository Guidelines

This repository implements a chess engine in Python. Use uv for dependency management and pytest for tests.

## Project Structure & Module Organization
- `src/`: Python source files (e.g., `dataloader.py`).
- `tests/`: Test suite and fixtures (e.g., `test_sanity.py`).
- `tests/data/`: Sample assets used by tests.
- `main.py`: Entry point for quick experiments.
- `pyproject.toml`: Project, build, and tooling config.

## Build, Test, and Development Commands
- `uv sync`: Create/update the virtual environment and install deps.
- `uv run test -q`: Run the test suite (alias for pytest).
- `uv run pytest -q`: Run pytest directly.
- `uv run python main.py`: Execute the example entry point.

## Coding Style & Naming Conventions
- **Python**: 4‑space indentation, type hints for new code.
- **Names**: modules `snake_case.py`, classes `CamelCase`, functions `snake_case`.
- **Formatting/Linting**: Keep imports tidy and lines reasonable (PEP 8). Add ruff/black later if needed.

## Testing Guidelines
- **Framework**: pytest; add tests under `tests/` with files named `test_*.py`.
- **Running**: `uv run test -q` (or `uv run pytest`).
- **Coverage**: Prefer meaningful unit tests around move generation, evaluation, and I/O.

## Commit & Pull Request Guidelines
- **Commits**: Clear, imperative subject (e.g., "Add move generator"), with a brief body explaining rationale.
- **PRs**: Link issues, describe changes, include screenshots or logs when helpful, and note any breaking changes.
- **Checks**: Ensure tests pass locally before requesting review.

## Security & Configuration Tips
- Avoid committing large datasets or secrets; keep credentials in env vars.
- Use Python ≥3.13; sync via `uv sync` to reproduce the environment.
