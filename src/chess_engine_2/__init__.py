"""Package placeholder for build backend.

This repository keeps modules (e.g., `dataloader.py`) at `src/` and uses
pytest's `pythonpath = ["src"]` for local imports. The empty package ensures
`uv_build` can produce a wheel for the project name `chess-engine-2`.
"""

__all__: list[str] = []

