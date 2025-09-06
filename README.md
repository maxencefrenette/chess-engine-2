# chess-engine-2
Yet another attempt at making a chess engine

## Acknowledgements

- The model is trained on [Leela Chess Zero](https://lczero.org)'s training data

## Weights & Biases Logging

This project integrates [Weights & Biases](https://wandb.ai) for experiment tracking.

- Install deps and login once: `uv sync` then `wandb login`.
- Configure the project name via `WANDB_PROJECT` (defaults to `chess-engine-2`).
- Optionally set a run name via `WANDB_NAME`.

Examples:

- CLI: `WANDB_PROJECT=chess-engine-2 WANDB_NAME="my-run" uv run train <baseline>`
- Python:
  ```py
  from chess_engine_2.hyperparameters import Hyperparameters
  from chess_engine_2.train import train

  hp = Hyperparameters.from_yaml("src/chess_engine_2/baselines/small.yaml")
  metrics = train(hp, name="experiment-001")
  print(metrics)
  ```
