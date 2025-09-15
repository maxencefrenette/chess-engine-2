from __future__ import annotations

import sys
from pathlib import Path

from chess_engine_2.hyperparameters import Hyperparameters
from chess_engine_2.train import train


def main() -> None:
    """Run five training jobs around a baseline learning rate.

    Usage: uv run src/chess_engine_2/experiments/tune_lr.py <baseline>

    Loads `src/chess_engine_2/baselines/<baseline>.yaml`, then launches runs at
    lr / 1.1**2, lr / 1.1, lr, lr * 1.1, lr * 1.1**2 with run names
    `<baseline>_tune_lr_<n>` for n in [0..4].
    """

    if len(sys.argv) < 2:
        base = Path(__file__).resolve().parents[1] / "baselines"
        available = sorted(p.stem for p in base.glob("*.yaml"))
        print("Usage: uv run python -m chess_engine_2.experiments.tune_lr <baseline>")
        if available:
            print("Available baselines:", ", ".join(available))
        raise SystemExit(2)

    baseline = sys.argv[1]
    baselines_dir = Path(__file__).resolve().parents[1] / "baselines"
    cfg_path = baselines_dir / f"{baseline}.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    hp = Hyperparameters.from_yaml(cfg_path)
    base_lr = float(hp.lr)

    # Scale factors around base LR: 1.1**e for e in [-2,-1,0,1,2]
    factors = [1.1**e for e in (-2, -1, 0, 1, 2)]

    for i, f in enumerate(factors):
        lr_i = base_lr * f
        run_name = f"{baseline}_tune_lr_{i}"
        hp_i = hp.model_copy(update={"lr": lr_i})
        print(f"Starting {run_name} with lr={lr_i:.8f}")
        res = train(run_name, hp_i)
        print(f"Finished {run_name}:")
        print(f"    Loss': {res['loss']:.6f}")
        print(f"    Policy Loss': {res['policy']:.6f}")
        print(f"    Value Loss': {res['value']:.6f}")


if __name__ == "__main__":
    main()
