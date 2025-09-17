from __future__ import annotations

import sys
from pathlib import Path

from chess_engine_2.hyperparameters import Hyperparameters
from chess_engine_2.model import MLPModel
from chess_engine_2.train import train


def main() -> None:
    """Run three training jobs that vary model width while matching total FLOPs.

    Usage: uv run src/chess_engine_2/experiments/tune_model_size.py <baseline>
    """

    if len(sys.argv) < 2:
        base = Path(__file__).resolve().parents[1] / "baselines"
        available = sorted(p.stem for p in base.glob("*.yaml"))
        print("Usage: uv run chess_engine_2/experiments/tune_model_size.py <baseline>")
        if available:
            print("Available baselines:", ", ".join(available))
        raise SystemExit(2)

    baseline = sys.argv[1]
    baselines_dir = Path(__file__).resolve().parents[1] / "baselines"
    cfg_path = baselines_dir / f"{baseline}.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    hp = Hyperparameters.from_yaml(cfg_path)
    base_flops_per_batch = MLPModel.flops_per_batch(hp)

    target_flops = base_flops_per_batch * hp.steps

    for idx, factor in enumerate([0.5, 1.0, 2.0]):
        model_dim = max(1, int(round(hp.model_dim * factor)))
        intermediate_dim = max(1, int(round(hp.intermediate_dim * factor)))

        hp_variant = hp.model_copy(
            update={
                "model_dim": model_dim,
                "intermediate_dim": intermediate_dim,
            }
        )

        run_name = f"{baseline}_tune_size_{idx}"
        flops_per_batch = MLPModel.flops_per_batch(hp_variant)

        steps = max(1, int(round(target_flops / flops_per_batch)))
        hp_variant = hp_variant.model_copy(update={"steps": steps})

        print(
            f"Starting {run_name} with model_dim={model_dim}, intermediate_dim={intermediate_dim}, steps={steps}"
        )
        res = train(run_name, hp_variant)
        print(f"Finished {run_name}:")
        print(f"    Loss': {res['loss']:.6f}")
        print(f"    Policy Loss': {res['policy']:.6f}")
        print(f"    Value Loss': {res['value']:.6f}")


if __name__ == "__main__":
    main()
