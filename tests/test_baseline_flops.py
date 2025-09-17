from __future__ import annotations

from pathlib import Path

import pytest

from chess_engine_2.hyperparameters import Hyperparameters
from chess_engine_2.model import MLPModel


@pytest.mark.parametrize(
    ("baseline_name", "flop_budget"),
    [
        ("e12", 1e12),
        ("e13", 1e13),
    ],
)
def test_baseline_training_flops_within_budget(
    baseline_name: str, flop_budget: float
) -> None:
    """Every baseline YAML should train under its FLOP budget.

    Uses PyTorch's flop counter to estimate per-batch FLOPs (forward+backward)
    on CPU for the configured batch size, then multiplies by the number of
    training steps from the YAML config.

    The test is skipped if ``torch.utils.flop_counter`` is unavailable.
    """

    pytest.importorskip("torch.utils.flop_counter")

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / f"src/chess_engine_2/baselines/{baseline_name}.yaml"
    hp = Hyperparameters.from_yaml(cfg_path)

    per_batch_flops = MLPModel.flops_per_batch(hp)
    total_training_flops = per_batch_flops * hp.steps

    assert (
        total_training_flops < flop_budget
    ), (
        f"Total FLOPs {total_training_flops:.2e} exceeds {flop_budget:.0e}"
        f" for {baseline_name}.yaml"
    )
