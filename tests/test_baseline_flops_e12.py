from __future__ import annotations

from pathlib import Path

import pytest

from chess_engine_2.hyperparameters import Hyperparameters
from chess_engine_2.model import MLPModel


def test_e10_baseline_training_flops_under_1e10() -> None:
    """e12.yaml baseline should train under 1e12 FLOPs.

    Uses PyTorch's flop counter to estimate per‑batch FLOPs (forward+backward)
    on CPU for the configured batch size, then multiplies by the number of
    training steps from the YAML config.

    The test is skipped if ``torch.utils.flop_counter`` is unavailable.
    """

    pytest.importorskip("torch.utils.flop_counter")

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "src/chess_engine_2/baselines/e12.yaml"
    hp = Hyperparameters.from_yaml(cfg_path)

    per_batch_flops = MLPModel.flops_per_batch(hp, device="cpu")
    total_training_flops = per_batch_flops * hp.steps

    assert (
        total_training_flops < 1e12
    ), f"Total FLOPs {total_training_flops:.2e} exceeds 1e12 for e12.yaml"
