from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader


# Ensure `src` is importable when running as a script
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataloader import Lc0V6Dataset  # noqa: E402
from model import (  # noqa: E402
    SimpleLinearModel,
    lc0_to_features,
    wdl_from_qd,
    cross_entropy_with_probs,
)


def _normalize_policy_target(policy: torch.Tensor, played_idx: torch.Tensor) -> torch.Tensor:
    """Normalize policy targets; fall back to one-hot on `played_idx` if all zeros."""
    sums = policy.sum(dim=-1, keepdim=True)
    norm = policy / (sums + 1e-8)
    # Fallback for zero rows
    zero_rows = (sums.squeeze(-1) <= 1e-8)
    if zero_rows.any():
        oh = torch.zeros_like(policy)
        oh.scatter_(1, played_idx.view(-1, 1).long(), 1.0)
        norm[zero_rows] = oh[zero_rows]
    return norm


def train(
    data_dir: str | os.PathLike[str] = "tests/data",
    batch_size: int = 64,
    max_steps: int = 25,
    lr: float = 1e-2,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = Lc0V6Dataset(data_dir)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0)

    model = SimpleLinearModel().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    step = 0
    last = {"loss": 0.0, "policy": 0.0, "value": 0.0}
    for batch in dl:
        # Move tensors we use to device
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        x = lc0_to_features(batch)
        out = model(x)

        # Targets
        pol_tgt = _normalize_policy_target(batch["policy"], batch["played_idx"])  # (B,1858)
        wdl_tgt = wdl_from_qd(batch["result_q"], batch["result_d"])  # (B,3)

        # Losses
        policy_loss = cross_entropy_with_probs(out["policy_logits"], pol_tgt)
        value_loss = cross_entropy_with_probs(out["value_logits"], wdl_tgt)
        loss = policy_loss + value_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        last = {
            "loss": float(loss.detach().cpu()),
            "policy": float(policy_loss.detach().cpu()),
            "value": float(value_loss.detach().cpu()),
        }
        step += 1
        if step >= max_steps:
            break

    return last


if __name__ == "__main__":
    res = train(
        data_dir=os.environ.get("LC0_DATA_DIR", "tests/data"),
        batch_size=int(os.environ.get("BATCH_SIZE", "64")),
        max_steps=int(os.environ.get("MAX_STEPS", "25")),
        lr=float(os.environ.get("LR", "1e-2")),
    )
    print({k: round(v, 6) for k, v in res.items()})

