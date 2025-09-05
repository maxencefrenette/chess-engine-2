from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv


# Load environment from project-level .env (repo root: two levels up from this file)
ROOT = Path(__file__).resolve()
load_dotenv(dotenv_path=(ROOT.parents[2] / ".env"))

from .dataloader import Lc0V6Dataset
from .model import SimpleLinearModel, lc0_to_features, wdl_from_qd, cross_entropy_with_probs
from .hyperparameters import Hyperparameters


def _normalize_policy_target(
    policy: torch.Tensor, played_idx: torch.Tensor
) -> torch.Tensor:
    """Normalize policy targets; fall back to one-hot on `played_idx` if all zeros."""
    sums = policy.sum(dim=-1, keepdim=True)
    norm = policy / (sums + 1e-8)
    # Fallback for zero rows
    zero_rows = sums.squeeze(-1) <= 1e-8
    if zero_rows.any():
        oh = torch.zeros_like(policy)
        oh.scatter_(1, played_idx.view(-1, 1).long(), 1.0)
        norm[zero_rows] = oh[zero_rows]
    return norm


def _resolve_training_data_path() -> Path:
    """Resolve the training data directory strictly from `TRAINING_DATA_PATH`.

    Raises a clear error if not set.
    """
    raw = os.environ.get("TRAINING_DATA_PATH")
    if not raw:
        raise RuntimeError(
            "TRAINING_DATA_PATH is not set. Define it in your .env file or export it."
        )
    expanded = os.path.expanduser(os.path.expandvars(raw))
    return Path(expanded)


def train(hp: Hyperparameters) -> Dict[str, float]:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    data_dir = _resolve_training_data_path()
    ds = Lc0V6Dataset(data_dir)
    dl = DataLoader(ds, batch_size=hp.batch_size, num_workers=0)

    model = SimpleLinearModel().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=hp.lr)

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
        pol_tgt = _normalize_policy_target(
            batch["policy"], batch["played_idx"]
        )  # (B,1858)
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
        if step >= hp.max_steps:
            break

    return last


def cli() -> None:
    """CLI: `uv run train <name-or-path>`

    - If `<name-or-path>` has no path separators, loads
      `chess_engine_2/baselines/<name>.yaml`.
    - Otherwise treats it as a path to a YAML file.
    """
    import sys

    if len(sys.argv) < 2:
        base = Path(__file__).parent / "baselines"
        available = sorted(p.stem for p in base.glob("*.yaml"))
        print("Usage: uv run train <baseline|/path/to/config.yaml>")
        if available:
            print("Available baselines:", ", ".join(available))
        raise SystemExit(2)

    arg = sys.argv[1]
    if "/" in arg or arg.endswith(".yaml") or "." in arg:
        cfg_path = Path(arg)
    else:
        cfg_path = Path(__file__).parent / "baselines" / f"{arg}.yaml"

    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    hp = Hyperparameters.from_yaml(cfg_path)
    res = train(hp)
    print({k: round(v, 6) for k, v in res.items()})


if __name__ == "__main__":
    cli()
