from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, Shuffler

from .dataloader import Lc0V6Dataset
from .hyperparameters import Hyperparameters
from .model import (
    MLPModel,
    cross_entropy_with_probs,
    lc0_to_features,
    wdl_from_qd,
)

# Load environment from project-level .env (repo root: two levels up from this file)
ROOT = Path(__file__).resolve()
load_dotenv(dotenv_path=(ROOT.parents[2] / ".env"))

WANDB_PROJECT = "chess-engine-2"


def _normalize_policy_target(
    policy: torch.Tensor, played_idx: torch.Tensor
) -> torch.Tensor:
    """Normalize policy targets; fall back to one-hot on `played_idx` if all zeros.

    Notes
    - Entries < 0 are treated as illegal and ignored (set to 0) before
      normalization, matching the docs where illegal moves are marked with -1.
    """
    # Clamp negatives (illegal) to zero before normalization
    policy_clamped = torch.clamp(policy, min=0.0)
    sums = policy_clamped.sum(dim=-1, keepdim=True)
    norm = policy_clamped / (sums + 1e-8)
    # Fallback for rows with no legal mass
    zero_rows = sums.squeeze(-1) <= 1e-8
    if zero_rows.any():
        oh = torch.zeros_like(policy_clamped)
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


def train(run_name: str, hp: Hyperparameters) -> dict[str, float]:
    """Train the model for ``hp.max_steps`` and return last-step metrics.

    Parameters
    - hp: Training hyperparameters.
    - name: Optional run name used for the Weights & Biases run.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    data_dir = _resolve_training_data_path()
    ds = Lc0V6Dataset(data_dir)
    datapipe = IterableWrapper(ds)
    datapipe = Shuffler(datapipe, buffer_size=hp.shuffle_buffer_size)
    dl = DataLoader(datapipe, batch_size=hp.batch_size, num_workers=0)

    model = MLPModel(hp).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=hp.lr)

    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config=hp.model_dump(),
        dir=Path(os.environ["WANDB_PATH"]).expanduser(),
    )

    per_batch_flops = MLPModel.flops_per_batch(hp)
    cumulative_flops = 0

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
        wdl_tgt = wdl_from_qd(batch["root_q"], batch["root_d"])  # (B,3)

        # Mask illegal moves at the logits level wherever policy < 0
        # This prevents illegal classes from contributing to the loss.
        illegal_mask = batch["policy"] < 0
        masked_policy_logits = out["policy_logits"].masked_fill(illegal_mask, -1e9)

        # Losses
        policy_loss = cross_entropy_with_probs(masked_policy_logits, pol_tgt)
        # Value loss masking: keep each sample with prob = value_sampling_rate
        B = pol_tgt.size(0)
        value_keep = (torch.rand(B, device=device) < float(hp.value_sampling_rate)).to(
            torch.float32
        )
        value_loss = cross_entropy_with_probs(
            out["value_logits"], wdl_tgt, weights=value_keep
        )
        loss = policy_loss + value_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        cumulative_flops += per_batch_flops

        last = {
            "loss": float(loss.detach().cpu()),
            "policy": float(policy_loss.detach().cpu()),
            "value": float(value_loss.detach().cpu()),
            "flops": cumulative_flops,
        }

        wandb_run.log(last, step=step)
        step += 1
        if step >= hp.steps:
            break

    wandb_run.finish()

    return last


def cli() -> None:
    """CLI: `uv run train <name-or-path>`

    Loads `chess_engine_2/baselines/<name>.yaml`.
    """

    if len(sys.argv) < 2:
        base = Path(__file__).parent / "baselines"
        available = sorted(p.stem for p in base.glob("*.yaml"))
        print("Usage: uv run train <baseline|/path/to/config.yaml>")
        if available:
            print("Available baselines:", ", ".join(available))
        raise SystemExit(2)

    baseline_name = sys.argv[1]
    cfg_path = Path(__file__).parent / "baselines" / f"{baseline_name}.yaml"

    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    hp = Hyperparameters.from_yaml(cfg_path)
    res = train(baseline_name, hp)
    print({k: round(v, 6) for k, v in res.items()})


if __name__ == "__main__":
    cli()
