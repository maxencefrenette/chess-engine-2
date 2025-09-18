from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, Shuffler

from .dataloader import Lc0V6Dataset
from .hyperparameters import Hyperparameters
from .model import (
    MLPModel,
    lc0_to_features,
    wdl_from_qd,
)

# Load environment from project-level .env (repo root: two levels up from this file)
ROOT = Path(__file__).resolve()
load_dotenv(dotenv_path=(ROOT.parents[2] / ".env"))

WANDB_PROJECT = "chess-engine-2"


def get_lr(hp: Hyperparameters, step: int) -> float:
    """Piecewise-linear learning rate schedule with an optional cooldown."""
    progress = step / hp.steps
    assert 0 <= progress < 1

    if progress < 1 - hp.lr_cooldown_frac:
        multiplier = 1.0
    else:
        decay = (1 - progress) / hp.lr_cooldown_frac
        multiplier = decay * 1.0 + (1 - decay) * 0.1

    return hp.lr * multiplier


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


def resolve_training_data_path() -> Path:
    """Resolve the training data directory from the `TRAINING_DATA_PATH` env variable."""
    expanded = os.path.expanduser(os.path.expandvars(os.environ["TRAINING_DATA_PATH"]))
    return Path(expanded)


def init_torch_backend():
    """Pick the best available PyTorch backend (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        print(
            f"Using CUDA {torch.version.cuda} with cuDNN {torch.backends.cudnn.version()}"
        )
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def cross_entropy_with_probs(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy for probabilistic targets with optional sample weights.

    Computes per-sample loss = -sum p * log_softmax(logits), and returns:
    - mean over batch if `weights` is None
    - weighted mean sum(w*l)/sum(w) if `weights` is provided (0-d if sum(w)==0 -> 0)
    """
    assert logits.device == target_probs.device

    logp = F.log_softmax(logits, dim=-1)
    per_sample = -(target_probs * logp).sum(dim=-1)
    if weights is None:
        return per_sample.mean()
    w = weights.to(per_sample.dtype)
    denom = w.sum()
    if denom.item() == 0.0:
        return per_sample.new_tensor(0.0)
    return (per_sample * w).sum() / denom


def train(run_name: str, hp: Hyperparameters) -> dict[str, float]:
    """Train the model for ``hp.max_steps`` and return last-step metrics.

    Parameters
    - hp: Training hyperparameters.
    - name: Optional run name used for the Weights & Biases run.
    """
    device = init_torch_backend()

    data_dir = resolve_training_data_path()
    ds = Lc0V6Dataset(data_dir)
    datapipe = IterableWrapper(ds)
    datapipe = Shuffler(datapipe, buffer_size=hp.shuffle_buffer_size)
    dl = DataLoader(datapipe, batch_size=hp.batch_size, num_workers=0)

    model = MLPModel(hp).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=get_lr(hp, 0))

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
        x = lc0_to_features(batch)
        out = model(x.to(device))

        # Targets
        pol_tgt = _normalize_policy_target(batch["policy"], batch["played_idx"]).to(
            device
        )  # (B,1858)
        wdl_tgt = wdl_from_qd(batch["root_q"], batch["root_d"]).to(
            device, torch.float32
        )  # (B,3)

        # Mask illegal moves at the logits level wherever policy < 0
        # This prevents illegal classes from contributing to the loss.
        illegal_mask = batch["policy"] < 0
        masked_policy_logits = out["policy_logits"].masked_fill(
            illegal_mask.to(device), -1e9
        )

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

        current_lr = get_lr(hp, step)
        for group in opt.param_groups:
            group["lr"] = current_lr

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        cumulative_flops += per_batch_flops

        last = {
            "loss": float(loss.detach().cpu()),
            "policy": float(policy_loss.detach().cpu()),
            "value": float(value_loss.detach().cpu()),
            "flops": cumulative_flops,
            "lr": current_lr,
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
