from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Features per docs/model.md
# - Board: (12, 8, 8) one-hot planes (PNBRQKpnbrqk) from STM perspective
# - Castling: (4,) in order KQkq (us_oo, us_ooo, them_oo, them_ooo)
# - En passant: (8,) file mask
# - Rule50: (101,) one-hot
# Total flattened = 12*64 + 4 + 8 + 101 = 881


def _bitboards12_to_planes(boards: torch.Tensor) -> torch.Tensor:
    """Convert 12 uint64 bitboards to (12, 8, 8) float planes.

    boards: Tensor shape (B, 12) dtype=uint64 where bit i corresponds to square index i.
    Returns: Tensor shape (B, 12, 8, 8) dtype=float32 with 0/1 values.
    """
    assert boards.dtype == torch.uint64 and boards.dim() == 2 and boards.size(1) == 12
    B = boards.size(0)
    device = boards.device
    idx = torch.arange(64, dtype=torch.int64, device=device)
    masks = (torch.ones(64, dtype=torch.int64, device=device) << idx)
    boards_i64 = boards.to(torch.int64)
    # (B, 12, 64) of bits via masking to avoid uint64 shifts on CPU
    bits = ((boards_i64.unsqueeze(-1) & masks) != 0).to(torch.float32)
    planes = bits.view(B, 12, 8, 8)
    return planes


def _castling_vector(us_oo: torch.Tensor, us_ooo: torch.Tensor,
                     them_oo: torch.Tensor, them_ooo: torch.Tensor) -> torch.Tensor:
    # Order: K Q k q
    return torch.stack([
        (us_oo > 0).to(torch.float32),
        (us_ooo > 0).to(torch.float32),
        (them_oo > 0).to(torch.float32),
        (them_ooo > 0).to(torch.float32),
    ], dim=-1)


def _enpassant_vector(side_to_move_or_ep: torch.Tensor, input_format: torch.Tensor) -> torch.Tensor:
    """Return 8-dim ep file mask.
    For input_format==3, the field contains an EP file bitmask; otherwise zeros.
    """
    mask = (input_format == 3).to(torch.uint8) * side_to_move_or_ep.to(torch.uint8)
    idx = torch.arange(8, dtype=torch.uint8, device=mask.device)
    ep = ((mask.unsqueeze(-1) >> idx) & 1).to(torch.float32)
    return ep


def _rule50_one_hot(rule50: torch.Tensor) -> torch.Tensor:
    # Clamp to [0, 100] to be safe.
    r = rule50.clamp(min=0, max=100).to(torch.long)
    B = r.numel()
    oh = torch.zeros(B, 101, dtype=torch.float32, device=r.device)
    oh.scatter_(1, r.view(-1, 1), 1.0)
    return oh


def lc0_to_features(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Transform a collated Lc0 batch (from Lc0V6Dataset) into 881-dim features per docs.

    Returns a tensor of shape (B, 881) dtype=float32.
    """
    planes_u64 = batch["planes"][:, :12]  # first 12 are PNBRQKpnbrqk (from STM perspective)
    board = _bitboards12_to_planes(planes_u64)

    cast = _castling_vector(
        batch["castling"][:, 1],  # us_oo is index 1 in dataloader order? We provided [us_ooo, us_oo, ...]
        batch["castling"][:, 0],
        batch["castling"][:, 3],
        batch["castling"][:, 2],
    )

    ep = _enpassant_vector(batch["side_to_move_or_enpassant"].to(torch.int64), batch["input_format"].to(torch.int64))
    r50 = _rule50_one_hot(batch["rule50"].to(torch.int64))

    feats = torch.cat([
        board.reshape(board.size(0), -1),  # 12*64
        cast.to(torch.float32),            # 4
        ep.to(torch.float32),              # 8
        r50.to(torch.float32),             # 101
    ], dim=1)
    assert feats.shape[1] == 881
    return feats


def wdl_from_qd(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Convert scalar expectation q in [-1,1] and draw prob d in [0,1] to W/D/L distribution.

    W - L = q, W + L = 1 - d  =>  W=(S+q)/2, L=(S-q)/2.
    """
    S = (1.0 - d).clamp(0.0, 1.0)
    W = (S + q).clamp(0.0, 2.0) * 0.5
    L = (S - q).clamp(0.0, 2.0) * 0.5
    D = d.clamp(0.0, 1.0)
    out = torch.stack([W, D, L], dim=-1)
    # Renormalize to sum to 1 to be safe.
    out = out / (out.sum(dim=-1, keepdim=True) + 1e-8)
    return out


@dataclass
class ModelConfig:
    in_dim: int = 881
    out_policy: int = 1858
    out_value: int = 3


class SimpleLinearModel(nn.Module):
    """Single linear layer mapping 881 -> (1858 policy logits, 3 value logits)."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.fc = nn.Linear(self.cfg.in_dim, self.cfg.out_policy + self.cfg.out_value)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.fc(x)
        policy_logits = out[:, : self.cfg.out_policy]
        value_logits = out[:, self.cfg.out_policy :]
        return {"policy_logits": policy_logits, "value_logits": value_logits}


def cross_entropy_with_probs(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for probabilistic targets: -sum p * log_softmax(logits)."""
    logp = F.log_softmax(logits, dim=-1)
    loss = -(target_probs * logp).sum(dim=-1).mean()
    return loss

