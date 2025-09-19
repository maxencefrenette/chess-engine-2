from __future__ import annotations

import os
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, computed_field


class Hyperparameters(BaseModel):
    """Typed training configuration loaded from YAML, validated by Pydantic.

    Extra keys in the YAML are ignored to keep configs forward‑compatible.
    """

    model_config = ConfigDict(extra="forbid")

    flops_budget: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    lr: float = Field(gt=0)
    lr_cooldown_frac: float = Field(default=0.0, ge=0.0, le=1.0)
    model_dim: int = Field(gt=0)
    intermediate_dim: int = Field(gt=0)
    layers: int = Field(gt=0)
    value_sampling_rate: float = Field(ge=0.0, le=1.0)
    shuffle_buffer_size: int = Field(gt=0)

    @computed_field
    @cached_property
    def steps(self) -> int:
        """Number of training steps to run, derived from the FLOPS budget."""
        from .train import MLPModel  # avoid circular import

        steps = int(self.flops_budget / MLPModel.flops_per_batch(self))
        if steps <= 0:
            raise ValueError(
                f"FLOPS budget too low for one training step: {self.flops_budget}"
            )
        return steps

    @classmethod
    def from_yaml(cls, path: os.PathLike[str] | str) -> Hyperparameters:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}
        if not isinstance(content, Mapping):
            raise ValueError(
                f"Expected a mapping at {p}, got: {type(content).__name__}"
            )
        return cls(**dict(content))


__all__ = ["Hyperparameters"]
