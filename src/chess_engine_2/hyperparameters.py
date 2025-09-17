from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Hyperparameters(BaseModel):
    """Typed training configuration loaded from YAML, validated by Pydantic.

    Extra keys in the YAML are ignored to keep configs forward‑compatible.
    """

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(gt=0)
    steps: int = Field(gt=0)
    lr: float = Field(gt=0)
    lr_cooldown_frac: float = Field(default=0.0, ge=0.0, le=1.0)
    model_dim: int = Field(gt=0)
    intermediate_dim: int = Field(gt=0)
    layers: int = Field(gt=0)
    value_sampling_rate: float = Field(ge=0.0, le=1.0)
    shuffle_buffer_size: int = Field(gt=0)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Hyperparameters:
        return cls(**dict(data))

    @classmethod
    def from_yaml(cls, path: os.PathLike[str] | str) -> Hyperparameters:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}
        if not isinstance(content, Mapping):
            raise ValueError(
                f"Expected a mapping at {p}, got: {type(content).__name__}"
            )
        return cls.from_dict(content)


__all__ = ["Hyperparameters"]
