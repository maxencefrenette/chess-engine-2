from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Hyperparameters:
    """Typed training configuration loaded from YAML."""

    batch_size: int
    max_steps: int
    lr: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Hyperparameters:
        # Accept a flat mapping; ignore unknown keys for forward-compatibility.
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**filtered)  # type: ignore[arg-type]

    @classmethod
    def from_yaml(cls, path: os.PathLike[str] | str) -> Hyperparameters:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}
        if not isinstance(content, Mapping):
            raise ValueError(f"Expected a mapping at {p}, got: {type(content).__name__}")
        return cls.from_dict(content)


__all__ = ["Hyperparameters"]
