from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class Hyperparameters:
    """Typed training configuration loaded from YAML.

    Note: the training data path, `device`, and `num_workers` are intentionally
    not part of this class; they are controlled inside the trainer code. The
    dataset path must be provided via the `TRAINING_DATA_PATH` environment
    variable.
    """

    batch_size: int = 64
    max_steps: int = 25
    lr: float = 1e-2

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Hyperparameters":
        # Accept a flat mapping; ignore unknown keys for forward-compatibility.
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**filtered)  # type: ignore[arg-type]

    @classmethod
    def from_yaml(cls, path: os.PathLike[str] | str) -> "Hyperparameters":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}
        if not isinstance(content, Mapping):
            raise ValueError(f"Expected a mapping at {p}, got: {type(content).__name__}")
        return cls.from_dict(content)


__all__ = ["Hyperparameters"]
