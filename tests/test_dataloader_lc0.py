import itertools
from pathlib import Path

import pytest
import torch

from chess_engine_2.dataloader import Lc0V6Dataset, LC0_V6_RECORD_SIZE


def test_dataset_reads_first_sample():
    root = Path("tests/data")
    ds = Lc0V6Dataset(root)
    it = iter(ds)
    sample = next(it)

    assert sample["version"] == 6
    assert sample["input_format"] in (0, 1)

    policy = sample["policy"]
    planes = sample["planes"]

    assert isinstance(policy, torch.Tensor)
    assert isinstance(planes, torch.Tensor)
    assert policy.shape == (1858,)
    assert planes.shape == (104,)
    assert policy.dtype == torch.float32
    assert planes.dtype == torch.uint64

    assert 0 <= sample["played_idx"] < 1858
    assert 0 <= sample["best_idx"] < 1858


def test_dataloader_batches_sequentially():
    root = Path("tests/data")
    ds = Lc0V6Dataset(root)
    dl = torch.utils.data.DataLoader(ds, batch_size=3, num_workers=0)

    batch = next(iter(dl))

    assert batch["policy"].shape == (3, 1858)
    assert batch["planes"].shape == (3, 104)
    # Make sure basic scalar fields batch up
    assert batch["version"].shape == (3,)  # tensorized by default collate


def test_record_size_constant():
    # Sanity: matches the official v6 record size
    assert LC0_V6_RECORD_SIZE == 8356
