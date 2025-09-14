from __future__ import annotations

import gzip
import io
import os
import struct
import tarfile
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

# Lc0 V6 training record description
# Source of the format specification: Leela Chess Zero wiki (V6TrainingData, size 8356).
# See: https://lczero.org/dev/wiki/training-data-format-versions/


LC0_V6_RECORD_SIZE = 8356
_LC0_V6_STRUCT = struct.Struct(
    "<"  # little endian, packed
    "II"  # version, input_format
    "1858f"  # probabilities
    "104Q"  # planes bitboards
    "8B"  # castling x4, side_or_ep, rule50, invariance_info, dummy
    "15f"  # root_q..plies_left (7) + result_q..orig_m (8) = 15 floats
    "I"  # visits
    "H"  # played_idx
    "H"  # best_idx
    "f"  # policy_kld
    "I"  # reserved
)

def _rev_bits_in_byte(b: int) -> int:
    # Reverse bit order in one byte (e.g., 0babcde... -> 0b...edcba)
    b = ((b & 0xF0) >> 4) | ((b & 0x0F) << 4)
    b = ((b & 0xCC) >> 2) | ((b & 0x33) << 2)
    b = ((b & 0xAA) >> 1) | ((b & 0x55) << 1)
    return b


def _reverse_bits_in_bytes64(x: int) -> int:
    """Invert lc0's ReverseBitsInBytes on a 64-bit mask.

    lc0 writes each plane with ReverseBitsInBytes applied (see lc0
    trainingdata/trainingdata.cc). We undo it here so downstream code can
    assume square index bit 0=a1 .. 63=h8 with normal a..h file order.
    """
    y = 0
    for i in range(8):
        byte = (x >> (i * 8)) & 0xFF
        y |= _rev_bits_in_byte(byte) << (i * 8)
    return y


def _read_exact(stream: io.BufferedReader, n: int) -> bytes | None:
    """Read exactly n bytes from stream; return None if EOF reached before any data."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if not chunks:
        return None
    data = b"".join(chunks)
    if len(data) != n:
        # Incomplete trailing record; treat as EOF.
        return None
    return data


class Lc0V6Dataset(IterableDataset):
    """Iterable dataset reading Lc0 v6 training data from *.tar archives.

    Parameters
    - root_dir: folder containing one or more .tar files. Each tar is expected
      to contain gzipped chunks named like training.*.gz. Files with the
      AppleDouble prefix (._*) are ignored.

    Notes
    - No shuffling is performed; samples are yielded in lexicographic order of
      tar files and members.
    - Only format version 6 is supported; other versions raise ValueError.
    """

    def __init__(self, root_dir: str | os.PathLike[str]):
        super().__init__()
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

    def _tar_paths(self) -> list[Path]:
        paths = sorted(self.root_dir.glob("*.tar"))
        return paths

    def _gz_members(self, tf: tarfile.TarFile) -> list[tarfile.TarInfo]:
        members = [
            m
            for m in tf.getmembers()
            if m.isfile()
            and m.name.endswith(".gz")
            and "/._" not in m.name
            and not m.name.split("/")[-1].startswith("._")
        ]
        members.sort(key=lambda m: m.name)
        return members

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | int | float]]:
        for tar_path in self._tar_paths():
            with tarfile.open(tar_path, mode="r") as tf:
                for m in self._gz_members(tf):
                    fobj = tf.extractfile(m)
                    if fobj is None:
                        continue
                    with gzip.GzipFile(fileobj=fobj) as gz:
                        while True:
                            raw = _read_exact(gz, LC0_V6_RECORD_SIZE)
                            if raw is None:
                                break

                            (
                                version,
                                input_format,
                                *rest,
                            ) = _LC0_V6_STRUCT.unpack(raw)

                            if version != 6:
                                raise ValueError(
                                    f"Unsupported training data version {version}; expected 6"
                                )

                            # Split fields according to the struct layout.
                            # rest contains: probabilities[1858], planes[104], 8B,
                            # 15f, visits (I), played_idx (H), best_idx (H), policy_kld (f),
                            # reserved (I)

                            probs = rest[:1858]
                            planes_raw = rest[1858 : 1858 + 104]
                            b8 = rest[1858 + 104 : 1858 + 104 + 8]
                            floats15 = rest[1858 + 104 + 8 : 1858 + 104 + 8 + 15]
                            after_floats = rest[1858 + 104 + 8 + 15 :]

                            (
                                castling_us_ooo,
                                castling_us_oo,
                                castling_them_ooo,
                                castling_them_oo,
                                side_to_move_or_enpassant,
                                rule50_count,
                                invariance_info,
                                _dummy,
                            ) = b8

                            (
                                root_q,
                                best_q,
                                root_d,
                                best_d,
                                root_m,
                                best_m,
                                plies_left,
                                result_q,
                                result_d,
                                played_q,
                                played_d,
                                played_m,
                                orig_q,
                                orig_d,
                                orig_m,
                            ) = floats15

                            visits = after_floats[0]
                            played_idx = after_floats[1]
                            best_idx = after_floats[2]
                            policy_kld = after_floats[3]
                            # reserved = after_floats[4]

                            rec = {
                                "version": version,
                                "input_format": input_format,
                                "policy": torch.tensor(probs, dtype=torch.float32),
                                # Undo lc0's ReverseBitsInBytes so bit 0=a1 .. 63=h8
                                "planes": torch.tensor(
                                    [_reverse_bits_in_bytes64(int(p)) for p in planes_raw],
                                    dtype=torch.uint64,
                                ),
                                "castling": torch.tensor(
                                    [
                                        castling_us_ooo,
                                        castling_us_oo,
                                        castling_them_ooo,
                                        castling_them_oo,
                                    ],
                                    dtype=torch.uint8,
                                ),
                                "side_to_move_or_enpassant": int(side_to_move_or_enpassant),
                                "rule50": int(rule50_count),
                                "invariance_info": int(invariance_info),
                                "root_q": float(root_q),
                                "best_q": float(best_q),
                                "root_d": float(root_d),
                                "best_d": float(best_d),
                                "root_m": float(root_m),
                                "best_m": float(best_m),
                                "plies_left": float(plies_left),
                                "result_q": float(result_q),
                                "result_d": float(result_d),
                                "played_q": float(played_q),
                                "played_d": float(played_d),
                                "played_m": float(played_m),
                                "orig_q": float(orig_q),
                                "orig_d": float(orig_d),
                                "orig_m": float(orig_m),
                                "visits": int(visits),
                                "played_idx": int(played_idx),
                                "best_idx": int(best_idx),
                                "policy_kld": float(policy_kld),
                            }
                            # Attach the source member name for grouping in snapshot tests
                            rec["source_member"] = m.name
                            yield rec


__all__ = ["Lc0V6Dataset", "LC0_V6_RECORD_SIZE"]
