from __future__ import annotations

from pathlib import Path
import tarfile

import chess  # type: ignore
import torch

from chess_engine_2.dataloader import Lc0V6Dataset
from chess_engine_2.utils import lc0_to_chess
from syrupy.extensions.single_file import SingleFileSnapshotExtension


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
TAR_PATH = DATA_ROOT / "lc0-data-sample.tar"


def _first_two_gz_members() -> list[str]:
    with tarfile.open(TAR_PATH, mode="r") as tf:
        m = [
            x
            for x in tf.getmembers()
            if x.isfile()
            and x.name.endswith(".gz")
            and "/._" not in x.name
            and not x.name.split("/")[-1].startswith("._")
        ]
        m.sort(key=lambda x: x.name)
        return [x.name for x in m[:2]]


def _rights_str(board: chess.Board) -> str:
    parts = []
    if board.has_kingside_castling_rights(chess.WHITE):
        parts.append("K")
    if board.has_queenside_castling_rights(chess.WHITE):
        parts.append("Q")
    if board.has_kingside_castling_rights(chess.BLACK):
        parts.append("k")
    if board.has_queenside_castling_rights(chess.BLACK):
        parts.append("q")
    return "".join(parts) or "-"


def _render_position(board: chess.Board, meta: dict[str, str | int]) -> str:
    lines = [str(board)]
    lines.append(f"# castling: {_rights_str(board)}")
    lines.append(f"# rule50: {meta['rule50']}")
    lines.append(f"# original_stm: {meta['original_stm']}")
    lines.append(f"# input_format: {meta['input_format']}")
    lines.append("# ep: -")
    return "\n".join(lines)


def _gather_game_text(target_member: str) -> str:
    ds = Lc0V6Dataset(DATA_ROOT)
    segments: list[str] = []
    for sample in ds:
        src = sample.get("source_member", "")  # type: ignore[assignment]
        if src != target_member:
            continue
        board = lc0_to_chess(sample)
        original_stm = "white" if int(sample["side_to_move_or_enpassant"]) == 0 else "black"
        # Always show from White's perspective: if the original side to move
        # was Black, mirror the canonical board so that actual White pieces are
        # shown as White at the bottom in the ASCII rendering.
        display = board if original_stm == "white" else board.mirror()
        meta = {
            "rule50": int(sample["rule50"]),
            "original_stm": original_stm,
            "input_format": int(sample["input_format"]),
        }
        segments.append(_render_position(display, meta))
    assert segments, f"No samples found for member {target_member}"
    return "\n\n".join(segments)


def test_snapshot_first_two_games(snapshot):
    # One snapshot per .gz member (treated as a game)
    members = _first_two_gz_members()
    for member in members:
        text = _gather_game_text(member)
        # Use single-file extension to create one snapshot file per game
        # and name the file after the member (bytes required by extension).
        snapshot(
            name=Path(member).name,
            extension_class=SingleFileSnapshotExtension,
        ).assert_match(text.encode("utf-8"))
