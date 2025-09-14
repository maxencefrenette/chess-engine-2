from __future__ import annotations

"""Utilities to convert training data samples to python-chess boards.

This module focuses on reconstructing chess positions from the Lc0 v6
training record as emitted by :class:`chess_engine_2.dataloader.Lc0V6Dataset`.

Design choices
- Input piece planes are assumed to follow the Lc0 convention for the first
  12 planes: "PNBRQKpnbrqk" from the side-to-move perspective. We reconstruct
  positions in a canonical view where the side to move is always White, i.e.,
  we map the first 6 planes (PNBRQK) to White pieces and the next 6 planes
  (pnbrqk) to Black pieces regardless of the original side to move. The
  original side to move is preserved in metadata for auditing.
- Castling rights from the dataset are given relative to the side to move as
  [us_ooo, us_oo, them_ooo, them_oo]. We convert these to absolute rights
  (KQkq) assuming "us" corresponds to the canonical White after the mapping
  above.
- En passant information is not available for input formats 0 or 1. For input
  format 3, the dataset encodes an en passant file bitmask and optional board
  symmetries via ``invariance_info``. Our test data uses formats 0/1, so we do
  not attempt to reconstruct EP squares here.

Notes on invariance_info (source: Lc0 wiki, training data v6)
- For input format 3 only, bits 0..2 of ``invariance_info`` describe a
  composition of horizontal mirror, vertical mirror, and diagonal transpose
  applied to the board as an augmentation; bit 7 indicates whether a side swap
  was included. Other bits 3..6 carry auxiliary flags. For formats 0/1 this
  information is not used in reconstruction.
"""

from typing import Iterable

import chess  # type: ignore
import torch


PIECE_ORDER = "PNBRQKpnbrqk"  # indices 0..11


def _empty_board() -> chess.Board:
    try:  # python-chess >= 1.999
        return chess.Board.empty()  # type: ignore[attr-defined]
    except Exception:
        return chess.Board(None)  # type: ignore[arg-type]


def _squares_from_bitboard(bb: int) -> Iterable[int]:
    # Enumerate squares a1=0 .. h8=63 for every set bit in bb.
    # This assumes bit 0 -> a1, which matches python-chess square indices.
    while bb:
        lsb = bb & -bb
        idx = (lsb.bit_length() - 1)
        yield idx
        bb ^= lsb


def lc0_to_chess(sample: dict[str, torch.Tensor | int | float]) -> chess.Board:
    """Convert a single Lc0 v6 training sample to a :class:`chess.Board`.

    The output board is canonicalized such that White is to move, and the first
    6 planes (PNBRQK) are interpreted as White pieces.
    """
    planes: torch.Tensor = sample["planes"]  # (104,) uint64
    castling: torch.Tensor = sample["castling"]  # (4,) uint8 [us_ooo, us_oo, them_ooo, them_oo]
    input_format = int(sample["input_format"])  # 0/1 in our test set

    board = _empty_board()
    board.clear_stack()  # ensure no move history
    board.turn = chess.WHITE  # canonicalize to White to move

    # Pieces: Always map PNBRQK -> White, pnbrqk -> Black
    piece_map = {
        "P": chess.PAWN,
        "N": chess.KNIGHT,
        "B": chess.BISHOP,
        "R": chess.ROOK,
        "Q": chess.QUEEN,
        "K": chess.KING,
        "p": chess.PAWN,
        "n": chess.KNIGHT,
        "b": chess.BISHOP,
        "r": chess.ROOK,
        "q": chess.QUEEN,
        "k": chess.KING,
    }

    for i, ch in enumerate(PIECE_ORDER):
        bb = int(planes[i].item())
        color = chess.WHITE if ch.isupper() else chess.BLACK
        ptype = piece_map[ch]
        for sq in _squares_from_bitboard(bb):
            board.set_piece_at(sq, chess.Piece(ptype, color))

    # Castling rights: map [us_ooo, us_oo, them_ooo, them_oo] -> KQkq where
    # "us" corresponds to canonical White after the mapping above.
    rights = []
    if int(castling[1]) > 0:
        rights.append("K")
    if int(castling[0]) > 0:
        rights.append("Q")
    if int(castling[3]) > 0:
        rights.append("k")
    if int(castling[2]) > 0:
        rights.append("q")
    fen_rights = "".join(rights) or "-"
    board.set_castling_fen(fen_rights)

    # Rule 50
    board.halfmove_clock = int(sample["rule50"]) if "rule50" in sample else 0

    # En passant: formats 0/1 don't encode EP file; leave unset.
    if input_format == 3 and "side_to_move_or_enpassant" in sample:
        # This path is for completeness; our tests use formats 0/1.
        mask = int(sample["side_to_move_or_enpassant"]) & 0xFF
        if mask:
            # Choose the first (lowest) file bit; rank depends on side to move,
            # but since we canonicalize to White, use rank 6 (from White's POV).
            file_idx = (mask & -mask).bit_length() - 1
            board.ep_square = chess.square(file_idx, 5)  # file, rank 6 -> index 5 (0-based)
        else:
            board.ep_square = None
    else:
        board.ep_square = None

    return board


def features_to_chess(*_args, **_kwargs):  # pragma: no cover - not part of this spec
    """Placeholder for features->chess reconstruction.

    The current spec only requires dataloader->lc0->chess snapshot tests.
    """
    raise NotImplementedError("features_to_chess is not implemented in this iteration")


__all__ = ["lc0_to_chess", "features_to_chess"]

