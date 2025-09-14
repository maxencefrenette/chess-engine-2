# Model

## Feature Encoding

- Board planes (12×64)
  - Order: `PNBRQKpnbrqk`, always from the perspective of the side to move (STM).
  - Uppercase planes are the STM (“ours”), lowercase planes are the opponent (“theirs”).
  - Bitboard layout: bit 0 is `a1`, …, bit 63 is `h8` (row‑major by rank, files a→h).

- Castling rights (4)
  - Vector in order `K Q k q` as floats in {0.0, 1.0}.

- En passant (8)
  - File mask for the EP target files. Bit 0 is file `a`, bit 7 is file `h`.

- Rule50 (101)
  - One‑hot encoding of the half‑move clock clamped to [0, 100]. A value of 100 indicates the rule‑50 draw threshold has been reached and that the current position is a draw.

- Flattened shape
  - All features are concatenated to a single vector of size 881: `12*64 + 4 + 8 + 101`.

## Backbone

The big bet of this project is that a chess position contains a small enough amount of information to fit in "one token" of a transformer. Therefore, we treat a position as a single “token”: a flat 881‑D vector fed to an MLP‑only network (a transformer with the attention blocks removed). Blocks use pre‑LN SwiGLU with residual connections.

## Heads (Outputs)

- Value (3)
  - W/D/L probabilities trained with cross‑entropy. We derive the training target WDL from lc0’s q (win‑loss expectation) and d (draw probability) during metrics where needed.

- Policy (1858)
  - Move logits over lc0’s classical 1858‑move encoding. Training uses cross‑entropy with illegal moves masked via target −1 and legal moves normalized as probabilities.
