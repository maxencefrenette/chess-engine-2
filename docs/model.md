# Model

## Features (Inputs)

- Board: A tensor of dimensions (12, 8, 8) representing the input board. Following the lc0 convention, pieces are encoded in this order: "PNBRQKpnbrqk" from the perspective of the side to move. Uppercase letters represent the side to move and lowercase letters represent the opponent's pieces.
- Castling rights: A tensor of dimensions (4) representing the castling rights of the current position. The order of the castling rights is "KQkq".
- En passant: A tensor of dimensions (8) representing the en passant squares of the current position.
- Rule50: A one-hot encoding of dimensions (101) representing the rule 50 counter from 0 to 100. Note that a value of 100 represents a game that is already drawn by the rule 50 rule.

All the features are flattened and concatenated into a single tensor of dimension 881.

## Backbone

The big bet of this project is that a chess position contains a small enough amount of information to fit in "one token" of a transformer. Therefore, the architecture is a transformer, but without the attention layers, which reduces to a MLP-only network.

## Heads (Outputs)

- Value: A tensor of dimensions (3) representing the expected outcome of the game as WDL probabilities. It is trained using cross-entropy loss.
- Policy: A tensor of dimensions (1858) representing the expected probabilities of each move being the best move. It is trained using cross-entropy loss.where illegal moves are masked.
