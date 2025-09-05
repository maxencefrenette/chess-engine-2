# Project Goals

The goal of this project is to produce a chess engine using modern deep learning techniques commonly used for LLMs.

## Planned architecture

* Training data: T80 LC0 training data.
* Training code: Bespoke code using PyTorch.
* Model architecture: A MLP-only network. Input features are fed in using a learned embedding layer, then passed through a series of residual MLP blocks, and finally outputting a policy and value head using an unembedding layer.
* Inference code: Same pytorch code used for training.
* MCTS search: Lc0 engine, with a custom backend to call python via RPC.
