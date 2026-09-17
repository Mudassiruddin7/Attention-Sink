"""Depth mixing rules.

StandardResidual reproduces the usual pre norm update, in which every
layer output is added into one running stream with unit weight.

BlockAttentionResidual reproduces Block Attention Residuals as described
by the Kimi team. Layers are partitioned into blocks, each block is
reduced to one representation by summation, and every layer builds its
input by attending over the token embedding plus the finished block
representations with a learned pseudo query.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class StandardResidual(nn.Module):
    """Running sum over depth. Kept as a module so both rules share an API."""

    def __init__(self, d_model: int, n_layers: int, block_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

    def reset(self, embedding: torch.Tensor):
        self.stream = embedding
        self.sources: List[torch.Tensor] = []

    def read(self, layer_idx: int) -> torch.Tensor:
        return self.stream

    def write(self, layer_idx: int, layer_out: torch.Tensor):
        self.stream = self.stream + layer_out

    def final(self) -> torch.Tensor:
        return self.stream

    def weights(self):
        return None


class BlockAttentionResidual(nn.Module):
    """Attention over block level representations instead of a running sum."""

    def __init__(self, d_model: int, n_layers: int, block_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.block_size = block_size
        self.n_blocks = (n_layers + block_size - 1) // block_size
        # One learnable pseudo query per layer, plus one for the output head.
        self.pseudo_q = nn.Parameter(torch.randn(n_layers + 1, d_model) * 0.02)
        self.scale = d_model ** -0.5
        self.last_weights = None

    def reset(self, embedding: torch.Tensor):
        # Source zero is always the token embedding. Finished sources are
        # normalised once, when they are added, rather than at every read.
        self.sources: List[torch.Tensor] = [embedding]
        self.normed: List[torch.Tensor] = [rms_norm(embedding)]
        self.partial = None
        self.stream = embedding
        self.last_weights = None

    def _attend(self, idx: int, sources, normed) -> torch.Tensor:
        q = self.pseudo_q[idx].to(sources[0].dtype)                 # (D,)
        logits = torch.stack([n @ q for n in normed], dim=0) * self.scale
        alpha = torch.softmax(logits.float(), dim=0).to(sources[0].dtype)   # (S, B, T)
        self.last_weights = alpha.detach().float().mean(dim=(1, 2))
        out = alpha[0, ..., None] * sources[0]
        for s in range(1, len(sources)):
            out = out + alpha[s, ..., None] * sources[s]
        return out

    def read(self, layer_idx: int) -> torch.Tensor:
        pos_in_block = layer_idx % self.block_size
        sources, normed = list(self.sources), list(self.normed)
        if pos_in_block > 0:
            # Layers after the first in a block also see the partial sum
            # accumulated so far inside the current block.
            sources.append(self.partial)
            normed.append(rms_norm(self.partial))
        self.stream = self._attend(layer_idx, sources, normed)
        return self.stream

    def write(self, layer_idx: int, layer_out: torch.Tensor):
        pos_in_block = layer_idx % self.block_size
        if pos_in_block == 0:
            self.partial = layer_out
        else:
            self.partial = self.partial + layer_out
        if pos_in_block == self.block_size - 1 or layer_idx == self.n_layers - 1:
            self.sources.append(self.partial)
            self.normed.append(rms_norm(self.partial))
            self.partial = None

    def final(self) -> torch.Tensor:
        return self._attend(self.n_layers, self.sources, self.normed)

    def weights(self):
        return self.last_weights
