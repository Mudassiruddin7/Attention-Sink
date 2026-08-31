"""Cache growth model for a hybrid attention stack.

The Kimi K3 report states the layer counts, the hidden size and the head
count, but it does not publish the latent rank of the Gated MLA layers or
the head dimensions of the Kimi Delta Attention layers. Those two numbers
are therefore inputs to this model and are listed with the results, so a
reader can substitute the published values once they appear and recompute
every figure.

Two quantities are exact and need no assumption at all.

  fraction of layers whose cache does not grow with context length
  ratio between the two cache terms at any given context length

Everything else scales linearly in the assumed dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class HybridSpec:
    # Layer counts, hidden size and head count are from Table 1 of the Kimi K3
    # technical report (arXiv:2607.24653): 93 layers, "69 KDA + 24 MLA",
    # hidden dimension 7,168, 96 attention heads. The report does not state
    # the MLA latent rank or the KDA state geometry, so those stay assumptions.
    name: str = "Kimi K3"
    n_layers: int = 93
    n_linear_layers: int = 69       # Kimi Delta Attention layers (reported)
    n_global_layers: int = 24       # Gated MLA layers (reported)
    hidden: int = 7168              # reported
    n_heads: int = 96               # reported
    kv_latent_rank: int = 512       # assumption, not published for K3
    state_dk: int = 128             # assumption, not published for K3
    state_dv: int = 128             # assumption, not published for K3
    bytes_per_elem: int = 2         # bf16

    def global_cache_bytes(self, tokens: int) -> int:
        """Latent key value cache of the global layers, linear in tokens."""
        return (self.n_global_layers * self.kv_latent_rank
                * self.bytes_per_elem * tokens)

    def linear_state_bytes(self, tokens: int = 0) -> int:
        """Recurrent state of the linear layers, constant in tokens."""
        return (self.n_linear_layers * self.n_heads * self.state_dk
                * self.state_dv * self.bytes_per_elem)

    def dense_cache_bytes(self, tokens: int) -> int:
        """Cache a fully dense stack of the same depth would need."""
        head_dim = self.hidden // self.n_heads
        return (self.n_layers * 2 * self.n_heads * head_dim
                * self.bytes_per_elem * tokens)

    def total_bytes(self, tokens: int) -> int:
        return self.global_cache_bytes(tokens) + self.linear_state_bytes(tokens)

    def constant_layer_fraction(self) -> float:
        return self.n_linear_layers / self.n_layers

    def summary(self):
        return asdict(self)


def gib(x: float) -> float:
    return x / (1024 ** 3)


def table(spec: HybridSpec, lengths):
    rows = []
    for t in lengths:
        g = spec.global_cache_bytes(t)
        s = spec.linear_state_bytes(t)
        d = spec.dense_cache_bytes(t)
        rows.append({
            "tokens": t,
            "global_cache_gib": gib(g),
            "linear_state_gib": gib(s),
            "hybrid_total_gib": gib(g + s),
            "dense_cache_gib": gib(d),
            "reduction_x": d / (g + s),
        })
    return rows
