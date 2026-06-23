"""
Alternative message-passing blocks for the InterGATE ablation pipeline.

Drop this file into the same package folder as model.py, for example:

    intergate/backbone_blocks.py

It does not replace model.py.  Instead, backbone_ablation.py temporarily patches
intergate.model.ResGATBlock at model-construction time, so the rest of the
pipeline keeps using the same gates, Top-K pruning, X_h conditioning, JK, pooling,
training loops and stability-selection code.

Backbones implemented here:
    - weighted_sage / graphsage
    - weighted_gin / gin
    - local_graph_transformer / graph_transformer

All blocks expose the same forward API as ResGATBlock:

    forward(h, edge_index, edge_w=None) -> h_new

where h has shape (B, N, H), edge_index is (2, E), and edge_w can be (E,) or
(B, E).  This is important because your model may use sample-conditioned edge
gates, in which case the effective edge weights are sample-specific.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Dict, Type
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Shared low-level helpers
# -------------------------------------------------------------------------


def _expand_edge_w(edge_w: Optional[torch.Tensor], B: int, E: int, device, dtype=None) -> Optional[torch.Tensor]:
    """Return flattened edge weights of shape (B*E,), accepting (E,) or (B,E)."""
    if edge_w is None:
        return None

    w = edge_w.to(device=device)
    if dtype is not None:
        w = w.to(dtype=dtype)

    if w.dim() == 1:
        if w.numel() != E:
            raise ValueError(f"edge_w shape mismatch: expected E={E}, got {tuple(w.shape)}")
        return w.view(1, E).expand(B, E).reshape(-1)

    if w.dim() == 2:
        if w.shape[0] != B or w.shape[1] != E:
            raise ValueError(f"edge_w shape mismatch: expected {(B, E)}, got {tuple(w.shape)}")
        return w.reshape(-1)

    raise ValueError(f"edge_w must be (E,) or (B,E), got {tuple(w.shape)}")


def _segment_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax over incoming edges for each destination node."""
    finfo = torch.finfo(scores.dtype)
    max_per = torch.full((num_nodes,), finfo.min, device=scores.device, dtype=scores.dtype)
    max_per.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)

    exp = torch.exp(scores - max_per[dst])
    sum_per = torch.zeros((num_nodes,), device=scores.device, dtype=scores.dtype)
    sum_per.scatter_add_(0, dst, exp)
    return exp / (sum_per[dst] + 1e-16)


def _replicate_edges(edge_index: torch.Tensor, B: int, N: int, device):
    """Replicate graph edges across a batch, returning flattened src_b and dst_b."""
    edge_index = edge_index.to(device=device, dtype=torch.long)
    E = edge_index.shape[1]
    src = edge_index[0]
    dst = edge_index[1]
    offsets = (torch.arange(B, device=device) * N).view(B, 1)
    src_b = (src.view(1, E) + offsets).reshape(-1)
    dst_b = (dst.view(1, E) + offsets).reshape(-1)
    return src_b, dst_b, E


def _weighted_neighbor_aggregate(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    edge_w: Optional[torch.Tensor] = None,
    *,
    normalize_by_abs_weight: bool = True,
) -> torch.Tensor:
    """
    Weighted aggregation of incoming messages.

    h:          (B, N, H)
    edge_index: (2, E), source -> destination
    edge_w:     None, (E,), or (B,E)

    If edge_w is signed, the sign is preserved in the message while the optional
    denominator uses |w| for numerical stability.
    """
    B, N, H = h.shape
    device = h.device
    dtype = h.dtype
    E = int(edge_index.shape[1])
    if E == 0:
        return torch.zeros_like(h)

    src_b, dst_b, E = _replicate_edges(edge_index, B=B, N=N, device=device)
    x = h.reshape(B * N, H)
    x_src = x[src_b]

    w = _expand_edge_w(edge_w, B=B, E=E, device=device, dtype=dtype)
    if w is None:
        msg = x_src
        denom_w = torch.ones((B * E,), device=device, dtype=dtype)
    else:
        msg = x_src * w.view(-1, 1)
        denom_w = w.abs() if normalize_by_abs_weight else torch.ones_like(w)

    out = torch.zeros((B * N, H), device=device, dtype=dtype)
    out.index_add_(0, dst_b, msg)

    if normalize_by_abs_weight:
        denom = torch.zeros((B * N,), device=device, dtype=dtype)
        denom.index_add_(0, dst_b, denom_w)
        out = out / denom.clamp_min(1e-6).view(-1, 1)

    return out.reshape(B, N, H)


# -------------------------------------------------------------------------
# Alternative blocks
# -------------------------------------------------------------------------


class WeightedGraphSAGEBlock(nn.Module):
    """
    Edge-weight-aware GraphSAGE block.

    This is not a plain fixed-prior GraphSAGE baseline. It receives the same
    gated/pruned effective edge weights as ResGATBlock, so it is suitable for a
    fair backbone-only ablation inside your proposed framework.
    """

    def __init__(
        self,
        hidden: int,
        heads: int = 4,              # accepted for API compatibility; unused
        dropout: float = 0.3,
        residual: bool = True,
        negative_slope: float = 0.2, # accepted for API compatibility; unused
        use_edge_weight: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.residual = bool(residual)
        self.use_edge_weight = bool(use_edge_weight)

        self.lin_self = nn.Linear(hidden, hidden, bias=False)
        self.lin_neigh = nn.Linear(hidden, hidden, bias=True)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_w: Optional[torch.Tensor] = None) -> torch.Tensor:
        ew = edge_w if self.use_edge_weight else None
        neigh = _weighted_neighbor_aggregate(h, edge_index, ew, normalize_by_abs_weight=True)
        out = self.lin_self(h) + self.lin_neigh(neigh)
        out = F.relu(out)
        out = self.drop(out)
        out = self.norm(out)
        return out + h if self.residual else out


class WeightedGINBlock(nn.Module):
    """
    Edge-weight-aware GIN-like block.

    Classical GIN uses unweighted sums. Here we use the effective gated/pruned
    signed edge weights so the graph selection mechanism is still active.
    """

    def __init__(
        self,
        hidden: int,
        heads: int = 4,              # accepted for API compatibility; unused
        dropout: float = 0.3,
        residual: bool = True,
        negative_slope: float = 0.2, # accepted for API compatibility; unused
        use_edge_weight: bool = True,
        train_eps: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.residual = bool(residual)
        self.use_edge_weight = bool(use_edge_weight)
        self.eps = nn.Parameter(torch.zeros(1)) if train_eps else torch.zeros(1)

        self.mlp = nn.Sequential(
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_w: Optional[torch.Tensor] = None) -> torch.Tensor:
        ew = edge_w if self.use_edge_weight else None
        agg = _weighted_neighbor_aggregate(h, edge_index, ew, normalize_by_abs_weight=False)
        out = (1.0 + self.eps.to(device=h.device, dtype=h.dtype)) * h + agg
        out = self.mlp(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.norm(out)
        return out + h if self.residual else out


class LocalGraphTransformerBlock(nn.Module):
    """
    Local graph-transformer-style attention block restricted to edge_index.

    This is deliberately NOT full N x N self-attention. It uses query-key-value
    attention only over existing graph edges, so memory stays comparable to your
    ResGAT block and the prior graph remains the computational support.
    """

    def __init__(
        self,
        hidden: int,
        heads: int = 4,
        dropout: float = 0.3,
        residual: bool = True,
        negative_slope: float = 0.2, # accepted for API compatibility; unused
        use_edge_weight: bool = True,
        **kwargs,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}")
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.dh = int(hidden // heads)
        self.residual = bool(residual)
        self.use_edge_weight = bool(use_edge_weight)

        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=True)

        # Per-head scalar edge bias. The edge weight is still also used as a
        # multiplicative message factor when use_edge_weight=True.
        self.edge_bias = nn.Parameter(torch.zeros(heads))
        self.drop_attn = nn.Dropout(dropout)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_w: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, H = h.shape
        device = h.device
        dtype = h.dtype
        E = int(edge_index.shape[1])
        if E == 0:
            return h

        src_b, dst_b, E = _replicate_edges(edge_index, B=B, N=N, device=device)
        x = h.reshape(B * N, H)

        q = self.q(x).reshape(B * N, self.heads, self.dh)
        k = self.k(x).reshape(B * N, self.heads, self.dh)
        v = self.v(x).reshape(B * N, self.heads, self.dh)

        q_dst = q[dst_b]
        k_src = k[src_b]
        v_src = v[src_b]

        scores = (q_dst * k_src).sum(dim=-1) / math.sqrt(float(self.dh))  # (B*E, heads)

        w = _expand_edge_w(edge_w, B=B, E=E, device=device, dtype=dtype) if self.use_edge_weight else None
        if w is not None:
            scores = scores + torch.tanh(w).view(-1, 1) * self.edge_bias.view(1, -1)

        num_nodes = B * N
        alpha_heads = []
        for head_id in range(self.heads):
            alpha_h = _segment_softmax(scores[:, head_id], dst_b, num_nodes=num_nodes)
            alpha_heads.append(alpha_h)
        alpha = torch.stack(alpha_heads, dim=1)  # (B*E, heads)
        alpha = self.drop_attn(alpha)

        msg = alpha.unsqueeze(-1) * v_src
        if w is not None:
            msg = msg * w.view(-1, 1, 1)

        out_flat = torch.zeros((num_nodes, self.heads, self.dh), device=device, dtype=dtype)
        for head_id in range(self.heads):
            out_flat[:, head_id, :].index_add_(0, dst_b, msg[:, head_id, :])

        out = out_flat.reshape(B, N, self.hidden)
        out = self.out(out)
        out = F.gelu(out)
        out = self.drop(out)
        out = self.norm(out)
        return out + h if self.residual else out


# -------------------------------------------------------------------------
# Patching utilities
# -------------------------------------------------------------------------


def normalize_backbone_name(name: str) -> str:
    name = str(name).strip().lower().replace("-", "_")
    aliases = {
        "gat": "gat",
        "resgat": "gat",
        "sage": "sage",
        "graphsage": "sage",
        "graph_sage": "sage",
        "gin": "gin",
        "transformer": "graph_transformer",
        "graphtransformer": "graph_transformer",
        "graph_transformer": "graph_transformer",
        "local_graph_transformer": "graph_transformer",
    }
    if name not in aliases:
        raise ValueError(
            f"Unknown backbone '{name}'. Valid values: gat, sage, gin, graph_transformer."
        )
    return aliases[name]


BACKBONE_TO_BLOCK: Dict[str, Optional[Type[nn.Module]]] = {
    "gat": None,  # keep the original ResGATBlock from model.py
    "sage": WeightedGraphSAGEBlock,
    "gin": WeightedGINBlock,
    "graph_transformer": LocalGraphTransformerBlock,
}


@contextmanager
def patch_backbone(backbone: str):
    """
    Temporarily replace intergate.model.ResGATBlock while constructing models.

    Use this context around calls to ablation.run_single_seed() or
    ablation.evaluate_bundle_on_test(). The original ResGATBlock is restored
    afterwards, even if an exception occurs.
    """
    backbone = normalize_backbone_name(backbone)

    # Import lazily to avoid circular imports at package import time.
    from . import model as model_module

    old_block = model_module.ResGATBlock
    new_block = BACKBONE_TO_BLOCK[backbone]

    if new_block is None:
        # Native model.py ResGATBlock
        yield backbone
        return

    model_module.ResGATBlock = new_block
    try:
        yield backbone
    finally:
        model_module.ResGATBlock = old_block
