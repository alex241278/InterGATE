"""
GNN model architecture:
  - sparse_mm_batch, NodeBatchNorm
  - ResGATBlock, SignedResGATBlock
  - ImprovedSharedGraphGNN
  - HybridGNNTabular, _make_mlp
"""

import math
import inspect
from typing import Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG


# ── Sparse helpers ─────────────────────────────────

def sparse_mm_batch(adj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    # h: (B, N, H)  adj: (N, N) sparse
    B, N, Hh = h.shape
    x = h.permute(1, 0, 2).reshape(N, B * Hh)        # (N, B*H)
    y = torch.sparse.mm(adj, x)                      # (N, B*H)
    out = y.reshape(N, B, Hh).permute(1, 0, 2).contiguous()
    return out

class NodeBatchNorm(nn.Module):
    """BatchNorm1d aplicada al último eje (features) de un tensor (B,N,H)."""
    def __init__(self, hidden: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, H = x.shape
        y = x.reshape(B * N, H)
        y = self.bn(y)
        return y.reshape(B, N, H)



# ── GAT block helpers ──────────────────────────────

def _expand_edge_w(edge_w: torch.Tensor, B: int, E: int, device, dtype=None) -> torch.Tensor:
    """Devuelve w_flat de shape (B*E,) aceptando edge_w (E,) o (B,E)."""
    if edge_w is None:
        return None
    w = edge_w.to(device=device)
    if dtype is not None:
        w = w.to(dtype=dtype)
    if w.dim() == 1:
        assert w.numel() == E, f"edge_w (E,) esperado E={E}, got {w.shape}"
        w = w.view(1, E).expand(B, E).reshape(-1)
    elif w.dim() == 2:
        assert w.shape[0] == B and w.shape[1] == E, f"edge_w (B,E) esperado {(B,E)}, got {tuple(w.shape)}"
        w = w.reshape(-1)
    else:
        raise ValueError(f"edge_w debe ser (E,) o (B,E), got {tuple(w.shape)}")
    return w

def _segment_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Softmax por grupos (por nodo destino), sin torch_scatter.
    scores: (E,) ; dst: (E,) en [0, num_nodes)
    """
    # max por dst
    finfo = torch.finfo(scores.dtype)
    max_per = torch.full((num_nodes,), finfo.min, device=scores.device, dtype=scores.dtype)
    max_per.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)

    exp = torch.exp(scores - max_per[dst])
    sum_per = torch.zeros((num_nodes,), device=scores.device, dtype=scores.dtype)
    sum_per.scatter_add_(0, dst, exp)
    return exp / (sum_per[dst] + 1e-16)


class ResGATBlock(nn.Module):
    """
    GAT multi-head (implementación propia) para un grafo fijo compartido por batch.
    - Atención por nodo destino (softmax sobre aristas entrantes)
    - Opción de usar edge_weight (ya con gate y normalización) como:
        (i) sesgo aditivo en el score de atención
        (ii) factor multiplicativo en el mensaje (permite signo)
    """
    def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.3,
                 residual: bool = True, negative_slope: float = 0.2,
                 use_edge_weight: bool = True):
        super().__init__()
        assert hidden % heads == 0, "hidden debe ser divisible por heads"
        self.hidden = hidden
        self.heads = heads
        self.dh = hidden // heads
        self.residual = residual
        self.negative_slope = negative_slope
        self.use_edge_weight = use_edge_weight

        self.lin = nn.Linear(hidden, hidden, bias=False)  # produce heads*dh
        # vectores de atención por head
        self.att_src = nn.Parameter(torch.empty(heads, self.dh))
        self.att_dst = nn.Parameter(torch.empty(heads, self.dh))
        # peso para edge_weight como feature (1 escalar por head)
        self.att_w   = nn.Parameter(torch.zeros(heads))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        h: (B, N, H)
        edge_index: (2, E) sobre N nodos
        edge_w: (E,) pesos por arista (opcional)
        """
        B, N, H = h.shape
        E = edge_index.shape[1]
        device = h.device

        # Flatten batch: (B*N, H)
        x = h.reshape(B * N, H)
        x = self.lin(x).reshape(B * N, self.heads, self.dh)  # (BN, heads, dh)

        src = edge_index[0].to(device)
        dst = edge_index[1].to(device)

        # Replicar aristas por batch
        offsets = (torch.arange(B, device=device) * N).view(B, 1)  # (B,1)
        src_b = (src.view(1, E) + offsets).reshape(-1)  # (B*E,)
        dst_b = (dst.view(1, E) + offsets).reshape(-1)

        x_src = x[src_b]  # (B*E, heads, dh)
        x_dst = x[dst_b]

        # score de atención
        e = (x_src * self.att_src.view(1, self.heads, self.dh)).sum(-1) \
          + (x_dst * self.att_dst.view(1, self.heads, self.dh)).sum(-1)   # (B*E, heads)

        if self.use_edge_weight and (edge_w is not None):
            w = _expand_edge_w(edge_w, B=B, E=E, device=device, dtype=h.dtype)  # (B*E,)
            e = e + w.view(-1, 1) * self.att_w.view(1, -1)                    # (B*E, heads)

        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # softmax por dst y head
        num_nodes = B * N
        alphas = []
        for h_i in range(self.heads):
            a = _segment_softmax(e[:, h_i], dst_b, num_nodes=num_nodes)  # (B*E,)
            alphas.append(a)
        alpha = torch.stack(alphas, dim=1)  # (B*E, heads)

        # mensaje: alpha * x_src  (y opcionalmente * edge_w para introducir signo/escala)
        msg = alpha.unsqueeze(-1) * x_src  # (B*E, heads, dh)
        if self.use_edge_weight and (edge_w is not None):
            w = _expand_edge_w(edge_w, B=B, E=E, device=device, dtype=h.dtype)  # (B*E,)
            msg = msg * w.view(-1, 1, 1)

        # agregación por dst
        out = torch.zeros((num_nodes, self.heads, self.dh), device=device, dtype=h.dtype)
        for h_i in range(self.heads):
            out[:, h_i, :].index_add_(0, dst_b, msg[:, h_i, :])

        out = out.reshape(B * N, self.hidden).reshape(B, N, self.hidden)
        out = F.elu(out)
        out = self.drop(out)
        out = self.norm(out)

        return (out + h) if self.residual else out


# ── SignedResGATBlock + ImprovedSharedGraphGNN ─────


class SignedResGATBlock(nn.Module):
    """
    Variante opcional: 2 canales (pos/neg) con parámetros separados.
    """
    def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.3,
                 negative_slope: float = 0.2, use_edge_weight: bool = True, residual: bool = True,
                 block_use_self: bool = True):  # <-- AÑADIDO
        super().__init__()
        self.residual = residual

        # Pasa use_self SOLO si ResGATBlock lo soporta
        resgat_params = inspect.signature(ResGATBlock.__init__).parameters
        kw = dict(heads=heads, dropout=dropout, residual=False,
                  negative_slope=negative_slope, use_edge_weight=use_edge_weight)
        if "use_self" in resgat_params:
            kw["use_self"] = block_use_self

        self.pos = ResGATBlock(hidden, **kw)
        self.neg = ResGATBlock(hidden, **kw)

        self.alpha_pos = nn.Parameter(torch.tensor(1.0))
        self.alpha_neg = nn.Parameter(torch.tensor(1.0))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor,
                edge_index_pos: torch.Tensor, edge_w_pos: Optional[torch.Tensor],
                edge_index_neg: torch.Tensor, edge_w_neg: Optional[torch.Tensor]) -> torch.Tensor:
        out_pos = self.pos(h, edge_index_pos, edge_w_pos) if edge_index_pos.numel() else torch.zeros_like(h)
        out_neg = self.neg(h, edge_index_neg, edge_w_neg) if edge_index_neg.numel() else torch.zeros_like(h)
        out = self.alpha_pos * out_pos + self.alpha_neg * out_neg
        out = F.elu(out)
        out = self.drop(out)
        out = self.norm(out)
        return (out + h) if self.residual else out





class ImprovedSharedGraphGNN(nn.Module):
    """
    Variante GAT del modelo compartido (sin PyG), extendida con:
    - Edge-type aware gating (HuRI vs OmniPath(+/-))
    - (opcional) Signed dual-backbone (pos/neg con parámetros separados)
    - (opcional) Sample-conditioned gating (dependiente de X_graph)
    - Hard-pruning top-K (robusto) + reporting
    - JK + multi-head attention pooling (genes) + (opcional) pooling jerárquico por reguladores

    NOTA: asume que existen:
      - ResGATBlock
      - SignedResGATBlock (si usas signed_channels_mode="dual_backbone")
      - NodeBatchNorm
    """

    def __init__(
        self,
        num_nodes: int,
        hidden: int,
        n_classes: int,
        graph_feat_dim: int = 0,
        num_layers: int = 6,
        dropout: float = 0.3,
        num_heads: int = 4,               # heads del POOLING
        edge_index: torch.Tensor = None,  # (2, E)
        edge_weight: torch.Tensor = None, # (E,)
        edge_type: Optional[torch.Tensor] = None,  # (E,) 0=HuRI,1=OP(+),2=OP(-)
        keep_self_loops: bool = True,

        # Bloques GAT
        block_residual: bool = True,
        block_use_self: bool = True,      # <- para compatibilidad (si el bloque lo soporta)
        xgraph_dropout: float = 0.0,
        gat_heads: int = 4,
        gat_use_edge_weight: bool = True,

        # --- toggles ---
        edge_type_gating: bool = True,
        sample_cond_gating: bool = True,
        sample_cond_mode: str = "per_type",          # "per_type" | "global"
        signed_channels_mode: str = "type_only",     # "type_only" | "dual_backbone"
        use_regulator_pool: bool = False,
        regulator_groups: Optional[List[torch.Tensor]] = None,

        # Para no romper notebooks viejos con args extra:
        **unused_kwargs,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.hidden = int(hidden)
        self.n_classes = int(n_classes)
        self.graph_feat_dim = int(graph_feat_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.gat_heads = int(gat_heads)
        self.block_use_self = bool(block_use_self)

        assert edge_index is not None and edge_weight is not None, "Pasa edge_index/edge_weight al modelo"
        edge_index = edge_index.long()
        edge_weight = edge_weight.float()

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight_base", edge_weight)

        # --- tipos de arista ---
        if edge_type is None:
            edge_type = torch.zeros(edge_weight.numel(), dtype=torch.long)
        self.register_buffer("edge_type", edge_type.long())
        self.num_edge_types = int(self.edge_type.max().item()) + 1 if self.edge_type.numel() else 1

        # --- self-loops mask (para hard-pruning) ---
        slm = (self.edge_index[0] == self.edge_index[1]).bool()
        if not keep_self_loops:
            slm = torch.zeros_like(slm, dtype=torch.bool)
        self.register_buffer("self_loop_mask", slm)

        # --- normalización base (una vez) para evitar recomputar grados en cada forward ---
        with torch.no_grad():
            idx = self.edge_index
            w0 = self.edge_weight_base
            N = self.num_nodes
            deg = torch.zeros(N, device=w0.device, dtype=w0.dtype)
            deg.index_add_(0, idx[0], w0.abs())
            deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
            w_norm = w0 * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
        self.register_buffer("edge_weight_norm_base", w_norm.float())

        # Gate aprendible (por arista)
        self.edge_logit = nn.Parameter(torch.zeros_like(self.edge_weight_base))
        self.gate_tau = 1.0

        # Edge-type gating (escala y sesgo por tipo)
        self.edge_type_gating = bool(edge_type_gating)
        if self.edge_type_gating:
            self.type_scale = nn.Parameter(torch.ones(self.num_edge_types))
            self.type_bias  = nn.Parameter(torch.zeros(self.num_edge_types))
        else:
            self.register_buffer("type_scale", torch.ones(self.num_edge_types))
            self.register_buffer("type_bias", torch.zeros(self.num_edge_types))

        # Sample-conditioned gating
        self.sample_cond_gating = bool(sample_cond_gating) and (self.graph_feat_dim > 0)
        self.sample_cond_mode = str(sample_cond_mode)
        if self.sample_cond_gating:
            out_dim = self.num_edge_types if self.sample_cond_mode == "per_type" else 1
            self.cond_mlp = nn.Sequential(
                nn.Linear(self.graph_feat_dim, max(8, self.graph_feat_dim // 2)),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(max(8, self.graph_feat_dim // 2), out_dim),
            )
        else:
            self.cond_mlp = None

        # Hard pruning (top-ratio)
        self.hard_keep_ratio = None
        self.hard_edge_ids_cpu = None
        self.hard_thr_logit = None
        self.hard_thr_gate = None
        self.hard_thr = None

        # Input MLP por nodo
        self.inp_lin1 = nn.Linear(1, hidden)
        self.inp_bn1  = NodeBatchNorm(hidden)
        self.inp_lin2 = nn.Linear(hidden, hidden)
        self.inp_bn2  = NodeBatchNorm(hidden)
        self.inp_drop = nn.Dropout(dropout)

        # ---------- Backbone (GAT) ----------
        self.signed_channels_mode = str(signed_channels_mode)

        # Construcción robusta: solo pasa "use_self"/"block_use_self" si el bloque lo acepta
        resgat_params = inspect.signature(ResGATBlock.__init__).parameters
        signed_params = None
        if "SignedResGATBlock" in globals():
            signed_params = inspect.signature(SignedResGATBlock.__init__).parameters

        def make_resgat_block():
            kw = dict(
                heads=gat_heads,
                dropout=dropout,
                residual=block_residual,
                use_edge_weight=gat_use_edge_weight,
            )
            # Algunos ResGATBlock usan use_self / add_self_loops / etc. Solo pasamos si existe.
            if "use_self" in resgat_params:
                kw["use_self"] = self.block_use_self
            if "block_use_self" in resgat_params:
                kw["block_use_self"] = self.block_use_self
            return ResGATBlock(hidden, **kw)

        def make_signed_block():
            # SignedResGATBlock puede (o no) aceptar block_use_self
            kw = dict(
                heads=gat_heads,
                dropout=dropout,
                residual=block_residual,
                use_edge_weight=gat_use_edge_weight,
            )
            if signed_params is not None and "block_use_self" in signed_params:
                kw["block_use_self"] = self.block_use_self
            return SignedResGATBlock(hidden, **kw)

        if self.signed_channels_mode == "dual_backbone":
            self.blocks = nn.ModuleList([make_signed_block() for _ in range(num_layers)])
            # máscaras por signo (según peso base)
            self.register_buffer("pos_mask", (self.edge_weight_base >= 0))
            self.register_buffer("neg_mask", (self.edge_weight_base < 0))
        else:
            self.blocks = nn.ModuleList([make_resgat_block() for _ in range(num_layers)])
            self.pos_mask = None
            self.neg_mask = None

        # JK
        self.jk = nn.Sequential(
            nn.Linear(hidden * (num_layers + 1), hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )

        # Pooling (genes)
        self.pool_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            ) for _ in range(num_heads)
        ])

        # Pooling jerárquico (reguladores) – opcional (no rompe la salida gene-attn)
        self.use_regulator_pool = bool(use_regulator_pool) and (regulator_groups is not None) and (len(regulator_groups) > 0)
        self.regulator_groups = regulator_groups if self.use_regulator_pool else None

        self.xgraph_drop = nn.Dropout(xgraph_dropout)

        head_in = hidden * num_heads + graph_feat_dim
        if self.use_regulator_pool:
            head_in += hidden * num_heads  # concatenamos g_reg
        self.trunk = nn.Sequential(
            nn.Linear(head_in, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head4 = nn.Linear(hidden, n_classes)
        self.head_AO = nn.Linear(hidden, 2)

        # Recon head (para pretraining masked gene modeling)
        self.recon_head = nn.Linear(hidden, 1)

    # ---------- gates ----------
    def _typed_logit(self):
        """Aplica escala/sesgo por tipo de arista en el MISMO device que edge_logit."""
        logit = self.edge_logit
        if not self.edge_type_gating:
            return logit

        dev = logit.device

        # edge_type debe vivir en el mismo device para indexar
        et = self.edge_type
        if isinstance(et, torch.Tensor) and et.device != dev:
            et = et.to(dev)

        ts = self.type_scale
        tb = self.type_bias

        # type_scale / type_bias pueden ser nn.Parameter (si edge_type_gating=True)
        if isinstance(ts, torch.nn.Parameter):
            if ts.data.device != dev:
                ts.data = ts.data.to(dev)
            ts_use = ts
        else:
            ts_use = ts.to(dev) if isinstance(ts, torch.Tensor) and ts.device != dev else ts

        if isinstance(tb, torch.nn.Parameter):
            if tb.data.device != dev:
                tb.data = tb.data.to(dev)
            tb_use = tb
        else:
            tb_use = tb.to(dev) if isinstance(tb, torch.Tensor) and tb.device != dev else tb

        return logit * ts_use[et] + tb_use[et]


    def edge_gate(self, x_graph: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Gate en (0,1). Devuelve:
          - (E,) si no hay conditioning
          - (B,E) si sample_cond_gating está activo y x_graph está presente
        """
        # device "fuente de verdad" del modelo
        dev = self.edge_logit.device
    
        # typed logit (E,) y SIEMPRE en dev (tu _typed_logit ya lo arregla)
        base = self._typed_logit() / float(self.gate_tau)  # (E,)
    
        # --- Asegura x_graph en el mismo device del modelo ---
        if x_graph is not None and isinstance(x_graph, torch.Tensor) and x_graph.device != dev:
            x_graph = x_graph.to(dev, non_blocking=True)
    
        # Conditioning por muestra (B,E)
        if self.sample_cond_gating and (x_graph is not None) and (x_graph.numel() > 0):
            # --- Asegura cond_mlp en el mismo device ---
            if hasattr(self, "cond_mlp") and (self.cond_mlp is not None):
                try:
                    pdev = next(self.cond_mlp.parameters()).device
                except StopIteration:
                    pdev = dev
                if pdev != dev:
                    self.cond_mlp = self.cond_mlp.to(dev)
    
            B = x_graph.shape[0]
            delta = self.cond_mlp(x_graph)  # (B,T) o (B,1) pero ya en dev
    
            # edge_type debe estar en el mismo device que delta para indexar columnas
            if isinstance(self.edge_type, torch.Tensor) and self.edge_type.device != delta.device:
                self.edge_type = self.edge_type.to(delta.device, non_blocking=True)
    
            if self.sample_cond_mode == "per_type":
                # gather por tipo -> (B,E)
                dt = delta[:, self.edge_type]
            else:
                # (B,1) broadcast a (B,E)
                dt = delta.view(B, 1)
    
            # base (E,) -> (B,E)
            base_b = base.view(1, -1).expand(B, -1)
            gate = torch.sigmoid(base_b + dt)
            return gate
    
        # Sin conditioning: (E,)
        return torch.sigmoid(base)


    def edge_l1_sum(self) -> torch.Tensor:
        g = self.edge_gate(None)  # soft, global
        return g.sum()

    def connectivity_penalty(self, min_deg: float = 0.05, use_abs: bool = True) -> torch.Tensor:
        """
        Penaliza nodos con grado esperado bajo usando *soft gates* (diferenciable).
        Grado esperado = suma de pesos (|w| opcional) de aristas incidentes (src y dst).
        Se calcula con gating global (sin conditioning) para ser estable.
        """
        g = self.edge_gate(None)  # (E,)
        w = self.edge_weight_norm_base
        if use_abs:
            w = w.abs()
        w = w * g
        N = self.num_nodes
        deg = torch.zeros(N, device=w.device, dtype=w.dtype)
        deg.index_add_(0, self.edge_index[0], w)
        deg.index_add_(0, self.edge_index[1], w)
        return F.relu(float(min_deg) - deg).mean()

    @torch.no_grad()
    def set_hard_keep_ratio(self, ratio: Optional[float]):
        """
        Hard-pruning robusto (top-k) para obtener un subgrafo compacto.
        - TOP-K sobre logits (pre-sigmoid) ajustados por tipo para ranking estable.
        - Mantiene SIEMPRE los self-loops (no cuentan para el TOP-K) si keep_self_loops estaba activo.
        - Si detecta pares u->v / v->u, conserva ambos.
        """
        self.hard_keep_ratio = ratio

        if ratio is None:
            self.hard_edge_ids_cpu = None
            self.hard_thr_logit = None
            self.hard_thr_gate = None
            self.hard_thr = None
            return

        score_full = (self._typed_logit().detach() / float(self.gate_tau))  # (E,)

        loop_ids = torch.where(self.self_loop_mask)[0]
        edge_ids = torch.where(~self.self_loop_mask)[0]

        if edge_ids.numel() == 0:
            kept_ids = loop_ids
            self.hard_edge_ids_cpu = kept_ids.detach().cpu()
            self.hard_thr_logit = float("nan")
            self.hard_thr_gate = float("nan")
            self.hard_thr = float("nan")
            return

        paired = False
        m = int(edge_ids.numel())
        if m % 2 == 0:
            half = m // 2
            a = edge_ids[:half]
            b = edge_ids[half:]
            ok = ((self.edge_index[0, a] == self.edge_index[1, b]) &
                  (self.edge_index[1, a] == self.edge_index[0, b])).float().mean().item()
            paired = (ok > 0.99)

        if paired:
            half = m // 2
            a = edge_ids[:half]
            b = edge_ids[half:]
            pair_score = torch.maximum(score_full[a], score_full[b])  # (half,)
            k_pairs = max(1, int(round(float(ratio) * half)))
            vals, sel = torch.topk(pair_score, k_pairs, largest=True, sorted=False)
            kept_ids = torch.cat([a[sel], b[sel], loop_ids], dim=0)
            thr_logit = float(vals.min().item())
        else:
            k = max(1, int(round(float(ratio) * m)))
            vals, sel = torch.topk(score_full[edge_ids], k, largest=True, sorted=False)
            kept_ids = torch.cat([edge_ids[sel], loop_ids], dim=0)
            thr_logit = float(vals.min().item())

        thr_gate = float(torch.sigmoid(torch.tensor(thr_logit)).item())
        self.hard_edge_ids_cpu = kept_ids.detach().cpu()
        self.hard_thr_logit = thr_logit
        self.hard_thr_gate  = thr_gate
        self.hard_thr = thr_gate

    def hard_prune_info(self):
        total = int(self.edge_weight_base.numel())
        if self.hard_keep_ratio is None or (self.hard_edge_ids_cpu is None):
            return {"keep_ratio": None, "thr_logit": None, "thr_gate": None, "thr": None, "kept": total, "total": total}

        kept = int(self.hard_edge_ids_cpu.numel())
        thr_logit = float(self.hard_thr_logit) if self.hard_thr_logit is not None else float("nan")
        thr_gate  = float(self.hard_thr_gate)  if self.hard_thr_gate  is not None else float("nan")
        return {"keep_ratio": float(self.hard_keep_ratio), "thr_logit": thr_logit, "thr_gate": thr_gate, "thr": thr_gate, "kept": kept, "total": total}

    # ---------- edges ----------
    def _select_edges(self, x_graph: Optional[torch.Tensor] = None):
        """Devuelve (edge_index_sel, w_sel) con gates + hard-pruning. w_sel puede ser (E,) o (B,E)."""
        g = self.edge_gate(x_graph)  # (E,) o (B,E)
        w_all = self.edge_weight_norm_base * g  # preserva signo (broadcast OK)

        if (self.hard_keep_ratio is not None) and (self.hard_edge_ids_cpu is not None):
            ids = self.hard_edge_ids_cpu.to(device=w_all.device, non_blocking=True)
            idx = self.edge_index[:, ids]
            if w_all.dim() == 1:
                w = w_all[ids]
            else:
                w = w_all[:, ids]
        else:
            idx = self.edge_index
            w = w_all
        return idx, w

    def build_adj(self) -> torch.Tensor:
        # solo válido para gating global (E,)
        idx, w = self._select_edges(None)
        if w.dim() != 1:
            raise RuntimeError("build_adj requiere gating global (edge_w 1D). Llama con sample_cond_gating=False o x_graph=None.")
        N = self.num_nodes
        return torch.sparse_coo_tensor(idx, w, size=(N, N), device=w.device, dtype=torch.float32).coalesce()

    def forward(self, x_gene: torch.Tensor, adj=None, x_graph: Optional[torch.Tensor] = None, return_attn: bool = False):
        # --- robustez CPU/GPU: mueve inputs al device real del modelo ---
        # device robusto: usa un parámetro "estable" (evita coger un parámetro perdido en CPU)
        try:
            dev = self.inp_lin1.weight.device
        except Exception:
            try:
                dev = self.edge_logit.device
            except Exception:
                dev = next(self.parameters()).device
        if isinstance(x_gene, torch.Tensor) and x_gene.device != dev:
            x_gene = x_gene.to(dev, non_blocking=True)
        if x_graph is not None and isinstance(x_graph, torch.Tensor) and x_graph.device != dev:
            x_graph = x_graph.to(dev, non_blocking=True)
        # edges con gate/pruning (y posible conditioning)
        edge_index_sel, edge_w_sel = self._select_edges(x_graph)
    
        if x_gene.dim() == 2:
            x_gene = x_gene.unsqueeze(-1)
    
        # Input MLP por nodo
        h = self.inp_lin1(x_gene)
        h = self.inp_bn1(h)
        h = F.relu(h)
        h = self.inp_drop(h)
        h = self.inp_lin2(h)
        h = self.inp_bn2(h)
        h0 = F.relu(h)
    
        hs = [h0]
        h = h0
    
        if self.signed_channels_mode == "dual_backbone":
            # divide edges por signo (según edge_weight_base) consistente con hard pruning
            if (self.hard_keep_ratio is not None) and (self.hard_edge_ids_cpu is not None):
                ids = self.hard_edge_ids_cpu.to(device=self.edge_weight_base.device, non_blocking=True)
                base_w = self.edge_weight_base[ids]
            else:
                base_w = self.edge_weight_base
    
            pos_ids = torch.where(base_w >= 0)[0]
            neg_ids = torch.where(base_w < 0)[0]
    
            ei_pos = edge_index_sel[:, pos_ids] if pos_ids.numel() else edge_index_sel[:, :0]
            ei_neg = edge_index_sel[:, neg_ids] if neg_ids.numel() else edge_index_sel[:, :0]
    
            if edge_w_sel.dim() == 1:
                w_pos = edge_w_sel[pos_ids] if pos_ids.numel() else edge_w_sel[:0]
                w_neg = (-edge_w_sel[neg_ids]) if neg_ids.numel() else edge_w_sel[:0]  # magnitud positiva
            else:
                w_pos = edge_w_sel[:, pos_ids] if pos_ids.numel() else edge_w_sel[:, :0]
                w_neg = (-edge_w_sel[:, neg_ids]) if neg_ids.numel() else edge_w_sel[:, :0]
    
            for blk in self.blocks:
                h = blk(h, ei_pos, w_pos, ei_neg, w_neg)
                hs.append(h)
        else:
            for blk in self.blocks:
                h = blk(h, edge_index_sel, edge_w_sel)
                hs.append(h)
    
        h_cat = torch.cat(hs, dim=-1)
        h = self.jk(h_cat)
    
        # --- pooling a nivel genes (SIN guardar attention si return_attn=False) ---
        pooled = []
        if return_attn:
            attn_all = []
            for pool_mlp in self.pool_mlps:
                score = pool_mlp(h).squeeze(-1)   # (B,N)
                attn = F.softmax(score, dim=1)    # (B,N)
                pooled.append(torch.einsum("bn,bnh->bh", attn, h))
                attn_all.append(attn)
            g_gene = torch.cat(pooled, dim=-1)
            attn_mean = torch.stack(attn_all, dim=0).mean(dim=0)  # (B,N)
        else:
            for pool_mlp in self.pool_mlps:
                score = pool_mlp(h).squeeze(-1)   # (B,N)
                attn = F.softmax(score, dim=1)    # (B,N) temporal
                pooled.append(torch.einsum("bn,bnh->bh", attn, h))
            g_gene = torch.cat(pooled, dim=-1)
            attn_mean = None
    
        # --- pooling jerárquico regulador (opcional) ---
        if self.use_regulator_pool:
            regs = self.regulator_groups
            B = h.shape[0]
            P = len(regs)
            h_reg = torch.zeros((B, P, self.hidden), device=h.device, dtype=h.dtype)
            for p, idxs in enumerate(regs):
                if idxs.numel() == 0:
                    continue
                h_reg[:, p, :] = h.index_select(1, idxs.to(h.device)).mean(dim=1)
    
            pooled_r = []
            for pool_mlp in self.pool_mlps:
                score_r = pool_mlp(h_reg).squeeze(-1)  # (B,P)
                attn_r = F.softmax(score_r, dim=1)
                pooled_r.append(torch.einsum("bp,bph->bh", attn_r, h_reg))
    
            g_reg = torch.cat(pooled_r, dim=-1)
            g = torch.cat([g_gene, g_reg], dim=-1)
        else:
            g = g_gene
    
        if x_graph is None:
            x_graph = torch.zeros((g.shape[0], 0), device=g.device, dtype=g.dtype)
    
        x_graph = self.xgraph_drop(x_graph)
        g2 = torch.cat([g, x_graph], dim=-1)
    
        z = self.trunk(g2)
        logits4 = self.head4(z)
        logits_ao = self.head_AO(z)
    
        return logits4, attn_mean, logits_ao


    def forward_reconstruct(self, x_gene: torch.Tensor, x_graph: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reconstrucción por nodo (para pretraining). Devuelve pred (B,N)."""
        edge_index_sel, edge_w_sel = self._select_edges(x_graph)

        if x_gene.dim() == 2:
            x_gene = x_gene.unsqueeze(-1)

        h = self.inp_lin1(x_gene)
        h = self.inp_bn1(h)
        h = F.relu(h)
        h = self.inp_drop(h)
        h = self.inp_lin2(h)
        h = self.inp_bn2(h)
        h0 = F.relu(h)

        hs = [h0]
        h = h0

        if self.signed_channels_mode == "dual_backbone":
            # (pretraining: usamos gating global)
            base_w = self.edge_weight_base
            pos_ids = torch.where(base_w >= 0)[0]
            neg_ids = torch.where(base_w < 0)[0]
            ei_pos = self.edge_index[:, pos_ids] if pos_ids.numel() else self.edge_index[:, :0]
            ei_neg = self.edge_index[:, neg_ids] if neg_ids.numel() else self.edge_index[:, :0]

            if edge_w_sel.dim() == 1:
                w_pos = edge_w_sel[pos_ids] if pos_ids.numel() else edge_w_sel[:0]
                w_neg = (-edge_w_sel[neg_ids]) if neg_ids.numel() else edge_w_sel[:0]
            else:
                w_pos = edge_w_sel[:, pos_ids] if pos_ids.numel() else edge_w_sel[:, :0]
                w_neg = (-edge_w_sel[:, neg_ids]) if neg_ids.numel() else edge_w_sel[:, :0]

            for blk in self.blocks:
                h = blk(h, ei_pos, w_pos, ei_neg, w_neg)
                hs.append(h)
        else:
            for blk in self.blocks:
                h = blk(h, edge_index_sel, edge_w_sel)
                hs.append(h)

        h_cat = torch.cat(hs, dim=-1)
        h = self.jk(h_cat)
        pred = self.recon_head(h).squeeze(-1)
        return pred


# ── HybridGNNTabular (Phase 2) ────────────────────




def _make_mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int, dropout: float) -> nn.Module:
    layers = []
    d = int(in_dim)
    n_layers = max(1, int(n_layers))
    for i in range(n_layers):
        h = int(hidden) if i < (n_layers - 1) else int(out_dim)
        layers.append(nn.Linear(d, h))
        if i < (n_layers - 1):
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(float(dropout)))
        d = h
    return nn.Sequential(*layers)


class HybridGNNTabular(nn.Module):
    """
    Phase 2 hybrid head:
      - rama GNN (grafo fijo/podado)
      - rama tabular MLP sobre x_gene compacto
      - fusión final (recomendado)
    Devuelve la misma API que el modelo original:
      (logits4, attn_mean, logits_ao)
    """

    def __init__(
        self,
        gnn: nn.Module,
        *,
        tab_in_dim: int,
        hidden: int,
        n_classes: int,
        tab_layers: int = 2,
        tab_dropout: float = 0.20,
        fusion_dropout: float = 0.20,
        use_fusion_head: bool = True,
        blend_logits: bool = False,
        blend_init: float = 0.70,
    ):
        super().__init__()
        self.gnn = gnn
        self.is_hybrid = True

        self.tab_in_dim = int(tab_in_dim)
        self.tab_hidden = int(hidden)
        self.n_classes = int(n_classes)
        self.tab_layers = int(tab_layers)
        self.tab_dropout = float(tab_dropout)
        self.fusion_dropout = float(fusion_dropout)
        self.use_fusion_head = bool(use_fusion_head)
        self.blend_logits = bool(blend_logits)
        self.blend_init = float(blend_init)

        self.tab_drop = nn.Dropout(self.tab_dropout)
        self.tab_trunk = _make_mlp(
            in_dim=self.tab_in_dim,
            hidden=self.tab_hidden,
            out_dim=self.tab_hidden,
            n_layers=self.tab_layers,
            dropout=self.tab_dropout,
        )

        if self.use_fusion_head:
            self.fusion = nn.Sequential(
                nn.Linear(2 * self.tab_hidden, self.tab_hidden),
                nn.LayerNorm(self.tab_hidden),
                nn.GELU(),
                nn.Dropout(self.fusion_dropout),
            )
            self.fusion_head4 = nn.Linear(self.tab_hidden, self.n_classes)
            self.fusion_headAO = nn.Linear(self.tab_hidden, 2)

        if self.blend_logits:
            p = min(max(float(blend_init), 1e-4), 1.0 - 1e-4)
            self.blend_logit = nn.Parameter(torch.tensor(math.log(p / (1.0 - p)), dtype=torch.float32))
            self.tab_head4 = nn.Linear(self.tab_hidden, self.n_classes)
            self.tab_headAO = nn.Linear(self.tab_hidden, 2)

    # -------- delegación mínima para compatibilidad --------
    @property
    def gate_tau(self):
        return getattr(self.gnn, "gate_tau", 1.0)

    @gate_tau.setter
    def gate_tau(self, v: float):
        if hasattr(self.gnn, "gate_tau"):
            self.gnn.gate_tau = float(v)

    @property
    def edge_logit(self):
        return getattr(self.gnn, "edge_logit")

    @property
    def edge_type_gating(self):
        return getattr(self.gnn, "edge_type_gating", True)

    @property
    def sample_cond_gating(self):
        return getattr(self.gnn, "sample_cond_gating", False)

    @property
    def sample_cond_mode(self):
        return getattr(self.gnn, "sample_cond_mode", "per_type")

    @property
    def signed_channels_mode(self):
        return getattr(self.gnn, "signed_channels_mode", "type_only")

    @property
    def use_regulator_pool(self):
        return getattr(self.gnn, "use_regulator_pool", False)

    def edge_l1_sum(self):
        return self.gnn.edge_l1_sum()

    def connectivity_penalty(self, *args, **kwargs):
        return self.gnn.connectivity_penalty(*args, **kwargs)

    def set_hard_keep_ratio(self, *args, **kwargs):
        return self.gnn.set_hard_keep_ratio(*args, **kwargs)

    def forward(self, x_gene: torch.Tensor, adj=None, x_graph: Optional[torch.Tensor] = None, return_attn: bool = False):
        z_holder = {}
        handle = None

        if self.use_fusion_head and hasattr(self.gnn, "trunk"):
            def _hook(_m, _inp, out):
                z_holder["z"] = out
            handle = self.gnn.trunk.register_forward_hook(_hook)

        logits4_gnn, attn_mean, logits_ao_gnn = self.gnn(x_gene, adj, x_graph, return_attn=return_attn)

        if handle is not None:
            handle.remove()

        z_tab = self.tab_trunk(self.tab_drop(x_gene))

        if self.use_fusion_head:
            z_gnn = z_holder.get("z", None)
            if z_gnn is None:
                z_gnn = torch.zeros_like(z_tab)
            z = self.fusion(torch.cat([z_gnn, z_tab], dim=-1))
            logits4 = self.fusion_head4(z)
            logits_ao = self.fusion_headAO(z)
            return logits4, attn_mean, logits_ao

        if self.blend_logits:
            tab_logits4 = self.tab_head4(z_tab)
            tab_logits_ao = self.tab_headAO(z_tab)
            alpha = torch.sigmoid(self.blend_logit)
            logits4 = alpha * logits4_gnn + (1.0 - alpha) * tab_logits4
            logits_ao = alpha * logits_ao_gnn + (1.0 - alpha) * tab_logits_ao
            return logits4, attn_mean, logits_ao

        return logits4_gnn, attn_mean, logits_ao_gnn


def build_pruned_model_from(
    model_gated,
    edge_index_p: torch.Tensor,
    edge_weight_p: torch.Tensor,
    edge_type_p: Optional[torch.Tensor] = None,
    reg_groups_t: Optional[List[torch.Tensor]] = None,
    *,
    num_nodes: int,
    hidden: int,
    n_classes: int,
    graph_feat_dim: int,
    num_layers: int,
    dropout: float,
    num_heads: int,
    xgraph_dropout: float = 0.0,
    block_use_self: bool = True,
    block_residual: bool = True,
    pool_level: str = "gene",
    reg_groups=None,
    reg_pool_min_targets: int = 3,
    device: Optional[torch.device] = None,
    # --- Hybrid knobs ---
    use_hybrid: bool = True,
    tab_layers: int = 2,
    tab_dropout: float = 0.20,
    fusion_dropout: float = 0.20,
    blend_logits: bool = False,
    blend_init: float = 0.70,
    **kwargs,
):
    """Crea el modelo de Phase 2 con grafo fijo; opcionalmente añade rama tabular híbrida."""
    assert device is not None, "device requerido"

    gnn_pruned = ImprovedSharedGraphGNN(
        num_nodes=int(num_nodes),
        hidden=int(hidden),
        n_classes=int(n_classes),
        graph_feat_dim=int(graph_feat_dim),
        num_layers=int(num_layers),
        dropout=float(dropout),
        num_heads=int(num_heads),
        edge_index=edge_index_p.to(device),
        edge_weight=(edge_weight_p.to(device) if isinstance(edge_weight_p, torch.Tensor) else None),
        edge_type=(edge_type_p.to(device) if isinstance(edge_type_p, torch.Tensor) else None),
        keep_self_loops=False,
        block_use_self=bool(block_use_self),
        block_residual=bool(block_residual),
        xgraph_dropout=float(xgraph_dropout),
        gat_heads=int(getattr(model_gated, "gat_heads", 4)),
        gat_use_edge_weight=bool(getattr(model_gated, "gat_use_edge_weight", True)),
        edge_type_gating=bool(getattr(model_gated, "edge_type_gating", True)),
        sample_cond_gating=bool(getattr(model_gated, "sample_cond_gating", False)),
        sample_cond_mode=str(getattr(model_gated, "sample_cond_mode", "per_type")),
        signed_channels_mode=str(getattr(model_gated, "signed_channels_mode", "type_only")),
        use_regulator_pool=bool(getattr(model_gated, "use_regulator_pool", False)),
        regulator_groups=(reg_groups_t if getattr(model_gated, "use_regulator_pool", False) else None),
        # kwargs extra se ignoran por compatibilidad de la clase
        edge_type_mode=str(getattr(model_gated, "edge_type_mode", "huri_op_signed")),
        signed_channels=bool(getattr(model_gated, "signed_channels", True)),
        add_connectivity_penalty=bool(getattr(model_gated, "add_connectivity_penalty", False)),
        connectivity_min_deg=float(getattr(model_gated, "connectivity_min_deg", CFG.CONNECTIVITY_MIN_DEG)),
        connectivity_use_abs=bool(getattr(model_gated, "connectivity_use_abs", CFG.CONNECTIVITY_USE_ABS)),
    ).to(device)

    def _move_attr_tensor(m, name: str, dev: torch.device):
        if hasattr(m, name):
            t = getattr(m, name)
            if isinstance(t, torch.Tensor) and t.device != dev:
                setattr(m, name, t.to(dev))

    for nm in ["type_scale", "type_bias", "edge_type", "edge_weight_norm_base"]:
        _move_attr_tensor(gnn_pruned, nm, device)

    old_sd = model_gated.state_dict()
    new_sd = gnn_pruned.state_dict()
    filtered = {k: v for k, v in old_sd.items() if (k in new_sd and new_sd[k].shape == v.shape)}
    gnn_pruned.load_state_dict(filtered, strict=False)

    # Grafo fijo real
    with torch.no_grad():
        gnn_pruned.edge_logit.data.fill_(20.0)
    gnn_pruned.edge_logit.requires_grad_(False)
    gnn_pruned.gate_tau = 1.0
    gnn_pruned.set_hard_keep_ratio(None)

    if not bool(use_hybrid):
        return gnn_pruned

    model_h = HybridGNNTabular(
        gnn=gnn_pruned,
        tab_in_dim=int(num_nodes),  # X_comp en Phase 2
        hidden=int(hidden),
        n_classes=int(n_classes),
        tab_layers=int(tab_layers),
        tab_dropout=float(tab_dropout),
        fusion_dropout=float(fusion_dropout),
        use_fusion_head=True,
        blend_logits=bool(blend_logits),
        blend_init=float(blend_init),
    ).to(device)
    return model_h
