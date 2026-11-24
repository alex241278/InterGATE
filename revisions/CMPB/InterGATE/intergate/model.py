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


@torch.no_grad()
def _topk_edge_ids(
    edge_index: torch.Tensor,
    score: torch.Tensor,
    ratio: float,
    self_loop_mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Selecciona TopK respetando pares bidireccionales cuando están ordenados.

    ``self_loop_mask`` solo marca autoaristas protegidas. Con
    ``KEEP_SELF_LOOPS=False`` es todo False y las autoaristas compiten dentro
    del mismo presupuesto que cualquier otra arista.
    """
    loop_ids = torch.where(self_loop_mask)[0]
    edge_ids = torch.where(~self_loop_mask)[0]
    if edge_ids.numel() == 0:
        return loop_ids, float("nan")

    m = int(edge_ids.numel())
    paired = False
    if m % 2 == 0:
        half = m // 2
        a = edge_ids[:half]
        b = edge_ids[half:]
        ok = (
            (edge_index[0, a] == edge_index[1, b])
            & (edge_index[1, a] == edge_index[0, b])
        ).float().mean().item()
        paired = ok > 0.99

    if paired:
        half = m // 2
        a = edge_ids[:half]
        b = edge_ids[half:]
        pair_score = torch.maximum(score[a], score[b])
        k_pairs = max(1, int(round(float(ratio) * half)))
        vals, sel = torch.topk(pair_score, k_pairs, largest=True, sorted=False)
        kept_ids = torch.cat([a[sel], b[sel], loop_ids], dim=0)
    else:
        k = max(1, int(round(float(ratio) * m)))
        vals, sel = torch.topk(score[edge_ids], k, largest=True, sorted=False)
        kept_ids = torch.cat([edge_ids[sel], loop_ids], dim=0)
    return kept_ids, float(vals.min().item())


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
    - covariables X_graph solo en el readout de Phase 2
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
        use_xgraph_in_head: bool = True,
        per_label_graphs: bool = False,
        class_names: Optional[List[str]] = None,
        signed_channels_mode: str = "type_only",     # "type_only" | "dual_backbone"
        use_regulator_pool: bool = False,
        regulator_groups: Optional[List[torch.Tensor]] = None,

        # Regularización de conectividad. Estos argumentos deben ser parte
        # explícita del modelo: si se absorben en **unused_kwargs, una
        # configuración ABL_no_conn queda registrada pero no cambia la pérdida.
        add_connectivity_penalty: bool = True,
        connectivity_min_deg: float = 0.05,
        connectivity_use_abs: bool = True,

        # V8-1. Debe ser explícito: si se absorbe en **unused_kwargs queda
        # registrado en el manifiesto y no hace nada.
        use_node_identity: bool = False,

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
        self.use_xgraph_in_head = bool(use_xgraph_in_head)
        self.add_connectivity_penalty = bool(add_connectivity_penalty)
        self.connectivity_min_deg = float(connectivity_min_deg)
        self.connectivity_use_abs = bool(connectivity_use_abs)

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

        # ── Grafo por etiqueta ────────────────────────────────────────────
        # Desplazamiento aprendido del logit de arista por clase: (n_classes, E).
        # Con per_label_graphs=False el parametro NO se crea (queda None) y
        # todos los caminos hacen short-circuit, asi que el comportamiento es
        # identico al de antes de esta feature (verificado en el smoke test).
        self.per_label_graphs = bool(per_label_graphs)
        self.class_names = (
            [str(x) for x in class_names]
            if class_names is not None
            else [f"class_{i}" for i in range(int(n_classes))]
        )
        if len(self.class_names) != int(n_classes):
            raise ValueError(
                f"class_names debe tener {n_classes} elementos, no "
                f"{len(self.class_names)}"
            )
        if self.per_label_graphs:
            self.edge_logit_delta = nn.Parameter(
                torch.zeros(int(n_classes), int(self.edge_weight_base.numel()))
            )
        else:
            self.register_parameter("edge_logit_delta", None)
        # Evidencia transcriptómica diferencial V8. Se calcula solo con TRAIN
        # de desarrollo y se adjunta después de construir el modelo. Un tensor
        # vacío mantiene compatibilidad con artefactos que no usan V8.
        self.register_buffer(
            "biology_edge_evidence", torch.empty(0, dtype=torch.float32),
            persistent=True,
        )
        self.gate_tau = 1.0

        # Edge-type gating (escala y sesgo por tipo)
        self.edge_type_gating = bool(edge_type_gating)
        if self.edge_type_gating:
            self.type_scale = nn.Parameter(torch.ones(self.num_edge_types))
            self.type_bias  = nn.Parameter(torch.zeros(self.num_edge_types))
        else:
            self.register_buffer("type_scale", torch.ones(self.num_edge_types))
            self.register_buffer("type_bias", torch.zeros(self.num_edge_types))

        # Hard pruning (top-ratio)
        self.hard_keep_ratio = None
        self.hard_edge_ids_cpu = None
        self.hard_thr_logit = None
        self.hard_thr_gate = None
        self.hard_thr = None
        self.hard_edge_ids_by_class_cpu = None
        self.hard_thr_logit_by_class = None
        self.hard_thr_gate_by_class = None
        self._fixed_class_edge_count = 0
        # Phase 2 contract. When enabled, the selected topology uses its fixed
        # signed, degree-normalized prior weights directly: structural gates
        # are neither evaluated nor multiplied into message-passing weights.
        self.fixed_graph_mode = False
        self.fixed_graph_bypass_gates = False
        self.last_class_embeddings = None
        self.last_class_attentions = None

        # Input MLP por nodo
        self.inp_lin1 = nn.Linear(1, hidden)
        self.inp_bn1  = NodeBatchNorm(hidden)
        self.inp_lin2 = nn.Linear(hidden, hidden)
        self.inp_bn2  = NodeBatchNorm(hidden)
        self.inp_drop = nn.Dropout(dropout)

        # V8-1: identidad de gen. El encoder heredado de V7 no tiene ningún
        # parámetro indexado por nodo (inp_lin1 y NodeBatchNorm son compartidos), así
        # que dos genes con la misma expresion y vecindario isomorfo son
        # indistinguibles. Inicializado a cero: en el paso 0 el modelo es
        # numéricamente idéntico al punto de partida de V7.
        self.use_node_identity = bool(use_node_identity)
        if self.use_node_identity:
            self.node_emb = nn.Embedding(self.num_nodes, hidden)
            nn.init.zeros_(self.node_emb.weight)
        else:
            self.node_emb = None

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


    def shared_edge_logits(self) -> torch.Tensor:
        """Puntuación del backbone común, incluida la calibración por tipo."""
        return self._typed_logit()

    def differential_edge_logits(self) -> torch.Tensor:
        """Desviaciones de subtipo identificables y centradas entre clases."""
        if not self.per_label_graphs or self.edge_logit_delta is None:
            return torch.zeros(
                (1, self.edge_logit.numel()),
                dtype=self.edge_logit.dtype,
                device=self.edge_logit.device,
            )
        delta = self.edge_logit_delta
        if bool(getattr(CFG, "CENTER_CLASS_DELTAS", False)):
            delta = delta - delta.mean(dim=0, keepdim=True)
        return delta

    def set_biology_evidence(self, evidence: torch.Tensor) -> None:
        """Adjunta evidencia clase×arista calculada sin validación ni test."""
        evidence = torch.as_tensor(
            evidence, dtype=self.edge_logit.dtype, device=self.edge_logit.device
        )
        expected = (self.n_classes, int(self.edge_logit.numel()))
        if tuple(evidence.shape) != expected:
            raise ValueError(
                f"biology_edge_evidence debe tener forma {expected}, no "
                f"{tuple(evidence.shape)}."
            )
        if not torch.isfinite(evidence).all():
            raise ValueError("biology_edge_evidence contiene valores no finitos.")
        # La evidencia se interpreta como contraste de clase; el centrado hace
        # que no pueda absorberse de forma arbitraria en el backbone común.
        evidence = evidence - evidence.mean(dim=0, keepdim=True)
        self.biology_edge_evidence = evidence.detach()

    def biology_alignment_penalty(self) -> torch.Tensor:
        """Distancia coseno entre deltas aprendidos y evidencia reproducible."""
        evidence = self.biology_edge_evidence
        if evidence.numel() == 0 or not self.per_label_graphs:
            return torch.zeros((), device=self.edge_logit.device)
        delta = self.differential_edge_logits()
        if tuple(delta.shape) != tuple(evidence.shape):
            raise RuntimeError("Deltas y evidencia biológica no están alineados.")
        penalties = []
        for class_id in range(self.n_classes):
            mask = evidence[class_id].ne(0)
            if int(mask.sum()) < 2:
                continue
            similarity = F.cosine_similarity(
                delta[class_id, mask].reshape(1, -1),
                evidence[class_id, mask].reshape(1, -1),
                dim=1,
                eps=1e-8,
            )[0]
            penalties.append(0.5 * (1.0 - similarity))
        if not penalties:
            return torch.zeros((), device=self.edge_logit.device)
        return torch.stack(penalties).mean()

    def all_class_edge_logits(self) -> torch.Tensor:
        """Scores de todos los expertos, sin ids de clase por muestra.

        Devuelve ``(n_classes, E)`` en modo por clase y ``(1, E)`` en modo
        compartido. Esta API no permite seleccionar rutas a partir del target.
        """
        base = self.shared_edge_logits()
        if not self.per_label_graphs or self.edge_logit_delta is None:
            return base.view(1, -1)
        return base.view(1, -1) + self.differential_edge_logits()

    def class_edge_logits(self, class_id: int) -> torch.Tensor:
        """Score estructural de una clase sin usar etiquetas de las muestras."""
        class_id = int(class_id)
        if not 0 <= class_id < self.n_classes:
            raise IndexError(
                f"class_id={class_id} fuera de [0, {self.n_classes})"
            )
        base = self.shared_edge_logits()
        if not self.per_label_graphs or self.edge_logit_delta is None:
            return base
        return base + self.differential_edge_logits()[class_id]

    def set_fixed_class_edge_ids(
        self,
        edge_ids_by_class: List[torch.Tensor],
    ) -> None:
        """Fija una máscara distinta por clase para la fase de fine-tuning.

        Los IDs indexan el ``edge_index`` unión almacenado en este modelo. Se
        registran como buffers para que viajen en ``state_dict`` y se restauren
        junto con el bundle.
        """
        if len(edge_ids_by_class) != self.n_classes:
            raise ValueError(
                f"Se esperaban {self.n_classes} máscaras, no "
                f"{len(edge_ids_by_class)}"
            )
        for c, ids in enumerate(edge_ids_by_class):
            ids_t = torch.as_tensor(
                ids, dtype=torch.long, device=self.edge_index.device
            ).reshape(-1)
            if ids_t.numel() == 0:
                raise ValueError(
                    f"El grafo fijo de {self.class_names[c]} está vacío."
                )
            if int(ids_t.min()) < 0 or int(ids_t.max()) >= self.edge_index.shape[1]:
                raise IndexError(
                    f"IDs fuera del grafo unión para {self.class_names[c]}"
                )
            name = f"fixed_class_edge_ids_{c}"
            if hasattr(self, name):
                setattr(self, name, ids_t)
            else:
                self.register_buffer(name, ids_t)
            raw_weight = self.edge_weight_base.index_select(0, ids_t)
            edge_index_c = self.edge_index.index_select(1, ids_t)
            out_degree = torch.zeros(
                self.num_nodes,
                dtype=raw_weight.dtype,
                device=raw_weight.device,
            )
            in_degree = torch.zeros_like(out_degree)
            out_degree.index_add_(0, edge_index_c[0], raw_weight.abs())
            in_degree.index_add_(0, edge_index_c[1], raw_weight.abs())
            out_inv_sqrt = out_degree.clamp_min(1e-12).pow(-0.5)
            in_inv_sqrt = in_degree.clamp_min(1e-12).pow(-0.5)
            norm_weight = (
                raw_weight
                * out_inv_sqrt[edge_index_c[0]]
                * in_inv_sqrt[edge_index_c[1]]
            )
            weight_name = f"fixed_class_edge_weight_norm_{c}"
            if hasattr(self, weight_name):
                setattr(self, weight_name, norm_weight)
            else:
                self.register_buffer(weight_name, norm_weight)
        self._fixed_class_edge_count = len(edge_ids_by_class)

    def fixed_class_edge_ids(self, class_id: int) -> Optional[torch.Tensor]:
        if int(class_id) >= int(self._fixed_class_edge_count):
            return None
        return getattr(self, f"fixed_class_edge_ids_{int(class_id)}", None)

    def fixed_class_edge_weight_norm(
        self, class_id: int
    ) -> Optional[torch.Tensor]:
        if int(class_id) >= int(self._fixed_class_edge_count):
            return None
        return getattr(
            self, f"fixed_class_edge_weight_norm_{int(class_id)}", None
        )

    def configure_fixed_graph_phase2(
        self,
        *,
        bypass_gates: bool = True,
        freeze_structure: bool = True,
    ) -> tuple[str, ...]:
        """Activate the auditable fixed-graph contract used in Phase 2.

        The class masks and signed prior weights are already materialized.
        With ``bypass_gates=True``, message passing uses those normalized
        weights directly and does not evaluate a sigmoid gate.  Structural
        parameters remain in the state dict for provenance but cannot affect
        the forward pass.  They are frozen as an additional fail-safe.
        """
        self.fixed_graph_mode = True
        self.fixed_graph_bypass_gates = bool(bypass_gates)
        frozen = []
        if freeze_structure:
            for name in (
                "edge_logit", "edge_logit_delta", "type_scale", "type_bias"
            ):
                value = getattr(self, name, None)
                if isinstance(value, nn.Parameter):
                    value.requires_grad_(False)
                    frozen.append(name)
        return tuple(frozen)

    def assert_fixed_graph_phase2(self) -> None:
        """Fail if a Phase 2 model can still alter or use structural gates."""
        if not self.fixed_graph_mode:
            raise RuntimeError("Phase 2 requiere fixed_graph_mode=True.")
        if not self.fixed_graph_bypass_gates:
            raise RuntimeError("Phase 2 debe omitir las compuertas estructurales.")
        trainable = [
            name for name in (
                "edge_logit", "edge_logit_delta", "type_scale", "type_bias"
            )
            if isinstance(getattr(self, name, None), nn.Parameter)
            and bool(getattr(self, name).requires_grad)
        ]
        if trainable:
            raise RuntimeError(
                "Parámetros estructurales entrenables en Phase 2: "
                f"{trainable}"
            )
        if self.per_label_graphs and self._fixed_class_edge_count != self.n_classes:
            raise RuntimeError(
                "Phase 2 requiere una máscara fija para cada clase: "
                f"{self._fixed_class_edge_count}/{self.n_classes}."
            )

    def _selected_ids_for_class(self, class_id: Optional[int]) -> Optional[torch.Tensor]:
        """Devuelve máscara fija o TopK activo para una ruta de clase."""
        if class_id is not None and self.per_label_graphs:
            fixed = self.fixed_class_edge_ids(int(class_id))
            if fixed is not None:
                return fixed
            if self.hard_edge_ids_by_class_cpu is not None:
                return self.hard_edge_ids_by_class_cpu[int(class_id)]
        if self.hard_keep_ratio is not None and self.hard_edge_ids_cpu is not None:
            return self.hard_edge_ids_cpu
        return None

    def per_label_delta_l1(self) -> torch.Tensor:
        """L1 media del delta por clase (0 si la feature esta apagada)."""
        if not self.per_label_graphs or self.edge_logit_delta is None:
            return torch.zeros((), device=self.edge_logit.device)
        return self.differential_edge_logits().abs().mean()

    def per_label_diversity(self) -> torch.Tensor:
        """Similitud coseno media entre los grafos de clases distintas.

        Sumada a la perdida, empuja a que cada clase seleccione aristas
        distintas. 0 si la feature esta apagada o hay menos de 2 clases.
        """
        if not self.per_label_graphs or self.edge_logit_delta is None:
            return torch.zeros((), device=self.edge_logit.device)
        d = self.differential_edge_logits()
        if d.shape[0] < 2:
            return torch.zeros((), device=d.device)
        dn = F.normalize(d, dim=1, eps=1e-8)
        sim = dn @ dn.t()
        off = ~torch.eye(d.shape[0], dtype=torch.bool, device=d.device)
        return sim[off].abs().mean()

    def edge_gate(
        self,
        x_graph: Optional[torch.Tensor] = None,
        class_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Gate global por experto en (0,1), siempre con forma (E,).
        """
        # device "fuente de verdad" del modelo
        dev = self.edge_logit.device
    
        # Cada experto usa base compartida + delta propio. class_id=None queda
        # reservado para preentrenamiento/compatibilidad con el grafo común.
        if class_id is not None and self.per_label_graphs:
            base = self.class_edge_logits(int(class_id))
        else:
            base = self._typed_logit()
        base = base / float(self.gate_tau)  # (E,)
    
        return torch.sigmoid(base)


    def edge_l1_sum(self) -> torch.Tensor:
        if self.per_label_graphs:
            gates = [
                self.edge_gate(None, class_id=c)
                for c in range(self.n_classes)
            ]
            # Mantiene la escala de la regularización del modelo compartido.
            return torch.stack(gates, dim=0).mean(dim=0).sum()
        return self.edge_gate(None).sum()

    def _mean_gate(self) -> torch.Tensor:
        if self.per_label_graphs:
            gates = [self.edge_gate(None, class_id=c) for c in range(self.n_classes)]
            return torch.stack(gates, dim=0).mean()
        return self.edge_gate(None).mean()

    def sparsity_penalty(self, target: float,
                         entropy_beta: float = 0.01) -> torch.Tensor:
        """Density-target regularizer, averaged over class-specific graphs.

        For each class ``c`` this implements

        ``(mean(g_c) - rho_t)^2 + beta * mean(H(g_c))``.

        The density and entropy are therefore computed separately for every
        label graph before averaging. This avoids a dense graph compensating
        for a sparse one and applies the sharpness term to all classes.
        """
        class_ids = range(self.n_classes) if self.per_label_graphs else (None,)
        terms = []
        for class_id in class_ids:
            g = self.edge_gate(None, class_id=class_id)
            density = (g.mean() - float(target)) ** 2
            if float(entropy_beta) > 0.0:
                gg = g.clamp(1e-6, 1.0 - 1e-6)
                entropy = -(
                    gg * gg.log() + (1.0 - gg) * (1.0 - gg).log()
                ).mean()
                density = density + float(entropy_beta) * entropy
            terms.append(density)
        return torch.stack(terms).mean()

    def connectivity_penalty(self, min_deg: float = 0.05, use_abs: bool = True) -> torch.Tensor:
        """
        Penaliza nodos con grado esperado bajo usando *soft gates* (diferenciable).
        Grado esperado = suma de pesos (|w| opcional) de aristas incidentes (src y dst).
        Se calcula con los gates globales por experto para ser estable.
        """
        expert_ids = range(self.n_classes) if self.per_label_graphs else [None]
        penalties = []
        for class_id in expert_ids:
            g = self.edge_gate(None, class_id=class_id)
            w = self.edge_weight_norm_base
            if use_abs:
                w = w.abs()
            w = w * g
            deg = torch.zeros(
                self.num_nodes, device=w.device, dtype=w.dtype
            )
            deg.index_add_(0, self.edge_index[0], w)
            deg.index_add_(0, self.edge_index[1], w)
            penalties.append(F.relu(float(min_deg) - deg).mean())
        return torch.stack(penalties).mean()

    @torch.no_grad()
    def set_hard_keep_ratio(self, ratio: Optional[float]):
        """
        Hard-pruning TopK. En modo predictivo por clase crea una máscara
        independiente para cada experto usando ``base + delta_clase``.
        """
        self.hard_keep_ratio = ratio

        if ratio is None:
            self.hard_edge_ids_cpu = None
            self.hard_thr_logit = None
            self.hard_thr_gate = None
            self.hard_thr = None
            self.hard_edge_ids_by_class_cpu = None
            self.hard_thr_logit_by_class = None
            self.hard_thr_gate_by_class = None
            return

        if not 0 < float(ratio) <= 1:
            raise ValueError(f"keep ratio debe estar en (0,1], no {ratio}")

        if self.per_label_graphs:
            ids_all, thr_all, gate_all = [], [], []
            for c in range(self.n_classes):
                score = (
                    self.class_edge_logits(c).detach()
                    / float(self.gate_tau)
                )
                ids, thr = _topk_edge_ids(
                    self.edge_index, score, float(ratio),
                    self.self_loop_mask,
                )
                ids_all.append(ids.detach().cpu())
                thr_all.append(float(thr))
                gate_all.append(
                    float(torch.sigmoid(torch.tensor(thr)).item())
                    if math.isfinite(float(thr)) else float("nan")
                )
            self.hard_edge_ids_by_class_cpu = ids_all
            self.hard_thr_logit_by_class = thr_all
            self.hard_thr_gate_by_class = gate_all
            # Campos de compatibilidad: primer experto. Las métricas completas
            # están en hard_prune_info()["per_class"].
            self.hard_edge_ids_cpu = ids_all[0]
            self.hard_thr_logit = thr_all[0]
            self.hard_thr_gate = gate_all[0]
            self.hard_thr = gate_all[0]
            return

        score = self._typed_logit().detach() / float(self.gate_tau)
        kept_ids, thr_logit = _topk_edge_ids(
            self.edge_index, score, float(ratio), self.self_loop_mask
        )
        thr_gate = float(torch.sigmoid(torch.tensor(thr_logit)).item())
        self.hard_edge_ids_cpu = kept_ids.detach().cpu()
        self.hard_thr_logit = thr_logit
        self.hard_thr_gate  = thr_gate
        self.hard_thr = thr_gate

    def hard_prune_info(self):
        total = int(self.edge_weight_base.numel())
        if self.per_label_graphs and self._fixed_class_edge_count:
            per_class = []
            for c in range(self.n_classes):
                ids = self.fixed_class_edge_ids(c)
                per_class.append({
                    "class_id": c,
                    "class_name": self.class_names[c],
                    "kept": int(ids.numel()),
                    "total": total,
                    "fixed": True,
                })
            return {
                "keep_ratio": None,
                "kept": sum(x["kept"] for x in per_class),
                "kept_per_class": [x["kept"] for x in per_class],
                "total": total,
                "per_class": per_class,
                "fixed": True,
            }

        if (
            self.per_label_graphs
            and self.hard_keep_ratio is not None
            and self.hard_edge_ids_by_class_cpu is not None
        ):
            per_class = []
            for c, ids in enumerate(self.hard_edge_ids_by_class_cpu):
                per_class.append({
                    "class_id": c,
                    "class_name": self.class_names[c],
                    "keep_ratio": float(self.hard_keep_ratio),
                    "thr_logit": float(self.hard_thr_logit_by_class[c]),
                    "thr_gate": float(self.hard_thr_gate_by_class[c]),
                    "kept": int(ids.numel()),
                    "total": total,
                })
            return {
                "keep_ratio": float(self.hard_keep_ratio),
                "kept": sum(x["kept"] for x in per_class),
                "kept_per_class": [x["kept"] for x in per_class],
                "total": total,
                "per_class": per_class,
            }

        if self.hard_keep_ratio is None or self.hard_edge_ids_cpu is None:
            return {"keep_ratio": None, "thr_logit": None, "thr_gate": None, "thr": None, "kept": total, "total": total}

        kept = int(self.hard_edge_ids_cpu.numel())
        thr_logit = float(self.hard_thr_logit) if self.hard_thr_logit is not None else float("nan")
        thr_gate  = float(self.hard_thr_gate)  if self.hard_thr_gate  is not None else float("nan")
        return {"keep_ratio": float(self.hard_keep_ratio), "thr_logit": thr_logit, "thr_gate": thr_gate, "thr": thr_gate, "kept": kept, "total": total}

    # ---------- edges ----------
    def _select_edges(
        self,
        x_graph: Optional[torch.Tensor] = None,
        class_id: Optional[int] = None,
    ):
        """Return selected edges and weights under Phase 1 or fixed Phase 2."""
        fixed_ids = (
            self.fixed_class_edge_ids(int(class_id))
            if class_id is not None and self.per_label_graphs
            else None
        )
        if fixed_ids is not None:
            ids = fixed_ids.to(device=self.edge_index.device, non_blocking=True)
            idx = self.edge_index[:, ids]
            norm = self.fixed_class_edge_weight_norm(int(class_id)).to(
                device=self.edge_weight_base.device, non_blocking=True
            )
            if self.fixed_graph_mode and self.fixed_graph_bypass_gates:
                # Exact Phase 2 definition: fixed class topology, signed prior
                # weight and class-specific directed degree normalization.
                # No learned gate magnitude is transferred from Phase 1.
                return idx, norm
            g = self.edge_gate(x_graph, class_id=class_id)
            if g.dim() == 1:
                w = norm * g[ids]
            else:
                w = norm.view(1, -1) * g[:, ids]
            return idx, w

        if self.fixed_graph_mode and self.fixed_graph_bypass_gates:
            # Shared-graph compatibility path. The multiclass protocol uses
            # the class-specific branch above.
            return self.edge_index, self.edge_weight_norm_base

        g = self.edge_gate(x_graph, class_id=class_id)  # (E,) o (B,E)
        w_all = self.edge_weight_norm_base * g  # preserva signo (broadcast OK)
        ids = self._selected_ids_for_class(class_id)
        if ids is not None:
            ids = ids.to(device=w_all.device, non_blocking=True)
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
            raise RuntimeError("build_adj requiere pesos de arista 1D.")
        N = self.num_nodes
        return torch.sparse_coo_tensor(idx, w, size=(N, N), device=w.device, dtype=torch.float32).coalesce()

    def _forward_single_graph(
        self,
        x_gene: torch.Tensor,
        x_graph: Optional[torch.Tensor],
        class_id: Optional[int],
        return_attn: bool,
    ):
        """Recorrido GNN de un experto; no recibe la etiqueta de la muestra."""
        # --- robustez CPU/GPU: mueve inputs al device real del modelo ---
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
        edge_index_sel, edge_w_sel = self._select_edges(
            x_graph, class_id=class_id
        )

        # La compactación física usa la unión de todos los grafos de clase,
        # pero cada experto solo puede observar los nodos incidentes a SUS
        # aristas. Sin esta máscara, un
        # nodo exclusivo del otro grafo todavía entraría por h0/JK/pooling
        # aunque no tuviera ninguna arista activa en esta ruta.
        active_nodes = torch.ones(
            self.num_nodes, dtype=torch.bool, device=edge_index_sel.device
        )
        if self.per_label_graphs and class_id is not None:
            active_nodes.zero_()
            if edge_index_sel.numel():
                active_nodes[torch.unique(edge_index_sel.reshape(-1))] = True
        if not bool(active_nodes.any()):
            raise RuntimeError(
                f"El experto {class_id} no contiene ningún nodo activo."
            )
        active_h = active_nodes.view(1, self.num_nodes, 1)

        # V8-P3. El loader de fase 2 entrega [genes_del_grafo | universo
        # completo]. Este es el unico embudo por el que pasa cualquier entrada
        # al GNN, incluida la fase 2A graph_only, que llama a forward()
        # directamente sin pasar por HybridGNNTabular. Se queda con el bloque
        # del grafo; el resto lo recoge la rama tabular del hibrido.
        if x_gene.shape[1] > self.num_nodes:
            x_gene = x_gene[:, : self.num_nodes]
        if x_gene.dim() == 2:
            x_gene = x_gene.unsqueeze(-1)
        x_gene = x_gene * active_h.to(dtype=x_gene.dtype)
    
        # Input MLP por nodo
        h = self.inp_lin1(x_gene)
        h = self.inp_bn1(h)
        h = F.relu(h)
        h = self.inp_drop(h)
        h = self.inp_lin2(h)
        h = self.inp_bn2(h)
        h0 = F.relu(h) * active_h.to(dtype=h.dtype)
        if self.use_node_identity:
            h0 = h0 + self.node_emb.weight.unsqueeze(0) * active_h.to(
                dtype=h0.dtype
            )

        hs = [h0]
        h = h0
    
        if self.signed_channels_mode == "dual_backbone":
            # divide edges por signo (según edge_weight_base) consistente con hard pruning
            ids = self._selected_ids_for_class(class_id)
            if ids is not None:
                ids = ids.to(
                    device=self.edge_weight_base.device, non_blocking=True
                )
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
                h = h * active_h.to(dtype=h.dtype)
                hs.append(h)
        else:
            for blk in self.blocks:
                h = blk(h, edge_index_sel, edge_w_sel)
                h = h * active_h.to(dtype=h.dtype)
                hs.append(h)
    
        h_cat = torch.cat(hs, dim=-1)
        h = self.jk(h_cat) * active_h.to(dtype=h_cat.dtype)
    
        # --- pooling a nivel genes (SIN guardar attention si return_attn=False) ---
        pooled = []
        if return_attn:
            attn_all = []
            for pool_mlp in self.pool_mlps:
                score = pool_mlp(h).squeeze(-1)   # (B,N)
                score = score.masked_fill(
                    ~active_nodes.view(1, -1),
                    torch.finfo(score.dtype).min,
                )
                attn = F.softmax(score, dim=1)    # (B,N)
                pooled.append(torch.einsum("bn,bnh->bh", attn, h))
                attn_all.append(attn)
            g_gene = torch.cat(pooled, dim=-1)
            attn_mean = torch.stack(attn_all, dim=0).mean(dim=0)  # (B,N)
        else:
            for pool_mlp in self.pool_mlps:
                score = pool_mlp(h).squeeze(-1)   # (B,N)
                score = score.masked_fill(
                    ~active_nodes.view(1, -1),
                    torch.finfo(score.dtype).min,
                )
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
    
        # X_h no modifica pesos. Solo puede entrar en el readout de Phase 2.
        if x_graph is None:
            x_graph_head = torch.zeros(
                (g.shape[0], self.graph_feat_dim),
                device=g.device,
                dtype=g.dtype,
            )
        elif self.use_xgraph_in_head:
            x_graph_head = x_graph
        else:
            x_graph_head = torch.zeros_like(x_graph)

        x_graph_head = self.xgraph_drop(x_graph_head)
        g2 = torch.cat([g, x_graph_head], dim=-1)
    
        z = self.trunk(g2)
        logits4 = self.head4(z)
        logits_ao = self.head_AO(z)

        return logits4, attn_mean, logits_ao, z

    def forward(
        self,
        x_gene: torch.Tensor,
        adj=None,
        x_graph: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """Predice evaluando todos los grafos de clase para cada muestra.

        En modo ``per_label_graphs`` el experto ``c`` produce únicamente el
        logit de la clase ``c``. Todos los expertos se ejecutan para todas las
        muestras; ``y`` no forma parte de esta API y no puede filtrar la ruta.
        """
        if not self.per_label_graphs:
            logits, attn, logits_ao, z = self._forward_single_graph(
                x_gene, x_graph, class_id=None, return_attn=return_attn
            )
            self.last_class_embeddings = z.unsqueeze(1)
            self.last_class_attentions = (
                attn.unsqueeze(1) if attn is not None else None
            )
            return logits, attn, logits_ao

        class_logits = []
        class_logits_full = []
        class_embeddings = []
        class_attn = []
        aux_logits = []
        for class_id in range(self.n_classes):
            logits_all, attn, logits_ao, z = self._forward_single_graph(
                x_gene, x_graph, class_id=class_id,
                return_attn=return_attn,
            )
            # V8-2: el vector completo de cada experto, no solo la diagonal.
            class_logits_full.append(logits_all)
            class_logits.append(logits_all[:, class_id])
            class_embeddings.append(z)
            aux_logits.append(logits_ao)
            if attn is not None:
                class_attn.append(attn)

        logits = torch.stack(class_logits, dim=1)
        # (B, n_expertos, n_clases). Lo consume train_one_epoch si
        # expert_lambda > 0; en inferencia es inerte.
        self.last_expert_logits = torch.stack(class_logits_full, dim=1)
        self.last_class_embeddings = torch.stack(class_embeddings, dim=1)
        self.last_class_attentions = (
            torch.stack(class_attn, dim=1) if class_attn else None
        )
        attn_mean = (
            self.last_class_attentions.mean(dim=1)
            if self.last_class_attentions is not None else None
        )
        logits_ao = torch.stack(aux_logits, dim=1).mean(dim=1)
        return logits, attn_mean, logits_ao


    def forward_reconstruct(self, x_gene: torch.Tensor, x_graph: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reconstrucción por nodo (para pretraining). Devuelve pred (B,N)."""
        edge_index_sel, edge_w_sel = self._select_edges(x_graph)

        # V8-P3: mismo truncado que en _forward_single_graph.
        if x_gene.shape[1] > self.num_nodes:
            x_gene = x_gene[:, : self.num_nodes]
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
        tab_hidden: Optional[int] = None,
        branch_dropout_tab: float = 0.0,
        branch_dropout_gnn: float = 0.0,
    ):
        super().__init__()
        self.gnn = gnn
        self.is_hybrid = True

        self.tab_in_dim = int(tab_in_dim)
        # V8: anchura de la rama tabular. Si es None o 0 cae en `hidden` y el
        # modelo es identico al de V7, incluidas las formas del state_dict.
        self.gnn_hidden = int(hidden)
        self.tab_hidden = int(tab_hidden) if tab_hidden else int(hidden)
        self.n_classes = int(n_classes)
        self.tab_layers = int(tab_layers)
        self.tab_dropout = float(tab_dropout)
        self.fusion_dropout = float(fusion_dropout)
        self.use_fusion_head = bool(use_fusion_head)
        self.blend_logits = bool(blend_logits)
        self.blend_init = float(blend_init)

        # V8. No reescalan por 1/(1-p): no son un regularizador sobre
        # activaciones sino la simulacion de una rama ausente, asi que la
        # fusion debe aprender a funcionar con el cero literal.
        self.branch_dropout_tab = float(branch_dropout_tab)
        self.branch_dropout_gnn = float(branch_dropout_gnn)

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
                # z_gnn mide gnn_hidden y z_tab mide tab_hidden. Cuando son
                # iguales esto es exactamente 2 * tab_hidden.
                nn.Linear(self.gnn_hidden + self.tab_hidden, self.tab_hidden),
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
    def per_label_graphs(self):
        return getattr(self.gnn, "per_label_graphs", False)

    @property
    def class_names(self):
        return getattr(self.gnn, "class_names", None)

    @property
    def signed_channels_mode(self):
        return getattr(self.gnn, "signed_channels_mode", "type_only")

    @property
    def use_regulator_pool(self):
        return getattr(self.gnn, "use_regulator_pool", False)

    def edge_l1_sum(self):
        return self.gnn.edge_l1_sum()

    def sparsity_penalty(self, *args, **kwargs):
        return self.gnn.sparsity_penalty(*args, **kwargs)

    def connectivity_penalty(self, *args, **kwargs):
        return self.gnn.connectivity_penalty(*args, **kwargs)

    @property
    def add_connectivity_penalty(self):
        return bool(getattr(self.gnn, "add_connectivity_penalty", False))

    @property
    def connectivity_min_deg(self):
        return float(getattr(
            self.gnn, "connectivity_min_deg", CFG.CONNECTIVITY_MIN_DEG
        ))

    @property
    def connectivity_use_abs(self):
        return bool(getattr(
            self.gnn, "connectivity_use_abs", CFG.CONNECTIVITY_USE_ABS
        ))

    def set_hard_keep_ratio(self, *args, **kwargs):
        return self.gnn.set_hard_keep_ratio(*args, **kwargs)

    def hard_prune_info(self):
        return self.gnn.hard_prune_info()

    def forward(self, x_gene: torch.Tensor, adj=None, x_graph: Optional[torch.Tensor] = None, return_attn: bool = False):
        # V8-P3: si x_gene trae mas columnas que nodos tiene el grafo, las
        # primeras num_nodes son la entrada del GNN y el resto es el universo
        # genico completo para la rama tabular.
        n_graph = int(self.gnn.num_nodes)
        if x_gene.shape[-1] > n_graph:
            x_tab = x_gene[:, n_graph:]
            x_gene = x_gene[:, :n_graph]
        else:
            x_tab = x_gene

        z_holder = {}
        handle = None

        if (
            self.use_fusion_head
            and hasattr(self.gnn, "trunk")
            and not self.per_label_graphs
        ):
            def _hook(_m, _inp, out):
                z_holder["z"] = out
            handle = self.gnn.trunk.register_forward_hook(_hook)

        logits4_gnn, attn_mean, logits_ao_gnn = self.gnn(
            x_gene, adj, x_graph, return_attn=return_attn
        )

        if handle is not None:
            handle.remove()

        z_tab = self.tab_trunk(self.tab_drop(x_tab))

        # V8: los logits propios del GNN, que hasta ahora se descartaban.
        # Los consume train_one_epoch para darle gradiente al grafo sin
        # pasar por la fusion.
        self.last_gnn_logits = logits4_gnn

        # V8: dropout de rama por muestra. Solo en entrenamiento.
        if self.training and self.branch_dropout_tab > 0.0:
            keep = (torch.rand(z_tab.shape[0], 1, device=z_tab.device)
                    >= self.branch_dropout_tab).to(z_tab.dtype)
            z_tab = z_tab * keep

        if self.use_fusion_head:
            if self.per_label_graphs:
                z_classes = getattr(
                    self.gnn, "last_class_embeddings", None
                )
                if z_classes is None:
                    raise RuntimeError(
                        "El GNN por clase no devolvió embeddings de expertos."
                    )
                if self.training and self.branch_dropout_gnn > 0.0:
                    keep_g = (
                        torch.rand(z_classes.shape[0], 1, 1,
                                   device=z_classes.device)
                        >= self.branch_dropout_gnn
                    ).to(z_classes.dtype)
                    z_classes = z_classes * keep_g

                logits_by_class = []
                aux_by_class = []
                expert_logits = []
                for class_id in range(self.n_classes):
                    z = self.fusion(torch.cat([
                        z_classes[:, class_id, :], z_tab
                    ], dim=-1))
                    logits_all = self.fusion_head4(z)
                    # V8-2: se calculan n_clases^2 logits; V7 usaba solo la
                    # diagonal. La matriz completa alimenta la supervision
                    # profunda de expertos.
                    expert_logits.append(logits_all)
                    logits_by_class.append(logits_all[:, class_id])
                    aux_by_class.append(self.fusion_headAO(z))
                self.last_expert_logits = torch.stack(expert_logits, dim=1)
                logits4 = torch.stack(logits_by_class, dim=1)
                logits_ao = torch.stack(aux_by_class, dim=1).mean(dim=1)
                return logits4, attn_mean, logits_ao

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
    class_edge_ids: Optional[List[torch.Tensor]] = None,
    class_names: Optional[List[str]] = None,
    per_label_graphs: Optional[bool] = None,
    # --- Hybrid knobs ---
    use_hybrid: bool = True,
    tab_layers: int = 2,
    tab_dropout: float = 0.20,
    fusion_dropout: float = 0.20,
    blend_logits: bool = False,
    blend_init: float = 0.70,
    # --- Extensiones V8 ---
    use_node_identity: bool = False,
    nodes_used_full=None,
    tab_in_dim: Optional[int] = None,
    tab_hidden_dim: Optional[int] = None,
    branch_dropout_tab: float = 0.0,
    branch_dropout_gnn: float = 0.0,
    **kwargs,
):
    """Crea el modelo de Phase 2 con grafo fijo; opcionalmente añade rama tabular híbrida."""
    assert device is not None, "device requerido"

    use_per_label = (
        bool(per_label_graphs)
        if per_label_graphs is not None
        else bool(getattr(model_gated, "per_label_graphs", False))
    )
    if class_names is None:
        class_names = list(getattr(
            model_gated, "class_names",
            [f"class_{i}" for i in range(int(n_classes))]
        ))

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
        use_xgraph_in_head=(
            str(getattr(CFG, "CLASS_GRAPH_XH_MODE", "zero")).lower()
            == "orig"
        ),
        per_label_graphs=use_per_label,
        class_names=class_names,
        use_node_identity=bool(use_node_identity),
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

    # V8-1. El filtro por forma descarta node_emb en silencio: fase 1 tiene
    # num_nodes completos y fase 2 solo |nodes_used|. Hay que reindexar.
    if (
        getattr(model_gated, "node_emb", None) is not None
        and getattr(gnn_pruned, "node_emb", None) is not None
    ):
        if nodes_used_full is None:
            raise ValueError(
                "use_node_identity exige nodes_used_full para compactar el "
                "embedding de gen entre fase 1 y fase 2."
            )
        node_idx = torch.as_tensor(
            np.asarray(nodes_used_full, dtype=np.int64), dtype=torch.long
        )
        with torch.no_grad():
            gnn_pruned.node_emb.weight.copy_(
                model_gated.node_emb.weight.detach().cpu()[node_idx].to(
                    gnn_pruned.node_emb.weight.device
                )
            )

    # Grafo fijo real. Los logits se conservan en el artefacto solo por
    # compatibilidad/provenance; la ruta fija no los usa en el forward.
    with torch.no_grad():
        gnn_pruned.edge_logit.data.fill_(20.0)
        if gnn_pruned.edge_logit_delta is not None:
            gnn_pruned.edge_logit_delta.zero_()
    gnn_pruned.gate_tau = 1.0
    gnn_pruned.set_hard_keep_ratio(None)
    if use_per_label:
        if class_edge_ids is None:
            raise ValueError(
                "La fase 2 por clase requiere class_edge_ids sobre el grafo unión."
            )
        gnn_pruned.set_fixed_class_edge_ids(class_edge_ids)
    gnn_pruned.configure_fixed_graph_phase2(
        bypass_gates=bool(CFG.PHASE2_FIXED_GRAPH_BYPASS_GATES),
        freeze_structure=bool(CFG.PHASE2_FREEZE_STRUCTURAL_PARAMETERS),
    )
    gnn_pruned.assert_fixed_graph_phase2()

    if not bool(use_hybrid):
        return gnn_pruned

    model_h = HybridGNNTabular(
        gnn=gnn_pruned,
        # V8-3: dimensión del universo completo cuando esa rama está activa.
        tab_in_dim=int(num_nodes if tab_in_dim is None else tab_in_dim),
        hidden=int(hidden),
        n_classes=int(n_classes),
        tab_layers=int(tab_layers),
        tab_dropout=float(tab_dropout),
        fusion_dropout=float(fusion_dropout),
        use_fusion_head=True,
        blend_logits=bool(blend_logits),
        blend_init=float(blend_init),
        tab_hidden=tab_hidden_dim,
        branch_dropout_tab=float(branch_dropout_tab),
        branch_dropout_gnn=float(branch_dropout_gnn),
    ).to(device)
    return model_h
