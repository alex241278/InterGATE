"""
Graph pruning: export_pruned_graph, evaluate_keep_ratios.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from .losses import compute_metrics_full


@torch.no_grad()
def evaluate_keep_ratios(model, ratios, adj, dl_va, device, label_map, xh_mode="zero"):
    from .training import predict_proba_xh_mode
    out = []
    for r in ratios:
        model.set_hard_keep_ratio(None if r == 1.0 else r)
        info = model.hard_prune_info()
        proba_va, y_va = predict_proba_xh_mode(model, adj, dl_va, device, xh_mode=xh_mode)
        m = compute_metrics_full(y_va, proba_va, label_map)
        print(f"[keep={r}] kept={info['kept']}/{info['total']} thr_logit={info.get('thr_logit', None)} thr_gate={info.get('thr_gate', None)} | auc_macro={m.get('auc_macro_ovr', float('nan')):.3f} acc={m.get('acc', float('nan')):.3f}")
        out.append((r, info, m))
    model.set_hard_keep_ratio(None)
    return out

@torch.no_grad()
def export_pruned_graph(
    model,
    keep_ratio: float,
    keep_self_loops: bool = False,
    pair_undirected: bool = True,
    use_typed_logits: bool = True,
    use_tau: bool = True,
    class_id: Optional[int] = None,
    return_edge_ids: bool = False,
):
    """
    Exporta el subgrafo top-k usando LOGITS (pre-sigmoid) para ranking estable.

    IMPORTANTE (comparabilidad):
      - A diferencia del novelty antiguo, NO se añaden self-loops "gratis" fuera del presupuesto.
      - Si keep_self_loops=False (default), se EXCLUYEN totalmente del ranking y de la exportación.

    Devuelve:
      edge_index_p, edge_weight_p, edge_type_p, thr_logit, kept, total_candidatas
    """

    core = getattr(model, "gnn", model)

    # Una máscara fija de fase 2 ya es el grafo exportable.
    if class_id is not None and hasattr(core, "fixed_class_edge_ids"):
        fixed = core.fixed_class_edge_ids(int(class_id))
        if fixed is not None:
            ids = fixed.to(core.edge_index.device)
            result = (
                core.edge_index[:, ids].detach().cpu(),
                core.edge_weight_base[ids].detach().cpu(),
                (
                    core.edge_type[ids].detach().cpu()
                    if getattr(core, "edge_type", None) is not None else None
                ),
                float("nan"),
                int(ids.numel()),
                int(core.edge_index.shape[1]),
            )
            return result + (ids.detach().cpu(),) if return_edge_ids else result

    # -------- 1) score base (logit tipado) / tau --------
    if (
        class_id is not None
        and bool(getattr(core, "per_label_graphs", False))
        and hasattr(core, "class_edge_logits")
    ):
        logit = core.class_edge_logits(int(class_id)).detach()
    elif use_typed_logits and hasattr(core, "_typed_logit"):
        logit = core._typed_logit().detach()  # (E,)
    else:
        logit = core.edge_logit.detach()
        if use_typed_logits and getattr(core, "edge_type_gating", False) and hasattr(core, "edge_type") and (core.edge_type is not None):
            logit = logit * core.type_scale[core.edge_type] + core.type_bias[core.edge_type]

    score = logit
    if use_tau and hasattr(core, "gate_tau"):
        score = score / float(core.gate_tau)

    edge_index = core.edge_index
    E_total = int(score.numel())

    # -------- 2) máscara de candidatos (self-loops opcionalmente excluidos) --------
    loop_mask = (edge_index[0] == edge_index[1])
    if keep_self_loops:
        cand_mask = torch.ones(E_total, dtype=torch.bool, device=edge_index.device)
    else:
        cand_mask = ~loop_mask

    cand_ids = torch.where(cand_mask)[0]
    m = int(cand_ids.numel())
    if m == 0:
        # nada que exportar
        edge_index_p = edge_index[:, :0].detach().cpu()
        edge_weight_p = core.edge_weight_base[:0].detach().cpu()
        edge_type_p = (core.edge_type[:0].detach().cpu()
                       if hasattr(core, "edge_type") and (core.edge_type is not None) else None)
        result = (
            edge_index_p, edge_weight_p, edge_type_p,
            float("nan"), 0, 0,
        )
        empty_ids = cand_ids[:0].detach().cpu()
        return result + (empty_ids,) if return_edge_ids else result

    # -------- 3) TopK robusto (con pairing opcional) --------
    paired = False
    if pair_undirected and (m % 2 == 0):
        half = m // 2
        a = cand_ids[:half]
        b = cand_ids[half:]
        ok = ((edge_index[0, a] == edge_index[1, b]) &
              (edge_index[1, a] == edge_index[0, b])).float().mean().item()
        paired = (ok > 0.99)

    if paired:
        half = m // 2
        a = cand_ids[:half]
        b = cand_ids[half:]
        pair_score = torch.maximum(score[a], score[b])  # (half,)

        k_pairs = max(1, int(round(float(keep_ratio) * half)))
        vals, sel = torch.topk(pair_score, k_pairs, largest=True, sorted=False)
        thr_logit = float(vals.min().item())

        kept_ids = torch.cat([a[sel], b[sel]], dim=0)
    else:
        k = max(1, int(round(float(keep_ratio) * m)))
        vals, sel = torch.topk(score[cand_ids], k, largest=True, sorted=False)
        thr_logit = float(vals.min().item())

        kept_ids = cand_ids[sel]

    kept = int(kept_ids.numel())

    # -------- 4) export --------
    edge_index_p = edge_index[:, kept_ids].detach().cpu()
    edge_weight_p = core.edge_weight_base[kept_ids].detach().cpu()
    edge_type_p = (core.edge_type[kept_ids].detach().cpu()
                   if hasattr(core, "edge_type") and (core.edge_type is not None) else None)

    result = (
        edge_index_p, edge_weight_p, edge_type_p,
        thr_logit, kept, m,
    )
    return (
        result + (kept_ids.detach().cpu(),)
        if return_edge_ids else result
    )


@torch.no_grad()
def export_pruned_graphs_by_class(
    model,
    keep_ratio: float,
    class_names=None,
    **kwargs,
):
    """Exporta un TopK independiente por experto de clase."""
    core = getattr(model, "gnn", model)
    n_classes = int(getattr(core, "n_classes", 1))
    per_label = bool(getattr(core, "per_label_graphs", False))
    if class_names is None:
        class_names = list(getattr(
            core, "class_names", [f"class_{i}" for i in range(n_classes)]
        ))
    # En modo compartido se repite la misma exportación bajo cada nombre para
    # conservar el contrato de salida; en modo por clase cada llamada usa su
    # delta independiente.
    ids = range(n_classes)
    out = {}
    for class_id in ids:
        ei, ew, et, thr, kept, total, edge_ids = export_pruned_graph(
            model,
            keep_ratio=keep_ratio,
            class_id=(class_id if per_label else None),
            return_edge_ids=True,
            **kwargs,
        )
        ids_device = edge_ids.to(core.edge_logit.device)
        if per_label and hasattr(core, "class_edge_logits"):
            logits = core.class_edge_logits(int(class_id)).detach()[
                ids_device
            ]
        elif hasattr(core, "_typed_logit"):
            logits = core._typed_logit().detach()[ids_device]
        else:
            logits = core.edge_logit.detach()[ids_device]
        gates = torch.sigmoid(logits / float(getattr(core, "gate_tau", 1.0)))
        out[int(class_id)] = {
            "class_id": int(class_id),
            "class_name": str(class_names[class_id]),
            "edge_index": ei,
            "edge_weight": ew,
            "edge_type": et,
            "candidate_edge_ids": edge_ids,
            "learned_logit": logits.detach().cpu(),
            "learned_gate": gates.detach().cpu(),
            "selection_frequency": torch.ones(
                int(kept), dtype=torch.float32
            ),
            "thr_logit": float(thr),
            "kept": int(kept),
            "total": int(total),
        }
    return out


@torch.no_grad()
def hard_mask_topk_from_logits(edge_logit: torch.Tensor, keep_ratio: float):
    # edge_logit: (E,)
    E = edge_logit.numel()
    k = max(1, int(round(float(keep_ratio) * E)))
    # topk sobre logits (robusto, sin saturación)
    vals, _ = torch.topk(edge_logit, k, largest=True, sorted=False)
    thr_logit = vals.min()  # umbral en logit
    mask = edge_logit >= thr_logit
    kept = int(mask.sum().item())
    return mask, float(thr_logit.item()), kept, int(E)
@torch.no_grad()
def hard_prune_info_logits(model, keep_ratio):
    mask, thr_logit, kept, total = hard_mask_topk_from_logits(model.edge_logit, keep_ratio)
    # Si quieres también un thr "en sigmoid" sólo para reporting:
    thr_gate = float(torch.sigmoid(torch.tensor(thr_logit)).item())
    return {"keep_ratio": keep_ratio, "thr_logit": thr_logit, "thr_gate": thr_gate, "kept": kept, "total": total}
