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
    x_graph_ref: torch.Tensor = None,   # opcional: "score típico" si sample-cond gating
):
    """
    Exporta el subgrafo top-k usando LOGITS (pre-sigmoid) para ranking estable.

    IMPORTANTE (comparabilidad):
      - A diferencia del novelty antiguo, NO se añaden self-loops "gratis" fuera del presupuesto.
      - Si keep_self_loops=False (default), se EXCLUYEN totalmente del ranking y de la exportación.

    Devuelve:
      edge_index_p, edge_weight_p, edge_type_p, thr_logit, kept, total_candidatas
    """

    # -------- 1) score base (logit tipado) / tau --------
    if use_typed_logits and hasattr(model, "_typed_logit"):
        logit = model._typed_logit().detach()  # (E,)
    else:
        logit = model.edge_logit.detach()
        if use_typed_logits and getattr(model, "edge_type_gating", False) and hasattr(model, "edge_type") and (model.edge_type is not None):
            logit = logit * model.type_scale[model.edge_type] + model.type_bias[model.edge_type]

    # (Opcional) sample-conditioned: añade mean(delta) para exportar un subgrafo "promedio"
    if x_graph_ref is not None and getattr(model, "sample_cond_gating", False) and (getattr(model, "cond_mlp", None) is not None):
        delta = model.cond_mlp(x_graph_ref)  # (B,T) o (B,1)
        if getattr(model, "sample_cond_mode", "per_type") == "per_type":
            dt = delta.mean(dim=0)[model.edge_type]
            logit = logit + dt
        else:
            logit = logit + delta.mean().view(())

    score = logit
    if use_tau and hasattr(model, "gate_tau"):
        score = score / float(model.gate_tau)

    edge_index = model.edge_index
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
        edge_weight_p = model.edge_weight_base[:0].detach().cpu()
        edge_type_p = (model.edge_type[:0].detach().cpu()
                       if hasattr(model, "edge_type") and (model.edge_type is not None) else None)
        return edge_index_p, edge_weight_p, edge_type_p, float("nan"), 0, 0

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
    edge_weight_p = model.edge_weight_base[kept_ids].detach().cpu()
    edge_type_p = (model.edge_type[kept_ids].detach().cpu()
                   if hasattr(model, "edge_type") and (model.edge_type is not None) else None)

    return edge_index_p, edge_weight_p, edge_type_p, thr_logit, kept, m

    # -------- 3) TopK robusto en NO-loops (con pairing opcional) --------
    m = int(edge_ids.numel())

    paired = False
    if pair_undirected and (m % 2 == 0):
        half = m // 2
        a = edge_ids[:half]
        b = edge_ids[half:]
        ok = ((edge_index[0, a] == edge_index[1, b]) &
              (edge_index[1, a] == edge_index[0, b])).float().mean().item()
        paired = (ok > 0.99)

    if paired:
        half = m // 2
        a = edge_ids[:half]
        b = edge_ids[half:]
        pair_score = torch.maximum(score[a], score[b])  # (half,)

        k_pairs = max(1, int(round(float(keep_ratio) * half)))
        vals, sel = torch.topk(pair_score, k_pairs, largest=True, sorted=False)
        thr_logit = float(vals.min().item())

        kept_ids = torch.cat([a[sel], b[sel], loop_ids], dim=0)
    else:
        k = max(1, int(round(float(keep_ratio) * m)))
        vals, sel = torch.topk(score[edge_ids], k, largest=True, sorted=False)
        thr_logit = float(vals.min().item())

        kept_ids = torch.cat([edge_ids[sel], loop_ids], dim=0)

    kept = int(kept_ids.numel())

    # -------- 4) export --------
    edge_index_p = edge_index[:, kept_ids].detach().cpu()
    edge_weight_p = model.edge_weight_base[kept_ids].detach().cpu()

    edge_type_p = (model.edge_type[kept_ids].detach().cpu()
                   if hasattr(model, "edge_type") and (model.edge_type is not None) else None)

    return edge_index_p, edge_weight_p, edge_type_p, thr_logit, kept, E_total


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

