"""
Stability selection: edge-set utilities, Jaccard metrics, stability_edge_sets.
"""

import os
import json
import gc
import random
from collections import Counter
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .config import CFG
from .utils import set_all_seeds, cleanup_memory


# ── Edge-set utilities & Jaccard ─────────────────


def edge_set_from_edge_index(edge_index_t: torch.Tensor) -> set:
    ei = edge_index_t.detach().cpu()
    return set(zip(ei[0].tolist(), ei[1].tolist()))


def jaccard(a: set, b: set) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    u = len(a | b)
    return float(len(a & b) / u) if u > 0 else 0.0


def mean_pairwise_jaccard(edge_sets: List[set]) -> float:
    if len(edge_sets) < 2:
        return float("nan")
    vals = []
    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            vals.append(jaccard(edge_sets[i], edge_sets[j]))
    return float(np.mean(vals)) if len(vals) > 0 else float("nan")


def pairwise_jaccard_stats(edge_sets: List[set]) -> Dict[str, float]:
    """Resumen de estabilidad (Jaccard) entre todos los pares de corridas."""
    n = len(edge_sets)
    if n < 2:
        return {
            "n_runs": float(n), "n_pairs": 0.0,
            "jaccard_mean": float("nan"), "jaccard_std": float("nan"),
            "jaccard_min": float("nan"), "jaccard_max": float("nan"),
        }
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(jaccard(edge_sets[i], edge_sets[j]))
    vals = np.asarray(vals, dtype=float)
    return {
        "n_runs": float(n),
        "n_pairs": float(vals.size),
        "jaccard_mean": float(np.mean(vals)) if vals.size else float("nan"),
        "jaccard_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "jaccard_min": float(np.min(vals)) if vals.size else float("nan"),
        "jaccard_max": float(np.max(vals)) if vals.size else float("nan"),
    }


def build_jaccard_df(edge_sets: List[set], labels: List[str], consensus: Optional[set] = None) -> pd.DataFrame:
    """Matriz Jaccard (labels x labels) + columna opcional to_consensus."""
    n = len(edge_sets)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            M[i, j] = 1.0 if i == j else jaccard(edge_sets[i], edge_sets[j])
    df = pd.DataFrame(M, index=labels, columns=labels)
    if consensus is not None:
        df["to_consensus"] = [jaccard(es, consensus) for es in edge_sets]
    return df


# ── Métricas ─────────────────────────────────────


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_pred = proba.argmax(axis=1)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    try:
        out["auc_ovr_macro"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_ovr_macro"] = float("nan")
    return out


# ── Stability edge sets ────────────────────────────


def stability_edge_sets(
    cfg: "AblationConfig",
    seeds: List[int],
    keep_ratio: float,
    epochs: int,
    class_weights_t: Optional[torch.Tensor],
    verbose: bool = False,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    # Lazy imports to avoid circular dependencies
    from .ablation import (
        get_full_gene_matrix_and_genes, build_model_from_cfg, make_loaders,
        make_prior_for_cfg, cfg_to_dict, _mkdir, _rt,
        SAVE_STAB_RUN_GRAPHS, STAB_RETURN_DETAILS,
    )
    from .training import train_graph_learning
    from .pruning import export_pruned_graph

    X_full, genes_full = get_full_gene_matrix_and_genes()

    edge_sets: List[set] = []
    rows: List[dict] = []

    want_run_graphs = bool(SAVE_STAB_RUN_GRAPHS and (STAB_RETURN_DETAILS or (save_dir is not None)))
    runs_graphs = [] if want_run_graphs else None

    # class_weights list
    cw_list = None
    if class_weights_t is not None:
        cw_list = class_weights_t.detach().cpu().tolist()

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_t) if class_weights_t is not None else torch.nn.CrossEntropyLoss()

    for i, seed in enumerate(seeds):
        set_all_seeds(seed)
        model = build_model_from_cfg(cfg, class_weights_t)

        dl_tr, dl_va, _ = make_loaders(X_full)
        model, best_state, best_macro_f1 = train_graph_learning(
            model, adj=None, dl_tr=dl_tr, dl_va=dl_va,
            device=CFG.DEVICE, label_map=_rt("label_map"),
            epochs=epochs, lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY,
            accum_steps=CFG.ACCUM_STEPS,
            class_weights=cw_list, class_weights_t=class_weights_t,
            edge_l1_per_edge=CFG.EDGE_L1_PER_EDGE,
            aux_lambda=CFG.AUX_LAMBDA, loss_fn=loss_fn,
            keep_min=keep_ratio,
        )
        model.load_state_dict(best_state)

        ei_p, ew_p, et_p, thr, kept, E_total = export_pruned_graph(model, keep_ratio=keep_ratio, keep_self_loops=False)
        es = edge_set_from_edge_index(ei_p)
        edge_sets.append(es)

        rows.append({
            "run": int(i), "seed": int(seed),
            "val_macro_f1": float(best_macro_f1),
            "thr_logit": float(thr), "n_edges": int(len(es)),
        })

        if want_run_graphs and runs_graphs is not None:
            runs_graphs.append({
                "run": int(i), "seed": int(seed),
                "edge_index_full": ei_p.detach().cpu(),
                "edge_weight_full": (ew_p.detach().cpu() if isinstance(ew_p, torch.Tensor) else None),
                "edge_type_full": (et_p.detach().cpu() if isinstance(et_p, torch.Tensor) else None),
                "thr_logit": float(thr), "kept": int(kept), "total": int(E_total),
            })

        if verbose:
            print(f"  run {i} seed {seed} edges={len(es)} valF1={best_macro_f1:.4f}")

        try:
            del model, best_state, ei_p, ew_p, et_p
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    # --- Jaccard ---
    jstats = pairwise_jaccard_stats(edge_sets)

    # Consenso por frecuencia >= STAB_FREQ_THR
    cnt = Counter()
    for es in edge_sets:
        cnt.update(es)
    runs = len(edge_sets)
    cons = {e for e, c in cnt.items() if (c / runs) >= CFG.STAB_FREQ_THR}

    # Stats vs consenso
    if len(cons) > 0:
        to_cons = np.array([jaccard(es, cons) for es in edge_sets], dtype=np.float32)
        to_cons_stats = {
            "to_consensus_mean": float(np.mean(to_cons)),
            "to_consensus_std": float(np.std(to_cons, ddof=1)) if to_cons.size > 1 else 0.0,
            "to_consensus_min": float(np.min(to_cons)),
            "to_consensus_max": float(np.max(to_cons)),
        }
    else:
        to_cons = None
        to_cons_stats = {
            "to_consensus_mean": float("nan"), "to_consensus_std": float("nan"),
            "to_consensus_min": float("nan"), "to_consensus_max": float("nan"),
        }

    out: Dict[str, Any] = {
        "jaccard_stats": jstats,
        "to_consensus_stats": to_cons_stats,
        "jaccard_mean": float(jstats["jaccard_mean"]),
        "jaccard_std": float(jstats["jaccard_std"]),
        "jaccard_min": float(jstats["jaccard_min"]),
        "jaccard_max": float(jstats["jaccard_max"]),
        "jaccard_n_pairs": int(jstats["n_pairs"]),
        "consensus_edges": cons,
        "consensus_size": int(len(cons)),
    }

    # Detalles opcionales
    need_tables = bool(STAB_RETURN_DETAILS or (save_dir is not None))
    if need_tables:
        runs_df = pd.DataFrame(rows)
        if to_cons is not None:
            runs_df["to_consensus"] = to_cons
        labels = [f"seed_{s}" for s in seeds]
        jacc_df = build_jaccard_df(edge_sets, labels=labels, consensus=(cons if len(cons) > 0 else None))
        out["runs_df"] = runs_df
        out["jaccard_df"] = jacc_df

    if STAB_RETURN_DETAILS:
        out["edge_sets"] = edge_sets
        if want_run_graphs and runs_graphs is not None:
            out["runs_graphs"] = runs_graphs

    # Guardado opcional
    if save_dir is not None:
        _mkdir(save_dir)
        try:
            out["runs_df"].to_csv(os.path.join(save_dir, "stability_runs.csv"), index=False)
            out["jaccard_df"].to_csv(os.path.join(save_dir, "stability_jaccard.csv"))
        except Exception:
            pass

        payload = {
            "cfg": cfg_to_dict(cfg),
            "seeds": list(seeds),
            "keep_ratio": float(keep_ratio),
            "epochs": int(epochs),
            "freq_thr": float(CFG.STAB_FREQ_THR),
            "consensus_edges": [(int(s), int(d)) for (s, d) in cons],
        }
        if want_run_graphs and runs_graphs is not None:
            payload["runs_graphs"] = runs_graphs
        torch.save(payload, os.path.join(save_dir, "stability_graphs.pt"))

        with open(os.path.join(save_dir, "stability_summary.json"), "w", encoding="utf-8") as f:
            json.dump({
                "jaccard_stats": jstats,
                "to_consensus_stats": to_cons_stats,
                "consensus_size": int(len(cons)),
            }, f, indent=2, ensure_ascii=False)

    return out
