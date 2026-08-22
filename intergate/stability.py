"""Multiseed selection reproducibility for edges and genes.

The development matrix and split remain fixed. Consequently, this module
quantifies stochastic-algorithm reproducibility and does not implement the
sample-resampling procedure or false-selection guarantees of formal stability
selection.
"""

import os
import json
import gc
import random
from collections import Counter
from typing import Optional, Dict, List, Any, Hashable, Iterable, Mapping

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


def nogueira_stability(
    selected_sets: List[set],
    universe: Iterable[Hashable],
) -> float:
    """Estabilidad de selección corregida por azar de Nogueira et al.

    ``universe`` debe ser el universo completo de características elegibles,
    no solo la unión de las características observadas. La función acepta
    conjuntos de tamaño fijo o variable. Con menos de dos corridas, o cuando
    todas/ninguna de las características son seleccionadas, la medida no está
    definida y devuelve ``nan``.
    """
    if len(selected_sets) < 2:
        return float("nan")
    universe = set(universe)
    d = int(len(universe))
    if d == 0:
        return float("nan")
    outside = set().union(*selected_sets) - universe
    if outside:
        raise ValueError(
            f"Hay {len(outside)} características seleccionadas fuera del universo."
        )

    n_runs = int(len(selected_sets))
    counts = Counter(item for selected in selected_sets for item in selected)
    probabilities = np.fromiter(
        (counts.get(item, 0) / n_runs for item in universe),
        dtype=float,
        count=d,
    )
    sample_variances = (
        (n_runs / (n_runs - 1.0)) * probabilities * (1.0 - probabilities)
    )
    mean_selected = float(np.mean([len(selected) for selected in selected_sets]))
    expected_variance = (mean_selected / d) * (1.0 - mean_selected / d)
    if expected_variance <= 0.0:
        return float("nan")
    return float(1.0 - float(np.mean(sample_variances)) / expected_variance)


def _median_observed(
    item: Hashable,
    value_maps: Optional[List[Mapping[Hashable, float]]],
    *,
    default: float,
) -> float:
    if not value_maps:
        return float(default)
    values = [float(values[item]) for values in value_maps if item in values]
    return float(np.median(values)) if values else float(default)


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


def _edge_rank_score_maps(graph: Mapping[str, Any]) -> tuple[dict, dict]:
    """Devuelve rango 1..K y logit de cada arista exportada."""
    edge_index = torch.as_tensor(graph["edge_index"]).detach().cpu()
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    raw_scores = graph.get("learned_logit", graph.get("learned_gate", None))
    if raw_scores is None:
        scores = np.full(len(edges), np.nan, dtype=float)
    else:
        scores = torch.as_tensor(raw_scores).detach().cpu().numpy().astype(float)
    if len(edges) != len(scores):
        raise ValueError("edge_index y learned_logit/gate no están alineados.")
    if len(set(edges)) != len(edges):
        raise ValueError("Una corrida TopK contiene aristas dirigidas duplicadas.")

    order = sorted(
        range(len(edges)),
        key=lambda i: (
            -scores[i] if np.isfinite(scores[i]) else float("inf"),
            edges[i][0], edges[i][1],
        ),
    )
    ranks = {edges[i]: float(rank) for rank, i in enumerate(order, start=1)}
    score_map = {edges[i]: float(scores[i]) for i in range(len(edges))}
    return ranks, score_map


def _gene_sets_and_counts(edge_sets: List[set]) -> tuple[List[set], List[Counter], List[dict]]:
    """Genes presentes, aristas incidentes y vecinos únicos por corrida."""
    gene_sets: List[set] = []
    incident_counts: List[Counter] = []
    neighbour_counts: List[dict] = []
    for edges in edge_sets:
        incident = Counter()
        neighbours: Dict[int, set] = {}
        for src, dst in edges:
            incident[src] += 1
            incident[dst] += 1
            neighbours.setdefault(src, set()).add(dst)
            neighbours.setdefault(dst, set()).add(src)
        genes = set(incident)
        gene_sets.append(genes)
        incident_counts.append(incident)
        neighbour_counts.append({gene: len(values) for gene, values in neighbours.items()})
    return gene_sets, incident_counts, neighbour_counts


def frequency_ranked_consensus(
    edge_sets: List[set],
    frequency_threshold: float,
    preserve_topk_budget: bool = True,
    explicit_target: Optional[int] = None,
    rank_maps: Optional[List[Mapping[tuple, float]]] = None,
    score_maps: Optional[List[Mapping[tuple, float]]] = None,
) -> Dict[str, Any]:
    """Construye un consenso sin cambiar inadvertidamente el K de TopK.

    Con ``preserve_topk_budget=True``, un ``explicit_target`` entero tiene
    prioridad. Si es ``None``, K se infiere del número de aristas realmente
    exportado por cada corrida, que a su vez procede de kappa. Todas las
    corridas deben tener el mismo K; una discrepancia indica que ya no se está
    comparando el mismo protocolo de selección.
    """
    if not edge_sets:
        raise ValueError("Se necesita al menos un conjunto de aristas.")
    if not 0.0 <= float(frequency_threshold) <= 1.0:
        raise ValueError("frequency_threshold debe estar en [0, 1].")
    if rank_maps is not None and len(rank_maps) != len(edge_sets):
        raise ValueError("rank_maps y edge_sets deben tener la misma longitud.")
    if score_maps is not None and len(score_maps) != len(edge_sets):
        raise ValueError("score_maps y edge_sets deben tener la misma longitud.")

    counter = Counter()
    for edge_set in edge_sets:
        counter.update(edge_set)
    n_runs = len(edge_sets)
    run_edge_sizes = [int(len(edge_set)) for edge_set in edge_sets]
    threshold_consensus = {
        edge for edge, count in counter.items()
        if (count / n_runs) >= float(frequency_threshold)
    }

    target_edges: Optional[int] = None
    target_source = "frequency_threshold"
    if preserve_topk_budget:
        if explicit_target is not None:
            target_edges = int(explicit_target)
            if target_edges <= 0:
                raise ValueError("explicit_target debe ser positivo o None.")
            if any(size != target_edges for size in run_edge_sizes):
                raise RuntimeError(
                    f"El override K={target_edges} no coincide con las "
                    f"corridas exportadas {run_edge_sizes}. Si se preservan "
                    "pares bidireccionales, usa un K compatible."
                )
            target_source = "explicit_override"
        else:
            unique_sizes = sorted(set(run_edge_sizes))
            if len(unique_sizes) != 1:
                raise RuntimeError(
                    "Las corridas de estabilidad no tienen el mismo K: "
                    f"{run_edge_sizes}. Revisa backbone, kappa y poda."
                )
            target_edges = unique_sizes[0]
            target_source = "kappa_topk"

        # Frecuencia es el criterio primario. Entre aristas con la misma
        # frecuencia, el rango mediano dentro de los TopK de las semillas en
        # las que apareció conserva más información que un desempate léxico.
        # La puntuación mediana y los endpoints completan un orden totalmente
        # determinista.
        median_rank = {
            edge: _median_observed(
                edge, rank_maps, default=float("inf")
            )
            for edge in counter
        }
        median_score = {
            edge: _median_observed(
                edge, score_maps, default=float("-inf")
            )
            for edge in counter
        }
        ranked = sorted(
            counter,
            key=lambda edge: (
                -counter[edge], median_rank[edge], -median_score[edge],
                edge[0], edge[1],
            ),
        )
        if len(ranked) < target_edges:
            raise RuntimeError(
                f"Solo hay {len(ranked)} aristas observadas, menos que "
                f"el presupuesto K={target_edges}."
            )
        consensus = set(ranked[:target_edges])
    else:
        consensus = threshold_consensus

    median_rank = {
        edge: _median_observed(edge, rank_maps, default=float("nan"))
        for edge in counter
    }
    median_score = {
        edge: _median_observed(edge, score_maps, default=float("nan"))
        for edge in counter
    }

    return {
        "consensus_edges": consensus,
        "threshold_consensus": threshold_consensus,
        "frequency_counter": counter,
        "run_edge_sizes": run_edge_sizes,
        "target_edges": target_edges,
        "target_source": target_source,
        "median_rank": median_rank,
        "median_score": median_score,
    }


# ── Métricas ─────────────────────────────────────


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_pred = proba.argmax(axis=1)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    try:
        if proba.ndim == 2 and proba.shape[1] == 2:
            out["auc_ovr_macro"] = float(
                roc_auc_score(y_true, proba[:, 1])
            )
        else:
            out["auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true, proba, multi_class="ovr", average="macro"
                )
            )
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
    """Estabilidad independiente de cada grafo predictivo de clase.

    Cada semilla entrena el clasificador completo y exporta un TopK para cada
    experto. Los consensos de las clases se construyen siempre por separado.
    """
    if not seeds:
        raise ValueError("stability_edge_sets necesita al menos una seed.")
    expected_runs = int(CFG.STAB_RUNS)
    if expected_runs != 8:
        raise ValueError(
            "El protocolo histórico está preespecificado para ocho corridas "
            "internas por cada semilla externa; "
            f"CFG.STAB_RUNS={expected_runs}."
        )
    if len(seeds) != expected_runs or len(set(map(int, seeds))) != expected_runs:
        raise ValueError(
            "Cada consenso externo requiere exactamente ocho semillas "
            "internas "
            f"distintas; recibidas {list(seeds)}."
        )

    # Lazy imports to avoid circular dependencies
    from .ablation import (
        get_full_gene_matrix_and_genes, build_model_from_cfg, make_loaders,
        cfg_to_dict, _mkdir, _rt, _RT,
        SAVE_STAB_RUN_GRAPHS, STAB_RETURN_DETAILS,
    )
    from .training import train_graph_learning
    from .pruning import export_pruned_graphs_by_class

    X_full, genes_full = get_full_gene_matrix_and_genes()
    label_map = _rt("label_map")
    n_classes = int(_rt("n_classes"))
    class_names = [None] * n_classes
    for name, class_id in label_map.items():
        class_names[int(class_id)] = str(name)
    if any(name is None for name in class_names):
        raise ValueError(f"label_map no cubre ids 0..{n_classes - 1}: {label_map}")

    edge_sets_by_class: Dict[int, List[set]] = {
        class_id: [] for class_id in range(n_classes)
    }
    edge_rank_maps_by_class: Dict[int, List[dict]] = {
        class_id: [] for class_id in range(n_classes)
    }
    edge_score_maps_by_class: Dict[int, List[dict]] = {
        class_id: [] for class_id in range(n_classes)
    }
    rows: List[dict] = []
    candidate_edge_universe: Optional[set] = None
    candidate_gene_universe: Optional[set] = None

    want_run_graphs = bool(SAVE_STAB_RUN_GRAPHS and (STAB_RETURN_DETAILS or (save_dir is not None)))
    runs_graphs = [] if want_run_graphs else None
    sample_weights_full = _RT.get("sample_weights", None)

    # class_weights list
    cw_list = None
    if class_weights_t is not None and sample_weights_full is None:
        cw_list = class_weights_t.detach().cpu().tolist()

    loss_fn = (
        torch.nn.CrossEntropyLoss(weight=class_weights_t)
        if class_weights_t is not None and sample_weights_full is None
        else torch.nn.CrossEntropyLoss()
    )

    for i, seed in enumerate(seeds):
        set_all_seeds(seed)
        model = build_model_from_cfg(cfg, class_weights_t)

        core_model = getattr(model, "gnn", model)
        current_candidate_edges = {
            (int(src), int(dst))
            for src, dst in edge_set_from_edge_index(core_model.edge_index)
            if int(src) != int(dst)
        }
        if candidate_edge_universe is None:
            candidate_edge_universe = current_candidate_edges
            candidate_gene_universe = {
                gene for edge in current_candidate_edges for gene in edge
            }
        elif current_candidate_edges != candidate_edge_universe:
            raise RuntimeError(
                "El universo candidato cambió entre semillas; la estabilidad "
                "no sería interpretable."
            )

        dl_tr, dl_va, _ = make_loaders(X_full)
        model, best_state, best_macro_f1 = train_graph_learning(
            model, adj=None, dl_tr=dl_tr, dl_va=dl_va,
            device=CFG.DEVICE, label_map=label_map,
            epochs=epochs, lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY,
            accum_steps=(
                CFG.GRAPH_ACCUM_STEPS
                if bool(getattr(model, "per_label_graphs", False))
                else CFG.ACCUM_STEPS
            ),
            class_weights=cw_list, class_weights_t=class_weights_t,
            edge_l1_per_edge=CFG.EDGE_L1_PER_EDGE,
            aux_lambda=CFG.AUX_LAMBDA, loss_fn=loss_fn,
            keep_min=keep_ratio,
            sample_weights_full=sample_weights_full,
        )
        model.load_state_dict(best_state)

        graphs = export_pruned_graphs_by_class(
            model,
            keep_ratio=keep_ratio,
            class_names=class_names,
            keep_self_loops=False,
        )
        if set(graphs) != set(range(n_classes)):
            raise RuntimeError(
                "La estabilidad por clase requiere un experto para cada "
                f"clase; recibidos {sorted(graphs)}."
            )

        if want_run_graphs and runs_graphs is not None:
            run_payload = {"run": int(i), "seed": int(seed), "classes": {}}

        for class_id, graph in graphs.items():
            es = edge_set_from_edge_index(graph["edge_index"])
            edge_sets_by_class[class_id].append(es)
            rank_map, score_map = _edge_rank_score_maps(graph)
            edge_rank_maps_by_class[class_id].append(rank_map)
            edge_score_maps_by_class[class_id].append(score_map)
            rows.append({
                "run": int(i),
                "seed": int(seed),
                "class_id": int(class_id),
                "class_name": class_names[class_id],
                "val_macro_f1": float(best_macro_f1),
                "thr_logit": float(graph["thr_logit"]),
                "n_edges": int(len(es)),
            })
            if want_run_graphs and runs_graphs is not None:
                run_payload["classes"][int(class_id)] = {
                    "class_name": class_names[class_id],
                    "edge_index_full": graph["edge_index"].detach().cpu(),
                    "edge_weight_full": (
                        graph["edge_weight"].detach().cpu()
                        if isinstance(graph["edge_weight"], torch.Tensor)
                        else None
                    ),
                    "edge_type_full": (
                        graph["edge_type"].detach().cpu()
                        if isinstance(graph["edge_type"], torch.Tensor)
                        else None
                    ),
                    "thr_logit": float(graph["thr_logit"]),
                    "kept": int(graph["kept"]),
                    "total": int(graph["total"]),
                }
            if verbose:
                print(
                    f"  run {i} seed {seed} {class_names[class_id]} "
                    f"edges={len(es)} valF1={best_macro_f1:.4f}"
                )
        if want_run_graphs and runs_graphs is not None:
            runs_graphs.append(run_payload)

        try:
            del model, best_state, graphs
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    # --- Consenso y estabilidad, siempre separados por clase ---
    if candidate_edge_universe is None or candidate_gene_universe is None:
        raise RuntimeError("No se pudo reconstruir el universo candidato.")
    per_class: Dict[int, Dict[str, Any]] = {}
    all_to_consensus: List[float] = []
    all_pairwise_means: List[float] = []
    all_pairwise_mins: List[float] = []
    all_pairwise_maxs: List[float] = []
    all_gene_pairwise_means: List[float] = []
    all_edge_nogueira: List[float] = []
    all_gene_nogueira: List[float] = []
    total_pairs = 0
    consensus_union: set = set()
    stable_core80_union: set = set()
    stable_core100_union: set = set()
    gene_core80_union: set = set()
    gene_core100_union: set = set()

    for class_id, edge_sets in edge_sets_by_class.items():
        edge_jstats = pairwise_jaccard_stats(edge_sets)
        n_runs = len(edge_sets)
        consensus_info = frequency_ranked_consensus(
            edge_sets=edge_sets,
            frequency_threshold=CFG.STAB_FREQ_THR,
            preserve_topk_budget=bool(getattr(
                CFG, "CLASS_GRAPH_STABILITY_FIXED_BUDGET", False
            )),
            explicit_target=getattr(
                CFG, "CLASS_GRAPH_TARGET_EDGES", None
            ),
            rank_maps=edge_rank_maps_by_class[class_id],
            score_maps=edge_score_maps_by_class[class_id],
        )
        counter = consensus_info["frequency_counter"]
        stable_core80 = consensus_info["threshold_consensus"]
        stable_core100 = {
            edge for edge, count in counter.items()
            if (count / n_runs) >= float(CFG.STAB_STRICT_FREQ_THR)
        }
        consensus = consensus_info["consensus_edges"]
        consensus_union |= consensus
        stable_core80_union |= stable_core80
        stable_core100_union |= stable_core100

        gene_sets, incident_counts, neighbour_counts = _gene_sets_and_counts(
            edge_sets
        )
        gene_counter = Counter(
            gene for selected_genes in gene_sets for gene in selected_genes
        )
        gene_core80 = {
            gene for gene, count in gene_counter.items()
            if (count / n_runs) >= float(CFG.STAB_FREQ_THR)
        }
        gene_core100 = {
            gene for gene, count in gene_counter.items()
            if (count / n_runs) >= float(CFG.STAB_STRICT_FREQ_THR)
        }
        edge_core80_genes = {
            gene for edge in stable_core80 for gene in edge
        }
        edge_core100_genes = {
            gene for edge in stable_core100 for gene in edge
        }
        gene_core80_union |= gene_core80
        gene_core100_union |= gene_core100

        gene_jstats = pairwise_jaccard_stats(gene_sets)
        edge_nogueira = nogueira_stability(
            edge_sets, candidate_edge_universe
        )
        gene_nogueira = nogueira_stability(
            gene_sets, candidate_gene_universe
        )
        if np.isfinite(edge_nogueira):
            all_edge_nogueira.append(float(edge_nogueira))
        if np.isfinite(gene_nogueira):
            all_gene_nogueira.append(float(gene_nogueira))
        if np.isfinite(gene_jstats["jaccard_mean"]):
            all_gene_pairwise_means.append(
                float(gene_jstats["jaccard_mean"])
            )

        if consensus:
            to_cons = np.asarray(
                [jaccard(es, consensus) for es in edge_sets],
                dtype=np.float32,
            )
            all_to_consensus.extend(to_cons.tolist())
            to_stats = {
                "to_consensus_mean": float(np.mean(to_cons)),
                "to_consensus_std": (
                    float(np.std(to_cons, ddof=1))
                    if to_cons.size > 1 else 0.0
                ),
                "to_consensus_min": float(np.min(to_cons)),
                "to_consensus_max": float(np.max(to_cons)),
            }
        else:
            to_cons = None
            to_stats = {
                "to_consensus_mean": float("nan"),
                "to_consensus_std": float("nan"),
                "to_consensus_min": float("nan"),
                "to_consensus_max": float("nan"),
            }

        if np.isfinite(edge_jstats["jaccard_mean"]):
            all_pairwise_means.append(float(edge_jstats["jaccard_mean"]))
            all_pairwise_mins.append(float(edge_jstats["jaccard_min"]))
            all_pairwise_maxs.append(float(edge_jstats["jaccard_max"]))
        total_pairs += int(edge_jstats["n_pairs"])
        per_class[class_id] = {
            "class_id": int(class_id),
            "class_name": class_names[class_id],
            "jaccard_stats": edge_jstats,
            "edge_jaccard_stats": edge_jstats,
            "gene_jaccard_stats": gene_jstats,
            "edge_nogueira_stability": float(edge_nogueira),
            "gene_nogueira_stability": float(gene_nogueira),
            "to_consensus_stats": to_stats,
            "consensus_edges": consensus,
            "exact_consensus_edges": consensus,
            # Alias histórico: desde esta versión siempre significa core80.
            "stable_core_edges": stable_core80,
            "stable_core80_edges": stable_core80,
            "stable_core100_edges": stable_core100,
            "gene_core80": gene_core80,
            "gene_core100": gene_core100,
            "edge_core80_genes": edge_core80_genes,
            "edge_core100_genes": edge_core100_genes,
            "consensus_size": int(len(consensus)),
            "exact_consensus_size": int(len(consensus)),
            "stable_core_size": int(len(stable_core80)),
            "stable_core80_size": int(len(stable_core80)),
            "stable_core100_size": int(len(stable_core100)),
            "gene_core80_size": int(len(gene_core80)),
            "gene_core100_size": int(len(gene_core100)),
            "edge_core80_gene_size": int(len(edge_core80_genes)),
            "edge_core100_gene_size": int(len(edge_core100_genes)),
            "frequency_threshold_size": int(len(stable_core80)),
            "target_edges": consensus_info["target_edges"],
            "target_source": consensus_info["target_source"],
            "run_edge_sizes": consensus_info["run_edge_sizes"],
            "run_gene_sizes": [int(len(genes)) for genes in gene_sets],
            "n_candidate_edges": int(len(candidate_edge_universe)),
            "n_candidate_genes": int(len(candidate_gene_universe)),
            "selection_frequency": {
                edge: float(counter[edge] / n_runs)
                for edge in counter
            },
            "selection_count": dict(counter),
            "selection_median_rank": consensus_info["median_rank"],
            "selection_median_score": consensus_info["median_score"],
            "gene_selection_frequency": {
                gene: float(gene_counter[gene] / n_runs)
                for gene in gene_counter
            },
            "gene_selection_count": dict(gene_counter),
            "gene_sets": gene_sets,
            "gene_incident_counts": incident_counts,
            "gene_neighbour_counts": neighbour_counts,
            "_to_consensus": to_cons,
        }

    if all_to_consensus:
        all_tc = np.asarray(all_to_consensus, dtype=np.float32)
        to_consensus_stats = {
            "to_consensus_mean": float(all_tc.mean()),
            "to_consensus_std": (
                float(all_tc.std(ddof=1)) if all_tc.size > 1 else 0.0
            ),
            "to_consensus_min": float(all_tc.min()),
            "to_consensus_max": float(all_tc.max()),
        }
    else:
        to_consensus_stats = {
            "to_consensus_mean": float("nan"),
            "to_consensus_std": float("nan"),
            "to_consensus_min": float("nan"),
            "to_consensus_max": float("nan"),
        }

    out: Dict[str, Any] = {
        "seeds": [int(seed) for seed in seeds],
        "n_runs": int(len(seeds)),
        "per_class": per_class,
        "consensus_edges_by_class": {
            class_id: details["consensus_edges"]
            for class_id, details in per_class.items()
        },
        "exact_consensus_edges_by_class": {
            class_id: details["exact_consensus_edges"]
            for class_id, details in per_class.items()
        },
        "stable_core_edges_by_class": {
            class_id: details["stable_core_edges"]
            for class_id, details in per_class.items()
        },
        "stable_core80_edges_by_class": {
            class_id: details["stable_core80_edges"]
            for class_id, details in per_class.items()
        },
        "stable_core100_edges_by_class": {
            class_id: details["stable_core100_edges"]
            for class_id, details in per_class.items()
        },
        "gene_core80_by_class": {
            class_id: details["gene_core80"]
            for class_id, details in per_class.items()
        },
        "gene_core100_by_class": {
            class_id: details["gene_core100"]
            for class_id, details in per_class.items()
        },
        "consensus_size_by_class": {
            class_id: details["consensus_size"]
            for class_id, details in per_class.items()
        },
        "target_edges_by_class": {
            class_id: details["target_edges"]
            for class_id, details in per_class.items()
        },
        # Compatibilidad: la unión sirve para compactar nodos, nunca para
        # sustituir los dos grafos predictivos.
        "consensus_edges": consensus_union,
        "consensus_size": int(len(consensus_union)),
        "exact_consensus_edges": consensus_union,
        "stable_core_edges": stable_core80_union,
        "stable_core_size": int(len(stable_core80_union)),
        "stable_core80_edges": stable_core80_union,
        "stable_core100_edges": stable_core100_union,
        "stable_core80_size": int(len(stable_core80_union)),
        "stable_core100_size": int(len(stable_core100_union)),
        "gene_core80": gene_core80_union,
        "gene_core100": gene_core100_union,
        "gene_core80_size": int(len(gene_core80_union)),
        "gene_core100_size": int(len(gene_core100_union)),
        "stability_scope": str(CFG.STABILITY_SCOPE),
        "data_resampling": bool(CFG.STABILITY_DATA_RESAMPLING),
        "jaccard_mean": (
            float(np.mean(all_pairwise_means))
            if all_pairwise_means else float("nan")
        ),
        "jaccard_std": (
            float(np.std(all_pairwise_means, ddof=1))
            if len(all_pairwise_means) > 1 else 0.0
        ),
        "jaccard_min": (
            float(np.min(all_pairwise_mins))
            if all_pairwise_mins else float("nan")
        ),
        "jaccard_max": (
            float(np.max(all_pairwise_maxs))
            if all_pairwise_maxs else float("nan")
        ),
        "jaccard_n_pairs": int(total_pairs),
        "gene_jaccard_mean": (
            float(np.mean(all_gene_pairwise_means))
            if all_gene_pairwise_means else float("nan")
        ),
        "edge_nogueira_mean": (
            float(np.mean(all_edge_nogueira))
            if all_edge_nogueira else float("nan")
        ),
        "gene_nogueira_mean": (
            float(np.mean(all_gene_nogueira))
            if all_gene_nogueira else float("nan")
        ),
        "to_consensus_stats": to_consensus_stats,
    }

    # Detalles opcionales
    need_tables = bool(STAB_RETURN_DETAILS or (save_dir is not None))
    if need_tables:
        runs_df = pd.DataFrame(rows)
        labels = [f"seed_{s}" for s in seeds]
        jaccard_by_class = {}
        gene_jaccard_by_class = {}
        for class_id, details in per_class.items():
            to_cons = details["_to_consensus"]
            if to_cons is not None:
                idx = runs_df["class_id"] == class_id
                runs_df.loc[idx, "to_consensus"] = to_cons
            consensus = details["consensus_edges"]
            jaccard_by_class[class_id] = build_jaccard_df(
                edge_sets_by_class[class_id],
                labels=labels,
                consensus=(consensus if consensus else None),
            )
            gene_jaccard_by_class[class_id] = build_jaccard_df(
                details["gene_sets"], labels=labels,
                consensus=(details["gene_core80"] or None),
            )
        out["runs_df"] = runs_df
        out["jaccard_df_by_class"] = jaccard_by_class
        out["gene_jaccard_df_by_class"] = gene_jaccard_by_class

    if STAB_RETURN_DETAILS:
        out["edge_sets_by_class"] = edge_sets_by_class
        out["gene_sets_by_class"] = {
            class_id: details["gene_sets"]
            for class_id, details in per_class.items()
        }
        if want_run_graphs and runs_graphs is not None:
            out["runs_graphs"] = runs_graphs

    # Guardado opcional
    if save_dir is not None:
        _mkdir(save_dir)
        out["runs_df"].to_csv(
            os.path.join(save_dir, "stability_runs_by_class.csv"),
            index=False,
        )
        candidate_incident = Counter()
        candidate_neighbours: Dict[int, set] = {}
        for src, dst in candidate_edge_universe:
            candidate_incident[src] += 1
            candidate_incident[dst] += 1
            candidate_neighbours.setdefault(src, set()).add(dst)
            candidate_neighbours.setdefault(dst, set()).add(src)

        for class_id, frame in out["jaccard_df_by_class"].items():
            safe = "".join(
                ch if ch.isalnum() or ch in "-_" else "_"
                for ch in class_names[class_id]
            )
            frame.to_csv(
                os.path.join(save_dir, f"stability_jaccard_edges_{safe}.csv")
            )
            # Alias histórico conservado para no romper lectores previos.
            frame.to_csv(
                os.path.join(save_dir, f"stability_jaccard_{safe}.csv")
            )
            out["gene_jaccard_df_by_class"][class_id].to_csv(
                os.path.join(save_dir, f"stability_jaccard_genes_{safe}.csv")
            )

            details = per_class[class_id]
            freq = pd.DataFrame([
                {
                    "src": int(src),
                    "dst": int(dst),
                    "selection_count": int(details["selection_count"][(src, dst)]),
                    "selection_frequency": float(value),
                    "median_within_seed_rank": float(
                        details["selection_median_rank"][(src, dst)]
                    ),
                    "median_learned_logit": float(
                        details["selection_median_score"][(src, dst)]
                    ),
                    "in_exact_k": (src, dst) in details["exact_consensus_edges"],
                    "in_stable_core": (src, dst) in details["stable_core80_edges"],
                    "in_edge_core80": (src, dst) in details["stable_core80_edges"],
                    "in_edge_core100": (src, dst) in details["stable_core100_edges"],
                }
                for (src, dst), value in details["selection_frequency"].items()
            ]).sort_values(
                ["selection_frequency", "median_within_seed_rank", "src", "dst"],
                ascending=[False, True, True, True],
            )
            freq.to_csv(
                os.path.join(save_dir, f"selection_frequency_{safe}.csv"),
                index=False,
            )

            gene_rows = []
            for gene, value in details["gene_selection_frequency"].items():
                incident = np.asarray([
                    counts.get(gene, 0)
                    for counts in details["gene_incident_counts"]
                ], dtype=float)
                neighbours = np.asarray([
                    counts.get(gene, 0)
                    for counts in details["gene_neighbour_counts"]
                ], dtype=float)
                row = {
                    "gene_idx": int(gene),
                    "gene": str(genes_full[int(gene)]),
                    "selection_count": int(
                        details["gene_selection_count"][gene]
                    ),
                    "selection_frequency": float(value),
                    "median_incident_edges": float(np.median(incident)),
                    "incident_edges_q25": float(np.quantile(incident, 0.25)),
                    "incident_edges_q75": float(np.quantile(incident, 0.75)),
                    "median_unique_neighbours": float(np.median(neighbours)),
                    "candidate_incident_edges": int(candidate_incident[gene]),
                    "candidate_unique_neighbours": int(
                        len(candidate_neighbours.get(gene, set()))
                    ),
                    "in_gene_core80": gene in details["gene_core80"],
                    "in_gene_core100": gene in details["gene_core100"],
                    "in_edge_core80": gene in details["edge_core80_genes"],
                    "in_edge_core100": gene in details["edge_core100_genes"],
                }
                for seed, selected_genes, counts in zip(
                    seeds, details["gene_sets"], details["gene_incident_counts"]
                ):
                    row[f"selected_seed_{int(seed)}"] = gene in selected_genes
                    row[f"incident_edges_seed_{int(seed)}"] = int(
                        counts.get(gene, 0)
                    )
                gene_rows.append(row)
            gene_freq = pd.DataFrame(gene_rows).sort_values(
                ["selection_frequency", "median_incident_edges", "gene"],
                ascending=[False, False, True],
            )
            gene_freq.to_csv(
                os.path.join(save_dir, f"gene_selection_frequency_{safe}.csv"),
                index=False,
            )

        payload = {
            "cfg": cfg_to_dict(cfg),
            "seeds": list(seeds),
            "keep_ratio": float(keep_ratio),
            "epochs": int(epochs),
            "freq_thr": float(CFG.STAB_FREQ_THR),
            "strict_freq_thr": float(CFG.STAB_STRICT_FREQ_THR),
            "stability_scope": str(CFG.STABILITY_SCOPE),
            "data_resampling": bool(CFG.STABILITY_DATA_RESAMPLING),
            "class_names": list(class_names),
            "target_edges_by_class": {
                int(class_id): details["target_edges"]
                for class_id, details in per_class.items()
            },
            "target_source_by_class": {
                int(class_id): details["target_source"]
                for class_id, details in per_class.items()
            },
            "consensus_edges_by_class": {
                int(class_id): [
                    (int(src), int(dst))
                    for src, dst in details["consensus_edges"]
                ]
                for class_id, details in per_class.items()
            },
            "exact_consensus_edges_by_class": {
                int(class_id): [
                    (int(src), int(dst))
                    for src, dst in details["exact_consensus_edges"]
                ]
                for class_id, details in per_class.items()
            },
            "stable_core_edges_by_class": {
                int(class_id): [
                    (int(src), int(dst))
                    for src, dst in details["stable_core_edges"]
                ]
                for class_id, details in per_class.items()
            },
            "stable_core80_edges_by_class": {
                int(class_id): [
                    (int(src), int(dst))
                    for src, dst in details["stable_core80_edges"]
                ]
                for class_id, details in per_class.items()
            },
            "stable_core100_edges_by_class": {
                int(class_id): [
                    (int(src), int(dst))
                    for src, dst in details["stable_core100_edges"]
                ]
                for class_id, details in per_class.items()
            },
            "gene_core80_by_class": {
                int(class_id): sorted(map(int, details["gene_core80"]))
                for class_id, details in per_class.items()
            },
            "gene_core100_by_class": {
                int(class_id): sorted(map(int, details["gene_core100"]))
                for class_id, details in per_class.items()
            },
            "selection_frequency_by_class": {
                int(class_id): [
                    (int(src), int(dst), float(freq))
                    for (src, dst), freq in details[
                        "selection_frequency"
                    ].items()
                ]
                for class_id, details in per_class.items()
            },
        }
        if want_run_graphs and runs_graphs is not None:
            payload["runs_graphs"] = runs_graphs
        torch.save(payload, os.path.join(save_dir, "stability_graphs.pt"))

        with open(os.path.join(save_dir, "stability_summary.json"), "w", encoding="utf-8") as f:
            json.dump({
                "class_names": list(class_names),
                "seeds": [int(seed) for seed in seeds],
                "n_runs": int(len(seeds)),
                "stability_level": "inner_within_outer_seed",
                "role": "build_one_outer_exact_k_graph_for_phase2",
                "stability_scope": str(CFG.STABILITY_SCOPE),
                "data_matrix_and_split_fixed": True,
                "data_resampling": bool(CFG.STABILITY_DATA_RESAMPLING),
                # Clave histórica conservada; este umbral es diagnóstico
                # dentro de las ocho corridas, no la firma biológica final.
                "primary_frequency_threshold": float(CFG.STAB_FREQ_THR),
                "inner_diagnostic_frequency_threshold": float(
                    CFG.STAB_FREQ_THR
                ),
                "strict_frequency_threshold": float(
                    CFG.STAB_STRICT_FREQ_THR
                ),
                "core80_min_runs": int(np.ceil(
                    float(CFG.STAB_FREQ_THR) * len(seeds)
                )),
                "core100_min_runs": int(np.ceil(
                    float(CFG.STAB_STRICT_FREQ_THR) * len(seeds)
                )),
                "interpretation": (
                    "inner multiseed algorithmic reproducibility used to build "
                    "one outer Exact-K graph; not sampling stability and not "
                    "formal stability-selection error control"
                ),
                "per_class": {
                    class_names[class_id]: {
                        "jaccard_stats": details["jaccard_stats"],
                        "edge_jaccard_stats": details["edge_jaccard_stats"],
                        "gene_jaccard_stats": details["gene_jaccard_stats"],
                        "edge_nogueira_stability": details[
                            "edge_nogueira_stability"
                        ],
                        "gene_nogueira_stability": details[
                            "gene_nogueira_stability"
                        ],
                        "to_consensus_stats": details[
                            "to_consensus_stats"
                        ],
                        "consensus_size": details["consensus_size"],
                        "exact_consensus_size": details["exact_consensus_size"],
                        "stable_core_size": details["stable_core_size"],
                        "stable_core80_size": details["stable_core80_size"],
                        "stable_core100_size": details[
                            "stable_core100_size"
                        ],
                        "gene_core80_size": details["gene_core80_size"],
                        "gene_core100_size": details["gene_core100_size"],
                        "edge_core80_gene_size": details[
                            "edge_core80_gene_size"
                        ],
                        "edge_core100_gene_size": details[
                            "edge_core100_gene_size"
                        ],
                        "frequency_threshold_size": details[
                            "frequency_threshold_size"
                        ],
                        "target_edges": details["target_edges"],
                        "target_source": details["target_source"],
                        "run_edge_sizes": details["run_edge_sizes"],
                        "run_gene_sizes": details["run_gene_sizes"],
                        "n_candidate_edges": details["n_candidate_edges"],
                        "n_candidate_genes": details["n_candidate_genes"],
                    }
                    for class_id, details in per_class.items()
                },
                "aggregate": {
                    "jaccard_mean": out["jaccard_mean"],
                    "jaccard_std": out["jaccard_std"],
                    "jaccard_min": out["jaccard_min"],
                    "jaccard_max": out["jaccard_max"],
                    "jaccard_n_pairs": out["jaccard_n_pairs"],
                    "to_consensus_stats": to_consensus_stats,
                    "consensus_union_size": out["consensus_size"],
                    "stable_core_union_size": out["stable_core_size"],
                    "stable_core80_union_size": out["stable_core80_size"],
                    "stable_core100_union_size": out[
                        "stable_core100_size"
                    ],
                    "gene_core80_union_size": out["gene_core80_size"],
                    "gene_core100_union_size": out["gene_core100_size"],
                    "gene_jaccard_mean": out["gene_jaccard_mean"],
                    "edge_nogueira_mean": out["edge_nogueira_mean"],
                    "gene_nogueira_mean": out["gene_nogueira_mean"],
                },
            }, f, indent=2, ensure_ascii=False)

    for details in per_class.values():
        details.pop("_to_consensus", None)
    return out
