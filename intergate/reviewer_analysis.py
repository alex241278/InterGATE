"""Reviewer-facing topology, stability and biological-recovery analyses.

All feature selection functions consume development data only. The external
test is intentionally absent from their APIs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from .stability import pairwise_jaccard_stats, nogueira_stability


def _edge_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["source_gene"].astype(str),
                   frame["target_gene"].astype(str)))


def consensus_from_seed_tables(seed_root: str | Path, *,
                               frequency_threshold: float = 0.80,
                               strict_threshold: float = 1.00,
                               expected_runs: int = 5,
                               expected_seed_values: Sequence[int] | None = None,
                               edge_universe: Iterable[tuple[str, str]] | None = None,
                               gene_universe: Iterable[str] | None = None,
                               output_dir: str | Path | None = None) -> dict:
    """Build Exact-K, edge cores and gene cores from saved TopK runs.

    This function performs the OUTER aggregation over the five saved
    seed-specific Exact-K consensus graphs. The primary biological signature
    is gene-core100 (5/5); gene-core80 (4/5) is sensitivity only. Runs differ
    only through stochastic training unless the caller explicitly generated
    resampled inputs.
    """
    if not 0.0 < float(frequency_threshold) <= 1.0:
        raise ValueError("frequency_threshold debe estar en (0, 1].")
    if not 0.0 < float(strict_threshold) <= 1.0:
        raise ValueError("strict_threshold debe estar en (0, 1].")
    if float(strict_threshold) < float(frequency_threshold):
        raise ValueError("strict_threshold debe ser >= frequency_threshold.")
    if int(expected_runs) != 5:
        raise ValueError(
            "El protocolo core80/core100 está preespecificado para cinco "
            f"corridas; expected_runs={expected_runs}."
        )
    edge_universe = set(edge_universe) if edge_universe is not None else None
    gene_universe = set(gene_universe) if gene_universe is not None else None
    seed_root = Path(seed_root)
    if expected_seed_values is not None:
        expected_seed_values = tuple(map(int, expected_seed_values))
        if len(expected_seed_values) != int(expected_runs) or (
            len(set(expected_seed_values)) != int(expected_runs)
        ):
            raise ValueError(
                "expected_seed_values debe contener exactamente las cinco "
                "semillas externas distintas."
            )
        paths = sorted(
            path
            for seed in expected_seed_values
            for path in (seed_root / f"seed_{seed}").glob("edges_*.csv")
        )
    else:
        paths = sorted(seed_root.glob("seed_*/edges_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No hay seed_*/edges_*.csv en {seed_root}")
    by_class: Dict[str, list[tuple[str, pd.DataFrame]]] = {}
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        name = str(frame["class_name"].iloc[0])
        by_class.setdefault(name, []).append((path.parent.name, frame))

    result = {}
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for class_name, seed_frames in sorted(by_class.items()):
        def _seed_order(item):
            text = str(item[0]).replace("seed_", "")
            return (0, int(text)) if text.isdigit() else (1, str(item[0]))

        seed_frames = sorted(seed_frames, key=_seed_order)
        seed_labels = [seed for seed, _ in seed_frames]
        if expected_seed_values is not None:
            expected_labels = {f"seed_{seed}" for seed in expected_seed_values}
            if set(seed_labels) != expected_labels:
                raise ValueError(
                    f"{class_name}: semillas encontradas {seed_labels} != "
                    f"preespecificadas {sorted(expected_labels)}."
                )
        frames = [frame for _, frame in seed_frames]
        edge_sets = [_edge_set(frame) for frame in frames]
        if len(edge_sets) != int(expected_runs):
            raise ValueError(
                f"{class_name}: se esperaban {expected_runs} semillas y se "
                f"encontraron {len(edge_sets)} ({seed_labels})."
            )
        sizes = [len(edges) for edges in edge_sets]
        if len(set(sizes)) != 1:
            raise ValueError(f"TopK no es exacto para {class_name}: {sizes}")
        k = sizes[0]
        counter = Counter(edge for edges in edge_sets for edge in edges)

        rank_maps, score_maps = [], []
        for frame in frames:
            frame = frame.copy()
            score_col = next((col for col in (
                              "inner_median_logit", "learned_logit", "learned_gate")
                              if col in frame and frame[col].notna().any()), None)
            if score_col is None:
                frame["_selection_score"] = np.nan
            else:
                frame["_selection_score"] = pd.to_numeric(
                    frame[score_col], errors="coerce"
                )
            pairs = list(zip(
                frame["source_gene"].astype(str),
                frame["target_gene"].astype(str),
            ))
            score_maps.append({
                edge: float(score) for edge, score in zip(
                    pairs, frame["_selection_score"].astype(float)
                )
            })
            if "consensus_rank" in frame:
                stored_rank = pd.to_numeric(
                    frame["consensus_rank"], errors="coerce"
                )
            else:
                stored_rank = pd.Series(np.nan, index=frame.index)
            valid_stored_rank = (
                stored_rank.notna().all()
                and (stored_rank > 0).all()
                and stored_rank.nunique() == len(frame)
            )
            if valid_stored_rank:
                rank_maps.append({
                    edge: float(rank)
                    for edge, rank in zip(pairs, stored_rank.astype(float))
                })
            else:
                ordered = frame.sort_values(
                    ["_selection_score", "source_gene", "target_gene"],
                    ascending=[False, True, True], na_position="last",
                )
                ordered_pairs = list(zip(
                    ordered["source_gene"].astype(str),
                    ordered["target_gene"].astype(str),
                ))
                rank_maps.append({
                    edge: rank for rank, edge in enumerate(ordered_pairs, 1)
                })

        median_rank = {
            edge: float(np.median([
                values[edge] for values in rank_maps if edge in values
            ]))
            for edge in counter
        }
        median_score = {
            edge: float(np.nanmedian([
                values[edge] for values in score_maps
                if edge in values and np.isfinite(values[edge])
            ])) if any(
                edge in values and np.isfinite(values[edge])
                for values in score_maps
            ) else float("nan")
            for edge in counter
        }
        ranked = sorted(counter, key=lambda e: (
            -counter[e], median_rank[e],
            -median_score[e] if np.isfinite(median_score[e]) else float("inf"),
            e[0], e[1],
        ))
        atomic_units = all(
            "structural_unit_id" in frame.columns
            and pd.to_numeric(
                frame["structural_unit_id"], errors="coerce"
            ).notna().all()
            and (
                pd.to_numeric(
                    frame["structural_unit_id"], errors="coerce"
                ) >= 0
            ).all()
            for frame in frames
        )
        k_structural_units = k
        edge_to_structural_unit: dict[tuple[str, str], int] = {}
        if atomic_units:
            unit_sets = []
            unit_counter: Counter = Counter()
            unit_rank_maps = []
            unit_score_maps = []
            unit_edges: dict[int, set[tuple[str, str]]] = {}
            unit_widths = set()
            for frame_id, frame in enumerate(frames):
                local_units = set()
                local_rank = {}
                local_score = {}
                ids = pd.to_numeric(
                    frame["structural_unit_id"], errors="raise"
                ).astype(np.int64)
                local = frame.assign(_structural_unit_id=ids)
                for unit_id, group in local.groupby(
                    "_structural_unit_id", sort=False
                ):
                    unit_id = int(unit_id)
                    edges = set(zip(
                        group["source_gene"].astype(str),
                        group["target_gene"].astype(str),
                    ))
                    unit_widths.add(len(edges))
                    if unit_id in unit_edges and unit_edges[unit_id] != edges:
                        raise ValueError(
                            f"{class_name}: la unidad estructural {unit_id} "
                            "cambia de aristas entre semillas."
                        )
                    unit_edges[unit_id] = edges
                    for edge in edges:
                        previous = edge_to_structural_unit.get(edge)
                        if previous is not None and previous != unit_id:
                            raise ValueError(
                                f"{class_name}: {edge} pertenece a dos "
                                "unidades estructurales."
                            )
                        edge_to_structural_unit[edge] = unit_id
                    local_units.add(unit_id)
                    local_rank[unit_id] = min(
                        rank_maps[frame_id][edge] for edge in edges
                    )
                    finite_scores = [
                        score_maps[frame_id][edge] for edge in edges
                        if np.isfinite(score_maps[frame_id][edge])
                    ]
                    local_score[unit_id] = (
                        max(finite_scores) if finite_scores else float("nan")
                    )
                unit_sets.append(local_units)
                unit_counter.update(local_units)
                unit_rank_maps.append(local_rank)
                unit_score_maps.append(local_score)
            if len(unit_widths) != 1:
                raise ValueError(
                    f"{class_name}: anchuras estructurales mezcladas "
                    f"{sorted(unit_widths)}."
                )
            unit_sizes = [len(values) for values in unit_sets]
            if len(set(unit_sizes)) != 1:
                raise ValueError(
                    f"{class_name}: la cuota por unidades no es exacta: "
                    f"{unit_sizes}."
                )
            k_structural_units = unit_sizes[0]
            unit_median_rank = {
                unit_id: float(np.median([
                    values[unit_id] for values in unit_rank_maps
                    if unit_id in values
                ]))
                for unit_id in unit_counter
            }
            unit_median_score = {
                unit_id: float(np.nanmedian([
                    values[unit_id] for values in unit_score_maps
                    if unit_id in values and np.isfinite(values[unit_id])
                ])) if any(
                    unit_id in values and np.isfinite(values[unit_id])
                    for values in unit_score_maps
                ) else float("nan")
                for unit_id in unit_counter
            }
            ranked_units = sorted(unit_counter, key=lambda unit_id: (
                -unit_counter[unit_id],
                unit_median_rank[unit_id],
                -unit_median_score[unit_id]
                if np.isfinite(unit_median_score[unit_id])
                else float("inf"),
                int(unit_id),
            ))
            selected_units = ranked_units[:k_structural_units]
            exact = set().union(*(unit_edges[value] for value in selected_units))
            if len(exact) != k:
                raise ValueError(
                    f"{class_name}: el consenso atómico produjo {len(exact)} "
                    f"aristas, pero cada semilla contiene {k}."
                )
        else:
            exact = set(ranked[:k])
        stable80 = {
            edge for edge, count in counter.items()
            if count / len(edge_sets) >= float(frequency_threshold)
        }
        stable100 = {
            edge for edge, count in counter.items()
            if count / len(edge_sets) >= float(strict_threshold)
        }

        gene_sets = [{gene for edge in edges for gene in edge}
                     for edges in edge_sets]
        gene_counter = Counter(gene for genes in gene_sets for gene in genes)
        gene_core80 = {
            gene for gene, count in gene_counter.items()
            if count / len(gene_sets) >= float(frequency_threshold)
        }
        gene_core100 = {
            gene for gene, count in gene_counter.items()
            if count / len(gene_sets) >= float(strict_threshold)
        }
        edge_core80_genes = {gene for edge in stable80 for gene in edge}
        edge_core100_genes = {gene for edge in stable100 for gene in edge}

        incident_by_seed = []
        neighbours_by_seed = []
        for edges in edge_sets:
            incident = Counter()
            neighbours: Dict[str, set[str]] = {}
            for src, dst in edges:
                incident[src] += 1
                incident[dst] += 1
                neighbours.setdefault(src, set()).add(dst)
                neighbours.setdefault(dst, set()).add(src)
            incident_by_seed.append(incident)
            neighbours_by_seed.append({
                gene: len(values) for gene, values in neighbours.items()
            })

        table = pd.DataFrame([
            {
                "class_name": class_name,
                "source_gene": src,
                "target_gene": dst,
                "selection_count": int(counter[(src, dst)]),
                "selection_frequency": float(counter[(src, dst)] / len(edge_sets)),
                "median_within_seed_rank": median_rank[(src, dst)],
                "median_learned_logit": median_score[(src, dst)],
                "structural_unit_id": edge_to_structural_unit.get(
                    (src, dst), -1
                ),
                "in_exact_k": (src, dst) in exact,
                "in_stable_core": (src, dst) in stable80,
                "in_edge_core80": (src, dst) in stable80,
                "in_edge_core100": (src, dst) in stable100,
            }
            for src, dst in ranked
        ])

        gene_rows = []
        for gene in sorted(gene_counter):
            incident = np.asarray([
                values.get(gene, 0) for values in incident_by_seed
            ], dtype=float)
            neighbours = np.asarray([
                values.get(gene, 0) for values in neighbours_by_seed
            ], dtype=float)
            row = {
                "class_name": class_name,
                "gene": gene,
                "selection_count": int(gene_counter[gene]),
                "selection_frequency": float(
                    gene_counter[gene] / len(gene_sets)
                ),
                "median_incident_edges": float(np.median(incident)),
                "incident_edges_q25": float(np.quantile(incident, 0.25)),
                "incident_edges_q75": float(np.quantile(incident, 0.75)),
                "median_unique_neighbours": float(np.median(neighbours)),
                "in_gene_core80": gene in gene_core80,
                "in_gene_core100": gene in gene_core100,
                "in_edge_core80": gene in edge_core80_genes,
                "in_edge_core100": gene in edge_core100_genes,
            }
            for seed, genes, counts in zip(
                seed_labels, gene_sets, incident_by_seed
            ):
                row[f"selected_{seed}"] = gene in genes
                row[f"incident_edges_{seed}"] = int(counts.get(gene, 0))
            gene_rows.append(row)
        gene_table = pd.DataFrame(gene_rows).sort_values(
            ["selection_frequency", "median_incident_edges", "gene"],
            ascending=[False, False, True],
        )

        edge_jaccard = pairwise_jaccard_stats(edge_sets)
        gene_jaccard = pairwise_jaccard_stats(gene_sets)
        edge_nogueira = (
            nogueira_stability(edge_sets, edge_universe)
            if edge_universe is not None else float("nan")
        )
        gene_nogueira = (
            nogueira_stability(gene_sets, gene_universe)
            if gene_universe is not None else float("nan")
        )
        result[class_name] = {
            "n_runs": len(edge_sets), "k": k,
            "k_structural_units": int(k_structural_units),
            "atomic_structural_units": bool(atomic_units),
            "seed_labels": list(seed_labels),
            "exact_edges": exact,
            "stable_edges": stable80,
            "stable_edges_80": stable80,
            "stable_edges_100": stable100,
            "frequency_table": table,
            "gene_frequency_table": gene_table,
            "exact_genes": sorted({g for edge in exact for g in edge}),
            "stable_genes": sorted(edge_core80_genes),
            "edge_core80_genes": sorted(edge_core80_genes),
            "edge_core100_genes": sorted(edge_core100_genes),
            "gene_core80": sorted(gene_core80),
            "gene_core100": sorted(gene_core100),
            "primary_gene_signature": sorted(gene_core100),
            "sensitivity_gene_signature_4of5": sorted(gene_core80),
            "edge_jaccard_stats": edge_jaccard,
            "gene_jaccard_stats": gene_jaccard,
            "edge_nogueira_stability": float(edge_nogueira),
            "gene_nogueira_stability": float(gene_nogueira),
            "stability_scope": "algorithmic_randomness_fixed_development_data",
        }
        if output_dir is not None:
            table.to_csv(output_dir / f"consensus_{class_name}.csv", index=False)
            gene_table.to_csv(
                output_dir / f"gene_stability_{class_name}.csv", index=False
            )
    return result


def topology_metrics(edges: Iterable[tuple[str, str]]) -> dict:
    """Directed topology plus weakly-connected giant-component diagnostics."""
    edges = {(str(a), str(b)) for a, b in edges if str(a) != str(b)}
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
    if n_nodes == 0:
        return {
            "n_nodes": 0, "n_edges": 0, "n_components": 0,
            "n_strong_components": 0, "isolated_dyads": 0,
            "giant_n": 0, "giant_fraction": np.nan,
            "density": np.nan, "reciprocity": np.nan,
            "mean_total_degree": np.nan, "degree_gini": np.nan,
            "out_in_assortativity": np.nan, "components_lower_bound": 0,
            "components_lower_bound_unique_undirected": 0,
        }
    components = list(nx.weakly_connected_components(graph))
    strong_components = list(nx.strongly_connected_components(graph))
    giant_n = max(map(len, components))
    undirected_edges = {
        tuple(sorted((a, b))) for a, b in edges if a != b
    }
    # Each unique undirected edge can reduce the component count by at most 1.
    lower = max(1, n_nodes - n_edges)
    lower_undirected = max(1, n_nodes - len(undirected_edges))
    reciprocity = nx.reciprocity(graph)
    degrees = np.asarray([d for _, d in graph.degree()], dtype=float)
    if degrees.mean() > 0:
        degree_gini = float(
            np.abs(degrees[:, None] - degrees[None, :]).mean()
            / (2.0 * degrees.mean())
        )
    else:
        degree_gini = np.nan
    try:
        assortativity = float(nx.degree_assortativity_coefficient(
            graph, x="out", y="in",
        ))
    except Exception:
        assortativity = np.nan
    return {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_unique_undirected_edges": int(len(undirected_edges)),
        "n_components": int(len(components)),
        "n_strong_components": int(len(strong_components)),
        "isolated_dyads": int(sum(len(component) == 2 for component in components)),
        "giant_n": int(giant_n),
        "giant_fraction": float(giant_n / n_nodes),
        "density": float(nx.density(graph)),
        "reciprocity": float(reciprocity) if reciprocity is not None else np.nan,
        "mean_total_degree": float(degrees.mean()),
        "degree_gini": degree_gini,
        "out_in_assortativity": assortativity,
        "components_lower_bound": int(lower),
        "components_lower_bound_unique_undirected": int(lower_undirected),
    }


def topology_consensus_table(consensus: Mapping[str, dict],
                             *, analysis: str = "FULL") -> pd.DataFrame:
    rows = []
    for class_name, data in consensus.items():
        for graph_set, key in (
            ("exact_k", "exact_edges"),
            ("edge_core80", "stable_edges_80"),
            ("edge_core100", "stable_edges_100"),
        ):
            row = {
                "analysis": analysis, "class_name": class_name,
                "graph_set": graph_set,
            }
            row.update(topology_metrics(data[key]))
            rows.append(row)
    return pd.DataFrame(rows)


def load_gmt(path: str | Path) -> dict[str, set[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    out: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                out[fields[0]] = {gene.upper() for gene in fields[2:] if gene}
    if not out:
        raise ValueError(f"GMT vacío: {path}")
    return out


def _bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


_ORA_COLUMNS = [
    "collection", "term", "panel_size", "set_size", "overlap_n",
    "fold_enrichment", "p_value", "fdr", "meets_min_overlap",
    "min_overlap", "n_tests", "overlap_genes",
]


def overrepresentation(
    genes: Iterable[str],
    gene_sets: Mapping[str, set[str]],
    universe: Iterable[str],
    *,
    min_overlap: int = 2,
    collection: str = "",
    return_reportable_only: bool = False,
) -> pd.DataFrame:
    """One-sided hypergeometric ORA with a fixed BH testing family.

    The family is defined *before looking at the selected panel*: every term
    with at least one member in ``universe`` is tested.  Terms with observed
    overlap below ``min_overlap`` therefore remain in the Benjamini--Hochberg
    correction.  ``min_overlap`` is only a reporting rule and never changes
    the multiplicity correction.

    Parameters
    ----------
    return_reportable_only
        If true, rows below ``min_overlap`` are removed only *after* BH has
        been calculated over the complete eligible family.  This is useful
        for the 1,000-panel null without changing its inferential definition.
    """
    min_overlap = int(min_overlap)
    if min_overlap < 0:
        raise ValueError("min_overlap debe ser >= 0.")

    universe = {str(g).upper() for g in universe}
    if not universe:
        raise ValueError("El universo de ORA no puede estar vacío.")
    genes = {str(g).upper() for g in genes} & universe

    # Eligibility depends only on the prespecified collection and universe,
    # never on the observed panel. Empty terms after restriction cannot be
    # tested and are excluded identically for every panel.
    eligible_terms = []
    for term, raw_members in sorted(gene_sets.items(), key=lambda item: str(item[0])):
        members = {str(g).upper() for g in raw_members} & universe
        if members:
            eligible_terms.append((str(term), members))

    n_tests = len(eligible_terms)
    if n_tests == 0:
        return pd.DataFrame(columns=_ORA_COLUMNS)

    rows = []
    for term, members in eligible_terms:
        overlap = genes & members
        overlap_n = len(overlap)
        p_value = hypergeom.sf(
            overlap_n - 1,
            len(universe),
            len(members),
            len(genes),
        )
        expected = len(genes) * len(members) / len(universe)
        rows.append({
            "collection": str(collection),
            "term": term,
            "panel_size": len(genes),
            "set_size": len(members),
            "overlap_n": overlap_n,
            "fold_enrichment": (
                float(overlap_n / expected) if expected > 0 else np.nan
            ),
            "p_value": float(p_value),
            "meets_min_overlap": bool(overlap_n >= min_overlap),
            "min_overlap": min_overlap,
            "n_tests": n_tests,
            "overlap_genes": ";".join(sorted(overlap)),
        })

    out = pd.DataFrame(rows)
    out["fdr"] = _bh_adjust(out["p_value"].to_numpy())
    out = out.loc[:, _ORA_COLUMNS]
    if return_reportable_only:
        out = out.loc[out["meets_min_overlap"]]
    return out.sort_values(["fdr", "p_value", "term"]).reset_index(drop=True)


def enrichment_for_panels(panels: Mapping[str, Iterable[str]], *,
                          universe: Iterable[str],
                          collections: Mapping[str, Mapping[str, set[str]]],
                          min_overlap: int = 2,
                          return_reportable_only: bool = False) -> pd.DataFrame:
    rows = []
    for panel_name, genes in panels.items():
        for collection_name, gene_sets in collections.items():
            result = overrepresentation(
                genes, gene_sets, universe, min_overlap=min_overlap,
                collection=collection_name,
                return_reportable_only=return_reportable_only,
            )
            result.insert(0, "panel", panel_name)
            rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_independent_rf_rankings(X_dev: pd.DataFrame, y_dev: Sequence[str],
                                connected_genes: Iterable[str], *,
                                classes: Sequence[str], seed: int = 1234,
                                n_estimators: int = 1000,
                                n_jobs: int = -1) -> pd.DataFrame:
    """Independent one-vs-rest RF ranking over all connected genes.

    No InterGATE-retained gene list is passed to the forest. The only feature
    restriction is membership in the candidate connected-gene universe.
    """
    genes = [g for g in map(str, connected_genes) if g in X_dev.columns]
    if not genes:
        raise ValueError("No hay genes conectados en X_dev.")
    x = X_dev.loc[:, genes].to_numpy(dtype=np.float32)
    y = np.asarray(y_dev).astype(str)
    rows = []
    for i, class_name in enumerate(classes):
        target = (y == str(class_name)).astype(np.int8)
        if np.unique(target).size != 2:
            raise ValueError(f"No hay positivos/negativos para {class_name}.")
        model = RandomForestClassifier(
            n_estimators=int(n_estimators), class_weight="balanced_subsample",
            max_features="sqrt", min_samples_leaf=2,
            random_state=int(seed) + i, n_jobs=n_jobs, oob_score=True,
        )
        model.fit(x, target)
        order = np.argsort(-model.feature_importances_, kind="stable")
        for rank, idx in enumerate(order, start=1):
            rows.append({
                "class_name": str(class_name), "gene": genes[idx],
                "importance": float(model.feature_importances_[idx]),
                "rank": int(rank), "oob_score": float(model.oob_score_),
                "feature_universe": "all_connected_genes",
            })
    return pd.DataFrame(rows)


def fit_independent_rf_rankings_loco(
    X_dev: pd.DataFrame,
    y_dev: Sequence[str],
    cohorts_dev: Sequence[str],
    connected_genes: Iterable[str],
    *,
    classes: Sequence[str],
    seed: int = 1234,
    n_estimators: int = 500,
    n_repeats: int = 3,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Cohort-rotated permutation-importance ranking over all connected genes.

    Each inner fit excludes one development cohort and permutation importance
    is measured only in that held-out cohort. Ranks, rather than raw scales,
    are averaged across eligible folds. The external test is not an argument.
    """
    genes = [g for g in map(str, connected_genes) if g in X_dev.columns]
    y = np.asarray(y_dev).astype(str)
    cohorts = np.asarray(cohorts_dev).astype(str)
    x = X_dev.loc[:, genes].to_numpy(dtype=np.float32)
    fold_rows = []
    for class_i, class_name in enumerate(classes):
        target = (y == str(class_name)).astype(np.int8)
        for fold_i, held_out in enumerate(sorted(np.unique(cohorts))):
            train = cohorts != held_out
            valid = ~train
            if np.unique(target[train]).size != 2 or np.unique(target[valid]).size != 2:
                continue
            model = RandomForestClassifier(
                n_estimators=int(n_estimators),
                class_weight="balanced_subsample", max_features="sqrt",
                min_samples_leaf=2,
                random_state=int(seed) + 100 * class_i + fold_i,
                n_jobs=n_jobs,
            )
            model.fit(x[train], target[train])
            imp = permutation_importance(
                model, x[valid], target[valid],
                scoring="balanced_accuracy", n_repeats=int(n_repeats),
                random_state=int(seed) + 1000 * class_i + fold_i,
                n_jobs=n_jobs,
            ).importances_mean
            order = np.argsort(-imp, kind="stable")
            rank = np.empty(len(genes), dtype=np.int64)
            rank[order] = np.arange(1, len(genes) + 1)
            fold_rows.extend({
                "class_name": str(class_name), "held_out_cohort": held_out,
                "gene": gene, "permutation_importance": float(imp[j]),
                "fold_rank": int(rank[j]),
            } for j, gene in enumerate(genes))
    folds = pd.DataFrame(fold_rows)
    if folds.empty:
        raise ValueError("Ningún fold LOCO contiene positivos y negativos.")
    summary = (
        folds.groupby(["class_name", "gene"], as_index=False)
        .agg(
            importance=("permutation_importance", "mean"),
            mean_rank=("fold_rank", "mean"),
            rank_sd=("fold_rank", "std"),
            n_folds=("held_out_cohort", "nunique"),
        )
    )
    missing_classes = sorted(set(map(str, classes)) - set(summary["class_name"]))
    if missing_classes:
        raise ValueError(
            "No hay ningún fold LOCO válido para estas clases del RF "
            f"independiente: {missing_classes}. Revisa la composición por cohorte."
        )
    summary = summary.sort_values(
        ["class_name", "mean_rank", "importance", "gene"],
        ascending=[True, True, False, True],
    )
    summary["rank"] = summary.groupby("class_name").cumcount() + 1
    summary["feature_universe"] = "all_connected_genes"
    return summary.reset_index(drop=True)


def size_matched_rf_panels(rankings: pd.DataFrame,
                           target_sizes: Mapping[str, int]) -> dict[str, list[str]]:
    panels = {}
    for class_name, size in target_sizes.items():
        sub = rankings[rankings["class_name"].astype(str) == str(class_name)]
        sub = sub.sort_values(["rank", "gene"])
        if len(sub) < int(size):
            raise ValueError(f"RF no tiene {size} genes para {class_name}.")
        panels[f"RF_{class_name}"] = sub.head(int(size))["gene"].tolist()
    return panels


def matched_random_panels(universe: Iterable[str], sizes: Mapping[str, int], *,
                          n_repeats: int = 1000, seed: int = 1234) -> dict:
    universe = np.asarray(sorted(set(map(str, universe))), dtype=object)
    rng = np.random.default_rng(seed)
    out = {}
    for name, size in sizes.items():
        if int(size) > len(universe):
            raise ValueError(f"Panel {name}: tamaño mayor que el universo.")
        out[name] = [
            sorted(rng.choice(universe, size=int(size), replace=False).tolist())
            for _ in range(int(n_repeats))
        ]
    return out


def panel_overlap_table(panels: Mapping[str, Iterable[str]],
                        universe: Iterable[str]) -> pd.DataFrame:
    universe = set(map(str, universe))
    names = list(panels)
    rows = []
    for i, a in enumerate(names):
        ga = set(map(str, panels[a])) & universe
        for b in names[i + 1:]:
            gb = set(map(str, panels[b])) & universe
            inter, union = ga & gb, ga | gb
            rows.append({
                "panel_a": a, "panel_b": b,
                "n_a": len(ga), "n_b": len(gb), "overlap": len(inter),
                "jaccard": len(inter) / len(union) if union else np.nan,
                "overlap_genes": ";".join(sorted(inter)),
            })
    return pd.DataFrame(rows)


def evaluate_panels_loco(X_dev: pd.DataFrame, y_dev: Sequence[str],
                         cohorts_dev: Sequence[str],
                         panels: Mapping[str, Iterable[str]], *,
                         seed: int = 1234) -> pd.DataFrame:
    """Descriptive post-selection readout under cohort rotations.

    Panels are fixed using the complete development block, so these rotations
    quantify retained signal but are not an unbiased estimate of transfer.
    Confirmatory panel transfer is evaluated only in the locked external block.
    """
    y = np.asarray(y_dev).astype(str)
    cohorts = np.asarray(cohorts_dev).astype(str)
    classes = sorted(np.unique(y))
    rows = []
    for panel_name, panel in panels.items():
        genes = [g for g in map(str, panel) if g in X_dev.columns]
        for fold, held_out in enumerate(sorted(np.unique(cohorts))):
            train = cohorts != held_out
            valid = ~train
            if not genes:
                rows.append({
                    "panel": panel_name, "held_out_cohort": held_out,
                    "n_genes": 0, "n_valid": int(valid.sum()),
                    "accuracy": np.nan, "balanced_accuracy": np.nan,
                    "macro_f1": np.nan, "auc_macro_ovr": np.nan,
                    "status": "empty_stable_core",
                    "role": "post_selection_descriptive",
                })
                continue
            if set(np.unique(y[train])) != set(classes):
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty="elasticnet", solver="saga", l1_ratio=0.5,
                    C=1.0, class_weight="balanced", max_iter=8000,
                    random_state=int(seed) + fold,
                ),
            )
            model.fit(X_dev.loc[train, genes], y[train])
            pred = model.predict(X_dev.loc[valid, genes])
            proba = model.predict_proba(X_dev.loc[valid, genes])
            row = {
                "panel": panel_name, "held_out_cohort": held_out,
                "n_genes": len(genes), "n_valid": int(valid.sum()),
                "accuracy": float(accuracy_score(y[valid], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y[valid], pred)),
                "macro_f1": float(f1_score(y[valid], pred, average="macro")),
                "status": "ok",
                "role": "post_selection_descriptive",
            }
            try:
                y_bin = label_binarize(y[valid], classes=model.classes_)
                row["auc_macro_ovr"] = float(roc_auc_score(
                    y_bin, proba, average="macro", multi_class="ovr",
                ))
            except Exception:
                row["auc_macro_ovr"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _prediction_metrics(y_true, pred, proba, classes) -> dict:
    row = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
    }
    try:
        y_bin = label_binarize(y_true, classes=classes)
        row["auc_macro_ovr"] = float(roc_auc_score(
            y_bin, proba, average="macro", multi_class="ovr",
        ))
    except Exception:
        row["auc_macro_ovr"] = np.nan
    return row


def evaluate_predictive_baselines_loco(
    X_dev: pd.DataFrame, y_dev: Sequence[str], cohorts_dev: Sequence[str],
    connected_genes: Iterable[str], *, seed: int = 1234,
    include_xgboost: bool = True,
) -> pd.DataFrame:
    """Development-only predictive baselines; none is a gene-recovery arm."""
    genes = [g for g in map(str, connected_genes) if g in X_dev.columns]
    y_text = np.asarray(y_dev).astype(str)
    cohorts = np.asarray(cohorts_dev).astype(str)
    classes = np.asarray(sorted(np.unique(y_text)), dtype=object)
    label_map = {name: i for i, name in enumerate(classes)}
    y_int = np.asarray([label_map[v] for v in y_text], dtype=np.int64)
    rows = []
    for fold, held_out in enumerate(sorted(np.unique(cohorts))):
        train, valid = cohorts != held_out, cohorts == held_out
        if set(np.unique(y_text[train])) != set(classes):
            continue
        models = {
            "elastic_net_all_connected": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty="elasticnet", solver="saga", l1_ratio=0.5,
                    C=1.0, class_weight="balanced", max_iter=8000,
                    random_state=seed + fold,
                ),
            ),
            "random_forest_all_connected": RandomForestClassifier(
                n_estimators=1000, class_weight="balanced_subsample",
                max_features="sqrt", min_samples_leaf=2,
                random_state=seed + fold, n_jobs=-1,
            ),
        }
        for name, model in models.items():
            model.fit(X_dev.loc[train, genes], y_text[train])
            pred = model.predict(X_dev.loc[valid, genes])
            proba = model.predict_proba(X_dev.loc[valid, genes])
            rows.append({
                "model": name, "held_out_cohort": held_out,
                "n_genes": len(genes), "role": "predictive_only",
                **_prediction_metrics(y_text[valid], pred, proba, model.classes_),
            })
        if include_xgboost:
            try:
                from xgboost import XGBClassifier
                xgb = XGBClassifier(
                    n_estimators=600, max_depth=4, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.5,
                    objective="multi:softprob", eval_metric="mlogloss",
                    random_state=seed + fold, n_jobs=-1,
                )
                xgb.fit(X_dev.loc[train, genes], y_int[train])
                pred_int = xgb.predict(X_dev.loc[valid, genes]).astype(int)
                proba = xgb.predict_proba(X_dev.loc[valid, genes])
                pred = classes[pred_int]
                rows.append({
                    "model": "xgboost_all_connected",
                    "held_out_cohort": held_out, "n_genes": len(genes),
                    "role": "predictive_only",
                    **_prediction_metrics(y_text[valid], pred, proba, classes),
                })
            except ImportError as exc:
                raise ImportError(
                    "Instala xgboost para ejecutar el baseline preespecificado."
                ) from exc
    return pd.DataFrame(rows)


def rewire_directed_stratified(edges: pd.DataFrame, *, nswap_multiplier: int = 10,
                               seed: int = 1234,
                               source_col: str = "source_gene",
                               target_col: str = "target_gene",
                               type_col: str = "edge_type",
                               sign_col: str = "sign") -> pd.DataFrame:
    """Double-edge swaps preserving in/out degree, edge type and sign exactly."""
    required = {source_col, target_col, type_col, sign_col}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Faltan columnas para rewiring: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    out = edges.copy().reset_index(drop=True)
    original_pairs = list(zip(
        out[source_col].astype(str), out[target_col].astype(str)
    ))
    occupied = set(zip(out[source_col].astype(str), out[target_col].astype(str)))
    total_accepted = 0
    total_attempts = 0
    total_target = 0
    for _, idx in out.groupby([type_col, sign_col], dropna=False).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        if len(idx) < 2:
            continue
        target_swaps = int(nswap_multiplier) * len(idx)
        total_target += target_swaps
        accepted, attempts = 0, 0
        max_attempts = max(100, target_swaps * 50)
        while accepted < target_swaps and attempts < max_attempts:
            attempts += 1
            total_attempts += 1
            i, j = rng.choice(idx, size=2, replace=False)
            a, b = str(out.at[i, source_col]), str(out.at[i, target_col])
            c, d = str(out.at[j, source_col]), str(out.at[j, target_col])
            new1, new2 = (a, d), (c, b)
            if a == d or c == b or new1 == new2:
                continue
            if new1 in occupied or new2 in occupied:
                continue
            occupied.remove((a, b)); occupied.remove((c, d))
            occupied.add(new1); occupied.add(new2)
            out.at[i, target_col] = d
            out.at[j, target_col] = b
            accepted += 1
            total_accepted += 1
    final_pairs = list(zip(
        out[source_col].astype(str), out[target_col].astype(str)
    ))
    out.attrs.update({
        "accepted_swaps": int(total_accepted),
        "attempted_swaps": int(total_attempts),
        "target_swaps": int(total_target),
        "changed_edge_rows": int(sum(
            before != after
            for before, after in zip(original_pairs, final_pairs)
        )),
    })
    return out


def edge_gene_set_coherence(edges: pd.DataFrame,
                            gene_sets: Mapping[str, set[str]], *,
                            source_col: str = "source_gene",
                            target_col: str = "target_gene") -> dict:
    """Fraction/count of retained links whose endpoints share a gene set."""
    sets_by_gene: Dict[str, set[str]] = {}
    for term, members in gene_sets.items():
        for gene in members:
            sets_by_gene.setdefault(str(gene).upper(), set()).add(str(term))
    coherent = []
    n_terms = []
    for src, dst in zip(edges[source_col], edges[target_col]):
        shared = sets_by_gene.get(str(src).upper(), set()) & sets_by_gene.get(
            str(dst).upper(), set()
        )
        coherent.append(bool(shared))
        n_terms.append(len(shared))
    return {
        "n_edges": int(len(edges)),
        "coherent_edges": int(np.sum(coherent)),
        "coherent_fraction": float(np.mean(coherent)) if coherent else np.nan,
        "mean_shared_terms_per_edge": float(np.mean(n_terms)) if n_terms else np.nan,
    }


def rewiring_coherence_null(edges: pd.DataFrame,
                            gene_sets: Mapping[str, set[str]], *,
                            n_repeats: int = 100,
                            nswap_multiplier: int = 10,
                            seed: int = 1234) -> pd.DataFrame:
    observed = edge_gene_set_coherence(edges, gene_sets)
    rows = [{"kind": "observed", "repeat": -1, **observed}]
    for repeat in range(int(n_repeats)):
        rewired = rewire_directed_stratified(
            edges, nswap_multiplier=nswap_multiplier,
            seed=int(seed) + repeat,
        )
        rows.append({
            "kind": "rewired", "repeat": repeat,
            "accepted_swaps": int(rewired.attrs.get("accepted_swaps", 0)),
            "attempted_swaps": int(rewired.attrs.get("attempted_swaps", 0)),
            "target_swaps": int(rewired.attrs.get("target_swaps", 0)),
            "changed_edge_rows": int(rewired.attrs.get("changed_edge_rows", 0)),
            **edge_gene_set_coherence(rewired, gene_sets),
        })
    out = pd.DataFrame(rows)
    null = out.loc[out["kind"] == "rewired", "coherent_fraction"]
    out.attrs["empirical_p"] = float(
        (1 + (null >= observed["coherent_fraction"]).sum()) / (1 + len(null))
    )
    return out


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_protocol(paths: Iterable[str | Path], output_json: str | Path,
                    extra: Mapping | None = None) -> dict:
    payload = {
        "files": {
            str(Path(path)): {"sha256": file_sha256(path),
                              "bytes": Path(path).stat().st_size}
            for path in paths
        },
        "extra": dict(extra or {}),
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return payload
