"""Final reviewer-requested analyses for the InterGATE V8 release.

This module contains only post-training analyses.  It never reads model
checkpoints, never refits a classifier and never changes an exported graph.
Its public functions implement:

* directionally coherent subtype-signature unions;
* candidate-matched edge nulls;
* a context-only (q) comparator;
* sensitivity of the exact edge budget to kappa;
* directed degree/type/sign-preserving rewiring; and
* fixed-universe over-representation analysis.

The implementation intentionally avoids NetworkX so that the final analyses
can run with the minimal locked scientific environment.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .biology import (
    _rank_scale,
    combine_biology_scores,
    module_aware_exact_k,
    weak_topology_summary,
)
from .biology_validation import bh_adjust, load_gmt


PRIMARY_SIGNATURE_TERMS: dict[str, tuple[str, ...]] = {
    "LumA": ("SMID_BREAST_CANCER_LUMINAL_A_UP",),
    "LumB": ("SMID_BREAST_CANCER_LUMINAL_B_UP",),
    "HER2-enriched": (
        "LANDIS_ERBB2_BREAST_PRENEOPLASTIC_UP",
        "LANDIS_ERBB2_BREAST_TUMORS_324_UP",
        "LANDIS_ERBB2_BREAST_TUMORS_65_UP",
        "SMID_BREAST_CANCER_ERBB2_UP",
    ),
    # Every term below is basal-high in its original contrast.  In particular,
    # LUMINAL_VS_BASAL_DN is the directionally valid counterpart of the
    # previously used (and directionally incorrect) UP set.
    "Basal-like": (
        "CHARAFE_BREAST_CANCER_BASAL_VS_MESENCHYMAL_UP",
        "CHARAFE_BREAST_CANCER_LUMINAL_VS_BASAL_DN",
        "HUPER_BREAST_BASAL_VS_LUMINAL_UP",
        "SMID_BREAST_CANCER_BASAL_UP",
    ),
}


BASAL_DIRECTIONLESS_SENSITIVITY_TERMS: tuple[str, ...] = (
    "FARMER_BREAST_CANCER_APOCRINE_VS_BASAL",
    "FARMER_BREAST_CANCER_BASAL_VS_LULMINAL",
)


def signature_unions(
    c2_cgp: Mapping[str, set[str]],
    *,
    include_directionless_basal: bool = False,
) -> dict[str, set[str]]:
    """Return class-specific unions after checking every requested term."""
    terms = {key: list(value) for key, value in PRIMARY_SIGNATURE_TERMS.items()}
    if include_directionless_basal:
        terms["Basal-like"].extend(BASAL_DIRECTIONLESS_SENSITIVITY_TERMS)
    missing = sorted({term for values in terms.values() for term in values} - set(c2_cgp))
    if missing:
        raise KeyError(f"MSigDB terms not found: {missing}")
    return {
        class_name: set().union(*(c2_cgp[term] for term in class_terms))
        for class_name, class_terms in terms.items()
    }


def edge_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame.source_gene.astype(str), frame.target_gene.astype(str)))


def jaccard(left: Iterable, right: Iterable) -> float:
    left, right = set(left), set(right)
    union = left | right
    return float(len(left & right) / len(union)) if union else float("nan")


def topology_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Directed diagnostics plus weak-component summaries without NetworkX."""
    if frame.empty:
        return {
            "n_edges": 0, "n_nodes": 0, "n_components": 0,
            "giant_nodes": 0, "giant_fraction": np.nan,
            "dyad_components": 0, "degree_one_nodes": 0,
            "density": np.nan, "reciprocity": np.nan,
            "mean_total_degree": np.nan, "degree_gini": np.nan,
        }
    source = frame.source_gene.astype(str).to_numpy()
    target = frame.target_gene.astype(str).to_numpy()
    genes = sorted(set(source) | set(target))
    gene_id = {gene: index for index, gene in enumerate(genes)}
    edge_index = np.vstack([
        np.fromiter((gene_id[value] for value in source), dtype=np.int64),
        np.fromiter((gene_id[value] for value in target), dtype=np.int64),
    ])
    summary = weak_topology_summary(edge_index)
    pairs = set(zip(source, target))
    reciprocal_rows = sum((target_value, source_value) in pairs
                          for source_value, target_value in pairs)
    degree = Counter(source)
    degree.update(target)
    values = np.asarray(list(degree.values()), dtype=float)
    degree_gini = (
        float(np.abs(values[:, None] - values[None, :]).mean()
              / (2.0 * values.mean()))
        if values.size and values.mean() > 0 else np.nan
    )
    n_nodes = len(genes)
    summary.update({
        "density": float(len(pairs) / (n_nodes * (n_nodes - 1)))
        if n_nodes > 1 else np.nan,
        "reciprocity": float(reciprocal_rows / len(pairs)) if pairs else np.nan,
        "mean_total_degree": float(values.mean()) if values.size else np.nan,
        "degree_gini": degree_gini,
    })
    return summary


def edge_signature_statistics(
    edges: pd.DataFrame,
    signatures: Mapping[str, Iterable[str]],
) -> pd.DataFrame:
    source = edges.source_gene.astype(str).str.upper().to_numpy()
    target = edges.target_gene.astype(str).str.upper().to_numpy()
    nodes = set(source) | set(target)
    rows: list[dict] = []
    for term, raw_members in signatures.items():
        members = {str(gene).upper() for gene in raw_members}
        source_hit = np.isin(source, list(members))
        target_hit = np.isin(target, list(members))
        rows.extend([
            {"term": term, "statistic": "at_least_one_endpoint",
             "value": float(np.mean(source_hit | target_hit))},
            {"term": term, "statistic": "both_endpoints",
             "value": float(np.mean(source_hit & target_hit))},
            {"term": term, "statistic": "unique_node_fraction",
             "value": float(len(nodes & members) / max(1, len(nodes)))},
        ])
    return pd.DataFrame(rows)


def _stratum_codes(candidates: pd.DataFrame) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    columns = ["edge_type", "sign", "source_degree_bin", "target_degree_bin"]
    tuples = pd.MultiIndex.from_frame(candidates[columns])
    codes, _ = pd.factorize(tuples, sort=True)
    pools = {
        int(code): np.flatnonzero(codes == code).astype(np.int64)
        for code in np.unique(codes)
    }
    return codes.astype(np.int64), pools


def matched_signature_null(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    signatures: Mapping[str, Iterable[str]],
    *,
    n_draws: int = 1000,
    seed: int = 1234,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidate null matched by type, sign and endpoint-degree bins.

    Sampling is without replacement within each stratum.  Candidate rows are
    retained in the pool, matching the prespecified primary biological null.
    """
    candidates = candidates.reset_index(drop=True).copy()
    id_to_row = pd.Series(
        np.arange(len(candidates), dtype=np.int64),
        index=candidates.candidate_edge_id.astype(int),
    )
    selected_rows = id_to_row.loc[selected.candidate_edge_id.astype(int)].to_numpy(int)
    codes, pools = _stratum_codes(candidates)
    required = Counter(codes[selected_rows].tolist())
    for code, count in required.items():
        if len(pools[code]) < count:
            raise RuntimeError(f"Null stratum {code}: {len(pools[code])} < {count}")

    observed = edge_signature_statistics(selected, signatures)
    observed.insert(0, "draw", -1)
    observed.insert(0, "kind", "observed")
    rng = np.random.default_rng(int(seed))
    null_rows: list[pd.DataFrame] = []
    for draw in range(int(n_draws)):
        rows = np.concatenate([
            rng.choice(pools[code], size=count, replace=False)
            for code, count in sorted(required.items())
        ])
        values = edge_signature_statistics(candidates.iloc[rows], signatures)
        values.insert(0, "draw", draw)
        values.insert(0, "kind", "matched_candidate")
        null_rows.append(values)
    distribution = pd.concat([observed, *null_rows], ignore_index=True)

    summary_rows: list[dict] = []
    for (term, statistic), block in distribution.groupby(["term", "statistic"]):
        observed_value = float(block.loc[block.kind.eq("observed"), "value"].iloc[0])
        null = block.loc[block.kind.eq("matched_candidate"), "value"].to_numpy(float)
        null_mean = float(null.mean())
        summary_rows.append({
            "term": term,
            "statistic": statistic,
            "observed": observed_value,
            "null_mean": null_mean,
            "null_sd": float(null.std(ddof=1)),
            "fold_over_null": observed_value / null_mean if null_mean > 0 else np.inf,
            "empirical_p": float((1 + np.sum(null >= observed_value)) / (1 + len(null))),
            "n_draws": int(len(null)),
        })
    summary = pd.DataFrame(summary_rows)
    summary["fdr_within_panel"] = bh_adjust(summary.empirical_p)
    return summary, distribution


def class_specificity_matrix(
    graphs: Mapping[str, pd.DataFrame],
    signatures: Mapping[str, Iterable[str]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for graph_class, graph in graphs.items():
        stats = edge_signature_statistics(graph, signatures)
        stats = stats.loc[stats.statistic.eq("unique_node_fraction")]
        for row in stats.itertuples(index=False):
            rows.append({
                "graph_class": graph_class,
                "annotation_class": row.term,
                "node_fraction": float(row.value),
            })
    table = pd.DataFrame(rows)
    summaries: list[dict] = []
    for graph_class in graphs:
        block = table.loc[table.graph_class.eq(graph_class)]
        focal = float(block.loc[
            block.annotation_class.eq(graph_class), "node_fraction"
        ].iloc[0])
        alternatives = block.loc[
            ~block.annotation_class.eq(graph_class), "node_fraction"
        ].to_numpy(float)
        summaries.append({
            "graph_class": graph_class,
            "focal_fraction": focal,
            "best_alternative_fraction": float(alternatives.max()),
            "specificity_difference": float(focal - alternatives.max()),
            "focal_is_largest": bool(focal > alternatives.max()),
        })
    return table.merge(pd.DataFrame(summaries), on="graph_class", how="left")


def select_exact_k(
    candidate_class: pd.DataFrame,
    scores: np.ndarray,
    *,
    k_edges: int,
    module_aware: bool,
) -> pd.DataFrame:
    candidate_class = candidate_class.sort_values("candidate_edge_id").reset_index(drop=True)
    scores = np.asarray(scores, dtype=float)
    if len(candidate_class) != len(scores):
        raise ValueError("Candidate table and score vector are not aligned.")
    eligible = np.flatnonzero(
        candidate_class.source_idx.to_numpy(int)
        != candidate_class.target_idx.to_numpy(int)
    )
    if module_aware:
        ids = module_aware_exact_k(
            np.vstack([
                candidate_class.source_idx.to_numpy(int),
                candidate_class.target_idx.to_numpy(int),
            ]),
            scores,
            k_edges=int(k_edges),
            pool_multiplier=6,
            target_components=12,
            min_component_edges=3,
        )
    else:
        order = eligible[np.lexsort((eligible, -scores[eligible]))]
        ids = order[:int(k_edges)]
    out = candidate_class.iloc[ids].copy()
    out["selection_score"] = scores[ids]
    out = out.sort_values(
        ["selection_score", "candidate_edge_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def aggregate_seed_graphs(
    seed_graphs: Mapping[int, pd.DataFrame],
    *,
    k_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Replicate the locked frequency-first, score-tie-break aggregation."""
    frames = []
    for seed, frame in seed_graphs.items():
        current = frame.copy()
        current["seed"] = int(seed)
        frames.append(current)
    all_edges = pd.concat(frames, ignore_index=True)
    grouped = (
        all_edges.groupby(["source_gene", "target_gene"], as_index=False)
        .agg(
            selection_count=("seed", "nunique"),
            median_biology_score=("biology_score", "median"),
            mean_context_score=("context_score", "mean"),
            mean_selection_frequency=("selection_frequency", "mean"),
            candidate_edge_id=("candidate_edge_id", "min"),
            source_idx=("source_idx", "first"),
            target_idx=("target_idx", "first"),
            prior_weight=("prior_weight", "median"),
            edge_type=("edge_type", "first"),
        )
    )
    n_seeds = len(seed_graphs)
    grouped["selection_frequency_seed"] = grouped.selection_count / float(n_seeds)
    frequency_rank = 2.0 * grouped.selection_frequency_seed.rank(
        method="average", pct=True
    ).to_numpy(float) - 1.0
    score_rank = _rank_scale(grouped.median_biology_score.to_numpy(float))
    aggregate_score = 2.0 * frequency_rank + score_rank
    genes = sorted(set(grouped.source_gene) | set(grouped.target_gene))
    gene_id = {gene: i for i, gene in enumerate(genes)}
    local_edges = np.vstack([
        grouped.source_gene.map(gene_id).to_numpy(int),
        grouped.target_gene.map(gene_id).to_numpy(int),
    ])
    ids = module_aware_exact_k(
        local_edges,
        aggregate_score,
        k_edges=int(k_edges),
        pool_multiplier=6,
        target_components=12,
        min_component_edges=3,
    )
    consensus = grouped.iloc[ids].copy()
    consensus["aggregate_biology_score"] = aggregate_score[ids]
    consensus = consensus.sort_values(
        ["aggregate_biology_score", "selection_count", "candidate_edge_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    consensus.insert(0, "consensus_rank", np.arange(1, len(consensus) + 1))
    strict = grouped.loc[grouped.selection_count.eq(n_seeds)].copy()

    sets = [edge_pairs(frame) for frame in seed_graphs.values()]
    pairwise = [jaccard(sets[i], sets[j])
                for i in range(len(sets)) for j in range(i + 1, len(sets))]
    summary = {
        "k_edges": int(k_edges),
        "n_union_edges": int(len(grouped)),
        "strict_edges": int(len(strict)),
        "strict_nodes": int(len(set(strict.source_gene) | set(strict.target_gene))),
        "mean_pairwise_jaccard": float(np.mean(pairwise)),
        **{f"consensus_{key}": value for key, value in topology_metrics(consensus).items()},
        **{f"strict_{key}": value for key, value in topology_metrics(strict).items()},
    }
    return consensus, strict, summary


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
    collection: str,
    min_overlap: int = 2,
) -> pd.DataFrame:
    universe = {str(gene).upper() for gene in universe}
    panel = {str(gene).upper() for gene in genes} & universe
    eligible = [
        (str(term), {str(gene).upper() for gene in members} & universe)
        for term, members in sorted(gene_sets.items())
    ]
    eligible = [(term, members) for term, members in eligible if members]
    rows: list[dict] = []
    for term, members in eligible:
        overlap = panel & members
        p_value = hypergeom.sf(
            len(overlap) - 1, len(universe), len(members), len(panel)
        )
        expected = len(panel) * len(members) / len(universe)
        rows.append({
            "collection": collection,
            "term": term,
            "panel_size": len(panel),
            "set_size": len(members),
            "overlap_n": len(overlap),
            "fold_enrichment": len(overlap) / expected if expected > 0 else np.nan,
            "p_value": float(p_value),
            "meets_min_overlap": bool(len(overlap) >= int(min_overlap)),
            "min_overlap": int(min_overlap),
            "n_tests": len(eligible),
            "overlap_genes": ";".join(sorted(overlap)),
        })
    out = pd.DataFrame(rows)
    out["fdr"] = bh_adjust(out.p_value)
    return out[_ORA_COLUMNS].sort_values(["fdr", "p_value", "term"])


def enrichment_for_panels(
    panels: Mapping[str, Iterable[str]],
    *,
    universe: Iterable[str],
    collections: Mapping[str, Mapping[str, set[str]]],
    min_overlap: int = 2,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for panel_name, genes in panels.items():
        for collection_name, gene_sets in collections.items():
            result = overrepresentation(
                genes, gene_sets, universe,
                collection=collection_name, min_overlap=min_overlap,
            )
            result.insert(0, "panel", panel_name)
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def hallmark_gene_bits(
    genes: Sequence[str],
    hallmark: Mapping[str, set[str]],
) -> list[int]:
    gene_id = {str(gene).upper(): index for index, gene in enumerate(genes)}
    bits = [0] * len(genes)
    for term_id, members in enumerate(hallmark.values()):
        bit = 1 << term_id
        for gene in members:
            index = gene_id.get(str(gene).upper())
            if index is not None:
                bits[index] |= bit
    return bits


def _coherence(source: np.ndarray, target: np.ndarray, bits: Sequence[int]) -> tuple[int, float]:
    shared = [(bits[int(a)] & bits[int(b)]).bit_count()
              for a, b in zip(source, target)]
    return int(sum(value > 0 for value in shared)), float(np.mean(shared))


def _rewire_one(
    source: np.ndarray,
    target: np.ndarray,
    strata: np.ndarray,
    *,
    rng: np.random.Generator,
    n_nodes: int,
    nswap_multiplier: int,
) -> tuple[np.ndarray, int, int, int]:
    target = target.copy()
    original_target = target.copy()
    occupied = set((source * n_nodes + target).tolist())
    accepted_total = 0
    attempts_total = 0
    target_total = 0
    for stratum in np.unique(strata):
        ids = np.flatnonzero(strata == stratum)
        if len(ids) < 2:
            continue
        target_swaps = int(nswap_multiplier) * len(ids)
        target_total += target_swaps
        accepted = 0
        attempts = 0
        max_attempts = max(100, target_swaps * 50)
        while accepted < target_swaps and attempts < max_attempts:
            attempts += 1
            i, j = rng.choice(ids, size=2, replace=False)
            a, c = int(source[i]), int(source[j])
            b, d = int(target[i]), int(target[j])
            if a == c or b == d or a == d or c == b:
                continue
            old1, old2 = a * n_nodes + b, c * n_nodes + d
            new1, new2 = a * n_nodes + d, c * n_nodes + b
            if new1 == new2 or new1 in occupied or new2 in occupied:
                continue
            occupied.remove(old1)
            occupied.remove(old2)
            occupied.add(new1)
            occupied.add(new2)
            target[i], target[j] = d, b
            accepted += 1
        accepted_total += accepted
        attempts_total += attempts
    changed = int(np.sum(target != original_target))
    return target, accepted_total, attempts_total, changed


def _rewire_graph_worker(payload: dict) -> tuple[str, list[dict]]:
    graph_class = payload["graph_class"]
    source = np.asarray(payload["source"], dtype=np.int64)
    target = np.asarray(payload["target"], dtype=np.int64)
    strata = np.asarray(payload["strata"], dtype=np.int64)
    bits = payload["bits"]
    n_nodes = int(payload["n_nodes"])
    n_repeats = int(payload["n_repeats"])
    nswap_multiplier = int(payload["nswap_multiplier"])
    base_seed = int(payload["seed"])
    coherent_edges, mean_terms = _coherence(source, target, bits)
    rows = [{
        "graph_class": graph_class,
        "kind": "observed",
        "repeat": -1,
        "n_edges": int(len(source)),
        "coherent_edges": coherent_edges,
        "coherent_fraction": coherent_edges / float(len(source)),
        "mean_shared_terms_per_edge": mean_terms,
        "accepted_swaps": np.nan,
        "attempted_swaps": np.nan,
        "target_swaps": np.nan,
        "changed_edge_rows": np.nan,
    }]
    for repeat in range(n_repeats):
        rewired_target, accepted, attempts, changed = _rewire_one(
            source, target, strata,
            rng=np.random.default_rng(base_seed + repeat),
            n_nodes=n_nodes,
            nswap_multiplier=nswap_multiplier,
        )
        coherent_edges, mean_terms = _coherence(source, rewired_target, bits)
        rows.append({
            "graph_class": graph_class,
            "kind": "rewired",
            "repeat": repeat,
            "n_edges": int(len(source)),
            "coherent_edges": coherent_edges,
            "coherent_fraction": coherent_edges / float(len(source)),
            "mean_shared_terms_per_edge": mean_terms,
            "accepted_swaps": accepted,
            "attempted_swaps": attempts,
            "target_swaps": nswap_multiplier * len(source),
            "changed_edge_rows": changed,
        })
    return graph_class, rows


def directed_rewiring_null(
    graphs: Mapping[str, pd.DataFrame],
    hallmark: Mapping[str, set[str]],
    *,
    n_repeats: int = 1000,
    nswap_multiplier: int = 10,
    seed: int = 303000,
    max_workers: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parallel graph-level rewiring with exact in/out/type/sign preservation."""
    all_genes = sorted({
        str(gene).upper()
        for graph in graphs.values()
        for gene in (*graph.source_gene.astype(str), *graph.target_gene.astype(str))
    })
    gene_id = {gene: index for index, gene in enumerate(all_genes)}
    bits = hallmark_gene_bits(all_genes, hallmark)
    payloads = []
    for graph_id, (graph_class, graph) in enumerate(graphs.items()):
        current = graph.copy()
        source = current.source_gene.astype(str).str.upper().map(gene_id).to_numpy(int)
        target = current.target_gene.astype(str).str.upper().map(gene_id).to_numpy(int)
        sign = np.sign(current.prior_weight.to_numpy(float)).astype(int)
        strata, _ = pd.factorize(pd.MultiIndex.from_arrays([
            current.edge_type.to_numpy(int), sign,
        ]), sort=True)
        payloads.append({
            "graph_class": graph_class,
            "source": source,
            "target": target,
            "strata": strata,
            "bits": bits,
            "n_nodes": len(all_genes),
            "n_repeats": int(n_repeats),
            "nswap_multiplier": int(nswap_multiplier),
            "seed": int(seed) + graph_id * 10000,
        })
    with ProcessPoolExecutor(max_workers=min(max_workers, len(payloads))) as executor:
        results = list(executor.map(_rewire_graph_worker, payloads))
    distribution = pd.DataFrame([
        row for _graph_class, rows in results for row in rows
    ])
    summaries: list[dict] = []
    for graph_class, block in distribution.groupby("graph_class"):
        observed = float(block.loc[block.kind.eq("observed"), "coherent_fraction"].iloc[0])
        null = block.loc[block.kind.eq("rewired"), "coherent_fraction"].to_numpy(float)
        summaries.append({
            "graph_class": graph_class,
            "observed_coherent_fraction": observed,
            "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "fold_over_null": observed / float(null.mean()),
            "empirical_p": float((1 + np.sum(null >= observed)) / (1 + len(null))),
            "n_rewirings": int(len(null)),
            "median_accepted_swaps": float(block.accepted_swaps.median()),
            "median_changed_edge_rows": float(block.changed_edge_rows.median()),
        })
    summary = pd.DataFrame(summaries)
    summary["fdr_four_graphs"] = bh_adjust(summary.empirical_p)
    return summary.sort_values("graph_class"), distribution

