"""Locked biological validation and candidate-matched topology nulls for V8."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def bh_adjust(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def load_gmt(path: str | Path) -> Dict[str, set[str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    out: Dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                out[str(fields[0])] = {
                    str(gene).strip().upper()
                    for gene in fields[2:]
                    if str(gene).strip()
                }
    if not out:
        raise ValueError(f"GMT vacío: {path}")
    return out


def candidate_edge_table(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    edge_type: np.ndarray,
    genes: Sequence[str],
    *,
    n_degree_bins: int = 10,
) -> pd.DataFrame:
    """Candidate prior with source/type/sign and endpoint-degree strata."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    edge_weight = np.asarray(edge_weight, dtype=float)
    edge_type = np.asarray(edge_type, dtype=np.int64)
    if edge_index.shape[1] != len(edge_weight) or len(edge_weight) != len(edge_type):
        raise ValueError("Prior desalineado.")
    frame = pd.DataFrame({
        "candidate_edge_id": np.arange(edge_index.shape[1], dtype=np.int64),
        "source_idx": edge_index[0],
        "target_idx": edge_index[1],
        "source_gene": [str(genes[int(value)]).upper() for value in edge_index[0]],
        "target_gene": [str(genes[int(value)]).upper() for value in edge_index[1]],
        "prior_weight": edge_weight,
        "edge_type": edge_type,
    })
    frame = frame.loc[frame.source_idx.ne(frame.target_idx)].copy()
    frame["sign"] = np.sign(frame.prior_weight).astype(np.int8)
    out_degree = frame.groupby("source_gene").size()
    in_degree = frame.groupby("target_gene").size()
    frame["source_out_degree"] = frame.source_gene.map(out_degree).astype(int)
    frame["target_in_degree"] = frame.target_gene.map(in_degree).astype(int)
    def _degree_bins(values: pd.Series) -> np.ndarray:
        """Quantile bins that never split equal-degree candidates."""
        array = values.to_numpy(dtype=float)
        if np.unique(array).size <= 1:
            return np.zeros(array.size, dtype=np.int64)
        cuts = np.unique(np.quantile(
            array, np.linspace(0.0, 1.0, int(n_degree_bins) + 1)
        ))
        if cuts.size <= 2:
            return np.zeros(array.size, dtype=np.int64)
        return np.searchsorted(cuts[1:-1], array, side="right").astype(np.int64)

    frame["source_degree_bin"] = _degree_bins(frame.source_out_degree)
    frame["target_degree_bin"] = _degree_bins(frame.target_in_degree)
    frame["null_stratum"] = list(zip(
        frame.edge_type.astype(int),
        frame.sign.astype(int),
        frame.source_degree_bin.astype(int),
        frame.target_degree_bin.astype(int),
    ))
    return frame.reset_index(drop=True)


def sample_candidate_matched_edges(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample without replacement within type/sign/source-degree/target-degree."""
    if "candidate_edge_id" not in selected:
        raise ValueError("selected requiere candidate_edge_id.")
    lookup = candidates.set_index("candidate_edge_id", drop=False)
    missing = set(selected.candidate_edge_id.astype(int)) - set(lookup.index.astype(int))
    if missing:
        raise ValueError(f"Aristas seleccionadas fuera del prior: {len(missing)}")
    selected_with_strata = lookup.loc[
        selected.candidate_edge_id.astype(int), ["null_stratum"]
    ]
    required = selected_with_strata.null_stratum.value_counts()
    sampled_ids: list[int] = []
    for stratum, count in required.items():
        pool = candidates.loc[
            candidates.null_stratum.map(lambda value: value == stratum),
            "candidate_edge_id",
        ].to_numpy(dtype=np.int64)
        if len(pool) < int(count):
            raise RuntimeError(
                f"Estrato {stratum}: pool={len(pool)} < requerido={count}."
            )
        sampled_ids.extend(
            rng.choice(pool, size=int(count), replace=False).astype(int).tolist()
        )
    return lookup.loc[sampled_ids].reset_index(drop=True)


def edge_signature_statistics(
    edges: pd.DataFrame,
    signatures: Mapping[str, Iterable[str]],
) -> pd.DataFrame:
    source = edges.source_gene.astype(str).str.upper().to_numpy()
    target = edges.target_gene.astype(str).str.upper().to_numpy()
    nodes = set(source) | set(target)
    rows = []
    for term, members_raw in signatures.items():
        members = {str(gene).upper() for gene in members_raw}
        source_hit = np.isin(source, list(members))
        target_hit = np.isin(target, list(members))
        rows.extend([
            {
                "term": str(term), "statistic": "both_endpoints",
                "value": float(np.mean(source_hit & target_hit)),
            },
            {
                "term": str(term), "statistic": "at_least_one_endpoint",
                "value": float(np.mean(source_hit | target_hit)),
            },
            {
                "term": str(term), "statistic": "unique_node_fraction",
                "value": float(len(nodes & members) / max(1, len(nodes))),
            },
        ])
    return pd.DataFrame(rows)


def matched_edge_signature_null(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    signatures: Mapping[str, Iterable[str]],
    *,
    n_draws: int = 1000,
    seed: int = 1234,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Empirical enrichment against prior-constrained matched edge samples."""
    observed = edge_signature_statistics(selected, signatures)
    observed.insert(0, "draw", -1)
    observed.insert(0, "kind", "observed")
    rng = np.random.default_rng(int(seed))
    null_rows = []
    for draw in range(int(n_draws)):
        sampled = sample_candidate_matched_edges(selected, candidates, rng=rng)
        values = edge_signature_statistics(sampled, signatures)
        values.insert(0, "draw", int(draw))
        values.insert(0, "kind", "matched_candidate")
        null_rows.append(values)
    distribution = pd.concat([observed, *null_rows], ignore_index=True)
    summary_rows = []
    for (term, statistic), block in distribution.groupby(["term", "statistic"]):
        observed_value = float(block.loc[block.kind.eq("observed"), "value"].iloc[0])
        null = block.loc[block.kind.eq("matched_candidate"), "value"].to_numpy(float)
        summary_rows.append({
            "term": str(term),
            "statistic": str(statistic),
            "observed": observed_value,
            "null_mean": float(np.mean(null)),
            "null_sd": float(np.std(null, ddof=1)) if len(null) > 1 else np.nan,
            "fold_over_null": (
                observed_value / float(np.mean(null))
                if float(np.mean(null)) > 0 else np.inf
            ),
            "empirical_p": float(
                (1 + np.sum(null >= observed_value)) / (1 + len(null))
            ),
            "n_draws": int(len(null)),
        })
    summary = pd.DataFrame(summary_rows)
    summary["fdr"] = bh_adjust(summary.empirical_p)
    return summary.sort_values(["fdr", "empirical_p"]), distribution


def class_specificity_matrix(
    graphs: Mapping[str, pd.DataFrame],
    class_signatures: Mapping[str, Iterable[str]],
) -> pd.DataFrame:
    """Cross every class graph with every locked class annotation."""
    rows = []
    for graph_class, edges in graphs.items():
        signatures = {
            str(annotation_class): class_signatures[annotation_class]
            for annotation_class in class_signatures
        }
        stats = edge_signature_statistics(edges, signatures)
        stats = stats.loc[stats.statistic.eq("unique_node_fraction")]
        for row in stats.itertuples(index=False):
            rows.append({
                "graph_class": str(graph_class),
                "annotation_class": str(row.term),
                "node_fraction": float(row.value),
            })
    table = pd.DataFrame(rows)
    direct_rows = []
    for graph_class in graphs:
        block = table.loc[table.graph_class.eq(str(graph_class))]
        focal = block.loc[block.annotation_class.eq(str(graph_class)), "node_fraction"]
        alternatives = block.loc[
            ~block.annotation_class.eq(str(graph_class)), "node_fraction"
        ]
        direct_rows.append({
            "graph_class": str(graph_class),
            "focal_fraction": float(focal.iloc[0]) if len(focal) else np.nan,
            "best_alternative_fraction": float(alternatives.max()) if len(alternatives) else np.nan,
            "specificity_difference": (
                float(focal.iloc[0] - alternatives.max())
                if len(focal) and len(alternatives) else np.nan
            ),
            "focal_is_largest": bool(
                len(focal) and len(alternatives)
                and float(focal.iloc[0]) > float(alternatives.max())
            ),
        })
    return table.merge(
        pd.DataFrame(direct_rows), on="graph_class", how="left"
    )
