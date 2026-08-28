"""Shared, read-only helpers for the locked InterGATE post-hoc analyses.

This file is an additive analysis module.  It does not replace or modify any
source file listed in ``protocol_lock_v8.json`` and it never trains a model.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .biology import weak_topology_summary
from .biology_validation import candidate_edge_table, load_gmt


LOCKED_CONTEXT_WEIGHT = 0.35
LOCKED_SUPPORT_WEIGHT = 0.075
LOCKED_TARGET_CLASS = "Basal-like"
LOCKED_SEEDS = (1234, 2025, 7, 99, 31337)
# Exact-K and strict-core size are observed outputs rather than protocol
# constants. Post-hoc analyses infer both quantities from the locked per-seed
# tables and therefore cannot inherit a stale topology.


def safe_name(value: Any) -> str:
    return "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(value)
    )


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_adjust(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def strict_edge_core_from_seed_tables(
    seed_root: str | Path,
    *,
    seeds: Sequence[int] = LOCKED_SEEDS,
    class_name: str = LOCKED_TARGET_CLASS,
    expected_k: int | None = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Return endpoints of identical directed candidate edges in all seeds."""

    seed_root = Path(seed_root)
    seeds = tuple(map(int, seeds))
    if seeds != LOCKED_SEEDS:
        raise ValueError(f"Semillas no bloqueadas: {seeds} != {LOCKED_SEEDS}.")
    frames: list[pd.DataFrame] = []
    inferred_k = None if expected_k is None else int(expected_k)
    safe = safe_name(class_name)
    for seed in seeds:
        path = seed_root / f"seed_{seed}" / f"biology_edges_{safe}.csv"
        if not path.is_file():
            raise FileNotFoundError(str(path))
        frame = pd.read_csv(path)
        required = {
            "candidate_edge_id", "source_gene", "target_gene", "edge_type",
            "prior_weight", "class_name", "biology_score",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: faltan columnas {sorted(missing)}.")
        if inferred_k is None:
            inferred_k = int(len(frame))
        if inferred_k <= 0:
            raise RuntimeError(f"{path}: Exact-K no puede ser cero.")
        if len(frame) != inferred_k:
            raise RuntimeError(
                f"{class_name}/seed={seed}: Exact-K={len(frame)}; "
                f"esperado={inferred_k}."
            )
        if frame.candidate_edge_id.astype(int).nunique() != inferred_k:
            raise RuntimeError(f"{path}: candidate_edge_id no es único.")
        if set(frame.class_name.astype(str)) != {str(class_name)}:
            raise RuntimeError(f"{path}: clase incoherente.")
        frame = frame.copy()
        frame["seed"] = int(seed)
        frames.append(frame)

    edge_sets = [set(frame.candidate_edge_id.astype(int)) for frame in frames]
    strict_ids = set.intersection(*edge_sets)
    stacked = pd.concat(frames, ignore_index=True)

    stable_annotation = (
        stacked.groupby("candidate_edge_id")
        .agg(
            source_gene_n=("source_gene", "nunique"),
            target_gene_n=("target_gene", "nunique"),
            edge_type_n=("edge_type", "nunique"),
            prior_weight_n=("prior_weight", "nunique"),
        )
    )
    bad = stable_annotation[
        stable_annotation[[
            "source_gene_n", "target_gene_n", "edge_type_n", "prior_weight_n",
        ]].max(axis=1).gt(1)
    ]
    if len(bad):
        raise RuntimeError(
            f"La anotación del prior cambió entre semillas en {len(bad)} aristas."
        )

    named_aggregations: dict[str, tuple[str, str]] = {
        "source_gene": ("source_gene", "first"),
        "target_gene": ("target_gene", "first"),
        "edge_type": ("edge_type", "first"),
        "prior_weight": ("prior_weight", "first"),
        "selection_count": ("seed", "nunique"),
        "median_biology_score": ("biology_score", "median"),
    }
    for source_column, output_column in (
        ("differential_logit", "median_differential_logit"),
        ("context_score", "mean_context_score"),
        ("selection_frequency", "mean_selection_frequency"),
    ):
        if source_column in stacked:
            named_aggregations[output_column] = (
                source_column,
                "median" if source_column == "differential_logit" else "mean",
            )

    grouped = (
        stacked.groupby("candidate_edge_id", as_index=False)
        .agg(**named_aggregations)
    )
    strict = grouped[grouped.candidate_edge_id.astype(int).isin(strict_ids)].copy()
    strict["class_name"] = str(class_name)
    strict["strict_edge_5of5"] = True
    strict = strict.sort_values(
        ["median_biology_score", "candidate_edge_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    strict.insert(0, "strict_rank", np.arange(1, len(strict) + 1))
    if not strict.selection_count.eq(len(seeds)).all():
        raise RuntimeError("El núcleo estricto contiene aristas con soporte <5/5.")
    return strict, frames


def edge_sets_pairwise_summary(
    frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    sets = [set(frame.candidate_edge_id.astype(int)) for frame in frames]
    seeds = [int(frame.seed.iloc[0]) for frame in frames]
    rows = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = sets[left] | sets[right]
            rows.append({
                "seed_a": seeds[left],
                "seed_b": seeds[right],
                "intersection": len(sets[left] & sets[right]),
                "union": len(union),
                "edge_jaccard": (
                    len(sets[left] & sets[right]) / len(union)
                    if union else np.nan
                ),
            })
    return pd.DataFrame(rows)


def topology_from_candidate_ids(
    candidate_ids: Iterable[int],
    edge_index: np.ndarray,
) -> dict[str, Any]:
    ids = np.asarray(sorted(set(map(int, candidate_ids))), dtype=np.int64)
    if ids.size == 0:
        return {
            "n_edges": 0, "n_nodes": 0, "n_components": 0,
            "giant_nodes": 0, "giant_fraction": np.nan,
            "dyad_components": 0, "degree_one_nodes": 0,
        }
    return weak_topology_summary(np.asarray(edge_index)[:, ids])


def build_candidates(runtime: Mapping[str, Any]) -> pd.DataFrame:
    candidates = candidate_edge_table(
        runtime["edge_index"], runtime["edge_weight"], runtime["edge_type"],
        runtime["genes"], n_degree_bins=10,
    )
    if candidates.candidate_edge_id.duplicated().any():
        raise RuntimeError("El universo candidato contiene identificadores duplicados.")
    return candidates


def matched_candidate_draw_ids(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    n_draws: int,
    seed: int,
    exclude_selected: bool,
) -> list[np.ndarray]:
    """Draw panels matched on type, sign and directed endpoint-degree bins."""

    selected_ids = set(selected.candidate_edge_id.astype(int))
    lookup = candidates.set_index("candidate_edge_id", drop=False)
    missing = selected_ids - set(lookup.index.astype(int))
    if missing:
        raise ValueError(f"Hay {len(missing)} aristas fuera del prior candidato.")
    strata = lookup.loc[sorted(selected_ids), "null_stratum"].value_counts()
    rng = np.random.default_rng(int(seed))
    draws: list[np.ndarray] = []
    for _ in range(int(n_draws)):
        sampled: list[int] = []
        for stratum, count in strata.items():
            mask = candidates.null_stratum.map(lambda value: value == stratum)
            pool = candidates.loc[mask, "candidate_edge_id"].to_numpy(np.int64)
            if exclude_selected:
                pool = pool[~np.isin(pool, list(selected_ids))]
            if pool.size < int(count):
                raise RuntimeError(
                    f"Estrato {stratum}: pool nulo={pool.size}, "
                    f"requerido={int(count)}."
                )
            sampled.extend(
                rng.choice(pool, size=int(count), replace=False).astype(int).tolist()
            )
        draw = np.asarray(sampled, dtype=np.int64)
        if draw.size != len(selected_ids) or np.unique(draw).size != draw.size:
            raise RuntimeError("El muestreo emparejado no preservó el tamaño exacto.")
        draws.append(draw)
    return draws


def load_locked_signature_resources(
    *,
    results_root: str | Path,
    resource_dir: str | Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]], dict]:
    """Reconstruct exactly the signatures already locked by Notebook 05."""

    results_root = Path(results_root)
    resource_dir = Path(resource_dir)
    manifest_path = (
        results_root / "05_biology_validation" / "locked_signature_manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Falta {manifest_path}. Ejecute antes el notebook 05."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release = str(manifest["msigdb_release"])
    msigdb = resource_dir / "msigdb"
    hallmark_path = msigdb / f"h.all.v{release}.symbols.gmt"
    cgp_path = msigdb / f"c2.cgp.v{release}.symbols.gmt"
    pam50_path = resource_dir / "PAM50_symbols.txt"
    missing = [
        str(path) for path in (hallmark_path, cgp_path, pam50_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Faltan recursos ya utilizados por el notebook 05:\n- "
            + "\n- ".join(missing)
        )
    hallmark = load_gmt(hallmark_path)
    cgp = load_gmt(cgp_path)
    pam50 = {
        line.strip().upper()
        for line in pam50_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    signatures: dict[str, set[str]] = {}
    for class_name, terms in manifest["terms"].items():
        absent = [term for term in terms if term not in cgp]
        if absent:
            raise RuntimeError(
                f"La colección C2 cambió; faltan términos bloqueados: {absent[:5]}."
            )
        members = set().union(*(cgp[term] for term in terms))
        signatures[str(class_name)] = {
            str(gene).upper() for gene in members if str(gene).upper() not in pam50
        }
        expected = int(manifest["sizes_primary"][class_name])
        if len(signatures[str(class_name)]) != expected:
            raise RuntimeError(
                f"Firma {class_name}: tamaño={len(signatures[str(class_name)])}; "
                f"bloqueado={expected}."
            )
    return signatures, hallmark, cgp, manifest


def output_hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    return {
        str(Path(path).name): sha256_file(path)
        for path in paths if Path(path).is_file()
    }
