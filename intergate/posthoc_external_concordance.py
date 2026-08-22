"""Locked external structural concordance for the Basal-like 5/5 core.

The analysis uses already projected external expression and observed external
labels only after Notebook 07.  No model weights are loaded, no graph is
refitted and no tuning or model selection is performed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .posthoc_biology_common import (
    EXPECTED_FULL_STRICT_EDGES,
    LOCKED_SEEDS,
    LOCKED_TARGET_CLASS,
    bh_adjust,
    build_candidates,
    matched_candidate_draw_ids,
    output_hashes,
    sha256_file,
    strict_edge_core_from_seed_tables,
)


def _edge_correlations(
    X: np.ndarray,
    sample_idx: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Vectorised Pearson correlations for all candidate prior rows."""

    X = np.asarray(X)
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    source = np.asarray(source, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    output = np.zeros(source.size, dtype=np.float64)
    if sample_idx.size < 4:
        return output
    for start in range(0, source.size, int(batch_size)):
        stop = min(source.size, start + int(batch_size))
        left = np.asarray(
            X[np.ix_(sample_idx, source[start:stop])], dtype=np.float64
        )
        right = np.asarray(
            X[np.ix_(sample_idx, target[start:stop])], dtype=np.float64
        )
        left -= left.mean(axis=0, keepdims=True)
        right -= right.mean(axis=0, keepdims=True)
        numerator = np.einsum("ij,ij->j", left, right, optimize=True)
        denominator = np.sqrt(
            np.einsum("ij,ij->j", left, left, optimize=True)
            * np.einsum("ij,ij->j", right, right, optimize=True)
        )
        correlations = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )
        output[start:stop] = np.clip(correlations, -0.999999, 0.999999)
    return output


def _one_vs_rest_differential_effect(
    X: np.ndarray,
    case_idx: np.ndarray,
    rest_idx: np.ndarray,
    edge_index: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    if len(case_idx) < 8 or len(rest_idx) < 8:
        raise ValueError(
            f"Grupo externo insuficiente: casos={len(case_idx)}, "
            f"resto={len(rest_idx)}; mínimo=8."
        )
    source, target = np.asarray(edge_index, dtype=np.int64)
    case_r = _edge_correlations(
        X, case_idx, source, target, batch_size=batch_size
    )
    rest_r = _edge_correlations(
        X, rest_idx, source, target, batch_size=batch_size
    )
    effect = np.arctanh(case_r) - np.arctanh(rest_r)
    variance = 1.0 / (len(case_idx) - 3.0) + 1.0 / (len(rest_idx) - 3.0)
    return effect.astype(np.float32), float(variance)


def compute_external_candidate_effects(
    runtime: Mapping[str, Any],
    *,
    target_class: str = LOCKED_TARGET_CLASS,
    external_cohorts: Sequence[str],
    batch_size: int = 4096,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Basal-like-vs-rest Fisher-z differences in each external cohort."""

    if not bool(runtime.get("external_opened", False)):
        raise RuntimeError("Este análisis requiere el runtime externo de Notebook 07.")
    classes = tuple(map(str, runtime["classes"]))
    if target_class not in classes:
        raise ValueError(f"{target_class} no está en {classes}.")
    target_id = classes.index(target_class)
    X = np.asarray(runtime["X_scaled"], dtype=np.float32)
    y = np.asarray(runtime["y"], dtype=np.int64)
    cohort = np.asarray(runtime["cohort"]).astype(str)
    test_idx = np.asarray(runtime["test_idx"], dtype=np.int64)
    normalized = np.char.upper(np.char.strip(cohort))

    effects: dict[str, np.ndarray] = {}
    variances: dict[str, float] = {}
    count_rows = []
    for cohort_name in external_cohorts:
        cohort_idx = test_idx[
            normalized[test_idx] == str(cohort_name).strip().upper()
        ]
        case_idx = cohort_idx[y[cohort_idx] == target_id]
        rest_idx = cohort_idx[y[cohort_idx] != target_id]
        effect, variance = _one_vs_rest_differential_effect(
            X,
            case_idx,
            rest_idx,
            runtime["edge_index"],
            batch_size=batch_size,
        )
        effects[str(cohort_name)] = effect
        variances[str(cohort_name)] = variance
        count_rows.append({
            "scope": str(cohort_name),
            "n_external": int(len(cohort_idx)),
            "n_basal_like": int(len(case_idx)),
            "n_rest": int(len(rest_idx)),
            "effect_variance": float(variance),
            "effect_se": float(np.sqrt(variance)),
        })

    weights = {
        name: 1.0 / variance for name, variance in variances.items()
    }
    total_weight = float(sum(weights.values()))
    meta_effect = sum(
        effects[name].astype(np.float64) * weights[name] for name in effects
    ) / total_weight
    meta_variance = 1.0 / total_weight
    effects["EXTERNAL_META"] = meta_effect.astype(np.float32)
    variances["EXTERNAL_META"] = float(meta_variance)
    count_rows.append({
        "scope": "EXTERNAL_META",
        "n_external": int(sum(row["n_external"] for row in count_rows)),
        "n_basal_like": int(sum(row["n_basal_like"] for row in count_rows)),
        "n_rest": int(sum(row["n_rest"] for row in count_rows)),
        "effect_variance": float(meta_variance),
        "effect_se": float(np.sqrt(meta_variance)),
    })

    edge_count = np.asarray(runtime["edge_index"]).shape[1]
    table = pd.DataFrame({
        "candidate_edge_id": np.arange(edge_count, dtype=np.int64),
    })
    for name, effect in effects.items():
        table[f"external_effect::{name}"] = effect
        table[f"external_z::{name}"] = effect / np.sqrt(variances[name])
    return table, pd.DataFrame(count_rows)


def load_median_learned_differential(
    run_root: str | Path,
    *,
    target_class: str = LOCKED_TARGET_CLASS,
    seeds: Sequence[int] = LOCKED_SEEDS,
) -> np.ndarray:
    """Read dense post-training deviations; model checkpoints are not loaded."""

    run_root = Path(run_root)
    arrays = []
    for seed in map(int, seeds):
        path = run_root / f"seed_{seed}" / "transport_support_dense.npz"
        if not path.is_file():
            raise FileNotFoundError(str(path))
        with np.load(path, allow_pickle=True) as dense:
            classes = list(map(str, dense["class_names"].tolist()))
            if target_class not in classes:
                raise ValueError(f"{path}: falta {target_class}.")
            class_id = classes.index(target_class)
            arrays.append(
                np.asarray(dense["differential_logits"], dtype=np.float32)[
                    class_id
                ]
            )
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise RuntimeError(f"Dense arrays desalineados: {sorted(shapes)}.")
    return np.median(np.stack(arrays), axis=0).astype(np.float32)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return np.nan
    result = spearmanr(left[valid], right[valid])
    value = getattr(result, "statistic", result[0])
    return float(value) if np.isfinite(value) else np.nan


def _direction_agreement(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right) & (left != 0) & (right != 0)
    if not valid.any():
        return np.nan
    return float(np.mean(np.sign(left[valid]) == np.sign(right[valid])))


def _panel_metrics(
    ids: np.ndarray,
    development_effect: np.ndarray,
    learned_differential: np.ndarray,
    external_effect: np.ndarray,
) -> dict[str, float]:
    ids = np.asarray(ids, dtype=np.int64)
    return {
        "spearman_development_external": _safe_spearman(
            development_effect[ids], external_effect[ids]
        ),
        "sign_agreement_development_external": _direction_agreement(
            development_effect[ids], external_effect[ids]
        ),
        "sign_agreement_learned_external": _direction_agreement(
            learned_differential[ids], external_effect[ids]
        ),
    }


def run_external_structural_concordance(
    runtime: Mapping[str, Any],
    *,
    run_root: str | Path,
    locked_seed_tables: str | Path,
    output_dir: str | Path,
    external_cohorts: Sequence[str],
    protocol_lock: str | Path,
    external_completion: str | Path,
    target_class: str = LOCKED_TARGET_CLASS,
    n_draws: int = 1000,
    random_seed: int = 20260821,
    batch_size: int = 4096,
) -> dict[str, pd.DataFrame]:
    """Run and persist the frozen external concordance analysis."""

    if int(n_draws) < 1000:
        raise ValueError("El análisis post-lock requiere al menos 1.000 nulos.")
    protocol_lock = Path(protocol_lock)
    external_completion = Path(external_completion)
    if not protocol_lock.is_file():
        raise FileNotFoundError(str(protocol_lock))
    if not external_completion.is_file():
        raise FileNotFoundError(
            f"Falta {external_completion}; ejecute primero el notebook 07."
        )
    completion = json.loads(external_completion.read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError("La evaluación externa bloqueada no figura completa.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strict, _ = strict_edge_core_from_seed_tables(
        locked_seed_tables, class_name=target_class
    )
    if len(strict) != EXPECTED_FULL_STRICT_EDGES:
        raise RuntimeError(
            f"Núcleo FULL inesperado: {len(strict)} != "
            f"{EXPECTED_FULL_STRICT_EDGES}."
        )

    candidates = build_candidates(runtime)
    candidate_effects, counts = compute_external_candidate_effects(
        runtime,
        target_class=target_class,
        external_cohorts=external_cohorts,
        batch_size=batch_size,
    )
    development_bundle = runtime.get("biology_evidence")
    if development_bundle is None:
        raise RuntimeError("Falta la evidencia diferencial bloqueada de desarrollo.")
    classes = list(map(str, development_bundle["class_names"].tolist()))
    class_id = classes.index(target_class)
    development_effect = np.asarray(
        development_bundle["meta_effect"], dtype=np.float32
    )[class_id]
    development_z = np.asarray(
        development_bundle["meta_z"], dtype=np.float32
    )[class_id]
    learned = load_median_learned_differential(
        run_root, target_class=target_class
    )
    edge_count = np.asarray(runtime["edge_index"]).shape[1]
    if not (
        len(development_effect) == len(development_z) == len(learned) == edge_count
    ):
        raise RuntimeError("Efectos de desarrollo, logits y prior no están alineados.")

    candidates = candidates.merge(
        candidate_effects, on="candidate_edge_id", how="left", validate="one_to_one"
    )
    candidates["development_meta_effect"] = development_effect[
        candidates.candidate_edge_id.to_numpy(np.int64)
    ]
    candidates["development_meta_z"] = development_z[
        candidates.candidate_edge_id.to_numpy(np.int64)
    ]
    candidates["median_learned_differential"] = learned[
        candidates.candidate_edge_id.to_numpy(np.int64)
    ]

    strict_ids = strict.candidate_edge_id.to_numpy(np.int64)
    draws = matched_candidate_draw_ids(
        strict,
        candidates,
        n_draws=n_draws,
        seed=random_seed,
        exclude_selected=True,
    )
    scopes = [*map(str, external_cohorts), "EXTERNAL_META"]
    distribution_rows = []
    summary_rows = []
    for scope in scopes:
        external = candidate_effects[f"external_effect::{scope}"].to_numpy(
            np.float32
        )
        observed = _panel_metrics(
            strict_ids, development_effect, learned, external
        )
        null_by_metric = {metric: [] for metric in observed}
        for draw_id, draw in enumerate(draws):
            metrics = _panel_metrics(draw, development_effect, learned, external)
            for metric, value in metrics.items():
                null_by_metric[metric].append(float(value))
                distribution_rows.append({
                    "scope": scope,
                    "metric": metric,
                    "kind": "degree_type_sign_matched",
                    "draw": int(draw_id),
                    "value": float(value),
                })
        for metric, observed_value in observed.items():
            null = np.asarray(null_by_metric[metric], dtype=float)
            null = null[np.isfinite(null)]
            empirical_p = float(
                (1 + np.sum(null >= observed_value)) / (1 + len(null))
            )
            summary_rows.append({
                "scope": scope,
                "metric": metric,
                "is_primary_metric": metric == "spearman_development_external",
                "observed": float(observed_value),
                "null_mean": float(np.mean(null)),
                "null_sd": float(np.std(null, ddof=1)),
                "null_q025": float(np.quantile(null, 0.025)),
                "null_q975": float(np.quantile(null, 0.975)),
                "observed_minus_null_mean": float(observed_value - np.mean(null)),
                "standardized_difference": float(
                    (observed_value - np.mean(null)) / np.std(null, ddof=1)
                ) if np.std(null, ddof=1) > 0 else np.nan,
                "empirical_p_upper": empirical_p,
                "n_draws": int(len(null)),
                "matching": (
                    "edge_type+prior_sign+source_out_degree_bin+"
                    "target_in_degree_bin"
                ),
                "selected_edges_excluded_from_null": True,
            })

    summary = pd.DataFrame(summary_rows)
    summary["fdr_all_12_tests"] = bh_adjust(summary.empirical_p_upper)
    summary["fdr_primary_four_scopes"] = np.nan
    primary = summary.is_primary_metric.to_numpy(bool)
    summary.loc[primary, "fdr_primary_four_scopes"] = bh_adjust(
        summary.loc[primary, "empirical_p_upper"]
    )
    distribution = pd.DataFrame(distribution_rows)

    effect_columns = [
        column for column in candidates.columns
        if column.startswith("external_effect::")
        or column.startswith("external_z::")
        or column in {
            "candidate_edge_id", "development_meta_effect",
            "development_meta_z", "median_learned_differential",
            "source_out_degree", "target_in_degree",
            "source_degree_bin", "target_degree_bin", "sign",
        }
    ]
    strict_effects = strict.merge(
        candidates.loc[:, effect_columns],
        on="candidate_edge_id",
        how="left",
        validate="one_to_one",
    )
    per_cohort_sign = []
    for scope in map(str, external_cohorts):
        per_cohort_sign.append(
            np.sign(strict_effects[f"external_effect::{scope}"].to_numpy(float))
            == np.sign(strict_effects.development_meta_effect.to_numpy(float))
        )
    strict_effects["external_cohorts_matching_development_sign"] = np.sum(
        np.vstack(per_cohort_sign), axis=0
    )

    candidate_file = output_dir / "external_candidate_differential_effects.csv.gz"
    strict_file = output_dir / "basal_strict_5of5_external_edge_effects.csv"
    count_file = output_dir / "external_cohort_counts.csv"
    summary_file = output_dir / "external_concordance_matched_null_summary.csv"
    distribution_file = (
        output_dir / "external_concordance_matched_null_distribution.csv.gz"
    )
    candidates.drop(columns=["null_stratum"]).to_csv(
        candidate_file, index=False, compression="gzip"
    )
    strict_effects.to_csv(strict_file, index=False)
    counts.to_csv(count_file, index=False)
    summary.to_csv(summary_file, index=False)
    distribution.to_csv(distribution_file, index=False, compression="gzip")

    manifest_file = output_dir / "external_structural_concordance_manifest.json"
    manifest = {
        "status": "complete",
        "analysis": "locked_external_structural_concordance",
        "target_class": target_class,
        "strict_core_definition": "identical directed candidate edge in all five seeds",
        "n_strict_edges": int(len(strict)),
        "external_cohorts": list(map(str, external_cohorts)),
        "primary_statistic": "spearman_development_external",
        "secondary_statistics": [
            "sign_agreement_development_external",
            "sign_agreement_learned_external",
        ],
        "matched_null_draws": int(n_draws),
        "matching": (
            "edge_type+prior_sign+source_out_degree_bin+target_in_degree_bin"
        ),
        "selected_edges_excluded_from_null": True,
        "model_refit": False,
        "model_weights_loaded": False,
        "hyperparameter_or_graph_selection_on_external": False,
        "external_expression_used_only_for_post_lock_effect_estimation": True,
        "protocol_lock_sha256": sha256_file(protocol_lock),
        "external_completion_sha256": sha256_file(external_completion),
        "output_sha256": output_hashes([
            candidate_file, strict_file, count_file, summary_file,
            distribution_file,
        ]),
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "summary": summary,
        "distribution": distribution,
        "strict_edges": strict_effects,
        "cohort_counts": counts,
    }
