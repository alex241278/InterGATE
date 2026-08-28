"""Matched biological-recovery comparison: XGBoost versus InterGATE.

The module deliberately reuses the five frozen XGBoost classifiers from the
paired predictive benchmark.  It never calls ``fit``.  Gene ranking is locked
on development validation by the signed TreeSHAP contrast for the Basal-like
margin (mean contribution in Basal-like samples minus the mean contribution in
the remaining classes).  The external cohorts are used only after the
development panel has been written to disk.

The common estimand is recovery of genes and biological programmes.  Molecular
edges are not inferred from TreeSHAP and therefore are not treated as an
XGBoost network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .biology_validation import bh_adjust, load_gmt


DEFAULT_CLASSES = ("LumA", "LumB", "HER2-enriched", "Basal-like")
DEFAULT_SEEDS = (1234, 2025, 7, 99, 31337)
TARGET_CLASS = "Basal-like"
PANEL_SIZE = 288
PROTOCOL_ID = "xgb-intergate-basal-biology-locked-v1"

PRIMARY_SIGNATURE_TERMS: dict[str, tuple[str, ...]] = {
    "LumA": ("SMID_BREAST_CANCER_LUMINAL_A_UP",),
    "LumB": ("SMID_BREAST_CANCER_LUMINAL_B_UP",),
    "HER2-enriched": (
        "LANDIS_ERBB2_BREAST_PRENEOPLASTIC_UP",
        "LANDIS_ERBB2_BREAST_TUMORS_324_UP",
        "LANDIS_ERBB2_BREAST_TUMORS_65_UP",
        "SMID_BREAST_CANCER_ERBB2_UP",
    ),
    "Basal-like": (
        "CHARAFE_BREAST_CANCER_BASAL_VS_MESENCHYMAL_UP",
        "CHARAFE_BREAST_CANCER_LUMINAL_VS_BASAL_DN",
        "HUPER_BREAST_BASAL_VS_LUMINAL_UP",
        "SMID_BREAST_CANCER_BASAL_UP",
    ),
}

PRESPECIFIED_PROGRAMS = {
    "Hallmark": (
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_NOTCH_SIGNALING",
        "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
        "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
        "HALLMARK_MYC_TARGETS_V1",
    ),
    "C2_CGP": ("SMID_BREAST_CANCER_BASAL_UP",),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ordered_text_sha256(values: Iterable[Any]) -> str:
    return sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def _file_sha256(path: Path | str) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))


def _runtime_genes(runtime: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact, ordered feature names used by the frozen benchmark."""
    genes = tuple(map(str, runtime["genes"]))
    if len(genes) != 7131:
        raise RuntimeError(f"Se esperaban 7.131 genes; se encontraron {len(genes)}.")
    if len(set(genes)) != len(genes):
        raise RuntimeError("El universo ordenado contiene genes duplicados.")
    _normalized_gene_symbols(genes)
    return genes


def _normalized_gene_symbols(genes: Sequence[str]) -> tuple[str, ...]:
    """Normalize symbols only for biological set operations and reporting."""
    normalized = tuple(str(gene).upper() for gene in genes)
    if len(set(normalized)) != len(normalized):
        raise RuntimeError(
            "La normalización de símbolos génicos genera identificadores duplicados."
        )
    return normalized


def _runtime_subset(
    runtime: Mapping[str, Any], indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    genes = _runtime_genes(runtime)
    X_df = runtime["X_df"]
    missing = [gene for gene in genes if gene not in X_df.columns]
    if missing:
        raise RuntimeError(f"Faltan genes en X_df; primeros={missing[:10]}.")
    indices = np.asarray(indices, dtype=np.int64)
    X = X_df.iloc[indices].loc[:, list(genes)].to_numpy(dtype=np.float32)
    y = np.asarray(runtime["y"], dtype=np.int64)[indices]
    cohort = np.asarray(runtime["cohort"]).astype(str)[indices]
    if not np.all(np.isfinite(X)):
        raise RuntimeError("La matriz para TreeSHAP contiene valores no finitos.")
    return X, y, cohort, genes


def _load_benchmark_manifest(benchmark_root: Path | str) -> dict[str, Any]:
    benchmark_root = Path(benchmark_root)
    path = benchmark_root / "development_benchmark_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "development_models_frozen":
        raise RuntimeError("Los modelos tabulares no constan como congelados.")
    if tuple(payload.get("class_names", ())) != DEFAULT_CLASSES:
        raise RuntimeError(f"Endpoint inesperado: {payload.get('class_names')}.")
    if tuple(map(int, payload.get("seeds", ()))) != DEFAULT_SEEDS:
        raise RuntimeError(f"Semillas inesperadas: {payload.get('seeds')}.")
    if int(payload.get("n_genes", -1)) != 7131:
        raise RuntimeError("El benchmark no contiene los 7.131 genes previstos.")
    return payload


def preflight_frozen_xgboost(
    runtime: Mapping[str, Any], benchmark_root: Path | str
) -> pd.DataFrame:
    """Verify feature order and every frozen XGBoost object before attribution."""
    benchmark_root = Path(benchmark_root)
    manifest = _load_benchmark_manifest(benchmark_root)
    genes = _runtime_genes(runtime)
    if _ordered_text_sha256(genes) != manifest["ordered_gene_sha256"]:
        raise RuntimeError("El orden génico no coincide con el benchmark congelado.")
    rows = []
    for seed in DEFAULT_SEEDS:
        relative = manifest["model_files"][str(seed)]["xgboost"]
        path = benchmark_root / relative
        expected = manifest["model_sha256"][str(seed)]["xgboost"]
        observed = _file_sha256(path) if path.is_file() else None
        rows.append({
            "seed": seed,
            "model_path": str(path),
            "exists": path.is_file(),
            "sha256_matches": observed == expected,
            "expected_sha256": expected,
            "observed_sha256": observed,
        })
    audit = pd.DataFrame(rows)
    if not audit["exists"].all() or not audit["sha256_matches"].all():
        raise RuntimeError("Falta o ha cambiado al menos un XGBoost congelado.")
    return audit


def _reshape_contributions(
    raw: np.ndarray, *, n_rows: int, n_classes: int, n_features: int
) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    expected_last = n_features + 1
    if raw.ndim == 3 and raw.shape == (n_rows, n_classes, expected_last):
        return raw
    if raw.ndim == 2 and raw.shape == (n_rows, n_classes * expected_last):
        return raw.reshape(n_rows, n_classes, expected_last)
    if n_classes == 1 and raw.ndim == 2 and raw.shape == (n_rows, expected_last):
        return raw[:, None, :]
    raise RuntimeError(
        "Forma TreeSHAP inesperada: "
        f"{raw.shape}; esperado {(n_rows, n_classes, expected_last)}."
    )


def _xgboost_scope_scores(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    genes: Sequence[str],
    scopes: Mapping[str, np.ndarray],
    *,
    seed: int,
    target_class_id: int,
    batch_size: int = 256,
    inference_device: str = "cpu",
) -> pd.DataFrame:
    """Exact TreeSHAP summaries without retaining sample-level tensors."""
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("Este análisis requiere xgboost>=2.0,<4.") from exc

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n_rows, n_features = X.shape
    n_classes = len(DEFAULT_CLASSES)
    if n_features != len(genes):
        raise RuntimeError("X y genes no están alineados.")
    scope_masks = {}
    for scope, mask in scopes.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (n_rows,):
            raise ValueError(f"Máscara {scope!r} desalineada: {mask.shape}.")
        scope_masks[str(scope)] = mask

    accumulators: dict[str, dict[str, Any]] = {}
    for scope in scope_masks:
        accumulators[scope] = {
            "sum_abs": np.zeros(n_features, dtype=np.float64),
            "sum_basal": np.zeros(n_features, dtype=np.float64),
            "sum_rest": np.zeros(n_features, dtype=np.float64),
            "n_all": 0,
            "n_basal": 0,
            "n_rest": 0,
        }

    booster = model.get_booster()
    inference_device = str(inference_device).lower()
    if inference_device != "cpu" and not inference_device.startswith("cuda"):
        raise ValueError("inference_device debe ser 'cpu' o comenzar por 'cuda'.")
    booster.set_param({"device": inference_device})
    verified_additivity = False
    for start in range(0, n_rows, int(batch_size)):
        stop = min(n_rows, start + int(batch_size))
        matrix = xgb.DMatrix(X[start:stop])
        raw = booster.predict(
            matrix,
            pred_contribs=True,
            approx_contribs=False,
            strict_shape=True,
            validate_features=False,
        )
        contributions = _reshape_contributions(
            raw,
            n_rows=stop - start,
            n_classes=n_classes,
            n_features=n_features,
        )
        phi = contributions[:, int(target_class_id), :-1]
        if not verified_additivity:
            margins = np.asarray(booster.predict(
                matrix,
                output_margin=True,
                strict_shape=True,
                validate_features=False,
            ))
            if margins.ndim == 1:
                margins = margins[:, None]
            reconstructed = contributions[:, int(target_class_id), :].sum(axis=1)
            if not np.allclose(
                reconstructed,
                margins[:, int(target_class_id)],
                rtol=1e-4,
                atol=1e-4,
            ):
                raise RuntimeError("TreeSHAP no reconstruye el margen XGBoost.")
            verified_additivity = True

        batch_y = y[start:stop]
        for scope, full_mask in scope_masks.items():
            local = full_mask[start:stop]
            if not np.any(local):
                continue
            local_phi = phi[local]
            local_y = batch_y[local]
            basal = local_y == int(target_class_id)
            rest = ~basal
            acc = accumulators[scope]
            acc["sum_abs"] += np.abs(local_phi).sum(axis=0)
            acc["n_all"] += int(len(local_phi))
            if np.any(basal):
                acc["sum_basal"] += local_phi[basal].sum(axis=0)
                acc["n_basal"] += int(np.sum(basal))
            if np.any(rest):
                acc["sum_rest"] += local_phi[rest].sum(axis=0)
                acc["n_rest"] += int(np.sum(rest))

    rows = []
    for scope, acc in accumulators.items():
        if acc["n_all"] == 0 or acc["n_basal"] == 0 or acc["n_rest"] == 0:
            raise RuntimeError(
                f"{scope}: se requieren muestras Basal-like y rest para el contraste."
            )
        mean_abs = acc["sum_abs"] / acc["n_all"]
        mean_basal = acc["sum_basal"] / acc["n_basal"]
        mean_rest = acc["sum_rest"] / acc["n_rest"]
        contrast = mean_basal - mean_rest
        order = np.lexsort((np.asarray(genes, dtype=str), -contrast))
        signed_rank = np.empty(n_features, dtype=np.int64)
        signed_rank[order] = np.arange(1, n_features + 1)
        abs_order = np.lexsort((np.asarray(genes, dtype=str), -mean_abs))
        abs_rank = np.empty(n_features, dtype=np.int64)
        abs_rank[abs_order] = np.arange(1, n_features + 1)
        rows.append(pd.DataFrame({
            "seed": int(seed),
            "scope": str(scope),
            "gene": list(genes),
            "mean_abs_shap": mean_abs,
            "mean_shap_basal": mean_basal,
            "mean_shap_rest": mean_rest,
            "basal_signed_contrast": contrast,
            "rank_signed_contrast": signed_rank,
            "rank_mean_abs_shap": abs_rank,
            "n_samples": int(acc["n_all"]),
            "n_basal": int(acc["n_basal"]),
            "n_rest": int(acc["n_rest"]),
        }))
    return pd.concat(rows, ignore_index=True)


def score_frozen_xgboost(
    runtime: Mapping[str, Any],
    benchmark_root: Path | str,
    *,
    indices: Sequence[int],
    scopes: Mapping[str, np.ndarray],
    batch_size: int = 256,
    inference_device: str = "cpu",
) -> pd.DataFrame:
    """Load the five immutable models and compute scope-specific gene scores."""
    benchmark_root = Path(benchmark_root)
    manifest = _load_benchmark_manifest(benchmark_root)
    audit = preflight_frozen_xgboost(runtime, benchmark_root)
    if not audit["sha256_matches"].all():
        raise RuntimeError("Preflight XGBoost incompleto.")
    X, y, _, feature_names = _runtime_subset(runtime, indices)
    if _ordered_text_sha256(feature_names) != manifest["ordered_gene_sha256"]:
        raise RuntimeError("El orden génico cambió antes de TreeSHAP.")
    gene_symbols = _normalized_gene_symbols(feature_names)
    target_class_id = DEFAULT_CLASSES.index(TARGET_CLASS)
    rows = []
    for seed in DEFAULT_SEEDS:
        relative = manifest["model_files"][str(seed)]["xgboost"]
        path = benchmark_root / relative
        model = joblib.load(path)
        if not hasattr(model, "get_booster"):
            raise TypeError(f"{path} no es un clasificador XGBoost compatible.")
        rows.append(_xgboost_scope_scores(
            model,
            X,
            y,
            gene_symbols,
            scopes,
            seed=seed,
            target_class_id=target_class_id,
            batch_size=batch_size,
            inference_device=inference_device,
        ))
    return pd.concat(rows, ignore_index=True)


def consensus_gene_scores(
    scores: pd.DataFrame,
    *,
    scope: str,
    panel_size: int = PANEL_SIZE,
) -> pd.DataFrame:
    block = scores.loc[scores["scope"].eq(str(scope))].copy()
    observed = tuple(sorted(block["seed"].astype(int).unique()))
    if observed != tuple(sorted(DEFAULT_SEEDS)):
        raise RuntimeError(f"{scope}: semillas encontradas={observed}.")
    if block.duplicated(["seed", "gene"]).any():
        raise RuntimeError(f"{scope}: genes duplicados dentro de una semilla.")
    block["selected_top_n"] = block["rank_signed_contrast"].le(int(panel_size))
    summary = block.groupby("gene", as_index=False).agg(
        mean_rank_signed_contrast=("rank_signed_contrast", "mean"),
        sd_rank_signed_contrast=("rank_signed_contrast", "std"),
        median_rank_signed_contrast=("rank_signed_contrast", "median"),
        mean_basal_signed_contrast=("basal_signed_contrast", "mean"),
        sd_basal_signed_contrast=("basal_signed_contrast", "std"),
        mean_abs_shap=("mean_abs_shap", "mean"),
        selected_seed_count=("selected_top_n", "sum"),
    )
    summary = summary.sort_values(
        ["mean_rank_signed_contrast", "mean_basal_signed_contrast", "gene"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    summary.insert(0, "consensus_rank", np.arange(1, len(summary) + 1))
    summary["size_matched_panel"] = summary["consensus_rank"].le(int(panel_size))
    summary["strict_5of5_panel"] = summary["selected_seed_count"].eq(len(DEFAULT_SEEDS))
    return summary


def cohort_balanced_seed_scores(
    scores: pd.DataFrame,
    *,
    cohort_scope_prefix: str = "cohort::",
    output_scope: str = "development_cohort_balanced",
) -> pd.DataFrame:
    """Give every development cohort equal weight within each model seed."""
    block = scores.loc[
        scores["scope"].astype(str).str.startswith(str(cohort_scope_prefix))
    ].copy()
    cohort_scopes = tuple(sorted(block["scope"].astype(str).unique()))
    if len(cohort_scopes) < 2:
        raise RuntimeError("Se requieren al menos dos cohortes para equilibrar rangos.")
    expected_rows = len(DEFAULT_SEEDS) * len(cohort_scopes)
    counts = block.groupby(["seed", "gene"])["scope"].nunique()
    if not counts.eq(len(cohort_scopes)).all():
        raise RuntimeError("No todos los genes aparecen en todas las cohortes/semillas.")
    summary = block.groupby(["seed", "gene"], as_index=False).agg(
        cohort_mean_rank_signed=("rank_signed_contrast", "mean"),
        cohort_mean_rank_abs=("rank_mean_abs_shap", "mean"),
        mean_abs_shap=("mean_abs_shap", "mean"),
        mean_shap_basal=("mean_shap_basal", "mean"),
        mean_shap_rest=("mean_shap_rest", "mean"),
        basal_signed_contrast=("basal_signed_contrast", "mean"),
        n_samples=("n_samples", "sum"),
        n_basal=("n_basal", "sum"),
        n_rest=("n_rest", "sum"),
    )
    if len(summary) * len(cohort_scopes) != len(block):
        raise RuntimeError(
            f"Agregación por cohorte inesperada: scopes={cohort_scopes}, "
            f"filas={len(block)}, esperadas por bloques={expected_rows}."
        )
    rows = []
    for seed, seed_frame in summary.groupby("seed", sort=True):
        seed_frame = seed_frame.copy()
        genes = seed_frame["gene"].astype(str).to_numpy()
        primary = seed_frame["cohort_mean_rank_signed"].to_numpy(float)
        tie = seed_frame["basal_signed_contrast"].to_numpy(float)
        order = np.lexsort((genes, -tie, primary))
        signed_rank = np.empty(len(seed_frame), dtype=np.int64)
        signed_rank[order] = np.arange(1, len(seed_frame) + 1)
        abs_primary = seed_frame["cohort_mean_rank_abs"].to_numpy(float)
        abs_tie = seed_frame["mean_abs_shap"].to_numpy(float)
        abs_order = np.lexsort((genes, -abs_tie, abs_primary))
        abs_rank = np.empty(len(seed_frame), dtype=np.int64)
        abs_rank[abs_order] = np.arange(1, len(seed_frame) + 1)
        seed_frame["rank_signed_contrast"] = signed_rank
        seed_frame["rank_mean_abs_shap"] = abs_rank
        seed_frame["scope"] = str(output_scope)
        seed_frame["n_cohorts_equal_weight"] = len(cohort_scopes)
        rows.append(seed_frame)
    return pd.concat(rows, ignore_index=True)


def load_intergate_basal_core(
    results_root: Path | str,
    locked_core_file: Path | str | None = None,
) -> tuple[set[str], Path]:
    """Load nodes induced by the 299 directed edges recovered in all 5 runs.

    This is intentionally distinct from the larger set of genes that happen to
    appear somewhere in every seed-specific graph.  If a locked 288-gene CSV is
    supplied, any result-derived edge core must reproduce it exactly.
    """
    results_root = Path(results_root)
    locked_genes: set[str] = set()
    locked_path = Path(locked_core_file) if locked_core_file is not None else None
    if locked_path is not None:
        if not locked_path.is_file():
            raise FileNotFoundError(locked_path)
        locked_frame = pd.read_csv(locked_path)
        gene_column = next(
            (name for name in ("gene", "gene_symbol") if name in locked_frame), None
        )
        if gene_column is None:
            raise KeyError(f"{locked_path} requiere una columna gene o gene_symbol.")
        locked_genes = set(locked_frame[gene_column].astype(str).str.upper())
        if len(locked_genes) != PANEL_SIZE:
            raise RuntimeError(
                f"El panel bloqueado debe contener {PANEL_SIZE} genes; "
                f"hay {len(locked_genes)} en {locked_path}."
            )

    edge_candidates = (
        results_root / "09_ablation_biology_w0075" / "arm_consensus"
        / "FULL" / "biology_consensus_Basal-like.csv",
        results_root / "04_controls" / "locked_biology_score_w0075"
        / "biology_consensus" / "biology_consensus_Basal-like.csv",
    )
    derived_genes: set[str] = set()
    derived_path: Path | None = None
    for candidate in edge_candidates:
        if not candidate.is_file():
            continue
        frame = pd.read_csv(candidate)
        required = {"source_gene", "target_gene", "selection_count"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{candidate} carece de {missing}.")
        strict_edges = frame.loc[
            pd.to_numeric(frame["selection_count"], errors="coerce").eq(5)
        ]
        derived_genes = set(strict_edges["source_gene"].astype(str).str.upper()) | set(
            strict_edges["target_gene"].astype(str).str.upper()
        )
        derived_path = candidate
        if derived_genes:
            break

    if derived_genes and len(derived_genes) != PANEL_SIZE:
        raise RuntimeError(
            f"El núcleo Basal-like estricto debe contener {PANEL_SIZE} genes; "
            f"se encontraron {len(derived_genes)} en {derived_path}."
        )
    if locked_genes and derived_genes and locked_genes != derived_genes:
        raise RuntimeError(
            "El panel bloqueado de 288 genes no coincide con los nodos de las "
            f"aristas 5/5 derivadas de {derived_path}."
        )
    if derived_genes:
        return derived_genes, derived_path  # type: ignore[arg-type]
    if locked_genes:
        return locked_genes, locked_path  # type: ignore[arg-type]
    raise FileNotFoundError(
        "No se localizó el consenso de aristas Basal-like ni un panel bloqueado."
    )


def load_locked_annotations(
    resource_dir: Path | str,
    biology_results_dir: Path | str,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]], dict[str, Any]]:
    resource_dir = Path(resource_dir)
    biology_results_dir = Path(biology_results_dir)
    manifest_path = biology_results_dir / "locked_signature_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release = str(manifest["msigdb_release"])
    hallmark_path = resource_dir / "msigdb" / f"h.all.v{release}.symbols.gmt"
    c2_path = resource_dir / "msigdb" / f"c2.cgp.v{release}.symbols.gmt"
    pam50_path = resource_dir / "PAM50_symbols.txt"
    hallmark = load_gmt(hallmark_path)
    c2 = load_gmt(c2_path)
    pam50 = {
        line.strip().upper()
        for line in pam50_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    terms = manifest.get("terms", PRIMARY_SIGNATURE_TERMS)
    signatures = {}
    for class_name in DEFAULT_CLASSES:
        requested = tuple(terms[class_name])
        missing = [term for term in requested if term not in c2]
        if missing:
            raise KeyError(f"Términos bloqueados ausentes para {class_name}: {missing}.")
        members = set().union(*(c2[term] for term in requested))
        signatures[class_name] = {
            str(gene).upper() for gene in members if str(gene).upper() not in pam50
        }
    return signatures, hallmark, c2, manifest


def _prior_degree(runtime: Mapping[str, Any], n_genes: int) -> np.ndarray:
    edge_index = np.asarray(runtime["edge_index"], dtype=np.int64)
    if edge_index.shape[0] != 2:
        raise RuntimeError("edge_index debe tener forma 2×E.")
    return (
        np.bincount(edge_index[0], minlength=n_genes)
        + np.bincount(edge_index[1], minlength=n_genes)
    ).astype(float)


def development_gene_covariates(runtime: Mapping[str, Any]) -> pd.DataFrame:
    """Train-only covariates for a common matched-gene null."""
    feature_names = _runtime_genes(runtime)
    gene_symbols = _normalized_gene_symbols(feature_names)
    train_idx = np.asarray(runtime["train_idx"], dtype=np.int64)
    X, _, cohorts, _ = _runtime_subset(runtime, train_idx)
    cohort_means = np.vstack([
        X[cohorts == cohort].mean(axis=0)
        for cohort in sorted(np.unique(cohorts))
    ])
    degree = _prior_degree(runtime, len(feature_names))
    return pd.DataFrame({
        "gene": gene_symbols,
        "expression_mean": X.mean(axis=0),
        "expression_sd": X.std(axis=0, ddof=1),
        "cohort_mean_sd": cohort_means.std(axis=0, ddof=1),
        "log1p_prior_degree": np.log1p(degree),
    })


def _matched_gene_draws(
    panel: set[str],
    covariates: pd.DataFrame,
    *,
    n_draws: int,
    random_seed: int,
    n_neighbors: int = 512,
) -> tuple[np.ndarray, pd.DataFrame]:
    genes = covariates["gene"].astype(str).to_numpy()
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    missing = sorted(set(panel) - set(gene_to_index))
    if missing:
        raise RuntimeError(f"Genes del panel fuera del universo: {missing[:10]}.")
    panel_indices = np.asarray([gene_to_index[gene] for gene in sorted(panel)], dtype=int)
    candidate_indices = np.asarray(
        [index for index, gene in enumerate(genes) if gene not in panel], dtype=int
    )
    columns = [
        "expression_mean", "expression_sd", "cohort_mean_sd", "log1p_prior_degree"
    ]
    values = covariates[columns].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(values)
    k = min(int(n_neighbors), len(candidate_indices))
    neighbours = NearestNeighbors(n_neighbors=k, algorithm="auto")
    neighbours.fit(scaled[candidate_indices])
    distances, local_indices = neighbours.kneighbors(scaled[panel_indices])
    neighbour_indices = candidate_indices[local_indices]
    rng = np.random.default_rng(int(random_seed))
    draws = np.empty((int(n_draws), len(panel_indices)), dtype=np.int64)
    for draw_id in range(int(n_draws)):
        used = np.zeros(len(genes), dtype=bool)
        selected = np.empty(len(panel_indices), dtype=np.int64)
        for target_position in rng.permutation(len(panel_indices)):
            candidates = neighbour_indices[target_position]
            available = candidates[~used[candidates]]
            if len(available) == 0:
                available = candidate_indices[~used[candidate_indices]]
                if len(available) == 0:
                    raise RuntimeError("No quedan genes disponibles para el nulo emparejado.")
                distance = np.square(
                    scaled[available] - scaled[panel_indices[target_position]]
                ).sum(axis=1)
                available = available[np.argsort(distance, kind="stable")[:64]]
            pool = available[: min(64, len(available))]
            chosen = int(rng.choice(pool))
            selected[target_position] = chosen
            used[chosen] = True
        draws[draw_id] = selected

    panel_mean = values[panel_indices].mean(axis=0)
    null_means = values[draws].mean(axis=1)
    universe_sd = values.std(axis=0, ddof=1)
    balance = pd.DataFrame({
        "covariate": columns,
        "panel_mean": panel_mean,
        "matched_null_mean": null_means.mean(axis=0),
        "matched_null_sd_across_draws": null_means.std(axis=0, ddof=1),
        "standardized_mean_difference": np.divide(
            panel_mean - null_means.mean(axis=0),
            universe_sd,
            out=np.zeros_like(panel_mean),
            where=universe_sd > 0,
        ),
    })
    return draws, balance


def signature_recovery_tables(
    panels: Mapping[str, Iterable[str]],
    signatures: Mapping[str, Iterable[str]],
    covariates: pd.DataFrame,
    *,
    n_draws: int = 1000,
    random_seed: int = 20260828,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Common hypergeometric and expression/degree-matched signature tests."""
    universe = set(covariates["gene"].astype(str))
    gene_array = covariates["gene"].astype(str).to_numpy()
    rows, distribution_rows, balances = [], [], []
    for panel_id, (panel_name, raw_panel) in enumerate(panels.items()):
        panel = {str(gene).upper() for gene in raw_panel} & universe
        if not panel:
            continue
        draws, balance = _matched_gene_draws(
            panel,
            covariates,
            n_draws=int(n_draws),
            random_seed=int(random_seed) + 1000 * panel_id,
        )
        balance.insert(0, "panel", str(panel_name))
        balances.append(balance)
        for class_name, raw_signature in signatures.items():
            signature = {str(gene).upper() for gene in raw_signature} & universe
            observed = len(panel & signature)
            expected = len(panel) * len(signature) / len(universe)
            p_hypergeom = hypergeom.sf(
                observed - 1, len(universe), len(signature), len(panel)
            )
            hit = np.isin(gene_array, list(signature))
            null_overlap = hit[draws].sum(axis=1).astype(int)
            null_mean = float(null_overlap.mean())
            rows.append({
                "panel": str(panel_name),
                "annotation_class": str(class_name),
                "panel_size": len(panel),
                "signature_size": len(signature),
                "overlap_n": int(observed),
                "overlap_fraction": observed / len(panel),
                "hypergeom_expected": float(expected),
                "fold_over_hypergeom_expected": (
                    observed / expected if expected > 0 else np.nan
                ),
                "hypergeom_p": float(p_hypergeom),
                "matched_null_mean_overlap": null_mean,
                "fold_over_matched_null": (
                    observed / null_mean if null_mean > 0 else np.inf
                ),
                "matched_empirical_p": float(
                    (1 + np.sum(null_overlap >= observed)) / (1 + len(null_overlap))
                ),
                "n_matched_draws": int(len(null_overlap)),
                "overlap_genes": ";".join(sorted(panel & signature)),
            })
            distribution_rows.append(pd.DataFrame({
                "panel": str(panel_name),
                "annotation_class": str(class_name),
                "draw": np.arange(len(null_overlap), dtype=int),
                "overlap_n": null_overlap,
            }))
    summary = pd.DataFrame(rows)
    summary["hypergeom_fdr_global"] = bh_adjust(summary["hypergeom_p"])
    summary["matched_fdr_global"] = bh_adjust(summary["matched_empirical_p"])
    distribution = pd.concat(distribution_rows, ignore_index=True)
    balance = pd.concat(balances, ignore_index=True)
    return summary, distribution, balance


def specificity_summary(signature_recovery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel, block in signature_recovery.groupby("panel", sort=False):
        basal = block.loc[block["annotation_class"].eq(TARGET_CLASS)].iloc[0]
        other = block.loc[~block["annotation_class"].eq(TARGET_CLASS)]
        rows.append({
            "panel": panel,
            "basal_fold_over_hypergeom_expected": basal["fold_over_hypergeom_expected"],
            "basal_fold_over_matched_null": basal["fold_over_matched_null"],
            "basal_matched_fdr_global": basal["matched_fdr_global"],
            "max_non_basal_fold_over_hypergeom_expected": other[
                "fold_over_hypergeom_expected"
            ].max(),
            "basal_specificity_margin": (
                basal["fold_over_hypergeom_expected"]
                - other["fold_over_hypergeom_expected"].max()
            ),
        })
    return pd.DataFrame(rows)


def matched_null_panel_contrast(
    signature_recovery: pd.DataFrame,
    null_distribution: pd.DataFrame,
    *,
    left_panel: str = "InterGATE_strict_5of5",
    right_panel: str = "XGBoost_TreeSHAP_top288",
) -> pd.DataFrame:
    """Covariate-matched descriptive contrast between equal-size panels.

    The statistic is derived from integer overlap-count differences before
    scaling by panel size.  Computing it by subtracting two independently
    rounded fractions can move an observed value by one floating-point ULP and
    incorrectly exclude exact ties from an empirical tail probability.
    """
    rows = []
    for class_name in DEFAULT_CLASSES:
        observed = signature_recovery.loc[
            signature_recovery["panel"].isin([left_panel, right_panel])
            & signature_recovery["annotation_class"].eq(class_name)
        ].set_index("panel")
        if set(observed.index) != {left_panel, right_panel}:
            continue
        if observed.loc[left_panel, "panel_size"] != observed.loc[right_panel, "panel_size"]:
            raise RuntimeError("El contraste directo requiere paneles del mismo tamaño.")
        panel_size = int(observed.loc[left_panel, "panel_size"])
        left_overlap = int(observed.loc[left_panel, "overlap_n"])
        right_overlap = int(observed.loc[right_panel, "overlap_n"])
        observed_count_difference = left_overlap - right_overlap
        observed_difference = observed_count_difference / panel_size
        block = null_distribution.loc[
            null_distribution["panel"].isin([left_panel, right_panel])
            & null_distribution["annotation_class"].eq(class_name)
        ]
        pivot = block.pivot(index="draw", columns="panel", values="overlap_n").dropna()
        null_count_difference = (
            pivot[left_panel].to_numpy(np.int64)
            - pivot[right_panel].to_numpy(np.int64)
        )
        null_difference = null_count_difference / panel_size
        null_mean = float(null_difference.mean())
        rows.append({
            "annotation_class": class_name,
            "left_panel": left_panel,
            "right_panel": right_panel,
            "panel_size": panel_size,
            "left_overlap_n": left_overlap,
            "right_overlap_n": right_overlap,
            "observed_fraction_difference_left_minus_right": observed_difference,
            "matched_null_mean_difference": null_mean,
            "empirical_p_left_greater": float(
                (1 + np.sum(null_count_difference >= observed_count_difference))
                / (1 + len(null_count_difference))
            ),
            "empirical_p_two_sided": float(
                (
                    1
                    + np.sum(
                        np.abs(null_difference - null_mean)
                        >= abs(observed_difference - null_mean)
                    )
                )
                / (1 + len(null_difference))
            ),
            "n_matched_draws": int(len(null_difference)),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_left_greater_four_signatures"] = bh_adjust(
            result["empirical_p_left_greater"]
        )
        result["fdr_two_sided_four_signatures"] = bh_adjust(
            result["empirical_p_two_sided"]
        )
    return result


def panel_overlap_table(panels: Mapping[str, Iterable[str]], universe: Iterable[str]) -> pd.DataFrame:
    universe = {str(gene).upper() for gene in universe}
    names = list(panels)
    rows = []
    for left_id, left_name in enumerate(names):
        left = {str(gene).upper() for gene in panels[left_name]} & universe
        for right_name in names[left_id + 1 :]:
            right = {str(gene).upper() for gene in panels[right_name]} & universe
            overlap = left & right
            union = left | right
            rows.append({
                "panel_left": left_name,
                "panel_right": right_name,
                "n_left": len(left),
                "n_right": len(right),
                "overlap_n": len(overlap),
                "jaccard": len(overlap) / len(union) if union else np.nan,
                "hypergeom_p": float(hypergeom.sf(
                    len(overlap) - 1, len(universe), len(left), len(right)
                )),
                "overlap_genes": ";".join(sorted(overlap)),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr"] = bh_adjust(out["hypergeom_p"])
    return out


def _overrepresentation(
    panel: Iterable[str],
    collection: Mapping[str, Iterable[str]],
    universe: Iterable[str],
    *,
    collection_name: str,
    min_overlap: int = 2,
) -> pd.DataFrame:
    """Complete-family one-sided ORA with BH correction within collection."""
    universe = {str(gene).upper() for gene in universe}
    panel = {str(gene).upper() for gene in panel} & universe
    rows = []
    for term, raw_members in sorted(collection.items(), key=lambda item: str(item[0])):
        members = {str(gene).upper() for gene in raw_members} & universe
        if not members:
            continue
        overlap = panel & members
        expected = len(panel) * len(members) / len(universe)
        rows.append({
            "collection": str(collection_name),
            "term": str(term),
            "panel_size": len(panel),
            "set_size": len(members),
            "overlap_n": len(overlap),
            "fold_enrichment": len(overlap) / expected if expected > 0 else np.nan,
            "p_value": float(hypergeom.sf(
                len(overlap) - 1, len(universe), len(members), len(panel)
            )),
            "meets_min_overlap": bool(len(overlap) >= int(min_overlap)),
            "min_overlap": int(min_overlap),
            "overlap_genes": ";".join(sorted(overlap)),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["n_tests"] = len(result)
    result["fdr"] = bh_adjust(result["p_value"])
    return result.sort_values(["fdr", "p_value", "term"]).reset_index(drop=True)


def programme_ora(
    panels: Mapping[str, Iterable[str]],
    genes: Sequence[str],
    hallmark: Mapping[str, set[str]],
    c2: Mapping[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for panel_name, panel in panels.items():
        for collection_name, collection in (
            ("Hallmark", hallmark), ("C2_CGP", c2)
        ):
            result = _overrepresentation(
                panel,
                collection,
                genes,
                collection_name=collection_name,
                min_overlap=2,
            )
            result.insert(0, "panel", str(panel_name))
            rows.append(result)
    complete = pd.concat(rows, ignore_index=True)
    requested = {
        (collection, term)
        for collection, terms in PRESPECIFIED_PROGRAMS.items()
        for term in terms
    }
    primary = complete.loc[
        [
            (str(collection), str(term)) in requested
            for collection, term in zip(complete["collection"], complete["term"])
        ]
    ].copy()
    primary["significant_fdr_005"] = (
        primary["meets_min_overlap"].astype(bool) & primary["fdr"].lt(0.05)
    )
    return complete, primary


def external_transportability(
    development_consensus: pd.DataFrame,
    external_scores: pd.DataFrame,
    *,
    panel_size: int = PANEL_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = development_consensus.set_index("gene")
    locked_panel = set(
        development_consensus.loc[
            development_consensus["size_matched_panel"], "gene"
        ].astype(str)
    )
    rows, external_consensus_rows = [], []
    for scope in external_scores["scope"].drop_duplicates():
        consensus = consensus_gene_scores(
            external_scores, scope=str(scope), panel_size=int(panel_size)
        )
        consensus.insert(0, "scope", str(scope))
        external_consensus_rows.append(consensus)
        ext = consensus.set_index("gene").reindex(development.index)
        all_rho = spearmanr(
            development["mean_basal_signed_contrast"],
            ext["mean_basal_signed_contrast"],
            nan_policy="omit",
        ).statistic
        dev_top = development.loc[sorted(locked_panel)]
        ext_top = ext.loc[sorted(locked_panel)]
        top_rho = spearmanr(
            dev_top["mean_basal_signed_contrast"],
            ext_top["mean_basal_signed_contrast"],
            nan_policy="omit",
        ).statistic
        dev_sign = np.sign(dev_top["mean_basal_signed_contrast"].to_numpy(float))
        ext_sign = np.sign(ext_top["mean_basal_signed_contrast"].to_numpy(float))
        informative = (dev_sign != 0) | (ext_sign != 0)
        sign_agreement = (
            float(np.mean(dev_sign[informative] == ext_sign[informative]))
            if np.any(informative) else np.nan
        )
        external_top = set(consensus.loc[consensus["size_matched_panel"], "gene"])
        rows.append({
            "scope": str(scope),
            "n_samples": int(external_scores.loc[
                external_scores["scope"].eq(scope), "n_samples"
            ].iloc[0]),
            "spearman_all_genes": float(all_rho),
            "spearman_locked_top288": float(top_rho),
            "sign_agreement_locked_top288": sign_agreement,
            "external_top288_overlap_n": len(locked_panel & external_top),
            "external_top288_jaccard": (
                len(locked_panel & external_top) / len(locked_panel | external_top)
            ),
        })
    return pd.DataFrame(rows), pd.concat(external_consensus_rows, ignore_index=True)


def _make_comparison_figure(
    signature_recovery: pd.DataFrame,
    primary_programmes: pd.DataFrame,
    transportability: pd.DataFrame,
    overlap: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    primary_panels = ["InterGATE_strict_5of5", "XGBoost_TreeSHAP_top288"]
    colors = {
        "InterGATE_strict_5of5": "#007C6C",
        "XGBoost_TreeSHAP_top288": "#D55E00",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    ax = axes[0, 0]
    block = signature_recovery.loc[signature_recovery["panel"].isin(primary_panels)]
    classes = list(DEFAULT_CLASSES)
    x = np.arange(len(classes))
    width = 0.36
    for offset, panel in zip((-width / 2, width / 2), primary_panels):
        values = block.loc[block["panel"].eq(panel)].set_index("annotation_class").reindex(classes)
        ax.bar(
            x + offset,
            values["fold_over_hypergeom_expected"],
            width,
            label=panel.replace("_", " "),
            color=colors[panel],
        )
    ax.axhline(1.0, color="#555555", lw=1, ls="--")
    ax.set_xticks(x, classes, rotation=20, ha="right")
    ax.set_ylabel("Fold over hypergeometric expectation")
    ax.set_title("A  Locked class-signature recovery")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    programmes = primary_programmes.loc[
        primary_programmes["panel"].isin(primary_panels)
    ].copy()
    programmes["short_term"] = programmes["term"].str.replace(
        "HALLMARK_", "", regex=False
    ).str.replace("SMID_BREAST_CANCER_", "", regex=False)
    terms = list(dict.fromkeys(programmes["short_term"]))
    y = np.arange(len(terms))
    for panel, marker in zip(primary_panels, ("o", "s")):
        values = programmes.loc[programmes["panel"].eq(panel)].set_index(
            "short_term"
        ).reindex(terms)
        size = 24 + 10 * values["overlap_n"].fillna(0).to_numpy(float)
        significant = values["significant_fdr_005"].fillna(False).to_numpy(bool)
        ax.scatter(
            values["fold_enrichment"], y,
            s=size,
            marker=marker,
            facecolors="white",
            edgecolors=colors[panel],
            linewidths=1.2,
            alpha=0.86,
            label=panel.replace("_", " "),
        )
        if np.any(significant):
            ax.scatter(
                values["fold_enrichment"].to_numpy(float)[significant],
                y[significant],
                s=size[significant],
                marker=marker,
                facecolors=colors[panel],
                edgecolors=colors[panel],
                linewidths=1.2,
                alpha=0.86,
            )
    ax.axvline(1.0, color="#555555", lw=1, ls="--")
    ax.set_yticks(y, terms)
    ax.invert_yaxis()
    ax.set_xlabel("Fold enrichment")
    ax.set_title("B  Prespecified Basal-like programmes")
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.99, 0.02, "Filled symbols: FDR < 0.05",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
    )

    ax = axes[1, 0]
    transport = transportability.copy()
    x = np.arange(len(transport))
    ax.bar(
        x - 0.18,
        transport["spearman_locked_top288"],
        0.36,
        label="Spearman",
        color="#56B4E9",
    )
    ax.bar(
        x + 0.18,
        transport["sign_agreement_locked_top288"],
        0.36,
        label="Sign agreement",
        color="#CC79A7",
    )
    scope_labels = transport["scope"].replace({
        "external_pooled": "Pooled external",
        "cohort::GSE1456": "GSE1456",
        "cohort::GSE21653": "GSE21653",
        "cohort::TCGA-BRCA": "TCGA-BRCA",
    })
    ax.set_xticks(x, scope_labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rank correlation / sign fraction")
    ax.set_title("C  External transportability of TreeSHAP")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    target = overlap.loc[
        overlap["panel_left"].isin(primary_panels)
        & overlap["panel_right"].isin(primary_panels)
    ]
    if target.empty:
        target = overlap.iloc[[0]]
    row = target.iloc[0]
    shared = int(row["overlap_n"])
    counts = [int(row["n_left"]) - shared, shared, int(row["n_right"]) - shared]
    ax.bar(
        [0, 1, 2], counts,
        color=[colors[primary_panels[0]], "#999999", colors[primary_panels[1]]],
    )
    ax.set_xticks([0, 1, 2], ["InterGATE only", "Shared", "XGBoost only"])
    ax.set_ylabel("Genes")
    ax.set_title("D  Overlap of the two 288-gene panels")
    for index, value in enumerate(counts):
        ax.text(index, value, str(value), ha="center", va="bottom")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("XGBoost versus InterGATE: Basal-like biological recovery", y=1.01)
    fig.tight_layout()
    pdf = output_dir / "Figure_XGBoost_vs_InterGATE_biology.pdf"
    png = output_dir / "Figure_XGBoost_vs_InterGATE_biology.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def run_development_comparison(
    runtime: Mapping[str, Any],
    *,
    benchmark_root: Path | str,
    results_root: Path | str,
    biology_results_dir: Path | str,
    resource_dir: Path | str,
    output_dir: Path | str,
    intergate_core_file: Path | str | None = None,
    n_matched_draws: int = 1000,
    random_seed: int = 20260828,
    batch_size: int = 256,
    tree_shap_device: str = "cpu",
) -> dict[str, Any]:
    """Lock XGBoost development ranking and perform the common biology tests."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protected = output_dir / "DEVELOPMENT_LOCK.json"
    if protected.exists():
        raise FileExistsError(
            f"La fase de desarrollo ya está bloqueada: {protected}. "
            "Use otra carpeta para una repetición deliberada."
        )
    if bool(runtime.get("external_opened", True)):
        raise RuntimeError("La fase de ranking requiere include_external=False.")

    audit = preflight_frozen_xgboost(runtime, benchmark_root)
    audit.to_csv(output_dir / "frozen_xgboost_hash_audit.csv", index=False)
    val_idx = np.asarray(runtime["val_idx"], dtype=np.int64)
    _, _, validation_cohort, _ = _runtime_subset(runtime, val_idx)
    development_scopes = {
        "development_pooled": np.ones(len(val_idx), dtype=bool),
        **{
            f"cohort::{name}": validation_cohort == name
            for name in sorted(np.unique(validation_cohort))
        },
    }
    raw_development_scores = score_frozen_xgboost(
        runtime,
        benchmark_root,
        indices=val_idx,
        scopes=development_scopes,
        batch_size=batch_size,
        inference_device=tree_shap_device,
    )
    balanced_development_scores = cohort_balanced_seed_scores(
        raw_development_scores,
        output_scope="development_cohort_balanced",
    )
    development_scores = pd.concat(
        [raw_development_scores, balanced_development_scores], ignore_index=True
    )
    development_scores_path = output_dir / "xgboost_development_gene_scores_by_seed.csv.gz"
    development_scores.to_csv(development_scores_path, index=False, compression="gzip")
    consensus = consensus_gene_scores(
        development_scores, scope="development_cohort_balanced", panel_size=PANEL_SIZE
    )
    consensus_path = output_dir / "xgboost_development_gene_consensus.csv"
    consensus.to_csv(consensus_path, index=False)
    pooled_consensus = consensus_gene_scores(
        development_scores, scope="development_pooled", panel_size=PANEL_SIZE
    )
    pooled_consensus_path = (
        output_dir / "xgboost_development_gene_consensus_pooled_sensitivity.csv"
    )
    pooled_consensus.to_csv(pooled_consensus_path, index=False)
    primary_panel_for_sensitivity = set(
        consensus.loc[consensus["size_matched_panel"], "gene"]
    )
    pooled_panel_for_sensitivity = set(
        pooled_consensus.loc[pooled_consensus["size_matched_panel"], "gene"]
    )
    rank_sensitivity = pd.DataFrame([{
        "comparison": "cohort_balanced_vs_sample_pooled",
        "spearman_all_gene_consensus_ranks": float(spearmanr(
            consensus.sort_values("gene")["consensus_rank"],
            pooled_consensus.sort_values("gene")["consensus_rank"],
        ).statistic),
        "top288_overlap_n": len(
            primary_panel_for_sensitivity & pooled_panel_for_sensitivity
        ),
        "top288_jaccard": (
            len(primary_panel_for_sensitivity & pooled_panel_for_sensitivity)
            / len(primary_panel_for_sensitivity | pooled_panel_for_sensitivity)
        ),
    }])
    rank_sensitivity.to_csv(
        output_dir / "xgboost_cohort_weighting_sensitivity.csv", index=False
    )

    xgb_panel = set(consensus.loc[consensus["size_matched_panel"], "gene"])
    xgb_strict = set(consensus.loc[consensus["strict_5of5_panel"], "gene"])
    intergate_panel, intergate_source = load_intergate_basal_core(
        results_root, locked_core_file=intergate_core_file
    )
    feature_names = _runtime_genes(runtime)
    gene_symbols = _normalized_gene_symbols(feature_names)
    if not (xgb_panel | xgb_strict | intergate_panel).issubset(set(gene_symbols)):
        raise RuntimeError("Al menos un panel contiene genes fuera del universo común.")
    panels = {
        "InterGATE_strict_5of5": intergate_panel,
        "XGBoost_TreeSHAP_top288": xgb_panel,
        "XGBoost_TreeSHAP_strict_5of5": xgb_strict,
    }
    panel_rows = [
        {"panel": name, "gene": gene}
        for name, members in panels.items()
        for gene in sorted(members)
    ]
    pd.DataFrame(panel_rows).to_csv(output_dir / "locked_gene_panels.csv", index=False)

    signatures, hallmark, c2, annotation_manifest = load_locked_annotations(
        resource_dir, biology_results_dir
    )
    covariates = development_gene_covariates(runtime)
    covariates.to_csv(output_dir / "development_gene_matching_covariates.csv.gz", index=False, compression="gzip")
    recovery, null_distribution, balance = signature_recovery_tables(
        panels,
        signatures,
        covariates,
        n_draws=n_matched_draws,
        random_seed=random_seed,
    )
    recovery.to_csv(output_dir / "locked_signature_recovery.csv", index=False)
    null_distribution.to_csv(
        output_dir / "locked_signature_matched_null_distribution.csv.gz",
        index=False,
        compression="gzip",
    )
    balance.to_csv(output_dir / "matched_null_balance.csv", index=False)
    specificity = specificity_summary(recovery)
    specificity.to_csv(output_dir / "locked_signature_specificity_summary.csv", index=False)
    direct_contrast = matched_null_panel_contrast(recovery, null_distribution)
    direct_contrast.to_csv(
        output_dir / "InterGATE_minus_XGBoost_signature_contrast.csv", index=False
    )
    overlap = panel_overlap_table(panels, gene_symbols)
    overlap.to_csv(output_dir / "panel_overlap.csv", index=False)
    ora, primary_programmes = programme_ora(panels, gene_symbols, hallmark, c2)
    ora.to_csv(output_dir / "hallmark_c2_ora_complete.csv.gz", index=False, compression="gzip")
    primary_programmes.to_csv(output_dir / "prespecified_basal_programmes.csv", index=False)

    development_lock = {
        "status": "development_panel_locked",
        "created_utc": _utc_now(),
        "protocol_id": PROTOCOL_ID,
        "analysis_role": "posthoc_secondary_biological_comparison",
        "target_class": TARGET_CLASS,
        "panel_size": PANEL_SIZE,
        "ranking": (
            "within each seed, equal-weight mean rank across development cohorts "
            "of the signed exact TreeSHAP contrast mean(Basal-like)-mean(rest); "
            "then mean rank across seeds"
        ),
        "xgboost_retrained": False,
        "tree_shap_device": str(tree_shap_device),
        "external_used_for_ranking": False,
        "development_validation_samples": int(len(val_idx)),
        "development_validation_cohorts": sorted(np.unique(validation_cohort).tolist()),
        "development_cohort_weighting": "equal",
        "seeds": list(DEFAULT_SEEDS),
        "ordered_gene_sha256": _ordered_text_sha256(feature_names),
        "intergate_core_source": str(intergate_source),
        "intergate_core_sha256": _ordered_text_sha256(sorted(intergate_panel)),
        "xgboost_top288_sha256": _ordered_text_sha256(sorted(xgb_panel)),
        "xgboost_strict_5of5_n": len(xgb_strict),
        "xgboost_strict_5of5_sha256": _ordered_text_sha256(sorted(xgb_strict)),
        "msigdb_release": annotation_manifest["msigdb_release"],
        "primary_excludes_pam50": True,
        "matched_null_draws": int(n_matched_draws),
        "output_sha256": {
            development_scores_path.name: _file_sha256(development_scores_path),
            consensus_path.name: _file_sha256(consensus_path),
            pooled_consensus_path.name: _file_sha256(pooled_consensus_path),
            "locked_gene_panels.csv": _file_sha256(output_dir / "locked_gene_panels.csv"),
        },
    }
    protected.write_text(
        json.dumps(development_lock, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "output_dir": output_dir,
        "development_lock": protected,
        "development_scores": development_scores,
        "development_consensus": consensus,
        "development_consensus_pooled_sensitivity": pooled_consensus,
        "cohort_weighting_sensitivity": rank_sensitivity,
        "panels": panels,
        "signature_recovery": recovery,
        "specificity": specificity,
        "direct_signature_contrast": direct_contrast,
        "overlap": overlap,
        "ora": ora,
        "primary_programmes": primary_programmes,
    }


def run_external_validation(
    runtime: Mapping[str, Any],
    *,
    benchmark_root: Path | str,
    output_dir: Path | str,
    batch_size: int = 256,
    tree_shap_device: str = "cpu",
) -> dict[str, Any]:
    """Validate the already locked XGBoost ranking in each external cohort."""
    output_dir = Path(output_dir)
    lock_path = output_dir / "DEVELOPMENT_LOCK.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Debe completarse y bloquearse primero la fase de desarrollo.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "development_panel_locked":
        raise RuntimeError("El bloqueo de desarrollo no está completo.")
    if not bool(runtime.get("external_opened", False)):
        raise RuntimeError("La validación externa requiere include_external=True.")
    completion_path = output_dir / "EXTERNAL_VALIDATION_COMPLETE.json"
    if completion_path.exists():
        raise FileExistsError(
            f"La validación externa ya está completa: {completion_path}. "
            "Use otra carpeta para una repetición deliberada."
        )
    genes = _runtime_genes(runtime)
    if _ordered_text_sha256(genes) != lock["ordered_gene_sha256"]:
        raise RuntimeError("El universo génico cambió después del bloqueo.")
    test_idx = np.asarray(runtime["test_idx"], dtype=np.int64)
    _, _, cohort, _ = _runtime_subset(runtime, test_idx)
    scopes = {"external_pooled": np.ones(len(test_idx), dtype=bool)}
    scopes.update({
        f"cohort::{name}": cohort == name for name in sorted(np.unique(cohort))
    })
    external_scores = score_frozen_xgboost(
        runtime,
        benchmark_root,
        indices=test_idx,
        scopes=scopes,
        batch_size=batch_size,
        inference_device=tree_shap_device,
    )
    scores_path = output_dir / "xgboost_external_gene_scores_by_seed_and_cohort.csv.gz"
    external_scores.to_csv(scores_path, index=False, compression="gzip")
    development_consensus = pd.read_csv(
        output_dir / "xgboost_development_gene_consensus.csv"
    )
    transport, external_consensus = external_transportability(
        development_consensus, external_scores, panel_size=PANEL_SIZE
    )
    transport.to_csv(output_dir / "xgboost_external_transportability.csv", index=False)
    external_consensus.to_csv(
        output_dir / "xgboost_external_gene_consensus_by_cohort.csv.gz",
        index=False,
        compression="gzip",
    )
    completion = {
        "status": "complete",
        "created_utc": _utc_now(),
        "protocol_id": PROTOCOL_ID,
        "development_lock_sha256": _file_sha256(lock_path),
        "external_samples": int(len(test_idx)),
        "external_cohorts": sorted(np.unique(cohort).tolist()),
        "external_refit": False,
        "tree_shap_device": str(tree_shap_device),
        "external_used_for_gene_selection": False,
        "external_scores_sha256": _file_sha256(scores_path),
    }
    completion_path.write_text(
        json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "external_scores": external_scores,
        "external_consensus": external_consensus,
        "transportability": transport,
        "completion": completion_path,
    }


def finalize_comparison(output_dir: Path | str) -> dict[str, Path]:
    """Create the review figure, compact summary and transportable ZIP."""
    output_dir = Path(output_dir)
    required = [
        output_dir / "DEVELOPMENT_LOCK.json",
        output_dir / "EXTERNAL_VALIDATION_COMPLETE.json",
        output_dir / "locked_signature_recovery.csv",
        output_dir / "prespecified_basal_programmes.csv",
        output_dir / "panel_overlap.csv",
        output_dir / "xgboost_external_transportability.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan resultados para cerrar la comparación:\n- " + "\n- ".join(missing))
    recovery = pd.read_csv(output_dir / "locked_signature_recovery.csv")
    programmes = pd.read_csv(output_dir / "prespecified_basal_programmes.csv")
    overlap = pd.read_csv(output_dir / "panel_overlap.csv")
    transport = pd.read_csv(output_dir / "xgboost_external_transportability.csv")
    pdf, png = _make_comparison_figure(
        recovery, programmes, transport, overlap, output_dir
    )

    primary = ["InterGATE_strict_5of5", "XGBoost_TreeSHAP_top288"]
    summary_rows = []
    for panel in primary:
        basal = recovery.loc[
            recovery["panel"].eq(panel)
            & recovery["annotation_class"].eq(TARGET_CLASS)
        ].iloc[0]
        panel_programmes = programmes.loc[programmes["panel"].eq(panel)]
        summary_rows.append({
            "panel": panel,
            "basal_signature_overlap_n": basal["overlap_n"],
            "basal_signature_fold_expected": basal["fold_over_hypergeom_expected"],
            "basal_signature_fold_matched_null": basal["fold_over_matched_null"],
            "basal_signature_matched_fdr": basal["matched_fdr_global"],
            "prespecified_programmes_significant": int(
                panel_programmes["significant_fdr_005"].sum()
            ),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "XGBoost_vs_InterGATE_biology_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "status": "complete",
        "created_utc": _utc_now(),
        "protocol_id": PROTOCOL_ID,
        "summary": summary_path.name,
        "figure_pdf": pdf.name,
        "figure_png": png.name,
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    archive_base = output_dir.parent / "InterGATE_XGBoost_biology_review"
    archive_path = Path(shutil.make_archive(
        str(archive_base), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
    ))
    return {
        "summary": summary_path,
        "figure_pdf": pdf,
        "figure_png": png,
        "manifest": manifest_path,
        "review_zip": archive_path,
    }
