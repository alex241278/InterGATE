"""Paired predictive benchmark and expanded reporting for InterGATE.

This module implements the predictive comparison that is appropriate for a
head-to-head Supplementary Table S12:

* identical development train/validation indices;
* identical locked external samples;
* the same ordered 7,131-gene feature universe;
* the same four-class endpoint and five prespecified seeds;
* no baseline refitting after the external block is opened.

The original leave-one-development-cohort-out (LOCO) controls answer a
different transportability question and should remain a separate analysis.

The workflow has three deliberately separated stages.  ``fit_development``
fits and freezes tabular models while the external block is closed.
``evaluate_external`` loads those frozen objects and only predicts the locked
external indices.  ``summarize_predictions`` reports seed distributions,
class-wise metrics, mean confusion matrices, LumA/LumB exchange rates,
stratified bootstrap confidence intervals, and paired bootstrap differences.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

DEFAULT_CLASSES = ("LumA", "LumB", "HER2-enriched", "Basal-like")
DEFAULT_SEEDS = (1234, 2025, 7, 99, 31337)
TABULAR_MODELS = ("elastic_net", "random_forest", "xgboost")
OVERALL_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "cohen_kappa",
    "macro_f1",
    "weighted_f1",
    "macro_auroc_ovr",
    "micro_auroc_ovr",
)
PER_CLASS_METRICS = ("precision", "recall", "f1", "auroc_ovr")
EXCHANGE_METRICS = (
    "luma_to_lumb_rate",
    "lumb_to_luma_rate",
    "bidirectional_exchange_rate",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _intergate_prediction_api():
    """Import torch-dependent InterGATE objects only for inference stages."""
    from .ablation import (
        load_bundle,
        predict_bundle_on_test,
        predict_bundle_on_validation,
    )
    from .config import CFG
    return load_bundle, predict_bundle_on_test, predict_bundle_on_validation, CFG


def _ordered_text_sha256(values: Iterable[Any]) -> str:
    payload = "\n".join(map(str, values)).encode("utf-8")
    return sha256(payload).hexdigest()


def _integer_array_sha256(values: Sequence[int]) -> str:
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    return sha256(array.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path | str) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probability_columns(class_names: Sequence[str]) -> list[str]:
    return [f"p_{name}" for name in class_names]


def _class_names_from_runtime(runtime: Mapping[str, Any]) -> tuple[str, ...]:
    classes = tuple(map(str, runtime["classes"]))
    if classes != tuple(DEFAULT_CLASSES):
        raise RuntimeError(
            f"Endpoint inesperado: {classes}; esperado={DEFAULT_CLASSES}."
        )
    return classes


def _coerce_encoded_labels(
    values: Sequence[Any], class_names: Sequence[str]
) -> np.ndarray:
    array = np.asarray(values)
    mapping = {str(name): class_id for class_id, name in enumerate(class_names)}
    if array.dtype.kind in "iu":
        encoded = array.astype(np.int64)
    elif array.dtype.kind == "f" and np.all(np.isfinite(array)):
        encoded = array.astype(np.int64)
        if not np.allclose(array, encoded, rtol=0.0, atol=0.0):
            raise ValueError("Las etiquetas numéricas no son enteros exactos.")
    else:
        unknown = sorted(set(map(str, array)) - set(mapping))
        if unknown:
            raise ValueError(f"Etiquetas desconocidas: {unknown}.")
        encoded = np.asarray([mapping[str(value)] for value in array], dtype=np.int64)
    if np.any(encoded < 0) or np.any(encoded >= len(class_names)):
        raise ValueError("Las etiquetas codificadas están fuera del endpoint.")
    return encoded


def _prediction_frame(
    *,
    model: str,
    split: str,
    seed: int,
    sample_index: Sequence[int],
    cohort: Sequence[str],
    y_true: Sequence[int],
    probability: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    probability = np.asarray(probability, dtype=float)
    y_true = _coerce_encoded_labels(y_true, class_names)
    sample_index = np.asarray(sample_index, dtype=np.int64).reshape(-1)
    cohort = np.asarray(cohort).astype(str).reshape(-1)
    expected_shape = (len(y_true), len(class_names))
    if probability.shape != expected_shape:
        raise ValueError(
            f"Probabilidades {probability.shape}; esperado={expected_shape}."
        )
    if len(sample_index) != len(y_true) or len(cohort) != len(y_true):
        raise ValueError("Índices, cohortes y etiquetas no están alineados.")
    if not np.all(np.isfinite(probability)):
        raise ValueError("Hay probabilidades no finitas.")
    if np.any(probability < -1e-7) or np.any(probability > 1.0 + 1e-7):
        raise ValueError("Hay probabilidades fuera de [0, 1].")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=2e-4, rtol=0.0):
        raise ValueError("Las probabilidades no suman uno por muestra.")

    frame = pd.DataFrame({
        "model": str(model),
        "split": str(split),
        "seed": int(seed),
        "sample_index": sample_index,
        "cohort": cohort,
        "y_true": y_true,
        "y_pred": probability.argmax(axis=1).astype(np.int64),
    })
    for class_id, class_name in enumerate(class_names):
        frame[f"p_{class_name}"] = probability[:, class_id]
    return frame


def _validate_prediction_frame(
    frame: pd.DataFrame, class_names: Sequence[str]
) -> pd.DataFrame:
    frame = frame.copy()
    required = {
        "model", "split", "seed", "sample_index", "cohort",
        "y_true", "y_pred", *_probability_columns(class_names),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas de predicción: {missing}.")
    frame["model"] = frame["model"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["seed"] = frame["seed"].astype(np.int64)
    frame["sample_index"] = frame["sample_index"].astype(np.int64)
    frame["cohort"] = frame["cohort"].astype(str)
    frame["y_true"] = _coerce_encoded_labels(frame["y_true"], class_names)
    frame["y_pred"] = _coerce_encoded_labels(frame["y_pred"], class_names)
    probability = frame[_probability_columns(class_names)].to_numpy(dtype=float)
    if not np.all(np.isfinite(probability)):
        raise ValueError("Hay probabilidades no finitas.")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=2e-4, rtol=0.0):
        raise ValueError("Las probabilidades no suman uno por muestra.")
    calculated = probability.argmax(axis=1)
    if not np.array_equal(calculated, frame["y_pred"].to_numpy(dtype=int)):
        raise ValueError("y_pred no coincide con argmax(probabilidades).")
    key = ["model", "split", "seed", "sample_index"]
    if frame.duplicated(key).any():
        raise ValueError(f"Hay predicciones duplicadas para {key}.")

    # Every seed of a model/split must refer to the same samples and truths.
    for (model, split), group in frame.groupby(["model", "split"], sort=False):
        reference = None
        for seed, seed_frame in group.groupby("seed", sort=True):
            current = seed_frame.sort_values("sample_index")[
                ["sample_index", "cohort", "y_true"]
            ].reset_index(drop=True)
            if reference is None:
                reference = current
            elif not current.equals(reference):
                raise RuntimeError(
                    f"{model}/{split}: las muestras cambian entre semillas; "
                    f"primera discrepancia en seed={seed}."
                )
    return frame.sort_values(key).reset_index(drop=True)


def _validate_paired_samples_and_seeds(
    frame: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
) -> None:
    """Require every model in a split to use the same five prediction rows."""
    expected_seed_set = set(map(int, expected_seeds))
    for split, split_frame in frame.groupby("split", sort=False):
        reference: pd.DataFrame | None = None
        reference_model: str | None = None
        for model, model_frame in split_frame.groupby("model", sort=True):
            observed_seeds = set(model_frame["seed"].astype(int).unique())
            if observed_seeds != expected_seed_set:
                raise RuntimeError(
                    f"{model}/{split}: semillas={sorted(observed_seeds)}; "
                    f"esperadas={sorted(expected_seed_set)}."
                )
            current = model_frame.loc[
                model_frame["seed"].eq(min(observed_seeds)),
                ["sample_index", "cohort", "y_true"],
            ].sort_values("sample_index").reset_index(drop=True)
            if reference is None:
                reference = current
                reference_model = str(model)
            elif not current.equals(reference):
                raise RuntimeError(
                    f"{model}/{split} no usa las mismas muestras/etiquetas que "
                    f"{reference_model}."
                )


def _runtime_feature_matrix(
    runtime: Mapping[str, Any]
) -> tuple[np.ndarray, tuple[str, ...]]:
    genes = tuple(map(str, runtime["genes"]))
    if len(genes) != 7131:
        raise RuntimeError(f"Se esperaban 7.131 genes conectados; hay {len(genes)}.")
    if len(set(genes)) != len(genes):
        raise RuntimeError("El universo génico contiene duplicados.")
    X_df = runtime["X_df"]
    missing = [gene for gene in genes if gene not in X_df.columns]
    if missing:
        raise RuntimeError(f"Faltan genes en X_df; primeros={missing[:10]}.")
    selected = X_df.loc[:, list(genes)]
    if tuple(map(str, selected.columns)) != genes:
        raise RuntimeError("Se alteró el orden génico del benchmark.")
    return selected.to_numpy(dtype=np.float32), genes


def build_tabular_models(
    seed: int,
    *,
    xgboost_device: str = "cpu",
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Construct the three fixed, prespecified tabular comparators."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "El benchmark pareado requiere xgboost>=2.0,<4."
        ) from exc

    models: dict[str, Any] = {
        "elastic_net": make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=1.0,
                class_weight="balanced",
                max_iter=8000,
                random_state=int(seed),
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=1000,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=int(seed),
            n_jobs=int(n_jobs),
        ),
    }
    xgb_kwargs = dict(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=5.0,
        subsample=0.80,
        colsample_bytree=0.35,
        reg_alpha=0.10,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=int(seed),
        n_jobs=int(n_jobs),
        verbosity=0,
    )
    if str(xgboost_device).lower() != "cpu":
        xgb_kwargs["device"] = str(xgboost_device)
    models["xgboost"] = XGBClassifier(**xgb_kwargs)
    return models


def _aligned_model_probability(
    model: Any, X: np.ndarray, n_classes: int
) -> np.ndarray:
    probability = np.asarray(model.predict_proba(X), dtype=float)
    observed_classes = np.asarray(model.classes_, dtype=np.int64)
    if probability.shape[1] != len(observed_classes):
        raise RuntimeError("predict_proba y model.classes_ no están alineados.")
    aligned = np.zeros((len(X), int(n_classes)), dtype=float)
    for column, class_id in enumerate(observed_classes):
        if class_id < 0 or class_id >= n_classes:
            raise RuntimeError(f"Clase del modelo fuera de rango: {class_id}.")
        aligned[:, int(class_id)] = probability[:, column]
    return aligned


def fit_development(
    runtime: Mapping[str, Any],
    *,
    intergate_bundle_root: Path | str,
    output_root: Path | str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    xgboost_device: str = "cpu",
    n_jobs: int = -1,
    external_complete_marker: Path | str | None = None,
    allow_posthoc: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    """Fit/freeze tabular models and predict the shared validation split.

    This stage must normally run before protocol freezing.  If an external
    completion marker already exists, execution is blocked unless
    ``allow_posthoc=True``; that choice is recorded in the manifest.
    """
    if bool(runtime.get("external_opened", True)):
        raise RuntimeError("El ajuste tabular requiere el externo cerrado.")
    load_bundle, _, predict_bundle_on_validation, cfg = (
        _intergate_prediction_api()
    )
    class_names = _class_names_from_runtime(runtime)
    seeds = tuple(map(int, seeds))
    if seeds != tuple(DEFAULT_SEEDS):
        raise RuntimeError(f"Semillas inesperadas: {seeds}.")

    posthoc = False
    if external_complete_marker is not None:
        marker = Path(external_complete_marker)
        posthoc = marker.is_file()
        if posthoc and not bool(allow_posthoc):
            raise RuntimeError(
                "La evaluación externa ya consta como completada. Para un "
                "análisis descriptivo añadido después, repita con "
                "allow_posthoc=True; el manifiesto lo declarará."
            )

    output_root = Path(output_root)
    models_root = output_root / "frozen_tabular_models"
    predictions_path = output_root / "development_predictions.csv.gz"
    manifest_path = output_root / "development_benchmark_manifest.json"
    protected = [predictions_path, manifest_path]
    if not force and any(path.exists() for path in protected):
        raise FileExistsError(
            "El benchmark de desarrollo ya existe. Use force=True solo para "
            "una repetición deliberada."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    X, genes = _runtime_feature_matrix(runtime)
    y = np.asarray(runtime["y"], dtype=np.int64)
    cohort = np.asarray(runtime["cohort"]).astype(str)
    train_idx = np.asarray(runtime["train_idx"], dtype=np.int64)
    val_idx = np.asarray(runtime["val_idx"], dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Train/validation no están definidos.")
    if set(np.unique(y[train_idx])) != set(range(len(class_names))):
        raise RuntimeError("El entrenamiento no contiene las cuatro clases.")

    bundle_root = Path(intergate_bundle_root)
    rows: list[pd.DataFrame] = []
    model_files: dict[str, dict[str, str]] = {}
    model_sha256: dict[str, dict[str, str]] = {}
    for seed in seeds:
        seed_dir = models_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model_files[str(seed)] = {}
        model_sha256[str(seed)] = {}

        bundle_dir = bundle_root / f"seed_{seed}"
        bundle = load_bundle(str(bundle_dir))
        bundle_genes = tuple(map(str, bundle["graph"]["genes_full"]))
        if bundle_genes != genes:
            raise RuntimeError(
                f"InterGATE seed={seed} y los baselines no comparten genes/orden."
            )
        neural = predict_bundle_on_validation(str(bundle_dir), device=cfg.DEVICE)
        if not np.array_equal(neural["val_idx"], val_idx):
            raise RuntimeError(f"InterGATE seed={seed}: val_idx no coincide.")
        rows.append(_prediction_frame(
            model="InterGATE",
            split="development_validation",
            seed=seed,
            sample_index=val_idx,
            cohort=neural["cohort"],
            y_true=neural["y_true"],
            probability=neural["proba"],
            class_names=class_names,
        ))

        for model_name, model in build_tabular_models(
            seed, xgboost_device=xgboost_device, n_jobs=n_jobs
        ).items():
            fit_kwargs: dict[str, Any] = {}
            if model_name == "xgboost":
                fit_kwargs["sample_weight"] = compute_sample_weight(
                    class_weight="balanced", y=y[train_idx]
                )
            model.fit(X[train_idx], y[train_idx], **fit_kwargs)
            probability = _aligned_model_probability(
                model, X[val_idx], len(class_names)
            )
            rows.append(_prediction_frame(
                model=model_name,
                split="development_validation",
                seed=seed,
                sample_index=val_idx,
                cohort=cohort[val_idx],
                y_true=y[val_idx],
                probability=probability,
                class_names=class_names,
            ))
            model_path = seed_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path, compress=3)
            model_files[str(seed)][model_name] = str(
                model_path.relative_to(output_root)
            )
            model_sha256[str(seed)][model_name] = _file_sha256(model_path)

    predictions = _validate_prediction_frame(pd.concat(rows), class_names)
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    manifest = {
        "status": "development_models_frozen",
        "created_utc": _utc_now(),
        "protocol": "paired-same-split-same-samples-v1",
        "comparison_role": "direct_predictive_benchmark",
        "loco_role": "separate_transportability_analysis",
        "class_names": list(class_names),
        "seeds": list(seeds),
        "models": list(TABULAR_MODELS),
        "n_genes": len(genes),
        "ordered_gene_sha256": _ordered_text_sha256(genes),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "train_index_sha256": _integer_array_sha256(train_idx),
        "validation_index_sha256": _integer_array_sha256(val_idx),
        "external_opened_during_fit": False,
        "external_evaluation_already_complete_at_declaration": bool(posthoc),
        "analysis_status": "posthoc_fixed_protocol" if posthoc else "prospective_locked",
        "hyperparameter_selection": "fixed_no_validation_or_external_tuning",
        "model_files": model_files,
        "model_sha256": model_sha256,
        "development_predictions": predictions_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "predictions": predictions_path,
        "manifest": manifest_path,
        "models_root": models_root,
    }


def _standardize_existing_intergate_external(
    path: Path | str,
    class_names: Sequence[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "test_index" in frame.columns and "sample_index" not in frame.columns:
        frame = frame.rename(columns={"test_index": "sample_index"})
    frame["model"] = "InterGATE"
    frame["split"] = "locked_external"
    frame["y_true"] = _coerce_encoded_labels(frame["y_true"], class_names)
    if "y_pred" not in frame.columns:
        frame["y_pred"] = frame[_probability_columns(class_names)].to_numpy().argmax(axis=1)
    else:
        frame["y_pred"] = _coerce_encoded_labels(frame["y_pred"], class_names)
    return frame[[
        "model", "split", "seed", "sample_index", "cohort",
        "y_true", "y_pred", *_probability_columns(class_names),
    ]]


def evaluate_external(
    runtime: Mapping[str, Any],
    *,
    development_output_root: Path | str,
    output_root: Path | str,
    existing_intergate_predictions: Path | str | None = None,
    intergate_bundle_root: Path | str | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Evaluate frozen tabular and InterGATE models on identical test rows."""
    if not bool(runtime.get("external_opened", False)):
        raise RuntimeError("El externo sigue cerrado; no puede evaluarse.")
    class_names = _class_names_from_runtime(runtime)
    development_output_root = Path(development_output_root)
    manifest_path = development_output_root / "development_benchmark_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Falta el manifiesto congelado: {manifest_path}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "development_models_frozen":
        raise RuntimeError("El benchmark tabular no está congelado.")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "external_predictions_paired.csv.gz"
    completion_path = output_root / "external_benchmark_COMPLETE.json"
    if not force and (predictions_path.exists() or completion_path.exists()):
        raise FileExistsError(
            "La evaluación externa pareada ya existe; no se repetirá sin force=True."
        )

    X, genes = _runtime_feature_matrix(runtime)
    if _ordered_text_sha256(genes) != manifest["ordered_gene_sha256"]:
        raise RuntimeError("El universo/orden génico cambió tras congelar baselines.")
    y = np.asarray(runtime["y"], dtype=np.int64)
    cohort = np.asarray(runtime["cohort"]).astype(str)
    test_idx = np.asarray(runtime["test_idx"], dtype=np.int64)
    if len(test_idx) == 0:
        raise RuntimeError("test_idx está vacío.")

    rows: list[pd.DataFrame] = []
    for seed_text, files in manifest["model_files"].items():
        seed = int(seed_text)
        for model_name, relative_path in files.items():
            model_path = development_output_root / relative_path
            expected_digest = manifest.get("model_sha256", {}).get(
                seed_text, {}
            ).get(model_name)
            if expected_digest is None:
                raise RuntimeError(
                    f"Falta el hash congelado de {model_name}/seed={seed}."
                )
            if _file_sha256(model_path) != expected_digest:
                raise RuntimeError(
                    f"El objeto congelado cambió: {model_name}/seed={seed}."
                )
            model = joblib.load(model_path)
            probability = _aligned_model_probability(
                model, X[test_idx], len(class_names)
            )
            rows.append(_prediction_frame(
                model=model_name,
                split="locked_external",
                seed=seed,
                sample_index=test_idx,
                cohort=cohort[test_idx],
                y_true=y[test_idx],
                probability=probability,
                class_names=class_names,
            ))

    if existing_intergate_predictions is not None:
        neural_frame = _standardize_existing_intergate_external(
            existing_intergate_predictions, class_names
        )
        rows.append(neural_frame)
    elif intergate_bundle_root is not None:
        _, predict_bundle_on_test, _, cfg = _intergate_prediction_api()
        bundle_root = Path(intergate_bundle_root)
        for seed in map(int, manifest["seeds"]):
            prediction = predict_bundle_on_test(
                str(bundle_root / f"seed_{seed}"), device=cfg.DEVICE
            )
            rows.append(_prediction_frame(
                model="InterGATE",
                split="locked_external",
                seed=seed,
                sample_index=prediction["test_idx"],
                cohort=prediction["cohort"],
                y_true=prediction["y_true"],
                probability=prediction["proba"],
                class_names=class_names,
            ))
    else:
        raise ValueError(
            "Indique existing_intergate_predictions o intergate_bundle_root."
        )

    predictions = _validate_prediction_frame(pd.concat(rows), class_names)
    _validate_paired_samples_and_seeds(
        predictions, expected_seeds=manifest["seeds"]
    )
    expected = pd.DataFrame({
        "sample_index": test_idx,
        "cohort": cohort[test_idx],
        "y_true": y[test_idx],
    }).sort_values("sample_index").reset_index(drop=True)
    for (model, seed), group in predictions.groupby(["model", "seed"]):
        observed = group[["sample_index", "cohort", "y_true"]].sort_values(
            "sample_index"
        ).reset_index(drop=True)
        if not observed.equals(expected):
            raise RuntimeError(
                f"{model}/seed={seed} no usa exactamente el mismo externo."
            )

    predictions.to_csv(predictions_path, index=False, compression="gzip")
    completion = {
        "status": "complete",
        "created_utc": _utc_now(),
        "protocol": manifest["protocol"],
        "analysis_status": manifest["analysis_status"],
        "models": sorted(predictions["model"].unique().tolist()),
        "seeds": list(map(int, manifest["seeds"])),
        "n_external_samples": int(len(test_idx)),
        "external_refit": False,
        "external_hyperparameter_selection": False,
        "same_samples_across_models": True,
        "same_genes_across_models": True,
        "ordered_gene_sha256": manifest["ordered_gene_sha256"],
        "external_index_sha256": _integer_array_sha256(test_idx),
        "predictions": predictions_path.name,
    }
    completion_path.write_text(
        json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"predictions": predictions_path, "completion": completion_path}


def _metrics_from_probability(
    y_true: np.ndarray,
    probability: np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict[str, float], list[dict[str, Any]], np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probability, dtype=float)
    n_classes = len(class_names)
    labels = np.arange(n_classes, dtype=np.int64)
    y_pred = probability.argmax(axis=1).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    total = float(cm.sum())
    accuracy = float(tp.sum() / total) if total else float("nan")
    present = support > 0
    balanced = float(np.mean(recall[present])) if np.any(present) else float("nan")
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support)) if total else float("nan")
    expected_agreement = (
        float(np.dot(support / total, predicted / total)) if total else float("nan")
    )
    kappa = (
        float((accuracy - expected_agreement) / (1.0 - expected_agreement))
        if total and expected_agreement < 1.0 else float("nan")
    )

    aucs = np.full(n_classes, np.nan, dtype=float)
    binary_truth: list[np.ndarray] = []
    binary_score: list[np.ndarray] = []
    for class_id in labels:
        target = (y_true == class_id).astype(np.int8)
        if target.min() == target.max():
            continue
        aucs[class_id] = float(roc_auc_score(target, probability[:, class_id]))
        binary_truth.append(target)
        binary_score.append(probability[:, class_id])
    macro_auc = float(np.nanmean(aucs)) if np.any(np.isfinite(aucs)) else float("nan")
    micro_auc = (
        float(roc_auc_score(np.concatenate(binary_truth), np.concatenate(binary_score)))
        if binary_truth else float("nan")
    )
    overall = {
        "n_samples": int(len(y_true)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "cohen_kappa": kappa,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_auroc_ovr": macro_auc,
        "micro_auroc_ovr": micro_auc,
    }
    per_class = [{
        "class_id": int(class_id),
        "class_name": str(class_names[class_id]),
        "precision": float(precision[class_id]),
        "recall": float(recall[class_id]),
        "f1": float(f1[class_id]),
        "support": int(support[class_id]),
        "auroc_ovr": float(aucs[class_id]),
    } for class_id in labels]
    row_normalized = np.divide(
        cm,
        support[:, None],
        out=np.full_like(cm, np.nan),
        where=support[:, None] > 0,
    )
    return overall, per_class, cm, row_normalized


def _exchange_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, float | int]:
    class_id = {name: index for index, name in enumerate(class_names)}
    luma = int(class_id["LumA"])
    lumb = int(class_id["LumB"])
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(probability).argmax(axis=1)
    n_luma = int(np.sum(y_true == luma))
    n_lumb = int(np.sum(y_true == lumb))
    a_to_b = int(np.sum((y_true == luma) & (y_pred == lumb)))
    b_to_a = int(np.sum((y_true == lumb) & (y_pred == luma)))
    return {
        "true_luma_support": n_luma,
        "true_lumb_support": n_lumb,
        "luma_to_lumb_count": a_to_b,
        "lumb_to_luma_count": b_to_a,
        "luma_to_lumb_rate": a_to_b / n_luma if n_luma else float("nan"),
        "lumb_to_luma_rate": b_to_a / n_lumb if n_lumb else float("nan"),
        "bidirectional_exchange_count": a_to_b + b_to_a,
        "bidirectional_exchange_rate": (
            (a_to_b + b_to_a) / (n_luma + n_lumb)
            if (n_luma + n_lumb) else float("nan")
        ),
    }


def _scope_frames(frame: pd.DataFrame):
    yield "pooled", frame
    for cohort_name, cohort_frame in frame.groupby("cohort", sort=True):
        yield f"cohort::{cohort_name}", cohort_frame


def _seed_tables(
    predictions: pd.DataFrame,
    class_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probability_columns = _probability_columns(class_names)
    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    exchange_rows: list[dict[str, Any]] = []
    for (model, split, seed), group in predictions.groupby(
        ["model", "split", "seed"], sort=True
    ):
        for scope, scoped in _scope_frames(group):
            y_true = scoped["y_true"].to_numpy(dtype=int)
            probability = scoped[probability_columns].to_numpy(dtype=float)
            overall, per_class, cm, normalized = _metrics_from_probability(
                y_true, probability, class_names
            )
            identifiers = {
                "model": model, "split": split, "seed": int(seed), "scope": scope,
            }
            metric_rows.append({**identifiers, **overall})
            class_rows.extend({**identifiers, **row} for row in per_class)
            for true_id, true_name in enumerate(class_names):
                for pred_id, pred_name in enumerate(class_names):
                    confusion_rows.append({
                        **identifiers,
                        "true_class_id": true_id,
                        "true_class": true_name,
                        "predicted_class_id": pred_id,
                        "predicted_class": pred_name,
                        "count": float(cm[true_id, pred_id]),
                        "row_fraction": float(normalized[true_id, pred_id]),
                    })
            exchange_rows.append({
                **identifiers,
                **_exchange_metrics(y_true, probability, class_names),
            })
    return tuple(map(pd.DataFrame, (
        metric_rows, class_rows, confusion_rows, exchange_rows
    )))


def _long_seed_summary(
    table: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    long = table.melt(
        id_vars=list(id_columns) + ["seed"],
        value_vars=list(metric_columns),
        var_name="metric",
        value_name="value",
    )
    return long.groupby(list(id_columns) + ["metric"], dropna=False)["value"].agg(
        mean="mean", sd="std", minimum="min", maximum="max", n_seeds="count"
    ).reset_index()


def _mean_confusion(confusions: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "model", "split", "scope", "true_class_id", "true_class",
        "predicted_class_id", "predicted_class",
    ]
    counts = confusions.groupby(identifiers, dropna=False).agg(
        mean_count=("count", "mean"),
        sd_count=("count", "std"),
        mean_row_fraction=("row_fraction", "mean"),
        sd_row_fraction=("row_fraction", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return counts


def _ensemble_predictions(
    predictions: pd.DataFrame,
    class_names: Sequence[str],
) -> pd.DataFrame:
    probability_columns = _probability_columns(class_names)
    identifiers = ["model", "split", "sample_index", "cohort", "y_true"]
    ensemble = predictions.groupby(identifiers, dropna=False).agg(
        **{column: (column, "mean") for column in probability_columns},
        n_seeds=("seed", "nunique"),
    ).reset_index()
    ensemble["y_pred"] = ensemble[probability_columns].to_numpy().argmax(axis=1)
    return ensemble


def _strata_indices(
    y_true: np.ndarray, cohort: np.ndarray
) -> list[np.ndarray]:
    keys = np.asarray([f"{c}\x1f{int(y)}" for c, y in zip(cohort, y_true)])
    return [np.flatnonzero(keys == key) for key in sorted(np.unique(keys))]


def _resample_strata(
    strata: Sequence[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    return np.concatenate([
        rng.choice(indices, size=len(indices), replace=True)
        for indices in strata
    ])


def _percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def _bootstrap_tables(
    ensemble: pd.DataFrame,
    class_names: Sequence[str],
    *,
    n_bootstrap: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probability_columns = _probability_columns(class_names)
    overall_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    exchange_rows: list[dict[str, Any]] = []
    generator = np.random.default_rng(int(random_seed))

    for (model, split), group in ensemble.groupby(["model", "split"], sort=True):
        for scope, scoped in _scope_frames(group):
            y_true = scoped["y_true"].to_numpy(dtype=int)
            probability = scoped[probability_columns].to_numpy(dtype=float)
            cohorts = scoped["cohort"].to_numpy(dtype=str)
            observed, observed_class, _, _ = _metrics_from_probability(
                y_true, probability, class_names
            )
            observed_exchange = _exchange_metrics(y_true, probability, class_names)
            overall_distribution = {
                metric: np.empty(n_bootstrap, dtype=float)
                for metric in OVERALL_METRICS
            }
            class_distribution = {
                (class_id, metric): np.empty(n_bootstrap, dtype=float)
                for class_id in range(len(class_names))
                for metric in PER_CLASS_METRICS
            }
            exchange_distribution = {
                metric: np.empty(n_bootstrap, dtype=float)
                for metric in EXCHANGE_METRICS
            }
            strata = _strata_indices(y_true, cohorts)
            for draw in range(n_bootstrap):
                sampled = _resample_strata(strata, generator)
                boot, boot_class, _, _ = _metrics_from_probability(
                    y_true[sampled], probability[sampled], class_names
                )
                boot_exchange = _exchange_metrics(
                    y_true[sampled], probability[sampled], class_names
                )
                for metric in OVERALL_METRICS:
                    overall_distribution[metric][draw] = boot[metric]
                for class_id, row in enumerate(boot_class):
                    for metric in PER_CLASS_METRICS:
                        class_distribution[(class_id, metric)][draw] = row[metric]
                for metric in EXCHANGE_METRICS:
                    exchange_distribution[metric][draw] = boot_exchange[metric]

            common = {
                "model": model,
                "split": split,
                "scope": scope,
                "n_samples": int(len(y_true)),
                "n_seeds_ensembled": int(scoped["n_seeds"].min()),
                "bootstrap_repetitions": int(n_bootstrap),
                "bootstrap_unit": "sample",
                "bootstrap_strata": "cohort_x_true_class",
            }
            for metric in OVERALL_METRICS:
                low, high = _percentile_interval(overall_distribution[metric])
                overall_rows.append({
                    **common, "metric": metric,
                    "ensemble_estimate": observed[metric],
                    "ci_95_low": low, "ci_95_high": high,
                })
            for class_id, class_name in enumerate(class_names):
                observed_row = observed_class[class_id]
                for metric in PER_CLASS_METRICS:
                    low, high = _percentile_interval(
                        class_distribution[(class_id, metric)]
                    )
                    class_rows.append({
                        **common,
                        "class_id": class_id,
                        "class_name": class_name,
                        "support": int(observed_row["support"]),
                        "metric": metric,
                        "ensemble_estimate": observed_row[metric],
                        "ci_95_low": low,
                        "ci_95_high": high,
                    })
            for metric in EXCHANGE_METRICS:
                low, high = _percentile_interval(exchange_distribution[metric])
                exchange_rows.append({
                    **common,
                    "metric": metric,
                    "ensemble_estimate": observed_exchange[metric],
                    "ci_95_low": low,
                    "ci_95_high": high,
                    "true_luma_support": observed_exchange["true_luma_support"],
                    "true_lumb_support": observed_exchange["true_lumb_support"],
                })
    return tuple(map(pd.DataFrame, (overall_rows, class_rows, exchange_rows)))


def _paired_bootstrap_differences(
    ensemble: pd.DataFrame,
    class_names: Sequence[str],
    *,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    probability_columns = _probability_columns(class_names)
    rows: list[dict[str, Any]] = []
    if "InterGATE" not in set(ensemble["model"]):
        return pd.DataFrame(rows)

    for split, split_frame in ensemble.groupby("split", sort=True):
        intergate = split_frame.loc[split_frame.model.eq("InterGATE")].copy()
        for comparator in sorted(set(split_frame["model"]) - {"InterGATE"}):
            other = split_frame.loc[split_frame.model.eq(comparator)].copy()
            merged = intergate.merge(
                other,
                on="sample_index",
                suffixes=("_intergate", "_comparator"),
                validate="one_to_one",
            )
            if len(merged) != len(intergate) or len(merged) != len(other):
                raise RuntimeError(
                    f"{split}: InterGATE y {comparator} no comparten muestras."
                )
            for field in ("cohort", "y_true"):
                if not np.array_equal(
                    merged[f"{field}_intergate"], merged[f"{field}_comparator"]
                ):
                    raise RuntimeError(
                        f"{split}: InterGATE y {comparator} discrepan en {field}."
                    )
            normalized = pd.DataFrame({
                "sample_index": merged["sample_index"],
                "cohort": merged["cohort_intergate"],
                "y_true": merged["y_true_intergate"],
            })
            for column in probability_columns:
                normalized[f"{column}_intergate"] = merged[f"{column}_intergate"]
                normalized[f"{column}_comparator"] = merged[f"{column}_comparator"]

            pair_digest = sha256(
                f"{split}|{comparator}|{random_seed}".encode("utf-8")
            ).digest()
            pair_seed = int.from_bytes(pair_digest[:8], "little")
            rng = np.random.default_rng(pair_seed)
            for scope, scoped in _scope_frames(normalized):
                y_true = scoped["y_true"].to_numpy(dtype=int)
                cohorts = scoped["cohort"].to_numpy(dtype=str)
                p_intergate = scoped[
                    [f"{column}_intergate" for column in probability_columns]
                ].to_numpy(dtype=float)
                p_comparator = scoped[
                    [f"{column}_comparator" for column in probability_columns]
                ].to_numpy(dtype=float)
                observed_intergate, _, _, _ = _metrics_from_probability(
                    y_true, p_intergate, class_names
                )
                observed_comparator, _, _, _ = _metrics_from_probability(
                    y_true, p_comparator, class_names
                )
                distributions = {
                    metric: np.empty(n_bootstrap, dtype=float)
                    for metric in OVERALL_METRICS
                }
                strata = _strata_indices(y_true, cohorts)
                for draw in range(n_bootstrap):
                    sampled = _resample_strata(strata, rng)
                    metric_intergate, _, _, _ = _metrics_from_probability(
                        y_true[sampled], p_intergate[sampled], class_names
                    )
                    metric_comparator, _, _, _ = _metrics_from_probability(
                        y_true[sampled], p_comparator[sampled], class_names
                    )
                    for metric in OVERALL_METRICS:
                        distributions[metric][draw] = (
                            metric_intergate[metric] - metric_comparator[metric]
                        )
                for metric in OVERALL_METRICS:
                    low, high = _percentile_interval(distributions[metric])
                    rows.append({
                        "split": split,
                        "scope": scope,
                        "reference_model": "InterGATE",
                        "comparator_model": comparator,
                        "metric": metric,
                        "reference_estimate": observed_intergate[metric],
                        "comparator_estimate": observed_comparator[metric],
                        "difference_reference_minus_comparator": (
                            observed_intergate[metric] - observed_comparator[metric]
                        ),
                        "ci_95_low": low,
                        "ci_95_high": high,
                        "n_samples": int(len(y_true)),
                        "bootstrap_repetitions": int(n_bootstrap),
                        "paired_resampling": True,
                        "bootstrap_strata": "cohort_x_true_class",
                    })
    return pd.DataFrame(rows)


def summarize_predictions(
    prediction_files: Sequence[Path | str],
    *,
    output_dir: Path | str,
    class_names: Sequence[str] = DEFAULT_CLASSES,
    n_bootstrap: int = 2000,
    random_seed: int = 20260822,
) -> dict[str, Path]:
    """Create every requested metric table from per-sample probabilities."""
    if int(n_bootstrap) < 100:
        raise ValueError("Use al menos 100 réplicas bootstrap.")
    frames = []
    for path in prediction_files:
        frame = pd.read_csv(path)
        if "model" not in frame.columns and "test_index" in frame.columns:
            frame = _standardize_existing_intergate_external(path, class_names)
        frames.append(frame)
    predictions = _validate_prediction_frame(pd.concat(frames), class_names)
    _validate_paired_samples_and_seeds(predictions)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, per_class, confusions, exchange = _seed_tables(
        predictions, class_names
    )
    metric_summary = _long_seed_summary(
        metrics,
        id_columns=("model", "split", "scope"),
        metric_columns=OVERALL_METRICS,
    )
    class_summary = _long_seed_summary(
        per_class,
        id_columns=("model", "split", "scope", "class_id", "class_name"),
        metric_columns=PER_CLASS_METRICS,
    )
    exchange_summary = _long_seed_summary(
        exchange,
        id_columns=("model", "split", "scope"),
        metric_columns=EXCHANGE_METRICS,
    )
    confusion_mean = _mean_confusion(confusions)
    ensemble = _ensemble_predictions(predictions, class_names)
    bootstrap_overall, bootstrap_class, bootstrap_exchange = _bootstrap_tables(
        ensemble,
        class_names,
        n_bootstrap=int(n_bootstrap),
        random_seed=int(random_seed),
    )
    paired = _paired_bootstrap_differences(
        ensemble,
        class_names,
        n_bootstrap=int(n_bootstrap),
        random_seed=int(random_seed),
    )

    outputs: dict[str, tuple[pd.DataFrame, str]] = {
        "metrics_by_seed": (metrics, "metrics_by_seed.csv"),
        "metrics_seed_summary": (metric_summary, "metrics_seed_summary.csv"),
        "class_metrics_by_seed": (per_class, "class_metrics_by_seed.csv"),
        "class_metrics_seed_summary": (
            class_summary, "class_metrics_seed_summary.csv"
        ),
        "confusion_by_seed": (confusions, "confusion_by_seed.csv"),
        "confusion_mean_across_seeds": (
            confusion_mean, "confusion_mean_across_seeds.csv"
        ),
        "luma_lumb_exchange_by_seed": (
            exchange, "luma_lumb_exchange_by_seed.csv"
        ),
        "luma_lumb_exchange_seed_summary": (
            exchange_summary, "luma_lumb_exchange_seed_summary.csv"
        ),
        "ensemble_predictions": (ensemble, "ensemble_mean_predictions.csv.gz"),
        "bootstrap_overall_95ci": (
            bootstrap_overall, "bootstrap_overall_95ci.csv"
        ),
        "bootstrap_class_95ci": (
            bootstrap_class, "bootstrap_class_95ci.csv"
        ),
        "bootstrap_luma_lumb_exchange_95ci": (
            bootstrap_exchange, "bootstrap_luma_lumb_exchange_95ci.csv"
        ),
        "paired_model_differences_95ci": (
            paired, "paired_model_differences_95ci.csv"
        ),
    }
    paths: dict[str, Path] = {}
    for key, (table, filename) in outputs.items():
        path = output_dir / filename
        compression = "gzip" if filename.endswith(".gz") else None
        table.to_csv(path, index=False, compression=compression)
        paths[key] = path

    # Publication-facing long tables combine seed variability and sample CIs
    # without pretending that seeds and patients are the same replication unit.
    s12_overall = metric_summary.merge(
        bootstrap_overall.rename(columns={"ensemble_estimate": "estimate"}),
        on=["model", "split", "scope", "metric"],
        how="left",
        validate="one_to_one",
    )
    s12_class = class_summary.merge(
        bootstrap_class.rename(columns={"ensemble_estimate": "estimate"}),
        on=["model", "split", "scope", "class_id", "class_name", "metric"],
        how="left",
        validate="one_to_one",
    )
    s12_overall_path = output_dir / "supplementary_table_s12_overall.csv"
    s12_class_path = output_dir / "supplementary_table_s12_by_class.csv"
    s12_overall.to_csv(s12_overall_path, index=False)
    s12_class.to_csv(s12_class_path, index=False)
    paths["supplementary_table_s12_overall"] = s12_overall_path
    paths["supplementary_table_s12_by_class"] = s12_class_path

    manifest = {
        "status": "complete",
        "created_utc": _utc_now(),
        "class_names": list(map(str, class_names)),
        "models": sorted(predictions["model"].unique().tolist()),
        "splits": sorted(predictions["split"].unique().tolist()),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_method": "percentile_stratified_by_cohort_and_true_class",
        "seed_summary": "mean_and_sample_sd_across_five_model_seeds",
        "ensemble_definition": "arithmetic_mean_probability_across_seeds",
        "confusion_definition": "mean_of_seed_specific_count_and_row_normalized_matrices",
        "paired_difference_definition": "InterGATE_minus_comparator_on_same_bootstrap_samples",
        "external_ci_scope": (
            "conditional_on_the_fixed_external_cohorts_not_new_cohort_sampling"
        ),
        "files": {key: path.name for key, path in paths.items()},
    }
    manifest_path = output_dir / "predictive_benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["manifest"] = manifest_path
    return paths


__all__ = [
    "DEFAULT_CLASSES",
    "DEFAULT_SEEDS",
    "TABULAR_MODELS",
    "build_tabular_models",
    "fit_development",
    "evaluate_external",
    "summarize_predictions",
]
