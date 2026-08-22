"""Controls used by Notebook 04.

This module keeps the notebook declarative and enforces three contracts:

1. every tabular baseline uses exactly the ordered connected-gene universe
   consumed by the full InterGATE model;
2. every trained arm differs from FULL in one and only one implementation
   component;
3. the biological readout in this notebook is restricted to the prespecified
   target class (Basal-like), while prediction remains the original four-class
   task needed to define class-contrastive topology.

The external block is never an input to any function in this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight

import intergate.transport as transport_module
from .ablation import (
    CFG_FULL as MODEL_CFG_FULL,
    load_bundle,
    make_prior_for_cfg,
)
from .biology import combine_biology_scores, module_aware_exact_k, weak_topology_summary
from .config import CFG
from .transport import TransportConfig


@dataclass(frozen=True)
class Notebook04Options:
    """Prespecified execution options for Notebook 04."""

    target_class: str = "Basal-like"
    expected_connected_genes: int = 7131
    ablation_run_mode: str = "pilot"
    run_xgboost: bool = True
    run_score_sensitivity: bool = True
    xgboost_device: str = "cpu"
    xgboost_n_jobs: int = -1

    def validate(self) -> None:
        if self.target_class != "Basal-like":
            raise ValueError(
                "Notebook 04 V6 está preespecificado para Basal-like."
            )
        if self.ablation_run_mode not in {"pilot", "confirmatory"}:
            raise ValueError(
                "ablation_run_mode debe ser 'pilot' o 'confirmatory'."
            )
        if int(self.expected_connected_genes) <= 0:
            raise ValueError("expected_connected_genes debe ser positivo.")


@dataclass(frozen=True)
class ArmSpec:
    artifact_name: str
    model_config: Any
    transport: TransportConfig
    changed_component: str
    reused: bool


def safe_name(value: Any) -> str:
    return "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(value)
    )


def _ordered_gene_hash(genes: Sequence[str]) -> str:
    return sha256("\n".join(map(str, genes)).encode("utf-8")).hexdigest()


def audit_baseline_feature_universe(
    runtime: Mapping[str, Any],
    *,
    full_bundle_dir: Path,
    expected_connected_genes: int = 7131,
) -> pd.DataFrame:
    """Fail unless baselines and FULL use the same 7,131 ordered genes."""

    if bool(runtime.get("external_opened", True)):
        raise RuntimeError("El runtime de Notebook 04 abrió el bloque externo.")

    genes = tuple(map(str, runtime["genes"]))
    if len(genes) != len(set(genes)):
        raise RuntimeError("El universo conectado contiene genes duplicados.")
    if len(genes) != int(expected_connected_genes):
        raise RuntimeError(
            "Universo génico inesperado: "
            f"{len(genes)} != {int(expected_connected_genes)}. "
            "No se ejecutarán baselines no comparables."
        )

    X_df = runtime["X_df"]
    missing = [gene for gene in genes if gene not in X_df.columns]
    if missing:
        raise RuntimeError(
            f"Faltan {len(missing)} genes conectados en X_df; "
            f"primeros={missing[:10]}."
        )

    edge_index = np.asarray(runtime["edge_index"], dtype=np.int64)
    observed_nodes = np.unique(edge_index.reshape(-1))
    expected_nodes = np.arange(len(genes), dtype=np.int64)
    if not np.array_equal(observed_nodes, expected_nodes):
        raise RuntimeError(
            "runtime['genes'] no coincide exactamente con los nodos con grado "
            "del prior tras CONNECTED_ONLY."
        )

    bundle = load_bundle(str(full_bundle_dir))
    bundle_genes = tuple(map(str, bundle["graph"]["genes_full"]))
    if bundle_genes != genes:
        raise RuntimeError(
            "El orden/universo de genes del FULL no coincide con el runtime "
            "que se pasaría a los baselines."
        )

    return pd.DataFrame([{
        "feature_universe": "prior_connected_genes_exact_order",
        "n_genes_runtime": int(len(genes)),
        "n_genes_full_bundle": int(len(bundle_genes)),
        "n_nodes_with_prior_degree": int(len(observed_nodes)),
        "expected_connected_genes": int(expected_connected_genes),
        "ordered_gene_sha256": _ordered_gene_hash(genes),
        "same_order_as_full_bundle": True,
        "external_opened": False,
    }])


def _prediction_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    classes: Sequence[str],
    *,
    target_class: str,
) -> dict[str, Any]:
    classes = np.asarray(list(map(str, classes)), dtype=object)
    y_true = np.asarray(y_true).astype(str)
    prediction = np.asarray(prediction).astype(str)
    probability = np.asarray(probability, dtype=float)
    if target_class not in set(classes):
        raise ValueError(f"La clase diana {target_class!r} no está en {classes}.")

    row: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
    }
    try:
        y_binary = label_binarize(y_true, classes=classes)
        row["auc_macro_ovr"] = float(roc_auc_score(
            y_binary, probability, average="macro", multi_class="ovr"
        ))
    except ValueError:
        row["auc_macro_ovr"] = np.nan

    y_target = y_true == str(target_class)
    pred_target = prediction == str(target_class)
    precision, recall, target_f1, _ = precision_recall_fscore_support(
        y_target,
        pred_target,
        average="binary",
        zero_division=0,
    )
    target_column = int(np.flatnonzero(classes == str(target_class))[0])
    row.update({
        "target_class": str(target_class),
        "target_precision": float(precision),
        "target_recall": float(recall),
        "target_f1": float(target_f1),
        "target_support": int(y_target.sum()),
    })
    try:
        row["target_auc_ovr"] = float(
            roc_auc_score(y_target.astype(np.int8), probability[:, target_column])
        )
    except ValueError:
        row["target_auc_ovr"] = np.nan
    return row


def evaluate_predictive_baselines_loco_exact_universe(
    runtime: Mapping[str, Any],
    *,
    seed: int,
    target_class: str,
    run_xgboost: bool = True,
    xgboost_device: str = "cpu",
    xgboost_n_jobs: int = -1,
) -> pd.DataFrame:
    """Four-class LOCO baselines over exactly ``runtime['genes']``."""

    genes = tuple(map(str, runtime["genes"]))
    X = runtime["X_df"].loc[:, list(genes)]
    if tuple(map(str, X.columns)) != genes:
        raise RuntimeError("Se alteró el orden de genes al construir X baseline.")

    y_text = np.asarray(runtime["y_str"]).astype(str)
    cohorts = np.asarray(runtime["cohort"]).astype(str)
    classes = np.asarray(sorted(np.unique(y_text)), dtype=object)
    label_map = {name: class_id for class_id, name in enumerate(classes)}
    y_integer = np.asarray(
        [label_map[value] for value in y_text], dtype=np.int64
    )

    XGBClassifier = None
    if run_xgboost:
        try:
            from xgboost import XGBClassifier as _XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "Falta XGBoost. Instálelo con: pip install 'xgboost>=2.0,<4'"
            ) from exc
        XGBClassifier = _XGBClassifier

    rows: list[dict[str, Any]] = []
    for fold, held_out in enumerate(sorted(np.unique(cohorts))):
        train = cohorts != held_out
        valid = cohorts == held_out
        if set(np.unique(y_text[train])) != set(classes):
            print(f"[baselines] {held_out}: omitida; faltan clases en train.")
            continue

        models = {
            "elastic_net_all_connected": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    class_weight="balanced",
                    max_iter=8000,
                    random_state=int(seed + fold),
                ),
            ),
            "random_forest_all_connected": RandomForestClassifier(
                n_estimators=1000,
                class_weight="balanced_subsample",
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=int(seed + fold),
                n_jobs=-1,
            ),
        }

        for model_name, model in models.items():
            model.fit(X.loc[train], y_text[train])
            prediction = model.predict(X.loc[valid])
            probability = model.predict_proba(X.loc[valid])
            model_classes = np.asarray(model.classes_, dtype=object)
            rows.append({
                "model": model_name,
                "held_out_cohort": str(held_out),
                "n_genes": int(len(genes)),
                "feature_universe": "prior_connected_genes_exact_order",
                "feature_universe_sha256": _ordered_gene_hash(genes),
                "role": "predictive_only_four_class",
                "class_balance": (
                    "class_weight_balanced" if model_name.startswith("elastic")
                    else "balanced_subsample"
                ),
                "hyperparameter_selection": "fixed_before_LOCO_no_heldout_tuning",
                **_prediction_metrics(
                    y_text[valid], prediction, probability, model_classes,
                    target_class=target_class,
                ),
            })

        if XGBClassifier is not None:
            kwargs = dict(
                n_estimators=600,
                max_depth=3,
                learning_rate=0.03,
                min_child_weight=5.0,
                subsample=0.80,
                colsample_bytree=0.35,
                reg_alpha=0.10,
                reg_lambda=2.0,
                objective="multi:softprob",
                num_class=int(len(classes)),
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=int(seed + fold),
                n_jobs=int(xgboost_n_jobs),
                verbosity=0,
            )
            if str(xgboost_device).lower() != "cpu":
                kwargs["device"] = str(xgboost_device)
            model = XGBClassifier(**kwargs)
            sample_weight = compute_sample_weight(
                class_weight="balanced", y=y_integer[train]
            )
            X_train = X.loc[train].to_numpy(dtype=np.float32)
            X_valid = X.loc[valid].to_numpy(dtype=np.float32)
            model.fit(
                X_train,
                y_integer[train],
                sample_weight=sample_weight,
            )
            probability = model.predict_proba(X_valid)
            prediction_integer = np.asarray(model.predict(X_valid), dtype=int)
            prediction = classes[prediction_integer]
            rows.append({
                "model": "xgboost_balanced_all_connected",
                "held_out_cohort": str(held_out),
                "n_genes": int(len(genes)),
                "feature_universe": "prior_connected_genes_exact_order",
                "feature_universe_sha256": _ordered_gene_hash(genes),
                "role": "predictive_only_four_class",
                "class_balance": "balanced_sample_weight_on_training_fold",
                "hyperparameter_selection": "fixed_before_LOCO_no_heldout_tuning",
                "xgboost_device": str(xgboost_device),
                **_prediction_metrics(
                    y_text[valid], prediction, probability, classes,
                    target_class=target_class,
                ),
            })

    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("No se generó ningún fold LOCO válido.")
    if set(output["n_genes"].astype(int)) != {len(genes)}:
        raise RuntimeError("Algún baseline no usó el universo génico completo.")
    output["evaluation_design"] = "leave_one_development_cohort_out"
    output["external_opened"] = False
    return output


def build_trained_arm_specs(
    base_transport: TransportConfig,
) -> dict[str, ArmSpec]:
    """Return FULL plus four single-component trained arms.

    ``NO_SIGNED_WEIGHT`` deliberately means removal of the scalar sign by
    replacing ``w`` with ``abs(w)``. Relation labels are retained, so this arm
    must not be described as removing every possible sign-related annotation.
    """

    return {
        "FULL": ArmSpec(
            artifact_name="INTERGATE_V6",
            model_config=MODEL_CFG_FULL,
            transport=base_transport,
            changed_component="none_reference",
            reused=True,
        ),
        "NO_CONTEXT_LOSS": ArmSpec(
            artifact_name="INTERGATE_V6_ABL_NO_CONTEXT_LOSS",
            model_config=replace(MODEL_CFG_FULL, name="NO_CONTEXT_LOSS"),
            transport=replace(base_transport, biology_context_lambda=0.0),
            changed_component="biology_context_loss_weight",
            reused=False,
        ),
        "NO_COHORT_QUERY_LOSS": ArmSpec(
            artifact_name="INTERGATE_V6_ABL_NO_COHORT_QUERY_LOSS",
            model_config=replace(
                MODEL_CFG_FULL, name="NO_COHORT_QUERY_LOSS"
            ),
            transport=replace(base_transport, meta_validation_weight=0.0),
            changed_component="heldout_cohort_query_loss_weight",
            reused=False,
        ),
        "NO_EDGE_TYPE_CALIBRATION": ArmSpec(
            artifact_name="INTERGATE_V6_ABL_NO_EDGE_TYPE_CALIBRATION",
            model_config=replace(
                MODEL_CFG_FULL,
                name="NO_EDGE_TYPE_CALIBRATION",
                edge_type_gating=False,
            ),
            transport=base_transport,
            changed_component="edge_type_gate_calibration",
            reused=False,
        ),
        "NO_SIGNED_WEIGHT": ArmSpec(
            artifact_name="INTERGATE_V6_ABL_NO_SIGNED_WEIGHT",
            model_config=replace(
                MODEL_CFG_FULL,
                name="NO_SIGNED_WEIGHT",
                use_signed_prior=False,
            ),
            transport=base_transport,
            changed_component="signed_scalar_prior_weight",
            reused=False,
        ),
    }


def _changed_fields(reference: Any, candidate: Any, ignored=()) -> set[str]:
    reference_dict = asdict(reference)
    candidate_dict = asdict(candidate)
    ignored = set(ignored)
    return {
        key for key in reference_dict
        if key not in ignored and reference_dict[key] != candidate_dict[key]
    }


def audit_single_component_ablations(
    specs: Mapping[str, ArmSpec],
) -> pd.DataFrame:
    """Prove at configuration and effective-prior level that arms are atomic."""

    expected = {
        "FULL": set(),
        "NO_CONTEXT_LOSS": {"transport.biology_context_lambda"},
        "NO_COHORT_QUERY_LOSS": {"transport.meta_validation_weight"},
        "NO_EDGE_TYPE_CALIBRATION": {"model.edge_type_gating"},
        "NO_SIGNED_WEIGHT": {"model.use_signed_prior"},
    }
    if set(specs) != set(expected):
        raise RuntimeError(
            f"Brazos inesperados: {sorted(specs)} != {sorted(expected)}."
        )

    full_ei, full_ew, full_et = make_prior_for_cfg(MODEL_CFG_FULL)
    full_ei = full_ei.detach().cpu()
    full_ew = full_ew.detach().cpu()
    full_et = full_et.detach().cpu()
    rows = []
    for arm, spec in specs.items():
        model_diffs = {
            f"model.{field}" for field in _changed_fields(
                MODEL_CFG_FULL, spec.model_config, ignored={"name"}
            )
        }
        transport_diffs = {
            f"transport.{field}" for field in _changed_fields(
                specs["FULL"].transport, spec.transport
            )
        }
        observed = model_diffs | transport_diffs
        if observed != expected[arm]:
            raise RuntimeError(
                f"{arm}: diferencias {sorted(observed)}; "
                f"esperadas={sorted(expected[arm])}."
            )

        edge_index, edge_weight, edge_type = make_prior_for_cfg(
            spec.model_config
        )
        edge_index = edge_index.detach().cpu()
        edge_weight = edge_weight.detach().cpu()
        edge_type = edge_type.detach().cpu()
        if not bool((edge_index == full_ei).all()):
            raise RuntimeError(f"{arm}: cambió edge_index.")
        if not bool((edge_type == full_et).all()):
            raise RuntimeError(f"{arm}: cambió edge_type.")
        if arm == "NO_SIGNED_WEIGHT":
            if not bool((edge_weight == full_ew.abs()).all()):
                raise RuntimeError(
                    "NO_SIGNED_WEIGHT no equivale exactamente a abs(w)."
                )
        elif not bool((edge_weight == full_ew).all()):
            raise RuntimeError(f"{arm}: cambió el peso del prior.")

        rows.append({
            "arm": arm,
            "changed_component": spec.changed_component,
            "changed_fields": ";".join(sorted(observed)),
            "n_changed_fields": int(len(observed)),
            "expected_n_changed_fields": int(0 if arm == "FULL" else 1),
            "edge_index_identical": True,
            "edge_type_tensor_identical": True,
            "edge_weight_transform": (
                "absolute_value_only" if arm == "NO_SIGNED_WEIGHT"
                else "identical"
            ),
            "single_component_contract_passed": True,
        })
    return pd.DataFrame(rows)


@contextmanager
def _temporary_transport_model_config(model_config: Any):
    original = transport_module.CFG_FULL
    transport_module.CFG_FULL = model_config
    try:
        yield
    finally:
        transport_module.CFG_FULL = original


def _read_completed_metrics(root: Path, seed: int) -> tuple[dict, dict]:
    meta_path = Path(root) / f"seed_{int(seed)}" / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Falta bundle completado para seed={seed}: {meta_path}."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta, dict(meta["metrics"])


def _assert_saved_arm_contract(
    meta: Mapping[str, Any], arm: str, spec: ArmSpec
) -> None:
    checks = {
        "edge_type_gating": bool(spec.model_config.edge_type_gating),
        "use_signed_prior": bool(spec.model_config.use_signed_prior),
    }
    for field, expected_value in checks.items():
        observed = bool(meta.get(field))
        if observed != expected_value:
            raise RuntimeError(
                f"{arm}: {field}={observed}, esperado={expected_value}."
            )
    protocol = meta.get("transport_protocol", {})
    for field in (
        "biology_context_lambda",
        "meta_validation_weight",
        "biology_score_context_weight",
        "biology_score_support_weight",
    ):
        observed = float(protocol.get(field, np.nan))
        expected_value = float(getattr(spec.transport, field))
        if not np.isclose(observed, expected_value, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"{arm}: transport.{field}={observed}, "
                f"esperado={expected_value}."
            )
    if bool(meta["metrics"].get("external_opened", True)):
        raise RuntimeError(f"{arm}: una salida abrió el bloque externo.")


def run_trained_single_component_ablations(
    *,
    specs: Mapping[str, ArmSpec],
    seeds: Sequence[int],
    artifacts_root: Path,
    target_class: str,
    run_mode: str,
) -> pd.DataFrame:
    """Run/reuse atomic arms and report four-class plus Basal-like F1."""

    rows = []
    target_metric = f"f1_{target_class}"
    for seed in map(int, seeds):
        for arm, spec in specs.items():
            root = Path(artifacts_root) / spec.artifact_name
            if spec.reused:
                meta, metrics = _read_completed_metrics(root, seed)
            else:
                print(f"\n[ablación atómica] arm={arm} seed={seed}")
                with _temporary_transport_model_config(spec.model_config):
                    returned = transport_module.run_transport_seed(
                        seed,
                        save_root=str(artifacts_root),
                        transport=spec.transport,
                        resume=True,
                        force=False,
                        artifact_name=spec.artifact_name,
                    )
                meta, metrics = _read_completed_metrics(root, seed)
                for metric_name in ("macro_f1", "bal_acc", target_metric):
                    if not np.isclose(
                        float(returned[metric_name]),
                        float(metrics[metric_name]),
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        raise RuntimeError(
                            f"{arm}/{seed}: retorno y bundle discrepan en "
                            f"{metric_name}."
                        )
            _assert_saved_arm_contract(meta, arm, spec)
            rows.append({
                "arm": arm,
                "seed": seed,
                "target_class": str(target_class),
                "changed_component": spec.changed_component,
                "artifact_name": spec.artifact_name,
                "run_mode": str(run_mode),
                "reused_full": bool(spec.reused),
                "edge_type_gating": bool(spec.model_config.edge_type_gating),
                "use_signed_prior": bool(spec.model_config.use_signed_prior),
                "biology_context_lambda": float(
                    spec.transport.biology_context_lambda
                ),
                "meta_validation_weight": float(
                    spec.transport.meta_validation_weight
                ),
                **metrics,
            })

    output = pd.DataFrame(rows)
    full = output.loc[output.arm.eq("FULL")].set_index("seed")
    output["full_macro_f1_same_seed"] = output.seed.map(full["macro_f1"])
    output["full_target_f1_same_seed"] = output.seed.map(full[target_metric])
    output["delta_macro_f1_vs_full_same_seed"] = (
        output.macro_f1 - output.full_macro_f1_same_seed
    )
    output["delta_target_f1_vs_full_same_seed"] = (
        output[target_metric] - output.full_target_f1_same_seed
    )
    output["external_opened"] = False
    return output


def _target_biology_edge_set(
    run_dir: Path, target_class: str
) -> tuple[pd.DataFrame, set[tuple[str, str, int]]]:
    path = Path(run_dir) / f"biology_edges_{safe_name(target_class)}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty or set(frame["class_name"].astype(str)) != {target_class}:
        raise RuntimeError(f"Contenido de clase inesperado en {path}.")
    if "module_aware_exact_k" in frame.columns:
        frame = frame.loc[frame["module_aware_exact_k"].astype(bool)].copy()
    keys = set(zip(
        frame["source_gene"].astype(str),
        frame["target_gene"].astype(str),
        frame["edge_type"].astype(int),
    ))
    return frame, keys


def summarise_target_biology_ablations(
    *,
    specs: Mapping[str, ArmSpec],
    seeds: Sequence[int],
    artifacts_root: Path,
    target_class: str,
) -> pd.DataFrame:
    """Compare only the Basal-like exact-K network against matched FULL."""

    rows = []
    for seed in map(int, seeds):
        full_dir = Path(artifacts_root) / specs["FULL"].artifact_name / f"seed_{seed}"
        _, full_set = _target_biology_edge_set(full_dir, target_class)
        for arm, spec in specs.items():
            run_dir = Path(artifacts_root) / spec.artifact_name / f"seed_{seed}"
            _, selected = _target_biology_edge_set(run_dir, target_class)
            union = selected | full_set
            support = json.loads(
                (run_dir / "transport_support_summary.json").read_text(
                    encoding="utf-8"
                )
            )["classes"][target_class]
            topology = support["biology_topology"]["module_aware"]
            rows.append({
                "arm": arm,
                "seed": seed,
                "class_name": target_class,
                "n_edges": int(len(selected)),
                "intersection_with_full_same_seed": int(
                    len(selected & full_set)
                ),
                "jaccard_with_full_same_seed": float(
                    len(selected & full_set) / len(union) if union else np.nan
                ),
                "strict_all_cohort_edges": int(
                    support["strict_all_cohort_edges"]
                ),
                "mean_transport_support_final": float(
                    support["mean_support_final"]
                ),
                "n_nodes": int(topology["n_nodes"]),
                "n_components": int(topology["n_components"]),
                "dyad_components": int(topology["dyad_components"]),
                "giant_fraction": float(topology["giant_fraction"]),
                "external_opened": False,
            })
    return pd.DataFrame(rows)


def _structural_units(dense: Any) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    unit_id = np.asarray(dense["structural_unit_id"], dtype=np.int64)
    values = np.unique(unit_id[unit_id >= 0])
    groups = [np.flatnonzero(unit_id == value) for value in values]
    widths = {len(group) for group in groups}
    if len(widths) != 1:
        raise RuntimeError(f"Anchuras estructurales mezcladas: {sorted(widths)}")
    first_ids = np.asarray([group[0] for group in groups], dtype=np.int64)
    return unit_id, groups, first_ids


def _rebuild_target_biology_network(
    runtime: Mapping[str, Any],
    *,
    run_dir: Path,
    seed: int,
    target_class: str,
    score_variant: str,
    context_weight: float,
    support_weight: float,
    cfg=CFG,
) -> tuple[pd.DataFrame, dict[str, Any], set[int]]:
    dense = np.load(
        Path(run_dir) / "transport_support_dense.npz", allow_pickle=True
    )
    class_names = list(map(str, dense["class_names"].tolist()))
    if target_class not in class_names:
        raise ValueError(f"{target_class} no está en {class_names}.")
    class_id = int(class_names.index(target_class))
    delta = np.asarray(dense["differential_logits"], dtype=float)
    context = np.asarray(dense["context_score"], dtype=float)
    support = np.asarray(dense["selection_frequency"], dtype=float)
    saved_selected = np.asarray(dense["biology_selected"], dtype=bool)
    unit_id, groups, first_ids = _structural_units(dense)
    selected_unit_count = len(np.unique(
        unit_id[saved_selected[class_id] & (unit_id >= 0)]
    ))
    if selected_unit_count <= 0:
        raise RuntimeError("No se pudo recuperar la cuota estructural Exact-K.")

    scores = combine_biology_scores(
        delta,
        context,
        support,
        context_weight=float(context_weight),
        support_weight=float(support_weight),
    )
    chosen_units = module_aware_exact_k(
        np.asarray(runtime["edge_index"])[:, first_ids],
        scores[class_id, first_ids],
        k_edges=selected_unit_count,
        pool_multiplier=int(cfg.BIOLOGY_MODULE_POOL_MULTIPLIER),
        target_components=int(cfg.BIOLOGY_MODULE_TARGET_COMPONENTS),
        min_component_edges=int(cfg.BIOLOGY_MODULE_MIN_EDGES),
    )
    selected_ids = np.concatenate([
        groups[int(unit_position)] for unit_position in chosen_units
    ]).astype(np.int64)
    selected_ids = selected_ids[
        np.lexsort((selected_ids, -scores[class_id, selected_ids]))
    ]
    expected_edges = int(saved_selected[class_id].sum())
    if len(selected_ids) != expected_edges:
        raise RuntimeError(
            f"Reconstrucción target {len(selected_ids)} != {expected_edges}."
        )

    saved_ids = set(map(int, np.flatnonzero(saved_selected[class_id])))
    selected_set = set(map(int, selected_ids))
    union = selected_set | saved_ids
    topology = weak_topology_summary(
        np.asarray(runtime["edge_index"])[:, selected_ids]
    )
    summary = {
        "seed": int(seed),
        "class_name": target_class,
        "score_variant": score_variant,
        "score_context_weight": float(context_weight),
        "score_support_weight": float(support_weight),
        "n_edges": int(len(selected_ids)),
        "reference_saved_graph_jaccard": float(
            len(selected_set & saved_ids) / len(union) if union else np.nan
        ),
        **topology,
        "external_opened": False,
    }

    edge_index = np.asarray(runtime["edge_index"], dtype=np.int64)
    edge_weight = np.asarray(runtime["edge_weight"], dtype=float)
    edge_type = np.asarray(runtime["edge_type"], dtype=np.int64)
    genes = np.asarray(list(map(str, runtime["genes"])), dtype=object)
    table = pd.DataFrame({
        "biology_rank": np.arange(1, len(selected_ids) + 1),
        "seed": int(seed),
        "class_name": target_class,
        "score_variant": score_variant,
        "candidate_edge_id": selected_ids,
        "source_gene": genes[edge_index[0, selected_ids]],
        "target_gene": genes[edge_index[1, selected_ids]],
        "prior_weight": edge_weight[selected_ids],
        "edge_type": edge_type[selected_ids],
        "differential_logit": delta[class_id, selected_ids],
        "differential_direction": np.where(
            delta[class_id, selected_ids] >= 0,
            "strengthened_for_class",
            "attenuated_for_class",
        ),
        "context_score": context[class_id, selected_ids],
        "transport_support": support[class_id, selected_ids],
        "biology_score": scores[class_id, selected_ids],
    })
    return table, summary, selected_set


def run_target_score_sensitivity(
    runtime: Mapping[str, Any],
    *,
    run_root: Path,
    output_dir: Path,
    seeds: Sequence[int],
    base_transport: TransportConfig,
    target_class: str,
    cfg=CFG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-hoc score sensitivity restricted to Basal-like."""

    variants = {
        "FULL_SCORE": (
            float(base_transport.biology_score_context_weight),
            float(base_transport.biology_score_support_weight),
        ),
        "NO_CONTEXT_SCORE": (
            0.0,
            float(base_transport.biology_score_support_weight),
        ),
        "NO_SUPPORT_SCORE": (
            float(base_transport.biology_score_context_weight),
            0.0,
        ),
        "PRIMARY_DIFFERENTIAL_ONLY": (0.0, 0.0),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    sets: dict[tuple[int, str], set[int]] = {}
    for seed in map(int, seeds):
        run_dir = Path(run_root) / f"seed_{seed}"
        for variant, (context_weight, support_weight) in variants.items():
            table, summary, selected = _rebuild_target_biology_network(
                runtime,
                run_dir=run_dir,
                seed=seed,
                target_class=target_class,
                score_variant=variant,
                context_weight=context_weight,
                support_weight=support_weight,
                cfg=cfg,
            )
            table.to_csv(
                output_dir /
                f"edges_seed_{seed}_{variant}_{safe_name(target_class)}.csv",
                index=False,
            )
            summaries.append(summary)
            sets[(seed, variant)] = selected

    summary_frame = pd.DataFrame(summaries)
    reference = summary_frame.loc[
        summary_frame.score_variant.eq("FULL_SCORE")
    ]
    if not reference.reference_saved_graph_jaccard.ge(1.0 - 1e-12).all():
        raise RuntimeError("FULL_SCORE no reconstruyó el grafo target guardado.")

    overlaps = []
    for seed in map(int, seeds):
        full = sets[(seed, "FULL_SCORE")]
        for variant in variants:
            selected = sets[(seed, variant)]
            union = full | selected
            overlaps.append({
                "seed": seed,
                "class_name": target_class,
                "score_variant": variant,
                "intersection_with_full_score": int(len(full & selected)),
                "jaccard_with_full_score": float(
                    len(full & selected) / len(union) if union else np.nan
                ),
                "external_opened": False,
            })
    overlap_frame = pd.DataFrame(overlaps)
    return summary_frame, overlap_frame


def save_notebook04_outputs(
    *,
    output_dir: Path,
    feature_audit: pd.DataFrame,
    baselines: pd.DataFrame,
    ablation_contract: pd.DataFrame,
    trained_metrics: pd.DataFrame,
    target_biology: pd.DataFrame,
    score_summary: Optional[pd.DataFrame] = None,
    score_overlap: Optional[pd.DataFrame] = None,
) -> None:
    """Write stable filenames consumed by the lock notebook."""

    output_dir = Path(output_dir)
    baseline_dir = output_dir / "predictive_baselines"
    ablation_dir = output_dir / "trained_ablations"
    score_dir = output_dir / "posthoc_score_sensitivity_Basal-like"
    for directory in (output_dir, baseline_dir, ablation_dir, score_dir):
        directory.mkdir(parents=True, exist_ok=True)

    feature_audit.to_csv(
        output_dir / "baseline_feature_universe_audit.csv", index=False
    )
    baselines.to_csv(
        baseline_dir / "predictive_baselines_loco.csv", index=False
    )
    baselines.to_csv(output_dir / "predictive_baselines_loco.csv", index=False)
    ablation_contract.to_csv(
        ablation_dir / "single_component_ablation_contract.csv", index=False
    )
    trained_metrics.to_csv(
        ablation_dir / "trained_ablation_predictive_metrics.csv", index=False
    )
    target_biology.to_csv(
        ablation_dir / "trained_ablation_biology_Basal-like.csv", index=False
    )

    first_seed = int(trained_metrics.seed.drop_duplicates().iloc[0])
    context_pair = trained_metrics.loc[
        trained_metrics.seed.eq(first_seed)
        & trained_metrics.arm.isin(["FULL", "NO_CONTEXT_LOSS"])
    ].copy()
    context_pair.to_csv(
        output_dir / "one_seed_context_ablation.csv", index=False
    )
    context_pair.to_csv(
        output_dir / "one_seed_context_loss_ablation.csv", index=False
    )
    if score_summary is not None:
        score_summary.to_csv(
            score_dir / "posthoc_score_network_summary_Basal-like.csv",
            index=False,
        )
    if score_overlap is not None:
        score_overlap.to_csv(
            score_dir / "posthoc_score_edge_overlap_Basal-like.csv",
            index=False,
        )


def options_from_environment() -> Notebook04Options:
    """Convenience constructor used by the compact notebook."""

    return Notebook04Options(
        target_class="Basal-like",
        expected_connected_genes=7131,
        ablation_run_mode=os.environ.get(
            "INTERGATE_ABLATION_RUN_MODE", "pilot"
        ),
        run_xgboost=os.environ.get("INTERGATE_RUN_XGBOOST", "1") != "0",
        run_score_sensitivity=(
            os.environ.get("INTERGATE_RUN_SCORE_SENSITIVITY", "1") != "0"
        ),
        xgboost_device=os.environ.get("INTERGATE_XGBOOST_DEVICE", "cpu"),
        xgboost_n_jobs=int(
            os.environ.get("INTERGATE_XGBOOST_N_JOBS", "-1")
        ),
    )
