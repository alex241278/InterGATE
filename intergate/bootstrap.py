"""
Bootstrap confidence intervals for classification metrics and AUC.

Usage from notebook::

    from intergate.bootstrap import (
        scores_from_proba, tune_class_biases, predict_with_class_bias,
        bootstrap_classification_ci, bootstrap_auc_ovr_ci,
    )

    # Class bias calibration
    scores_va = scores_from_proba(proba_va)
    scores_te = scores_from_proba(proba_te)
    bias_opt, best_val_f1 = tune_class_biases(scores_va, y_va, classes=classes)
    pred_te_cal = predict_with_class_bias(scores_te, bias_opt)

    # Bootstrap CI (hard labels)
    boot_cls = bootstrap_classification_ci(y_te, pred_te_cal, classes, B=10000)
    display(boot_cls["overall"])

    # Bootstrap CI (AUC)
    boot_auc = bootstrap_auc_ovr_ci(y_te, proba_te, classes, B=10000)
    display(boot_auc["macro"])
"""

from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt


# ── Scores from proba ────────────────────────────


def scores_from_proba(proba, eps=1e-12):
    """
    Convierte probabilidades en scores comparables para argmax.
    log(proba) sirve porque softmax(logits) => logits = log(proba) + cte por muestra.
    """
    p = np.clip(np.asarray(proba, dtype=float), eps, 1.0)
    return np.log(p)


def predict_with_class_bias(scores, bias):
    """
    scores: (n, C)  -> logits o log-probs
    bias:   (C,)
    """
    scores = np.asarray(scores, dtype=float)
    bias = np.asarray(bias, dtype=float)
    return np.argmax(scores + bias[None, :], axis=1)


def tune_class_biases(
    scores_val,
    y_val,
    classes=None,
    init_bias=None,
    ref_class=None,
    coarse_grid=None,
    fine_grid=None,
    n_passes=3,
    verbose=True,
):
    """
    Búsqueda greedy por coordenadas para maximizar macro-F1 en validación.

    - Ajusta un bias por clase.
    - Fija una clase de referencia (bias=0) para evitar indeterminación.
    - Optimiza macro-F1 MULTICLASE real, no F1 binario por clase.
    """
    scores_val = np.asarray(scores_val, dtype=float)
    y_val = np.asarray(y_val, dtype=int)

    n_classes = scores_val.shape[1]

    if coarse_grid is None:
        coarse_grid = np.linspace(-1.5, 1.5, 61)   # paso 0.05
    if fine_grid is None:
        fine_grid = np.linspace(-0.30, 0.30, 61)   # paso 0.01

    if init_bias is None:
        bias = np.zeros(n_classes, dtype=float)
    else:
        bias = np.asarray(init_bias, dtype=float).copy()
        assert bias.shape == (n_classes,)

    # clase de referencia: por defecto la más frecuente
    if ref_class is None:
        ref_class = int(np.argmax(np.bincount(y_val, minlength=n_classes)))

    # eval inicial
    pred0 = predict_with_class_bias(scores_val, bias)
    best_f1 = f1_score(y_val, pred0, average="macro")

    if verbose:
        ref_name = classes[ref_class] if classes is not None else ref_class
        print(f"[BIAS] Clase de referencia fija: {ref_name}")
        print(f"[BIAS] Macro-F1 inicial (sin/antes de calibrar): {best_f1:.4f}")

    order = [c for c in range(n_classes) if c != ref_class]

    for p in range(n_passes):
        grid = coarse_grid if p == 0 else fine_grid
        improved_any = False

        for c in order:
            current = bias[c]
            local_best_val = current
            local_best_f1 = best_f1

            for delta in grid:
                cand = bias.copy()
                cand[c] = current + float(delta)

                pred = predict_with_class_bias(scores_val, cand)
                f1m = f1_score(y_val, pred, average="macro")

                if f1m > local_best_f1 + 1e-12:
                    local_best_f1 = f1m
                    local_best_val = cand[c]

            if local_best_val != current:
                bias[c] = local_best_val
                best_f1 = local_best_f1
                improved_any = True

                if verbose:
                    cname = classes[c] if classes is not None else c
                    print(f"[BIAS] pass={p+1} clase={cname} -> bias={bias[c]:+.3f} | macroF1={best_f1:.4f}")

        if not improved_any:
            if verbose:
                print(f"[BIAS] Sin mejoras en pass {p+1}; paro.")
            break

    return bias, best_f1


# ── Bootstrap CI (classification metrics) ────────


def _metrics_from_confusion(cm: np.ndarray):
    """
    Calcula métricas globales y por clase a partir de una matriz de confusión CxC.
    Filas = clase real, columnas = clase predicha.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)   # reales por clase
    pred_pos = cm.sum(axis=0)  # predichos por clase
    total = cm.sum()

    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall    = np.divide(tp, support,  out=np.zeros_like(tp), where=support > 0)

    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    acc = float(tp.sum() / total) if total > 0 else np.nan
    macro_precision = float(np.mean(precision)) if precision.size else np.nan
    macro_recall    = float(np.mean(recall)) if recall.size else np.nan
    macro_f1        = float(np.mean(f1)) if f1.size else np.nan
    weighted_f1     = float(np.average(f1, weights=support)) if total > 0 else np.nan

    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support_per_class": support,
        "confusion_matrix": cm,
    }


def _confusion_from_labels(y_true_enc: np.ndarray, y_pred_enc: np.ndarray, n_classes: int):
    """Construye matriz de confusión rápida con np.bincount."""
    idx = n_classes * y_true_enc + y_pred_enc
    cm = np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    return cm


def bootstrap_classification_ci(
    y_true,
    y_pred,
    class_names=None,
    B: int = 10_000,
    seed: int = 42,
    ci=(2.5, 97.5),
    return_distributions: bool = False,
):
    """
    Bootstrap no paramétrico para métricas discretas de clasificación multiclase.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.sort(labels)

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y_true_enc = np.array([label_to_idx[v] for v in y_true], dtype=np.int64)
    y_pred_enc = np.array([label_to_idx[v] for v in y_pred], dtype=np.int64)

    n = len(y_true_enc)
    C = len(labels)

    if class_names is None:
        class_names = [str(l) for l in labels]
    else:
        class_names = list(class_names)
        if len(class_names) != C:
            raise ValueError(f"class_names tiene longitud {len(class_names)} y se esperaban {C}")

    # Métricas observadas (point estimate)
    cm_obs = _confusion_from_labels(y_true_enc, y_pred_enc, C)
    obs = _metrics_from_confusion(cm_obs)

    # Arrays bootstrap
    overall_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    overall_boot = {m: np.empty(B, dtype=np.float64) for m in overall_names}
    per_class_boot = {
        "precision": np.empty((B, C), dtype=np.float64),
        "recall":    np.empty((B, C), dtype=np.float64),
        "f1":        np.empty((B, C), dtype=np.float64),
    }

    rng = np.random.default_rng(seed)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yt_b = y_true_enc[idx]
        yp_b = y_pred_enc[idx]

        cm_b = _confusion_from_labels(yt_b, yp_b, C)
        mb = _metrics_from_confusion(cm_b)

        for m in overall_names:
            overall_boot[m][b] = mb[m]

        per_class_boot["precision"][b, :] = mb["precision_per_class"]
        per_class_boot["recall"][b, :]    = mb["recall_per_class"]
        per_class_boot["f1"][b, :]        = mb["f1_per_class"]

    alpha_low, alpha_high = ci

    # Tabla global
    overall_rows = []
    for m in overall_names:
        low, high = np.percentile(overall_boot[m], [alpha_low, alpha_high])
        overall_rows.append({
            "metric": m,
            "point_estimate": float(obs[m]),
            "ci_low": float(low),
            "ci_high": float(high),
            "ci_95_percentile": f"[{low:.4f}, {high:.4f}]",
        })
    df_overall = pd.DataFrame(overall_rows)

    # Tabla por clase
    per_class_rows = []
    for j, cls in enumerate(class_names):
        for metric_key, obs_key in [
            ("precision", "precision_per_class"),
            ("recall", "recall_per_class"),
            ("f1", "f1_per_class"),
        ]:
            vals = per_class_boot[metric_key][:, j]
            low, high = np.percentile(vals, [alpha_low, alpha_high])
            per_class_rows.append({
                "class": cls,
                "metric": metric_key,
                "point_estimate": float(obs[obs_key][j]),
                "ci_low": float(low),
                "ci_high": float(high),
                "ci_95_percentile": f"[{low:.4f}, {high:.4f}]",
                "support": int(obs["support_per_class"][j]),
            })
    df_per_class = pd.DataFrame(per_class_rows)

    result = {
        "overall": df_overall,
        "per_class": df_per_class,
        "observed_confusion_matrix": pd.DataFrame(cm_obs, index=class_names, columns=class_names),
    }

    if return_distributions:
        result["bootstrap_distributions"] = {
            "overall": overall_boot,
            "per_class": per_class_boot,
        }

    return result


# ── Bootstrap CI (AUC OVR) ─────────────────────


def _compute_ovr_auc_observed(y_true_enc: np.ndarray, proba: np.ndarray):
    """
    Calcula AUC OVR por clase y macro sobre el conjunto observado.
    """
    n_classes = proba.shape[1]
    auc_per_class = np.full(n_classes, np.nan, dtype=float)

    for c in range(n_classes):
        y_bin = (y_true_enc == c).astype(int)
        if np.unique(y_bin).size < 2:
            continue
        auc_per_class[c] = roc_auc_score(y_bin, proba[:, c])

    auc_macro = float(np.nanmean(auc_per_class)) if np.any(~np.isnan(auc_per_class)) else np.nan
    return auc_macro, auc_per_class


def bootstrap_auc_ovr_ci(
    y_true,
    proba,
    class_names=None,
    B: int = 10_000,
    seed: int = 42,
    ci=(2.5, 97.5),
    return_distributions: bool = False,
):
    """
    Bootstrap no paramétrico para AUC one-vs-rest (multiclase).

    Parámetros
    ----------
    y_true : array-like shape (n_samples,)
        Etiquetas reales (enteros o strings).
    proba : array-like shape (n_samples, n_classes)
        Probabilidades por clase.
    class_names : list[str] o None
        Nombres de clase alineados con columnas de proba.
    """
    y_true = np.asarray(y_true).ravel()
    proba = np.asarray(proba, dtype=float)

    if proba.ndim != 2:
        raise ValueError(f"`proba` debe ser 2D, recibido shape={proba.shape}")
    if len(y_true) != proba.shape[0]:
        raise ValueError(f"n_samples inconsistente: len(y_true)={len(y_true)} vs proba.shape[0]={proba.shape[0]}")

    n_samples, n_classes = proba.shape

    unique_labels = np.sort(np.unique(y_true))

    if np.array_equal(unique_labels, np.arange(n_classes)):
        y_true_enc = y_true.astype(int)
        labels = list(range(n_classes))
    else:
        if len(unique_labels) != n_classes:
            raise ValueError(
                f"Hay {len(unique_labels)} etiquetas únicas en y_true pero {n_classes} columnas en proba."
            )
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        y_true_enc = np.array([label_to_idx[v] for v in y_true], dtype=np.int64)
        labels = list(unique_labels)

    if class_names is None:
        class_names = [str(x) for x in labels]
    else:
        class_names = list(class_names)
        if len(class_names) != n_classes:
            raise ValueError(f"class_names tiene longitud {len(class_names)} y proba tiene {n_classes} columnas")

    # Métricas observadas
    obs_macro, obs_per_class = _compute_ovr_auc_observed(y_true_enc, proba)

    # Bootstrap
    rng = np.random.default_rng(seed)
    auc_macro_boot = np.empty(B, dtype=float)
    auc_per_class_boot = np.full((B, n_classes), np.nan, dtype=float)

    for b in range(B):
        idx = rng.integers(0, n_samples, size=n_samples)
        y_b = y_true_enc[idx]
        p_b = proba[idx]

        aucs_b = np.full(n_classes, np.nan, dtype=float)
        for c in range(n_classes):
            y_bin = (y_b == c).astype(int)
            if np.unique(y_bin).size < 2:
                continue
            try:
                aucs_b[c] = roc_auc_score(y_bin, p_b[:, c])
            except ValueError:
                aucs_b[c] = np.nan

        auc_per_class_boot[b, :] = aucs_b
        auc_macro_boot[b] = np.nanmean(aucs_b) if np.any(~np.isnan(aucs_b)) else np.nan

    alpha_low, alpha_high = ci

    # Resumen macro
    valid_macro = auc_macro_boot[~np.isnan(auc_macro_boot)]
    if valid_macro.size == 0:
        macro_low, macro_high = np.nan, np.nan
    else:
        macro_low, macro_high = np.percentile(valid_macro, [alpha_low, alpha_high])

    df_macro = pd.DataFrame([{
        "metric": "auc_ovr_macro",
        "point_estimate": float(obs_macro) if not np.isnan(obs_macro) else np.nan,
        "ci_low": float(macro_low) if not np.isnan(macro_low) else np.nan,
        "ci_high": float(macro_high) if not np.isnan(macro_high) else np.nan,
        "ci_95_percentile": (
            f"[{macro_low:.4f}, {macro_high:.4f}]"
            if (not np.isnan(macro_low) and not np.isnan(macro_high)) else "NA"
        ),
        "n_boot_valid": int(valid_macro.size),
        "n_boot_total": int(B),
    }])

    # Resumen por clase
    rows = []
    for c, cls_name in enumerate(class_names):
        vals = auc_per_class_boot[:, c]
        vals = vals[~np.isnan(vals)]

        if vals.size == 0:
            low, high = np.nan, np.nan
            ci_txt = "NA"
        else:
            low, high = np.percentile(vals, [alpha_low, alpha_high])
            ci_txt = f"[{low:.4f}, {high:.4f}]"

        rows.append({
            "class": cls_name,
            "metric": "auc_ovr",
            "point_estimate": float(obs_per_class[c]) if not np.isnan(obs_per_class[c]) else np.nan,
            "ci_low": float(low) if not np.isnan(low) else np.nan,
            "ci_high": float(high) if not np.isnan(high) else np.nan,
            "ci_95_percentile": ci_txt,
            "n_boot_valid": int(vals.size),
            "n_boot_total": int(B),
            "n_positive_obs": int((y_true_enc == c).sum()),
            "n_negative_obs": int((y_true_enc != c).sum()),
        })

    df_per_class = pd.DataFrame(rows)

    out = {
        "macro": df_macro,
        "per_class": df_per_class,
    }

    if return_distributions:
        out["bootstrap_distributions"] = {
            "auc_macro": auc_macro_boot,
            "auc_per_class": auc_per_class_boot,
        }

    return out
