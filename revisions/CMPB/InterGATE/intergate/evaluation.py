"""
evaluation.py
=============

Rendimiento del modelo a partir de un bundle de `run_ablation`.

Dos niveles:

1. **Métricas ya calculadas** (`metrics_from_bundle`). Están en `meta.json`, no
   hace falta re-inferir nada. Es lo que devuelve `run_ablation` en `df_abla`.

2. **Matriz de confusión y curvas** (`confusion_from_bundle`, `plot_roc_pr`,
   `threshold_sweep`). Necesitan las predicciones de test, que se guardan en
   `test_predictions.npz` desde el parche de `save_bundle`. Los bundles
   anteriores a ese parche no lo tienen: hay que relanzar, o reconstruir el
   dataloader compacto a mano.

El punto de operación importa: si reportas F1 de Basal-like, argmax no tiene por qué
ser el umbral correcto. `threshold_sweep` muestra el trade-off completo, y
`ovr_thresholds_from_val` (en postprocessing) ajusta el umbral en validación.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "metrics_from_bundle",
    "load_test_predictions",
    "confusion_from_bundle",
    "plot_confusion",
    "plot_roc_pr",
    "threshold_sweep",
    "classification_report_table",
    "compare_runs",
]


# ══════════════════════════════════════════════════════════════════════
# 1) Métricas ya calculadas
# ══════════════════════════════════════════════════════════════════════

def metrics_from_bundle(bundle_dir: str, verbose: bool = True) -> pd.Series:
    """Métricas de test guardadas en `meta.json`. No re-infiere nada."""
    with open(os.path.join(bundle_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    met = meta.get("metrics", meta)
    flat = {k: v for k, v in met.items() if isinstance(v, (int, float, bool))}
    s = pd.Series(flat, name=os.path.basename(bundle_dir.rstrip("/")))
    if verbose:
        head = [k for k in ["acc", "bal_acc", "macro_f1", "weighted_f1",
                            "auc_macro_ovr", "auc_micro_ovr"] if k in s]
        cls = [k for k in s.index
               if k.startswith(("f1_", "precision_", "recall_", "auc_", "support_"))
               and k not in head]
        print("── Métricas globales ──")
        for k in head:
            print(f"  {k:24s} {s[k]:.4f}")
        if cls:
            print("── Por clase reportada ──")
            for k in cls:
                v = s[k]
                print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
    return s


def compare_runs(bundle_dirs: Sequence[str],
                 metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Tabla comparativa entre corridas (p. ej. un barrido de lambda)."""
    rows = [metrics_from_bundle(d, verbose=False) for d in bundle_dirs]
    df = pd.DataFrame(rows)
    if metrics:
        keep = [m for m in metrics if m in df.columns]
        df = df[keep]
    return df


# ══════════════════════════════════════════════════════════════════════
# 2) Predicciones de test
# ══════════════════════════════════════════════════════════════════════

def load_test_predictions(bundle_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Devuelve (proba, y_true, class_names) desde `test_predictions.npz`."""
    path = os.path.join(bundle_dir, "test_predictions.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No hay {path}.\n"
            "Este bundle se generó antes del parche que guarda las predicciones "
            "de test en save_bundle(). Relanza la corrida con el ablation.py "
            "parcheado, o usa metrics_from_bundle() para las métricas globales, "
            "que sí están en meta.json."
        )
    d = np.load(path, allow_pickle=True)
    names = [str(c) for c in d["class_names"]] if "class_names" in d else None
    proba = np.asarray(d["proba"], dtype=float)
    y = np.asarray(d["y_true"]).reshape(-1).astype(int)
    if not names:
        names = [str(i) for i in range(proba.shape[1])]
    return proba, y, names


def confusion_from_bundle(
    bundle_dir: str,
    *,
    thresholds: Optional[np.ndarray] = None,
    normalize: Optional[str] = None,
) -> pd.DataFrame:
    """Matriz de confusión de test.

    `thresholds` aplica umbrales one-vs-rest en vez de argmax; útil si la
    métrica que reportas es F1 de la clase positiva.
    `normalize`: None | "true" (por fila = recall) | "pred" (por columna).
    """
    from sklearn.metrics import confusion_matrix

    proba, y, names = load_test_predictions(bundle_dir)
    if thresholds is None:
        pred = proba.argmax(axis=1)
    else:
        from .postprocessing import predict_with_ovr_thresholds

        pred = predict_with_ovr_thresholds(proba, np.asarray(thresholds, dtype=float))

    cm = confusion_matrix(y, pred, labels=np.arange(len(names)))
    if normalize == "true":
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    elif normalize == "pred":
        cm = cm / np.maximum(cm.sum(axis=0, keepdims=True), 1)
    return pd.DataFrame(cm, index=[f"true {n}" for n in names],
                        columns=[f"pred {n}" for n in names])


def plot_confusion(
    cm_df: pd.DataFrame,
    *,
    title: str = "Test confusion matrix",
    cmap: str = "Blues",
    figsize=(5.4, 4.6),
    fmt: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    import matplotlib.pyplot as plt

    cm = cm_df.to_numpy(dtype=float)
    is_norm = cm.max() <= 1.0 + 1e-9
    fmt = fmt or ("{:.2f}" if is_norm else "{:.0f}")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max() if cm.max() else 1)
    ax.set_xticks(range(cm.shape[1]))
    ax.set_xticklabels([c.replace("pred ", "") for c in cm_df.columns])
    ax.set_yticks(range(cm.shape[0]))
    ax.set_yticklabels([r.replace("true ", "") for r in cm_df.index])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, pad=10)
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, fmt.format(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "#222222", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
    return fig, ax


def classification_report_table(
    bundle_dir: str,
    *,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Precisión, recall, F1 y soporte por clase, más los agregados."""
    from sklearn.metrics import classification_report

    proba, y, names = load_test_predictions(bundle_dir)
    if thresholds is None:
        pred = proba.argmax(axis=1)
    else:
        from .postprocessing import predict_with_ovr_thresholds

        pred = predict_with_ovr_thresholds(proba, np.asarray(thresholds, dtype=float))
    rep = classification_report(
        y, pred, labels=np.arange(len(names)), target_names=names,
        output_dict=True, zero_division=0,
    )
    return pd.DataFrame(rep).T


def plot_roc_pr(
    bundle_dir: str,
    *,
    positive_class: str = "Basal-like",
    figsize=(11, 4.4),
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """ROC y precision-recall de la clase positiva.

    Son independientes del umbral, así que separan la calidad del modelo de la
    elección del punto de operación.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
    )

    proba, y, names = load_test_predictions(bundle_dir)
    if positive_class not in names:
        raise KeyError(f"{positive_class!r} no está en {names}")
    ci = names.index(positive_class)
    score = proba[:, ci]
    y_bin = (y == ci).astype(int)

    fpr, tpr, _ = roc_curve(y_bin, score)
    auc = roc_auc_score(y_bin, score)
    prec, rec, _ = precision_recall_curve(y_bin, score)
    ap = average_precision_score(y_bin, score)
    base = y_bin.mean()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(fpr, tpr, color="#7B61A8", lw=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC — {positive_class} vs rest")

    axes[1].plot(rec, prec, color="#1a9850", lw=2, label=f"AP = {ap:.3f}")
    axes[1].axhline(base, ls="--", color="#999", lw=1,
                    label=f"prevalence = {base:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall — {positive_class}")

    for a in axes:
        a.legend(fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
    return fig, axes


def threshold_sweep(
    bundle_dir: str,
    *,
    positive_class: str = "Basal-like",
    grid: Optional[np.ndarray] = None,
    plot: bool = True,
    figsize=(7, 4.4),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> pd.DataFrame:
    """Precisión, recall, F1 y macro-F1 en función del umbral.

    Muestra el intercambio entre la sensibilidad de la clase elegida y los
    errores introducidos en el resto de clases.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    proba, y, names = load_test_predictions(bundle_dir)
    ci = names.index(positive_class)
    score = proba[:, ci]
    y_bin = (y == ci).astype(int)
    grid = np.linspace(0.05, 0.95, 91) if grid is None else np.asarray(grid, float)

    rows = []
    for t in grid:
        pred_bin = (score >= t).astype(int)
        pred_multi = np.where(pred_bin == 1, ci, 1 - ci) if len(names) == 2 else None
        rows.append({
            "threshold": float(t),
            "precision": precision_score(y_bin, pred_bin, zero_division=0),
            "recall": recall_score(y_bin, pred_bin, zero_division=0),
            "f1": f1_score(y_bin, pred_bin, zero_division=0),
            "macro_f1": (f1_score(y, pred_multi, average="macro", zero_division=0)
                         if pred_multi is not None else np.nan),
            "n_predicted_positive": int(pred_bin.sum()),
        })
    df = pd.DataFrame(rows)

    if plot:
        import matplotlib.pyplot as plt

        best = df.loc[df["f1"].idxmax()]
        fig, ax = plt.subplots(figsize=figsize)
        for col, c in [("precision", "#4393c3"), ("recall", "#d73027"),
                       ("f1", "#1a9850"), ("macro_f1", "#7B61A8")]:
            if df[col].notna().any():
                ax.plot(df["threshold"], df[col], label=col, color=c, lw=1.8)
        ax.axvline(0.5, ls="--", color="#999", lw=1, label="argmax (0.5)")
        ax.axvline(best["threshold"], ls=":", color="#1a9850", lw=1.6,
                   label=f"best f1 = {best['f1']:.3f} @ {best['threshold']:.2f}")
        ax.set_xlabel(f"Decision threshold on P({positive_class})")
        ax.set_ylabel("Score")
        ax.set_title(f"Operating point trade-off — {positive_class}")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[SAVE] {save_path}")
    return df
