#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""QC de lote para el flujo definitivo de M-ComBat de referencia.

Este módulo no implementa ni ejecuta rutas alternativas de armonización. Solo
compara la matriz inmediatamente anterior a M-ComBat con su salida congelada,
por separado en desarrollo y evaluación externa, y verifica el contrato de
proyección sin fuga.

Las métricas de lote son diagnósticas: no se usan para seleccionar el endpoint,
la partición, kappa ni el checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _as_sample_matrix(expr: pd.DataFrame) -> np.ndarray:
    """Convierte genes×muestras a muestras×genes y estandariza por gen."""
    x = expr.T.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(x).all():
        raise ValueError("La expresión contiene NaN o infinitos.")
    return StandardScaler().fit_transform(x)


def reference_pca_scores(expr_train: pd.DataFrame, expr_test: pd.DataFrame,
                         n_pc: int = 20, seed: int = 42):
    """Ajusta escalado y PCA en desarrollo y proyecta evaluación externa.

    Ambas matrices deben ser genes×muestras y compartir exactamente el mismo
    orden de genes. Ningún parámetro se vuelve a estimar con el bloque externo.
    """
    if not expr_train.index.equals(expr_test.index):
        raise ValueError(
            "Train y test deben tener los mismos genes y en el mismo orden."
        )
    x_train = expr_train.T.to_numpy(dtype=np.float64, copy=True)
    x_test = expr_test.T.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("La expresión contiene NaN o infinitos.")
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)
    n = max(1, min(int(n_pc), x_train_s.shape[0] - 1, x_train_s.shape[1]))
    pca = PCA(n_components=n, random_state=seed).fit(x_train_s)
    return (
        pca.transform(x_train_s),
        pca.transform(x_test_s),
        pca.explained_variance_ratio_,
    )


def pca_scores(expr: pd.DataFrame, n_pc: int = 20, seed: int = 42):
    x = _as_sample_matrix(expr)
    n = max(1, min(int(n_pc), x.shape[0] - 1, x.shape[1]))
    pca = PCA(n_components=n, random_state=seed)
    return pca.fit_transform(x), pca.explained_variance_ratio_


def _safe_silhouette(z: np.ndarray, groups: Iterable[str]) -> float:
    groups = np.asarray(list(groups)).astype(str)
    if len(np.unique(groups)) < 2 or len(groups) <= len(np.unique(groups)):
        return float("nan")
    return float(silhouette_score(z, groups))


def batch_predictability_cv(z: np.ndarray, batch: Iterable[str],
                            seed: int = 42) -> float:
    """Balanced accuracy al predecir cohorte desde PCs (menor es mejor)."""
    batch = np.asarray(list(batch)).astype(str)
    counts = pd.Series(batch).value_counts()
    n_splits = int(min(5, counts.min()))
    if len(counts) < 2 or n_splits < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = LogisticRegression(
        max_iter=4000, class_weight="balanced", random_state=seed,
    )
    scores = cross_val_score(clf, z, batch, cv=cv, scoring="balanced_accuracy")
    return float(scores.mean())


def knn_batch_entropy(z: np.ndarray, batch: Iterable[str], k: int = 30) -> float:
    """Entropía local de cohortes normalizada por la mezcla global.

    Cero indica vecindarios de una sola cohorte. Valores cercanos a uno son
    compatibles con la mezcla esperada dada la composición global.
    """
    batch = np.asarray(list(batch)).astype(str)
    n = len(batch)
    if n < 3 or len(np.unique(batch)) < 2:
        return float("nan")
    # kneighbors() autoconsultado devuelve la propia muestra como primer
    # vecino, de modo que exige n_neighbors = k + 1 < n.
    k = max(1, min(int(k), n - 2))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    ind = nn.kneighbors(return_distance=False)[:, 1:]
    levels = sorted(np.unique(batch))

    def entropy(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    global_p = pd.Series(batch).value_counts(normalize=True).reindex(
        levels, fill_value=0.0
    ).to_numpy()
    h_global = entropy(global_p)
    if h_global <= 0:
        return float("nan")
    local = []
    for row in ind:
        p = pd.Series(batch[row]).value_counts(normalize=True).reindex(
            levels, fill_value=0.0
        ).to_numpy()
        local.append(entropy(p) / h_global)
    return float(np.mean(local))


def gene_batch_eta2(expr: pd.DataFrame, batch: Iterable[str]) -> np.ndarray:
    """Fracción de varianza por gen atribuible a la cohorte (eta² ANOVA)."""
    x = expr.T.to_numpy(dtype=np.float64)
    batch = np.asarray(list(batch)).astype(str)
    grand = x.mean(axis=0)
    ss_total = ((x - grand) ** 2).sum(axis=0)
    ss_batch = np.zeros(x.shape[1], dtype=np.float64)
    for level in np.unique(batch):
        xb = x[batch == level]
        ss_batch += len(xb) * (xb.mean(axis=0) - grand) ** 2
    return np.divide(
        ss_batch, ss_total, out=np.full_like(ss_batch, np.nan),
        where=ss_total > 0,
    )


def diagnostico_bloque(expr: pd.DataFrame, meta: pd.DataFrame, *,
                       split: str, stage: str, cohort_col: str = "dataset",
                       label_col: str = "label", n_pc: int = 20,
                       seed: int = 42, z_override: np.ndarray | None = None,
                       explained_override: np.ndarray | None = None) -> dict:
    """Resume efecto de lote y conservación de señal biológica de un bloque."""
    samples = expr.columns.astype(str)
    meta = meta.copy()
    if not meta.index.equals(pd.Index(samples)):
        if "sample" in meta.columns:
            meta = meta.set_index(meta["sample"].astype(str), drop=False)
        meta = meta.loc[samples]
    if z_override is None:
        z, explained = pca_scores(expr, n_pc=n_pc, seed=seed)
    else:
        z = np.asarray(z_override, dtype=np.float64)
        if z.shape[0] != expr.shape[1]:
            raise ValueError("Los scores PCA no están alineados con las muestras.")
        explained = np.asarray(explained_override, dtype=np.float64)
    batch = meta[cohort_col].astype(str).to_numpy()
    labels = meta[label_col].astype(str).to_numpy()
    eta2 = gene_batch_eta2(expr, batch)
    return {
        "split": str(split),
        "stage": str(stage),
        "n_samples": int(expr.shape[1]),
        "n_genes": int(expr.shape[0]),
        "n_cohorts": int(len(np.unique(batch))),
        "n_labels": int(len(np.unique(labels))),
        "pc_variance_20": float(np.sum(explained)),
        "batch_silhouette": _safe_silhouette(z, batch),
        "label_silhouette": _safe_silhouette(z, labels),
        "batch_balanced_accuracy_cv": batch_predictability_cv(z, batch, seed),
        "knn_batch_entropy": knn_batch_entropy(z, batch),
        "gene_batch_eta2_median": float(np.nanmedian(eta2)),
        "gene_batch_eta2_q90": float(np.nanquantile(eta2, 0.90)),
    }


def comparar_pre_post(expr_pre: pd.DataFrame, meta_all: pd.DataFrame,
                      expr_train: pd.DataFrame, meta_train: pd.DataFrame,
                      expr_test: pd.DataFrame, meta_test: pd.DataFrame,
                      *, cohort_col: str = "dataset", label_col: str = "label",
                      n_pc: int = 20, seed: int = 42) -> pd.DataFrame:
    """Compara pre-M-ComBat y post-M-ComBat dentro de cada bloque fijo."""
    meta_all = meta_all.copy()
    if "sample" in meta_all.columns:
        meta_all = meta_all.set_index(meta_all["sample"].astype(str), drop=False)
    common_genes = expr_train.index.intersection(expr_test.index)
    if len(common_genes) == 0:
        raise ValueError("Train y test no comparten genes.")
    train_post = expr_train.loc[common_genes]
    test_post = expr_test.loc[common_genes]
    train_pre = expr_pre.loc[common_genes, train_post.columns.astype(str)]
    test_pre = expr_pre.loc[common_genes, test_post.columns.astype(str)]

    rows = []
    for stage, train_expr, test_expr in (
        ("pre_mcombat", train_pre, test_pre),
        ("post_mcombat", train_post, test_post),
    ):
        z_train, z_test, explained = reference_pca_scores(
            train_expr, test_expr, n_pc=n_pc, seed=seed,
        )
        for split, block_expr, block_meta, z in (
            ("train", train_expr, meta_train, z_train),
            ("test", test_expr, meta_test, z_test),
        ):
            rows.append(diagnostico_bloque(
                block_expr, block_meta, split=split, stage=stage,
                cohort_col=cohort_col, label_col=label_col,
                n_pc=n_pc, seed=seed, z_override=z,
                explained_override=explained,
            ))
    out = pd.DataFrame(rows)
    return out.sort_values(["split", "stage"]).reset_index(drop=True)


def cramer_v_batch_label(meta: pd.DataFrame, cohort_col: str = "dataset",
                         label_col: str = "label") -> float:
    """Cuantifica confusión estructural cohorte×etiqueta (no corregible)."""
    tab = pd.crosstab(meta[cohort_col].astype(str), meta[label_col].astype(str))
    if min(tab.shape) < 2:
        return float("nan")
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    phi2 = chi2 / n
    r, k = tab.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else float("nan")


def tabla_composicion(meta: pd.DataFrame, cohort_col: str = "dataset",
                      label_col: str = "label") -> pd.DataFrame:
    """Tabla de muestras por cohorte y clase para interpretar el QC de lote."""
    tab = pd.crosstab(meta[cohort_col].astype(str), meta[label_col].astype(str))
    tab["n_samples"] = tab.sum(axis=1)
    tab["n_labels"] = (tab.drop(columns="n_samples") > 0).sum(axis=1)
    return tab.sort_values("n_samples", ascending=False)


def cargar_salidas(data_prepare_dir: str | Path = ".") -> dict:
    """Carga las entradas pre y post del flujo definitivo para el notebook QC.

    Los cuatro CSV de ``output_mcombat`` los escribe ``fit_project_prepared``,
    que solo se invoca desde ``03_split_y_mcombat_ref.ipynb``. Si faltan, el
    error indica qué notebook queda por ejecutar en lugar de un numerado
    genérico.
    """
    root = Path(data_prepare_dir)
    pre_dir = root / "output_prepared"
    post_dir = root / "output_mcombat"

    etapas = {
        "02_preparacion_y_suelo.ipynb": {
            "expr_pre": pre_dir / "expr_prepared.csv",
            "meta_all": pre_dir / "metadata_prepared.csv",
        },
        "03_split_y_mcombat_ref.ipynb": {
            "expr_train": post_dir / "expr_train_mcombat.csv",
            "expr_test": post_dir / "expr_test_mcombat.csv",
            "meta_train": post_dir / "meta_train.csv",
            "meta_test": post_dir / "meta_test.csv",
        },
    }
    # Se comprueba pero no se carga: el notebook QC lo lee por su cuenta.
    solo_existencia = {
        "03_split_y_mcombat_ref.ipynb": (
            post_dir / "verificacion_invariancia.csv",
        ),
    }

    faltantes: dict[str, list[str]] = {}
    for notebook, archivos in etapas.items():
        pendientes = [str(p) for p in archivos.values() if not p.exists()]
        pendientes += [
            str(p) for p in solo_existencia.get(notebook, ()) if not p.exists()
        ]
        if pendientes:
            faltantes[notebook] = pendientes
    if faltantes:
        detalle = "\n".join(
            f"Ejecuta antes {notebook}. Faltan:\n- " + "\n- ".join(pendientes)
            for notebook, pendientes in faltantes.items()
        )
        raise FileNotFoundError(detalle)

    files = {
        name: path
        for archivos in etapas.values()
        for name, path in archivos.items()
    }
    return {
        name: pd.read_csv(path, index_col=0)
        for name, path in files.items()
    }
