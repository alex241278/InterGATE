"""
Data loading, gene preparation, cohort-based splitting, scaling, and DataLoader creation.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from torch.utils.data import Dataset, DataLoader

from .config import CFG as C
from .prepared_data import load_prepared_bundle


# ────────────────────────────────────────────────────────────
# 1) Load expression + metadata
# ────────────────────────────────────────────────────────────
def load_expression_and_metadata(
    expr_csv: Optional[Path] = None,
    meta_csv: Optional[Path] = None,
    sample_col: str = "sample",
    label_col: str = "label",
    cohort_col: str = "dataset",
    include_external: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the immutable train and external-test outputs of data_prepare.

    ``expr_csv``/``meta_csv`` are intentionally not accepted in protocol V8:
    a single combined file would erase the split boundary. Paths are taken
    from ``CFG.TRAIN_*`` and ``CFG.TEST_*`` (or their environment overrides).

    Returns
    -------
    X_df : pd.DataFrame   (samples × genes, numeric)
    y_str : np.ndarray    string labels
    cohort : np.ndarray   string cohort/batch ids
    """
    if expr_csv is not None or meta_csv is not None:
        raise ValueError(
            "El protocolo V8 no admite una matriz combinada. Configura las "
            "cuatro rutas TRAIN/TEST de M-ComBat en CFG."
        )
    if (sample_col, label_col, cohort_col) != (
        C.SAMPLE_COL, C.LABEL_COL, C.COHORT_COL
    ):
        raise ValueError(
            "Los nombres de columnas deben configurarse centralmente en CFG."
        )

    if include_external:
        bundle = load_prepared_bundle(C)
    else:
        from .prepared_data import load_development_bundle
        bundle = load_development_bundle(C)
    meta = bundle.metadata()
    X_df = bundle.expression_samples()
    meta = meta.loc[X_df.index]
    y_str = meta[label_col].astype(str).to_numpy()
    cohort = meta[cohort_col].astype(str).to_numpy()
    X_df = X_df.reset_index(drop=True)

    print(f"X_df: {X_df.shape} | y: {y_str.shape} | cohort: {cohort.shape}")
    print("Targets:\n", pd.Series(y_str).value_counts())

    return X_df, y_str, cohort


# ────────────────────────────────────────────────────────────
# 2) Gene name canonicalisation
# ────────────────────────────────────────────────────────────
def canonical_gene(g: str) -> str:
    if g is None:
        return ""
    g = str(g).strip()
    g = re.sub(r"^hsa:", "", g, flags=re.I)
    for sep in ["///", "|", ";", ","]:
        if sep in g:
            g = g.split(sep, 1)[0].strip()
    if " " in g:
        g = g.split(" ", 1)[0].strip()
    g = re.sub(r"^(ENSG\d+)\.\d+$", r"\1", g, flags=re.I)
    g = re.sub(r"^(ENST\d+)\.\d+$", r"\1", g, flags=re.I)
    return g


def prepare_genes(X_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Canonicalise gene names and aggregate duplicates.

    Returns
    -------
    X_df_kegg : pd.DataFrame  (samples × genes, clean)
    genes_kegg : list[str]     ordered gene symbols
    """
    gene_raw = pd.Index(X_df.columns.astype(str))
    gene_clean = gene_raw.map(canonical_gene)

    if gene_clean.duplicated().any():
        print("[WARN] Duplicados tras limpiar genes -> agrupando por media.")
        X_df_sym = X_df.copy()
        X_df_sym.columns = gene_clean
        X_df_sym = X_df_sym.groupby(axis=1, level=0).mean()
    else:
        X_df_sym = X_df.copy()
        X_df_sym.columns = gene_clean

    genes_kegg = list(X_df_sym.columns.astype(str))
    print(f"Genes totales (nodos): {len(genes_kegg)}")
    return X_df_sym, genes_kegg


# ────────────────────────────────────────────────────────────
# 3) Encode labels
# ────────────────────────────────────────────────────────────
def encode_labels(y_str: np.ndarray) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Codifica las clases en el orden único declarado por ``CFG``.

    Ordenarlas alfabéticamente hacía que los logits, los nombres exportados y
    la comprobación del notebook externo usasen órdenes distintos. El orden
    del contrato es parte del modelo y debe mantenerse desde el entrenamiento
    hasta la evaluación bloqueada.
    """
    labels = y_str.astype(str)
    observed = set(pd.unique(labels).tolist())
    declared = list(map(str, C.MULTICLASS_LABELS))
    if observed != set(declared):
        raise ValueError(
            "Las etiquetas observadas no coinciden con CFG: "
            f"faltan={sorted(set(declared)-observed)}, "
            f"sobran={sorted(observed-set(declared))}"
        )
    classes = declared
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[c] for c in labels], dtype=np.int64)
    print(f"y: {y.shape}, n_classes: {len(classes)}, classes: {classes}")
    return y, classes, label_map


# ────────────────────────────────────────────────────────────
# 3b) Endpoint multiclase y filtro de cohortes
# ────────────────────────────────────────────────────────────
def cohort_label_table(cohort, y_str, verbose: bool = True) -> pd.DataFrame:
    """Cross-tab cohorte x etiqueta con el numero de clases distintas."""
    tab = pd.crosstab(pd.Series(np.asarray(cohort).astype(str), name="cohort"),
                      pd.Series(np.asarray(y_str).astype(str), name="label"))
    tab["n_samples"] = tab.sum(axis=1)
    tab["n_labels"] = (tab.drop(columns="n_samples") > 0).sum(axis=1)
    tab = tab.sort_values("n_samples", ascending=False)
    if verbose:
        print(tab.to_string())
    return tab


def preview_cohort_filter(cohort, y_str, verbose: bool = True) -> pd.DataFrame:
    """Coste de cada umbral de min_labels antes de fijarlo."""
    cohort = np.asarray(cohort).astype(str)
    y_str = np.asarray(y_str).astype(str)
    tab = cohort_label_table(cohort, y_str, verbose=False)
    n_classes = len(set(y_str.tolist()))
    rows = []
    for thr in range(1, n_classes + 1):
        drop = sorted(tab.index[tab["n_labels"] < thr].astype(str).tolist())
        mask = ~pd.Series(cohort).isin(drop).values
        rows.append({
            "min_labels": thr,
            "cohortes_ok": int(len(tab) - len(drop)),
            "muestras": int(mask.sum()),
            "pct_muestras": round(100 * mask.sum() / len(cohort), 1),
            "clases_vivas": len(set(y_str[mask].tolist())) if mask.any() else 0,
            "descartadas": ", ".join(drop) if drop else "-",
        })
    out = pd.DataFrame(rows).set_index("min_labels")
    if verbose:
        print(f"Cohortes: {len(tab)} | muestras: {len(cohort)} | clases: {n_classes}")
        print(out.to_string())
    return out


def drop_task_labels(X_df, y_str, cohort, cfg=None, verbose: bool = True):
    """Descarta controles y restringe al endpoint multiclase declarado."""
    if cfg is None:
        cfg = C
    X_df = X_df.reset_index(drop=True)
    y_str = np.asarray(y_str).astype(str)
    cohort = np.asarray(cohort).astype(str)
    n0, dropped = len(y_str), []

    def _keep(mask, why):
        nonlocal X_df, y_str, cohort
        if mask.all():
            return
        if mask.sum() == 0:
            raise ValueError(f"{why} ha dejado 0 muestras.")
        rows = np.where(mask)[0]
        X_df = X_df.iloc[rows].reset_index(drop=True)
        y_str, cohort = y_str[rows], cohort[rows]

    drop = tuple(cfg.DROP_LABELS)
    if drop:
        _keep(~np.isin(y_str, list(drop)), f"DROP_LABELS={drop}")
        dropped += list(drop)
        if verbose:
            print(f"[task] descartadas {list(drop)}: {n0} -> {len(y_str)} muestras")

    keep = set(cfg.MULTICLASS_LABELS)
    extra = sorted(set(y_str.tolist()) - keep)
    if extra:
        _keep(np.isin(y_str, list(keep)), "MULTICLASS_LABELS")
        dropped += extra
        if verbose:
            print(f"[task] fuera de MULTICLASS_LABELS -> descartadas {extra}")
    missing = sorted(keep - set(y_str.tolist()))
    if missing:
        raise ValueError(
            f"MULTICLASS_LABELS pide {sorted(keep)} pero faltan {missing}."
        )

    return X_df, y_str, cohort, sorted(set(dropped))


def prepare_task(X_df, y_str, cohort, cfg=None, verbose: bool = True):
    """Aplica el endpoint y después el mínimo de etiquetas por cohorte."""
    if cfg is None:
        cfg = C
    n0 = len(np.asarray(y_str))
    X_df, y_str, cohort, dropped = drop_task_labels(X_df, y_str, cohort, cfg,
                                                    verbose=verbose)
    X_df, y_str, cohort, dropped_cohorts = filter_cohorts(X_df, y_str, cohort, cfg,
                                                          verbose=verbose)
    info = {
        "task_mode": "multiclass",
        "n_before": n0,
        "n_after": len(y_str),
        "dropped_labels": dropped,
        "dropped_cohorts": dropped_cohorts,
        "classes": sorted(set(np.asarray(y_str).astype(str).tolist())),
    }
    if verbose:
        print(f"[task] {info['n_before']} -> {info['n_after']} muestras | "
              f"clases finales: {info['classes']}")
    return X_df, y_str, cohort, info


def filter_cohorts(X_df, y_str, cohort, cfg=None, *, min_labels=None,
                   require_classes=None, drop=None, protect=None,
                   verbose: bool = True):
    """Descarta cohortes enteras (filas), nunca columnas.

    Como solo elimina muestras, el universo de genes no cambia: el backbone y
    su cache siguen siendo validos. Lo que si cambia es el scaler (se reajusta
    en el nuevo TRAIN) y por tanto Xs_gene y X_h.

    Se llama después de fijar el endpoint, para que el mínimo de etiquetas se
    evalúe exactamente sobre las clases activas.
    """
    if cfg is None:
        cfg = C
    X_df = X_df.reset_index(drop=True)
    y_str = np.asarray(y_str).astype(str)
    cohort = np.asarray(cohort).astype(str)
    up = pd.Series(cohort).str.upper().str.strip()

    min_labels = int(cfg.MIN_LABELS_PER_COHORT if min_labels is None else min_labels)
    require_classes = list(cfg.REQUIRE_CLASSES if require_classes is None
                           else require_classes)
    drop = list(cfg.DROP_COHORTS if drop is None else drop)
    protect = list(cfg.PROTECT_COHORTS if protect is None else protect)

    tab = cohort_label_table(up.values, y_str, verbose=False)
    label_cols = [c for c in tab.columns if c not in ("n_samples", "n_labels")]
    to_drop = set()

    if min_labels > 1:
        low = sorted(tab.index[tab["n_labels"] < min_labels].astype(str).tolist())
        to_drop |= set(low)
        if verbose and low:
            print(f"[cohorts] menos de {min_labels} etiquetas -> descartadas:")
            for c in low:
                labs = [l for l in label_cols if tab.loc[c, l] > 0]
                print(f"    {c:<12} n={int(tab.loc[c, 'n_samples']):>5}  "
                      f"{len(labs)}: {', '.join(labs)}")

    if require_classes:
        missing_cols = [c for c in require_classes if c not in tab.columns]
        if missing_cols:
            raise ValueError(
                f"REQUIRE_CLASSES pide {missing_cols}, ausentes de las "
                f"etiquetas {sorted(set(y_str.tolist()))}"
            )
        lacking = sorted(tab.index[(tab[require_classes] == 0).any(axis=1)]
                         .astype(str).tolist())
        to_drop |= set(lacking)
        if verbose and lacking:
            print(f"[cohorts] sin {require_classes} -> descartadas: {lacking}")

    to_drop |= {str(c).upper().strip() for c in drop}

    if protect:
        clash = sorted({str(c).upper().strip() for c in protect} & to_drop)
        if clash:
            raise ValueError(
                f"Cohortes protegidas que el filtro descartaria: {clash}"
            )

    mask = ~up.isin(to_drop)
    if mask.sum() == 0:
        raise ValueError("El filtro de cohortes ha dejado 0 muestras.")
    lost = sorted(set(y_str.tolist()) - set(y_str[mask.values].tolist()))
    if lost:
        raise ValueError(f"El filtro elimina clases enteras: {lost}")

    if verbose:
        print(f"[cohorts] descartadas {sorted(to_drop) or '-'}: "
              f"{len(up)} -> {int(mask.sum())} muestras")
        print(f"[cohorts] quedan: {sorted(up[mask].unique())}")

    rows = np.where(mask.values)[0]
    return (X_df.iloc[rows].reset_index(drop=True), y_str[rows], cohort[rows],
            sorted(to_drop))


# ────────────────────────────────────────────────────────────
# 4) Cohort-based split
# ────────────────────────────────────────────────────────────
def cohort_split(
    cohort: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reuse the immutable 4/3 cohort partition, then create validation only
    inside the development block. External-test samples never enter this
    internal split or any fitted scaler.

    Returns  train_idx, val_idx, test_idx
    """
    cohort_s = pd.Series(cohort).astype(str)
    cohort_up = cohort_s.str.upper().str.strip()
    all_cohorts = set(pd.unique(cohort_up))
    train_cohorts = {str(c).upper().strip() for c in C.TRAIN_COHORTS}
    test_cohorts = {str(c).upper().strip() for c in C.TEST_COHORTS}
    declared = train_cohorts | test_cohorts
    development_only = all_cohorts == train_cohorts
    if not development_only and all_cohorts != declared:
        raise ValueError(
            "Las cohortes cargadas no coinciden con data_prepare: "
            f"faltan={sorted(declared-all_cohorts)}, "
            f"sobran={sorted(all_cohorts-declared)}"
        )
    train_pool_mask = cohort_up.isin(train_cohorts)
    test_mask = cohort_up.isin(test_cohorts)

    assert not (train_pool_mask & test_mask).any(), "Overlap TRAIN/TEST"

    train_pool_idx = np.where(train_pool_mask.values)[0]
    test_idx = np.where(test_mask.values)[0]

    y = np.asarray(y)
    stratify = y[train_pool_idx]
    counts = pd.Series(stratify).value_counts()
    if (counts < 2).any():
        raise ValueError(
            "No hay al menos dos muestras por clase en el bloque de desarrollo."
        )
    train_idx, val_idx = train_test_split(
        train_pool_idx, test_size=val_size, random_state=seed + 1,
        shuffle=True, stratify=stratify,
    )

    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(set(train_idx) & set(val_idx)) == 0

    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Development cohorts (fixed): {list(C.TRAIN_COHORTS)}")
    if development_only:
        print("External test: BLOQUEADO; no se han leído sus matrices ni etiquetas.")
    else:
        print(f"External-test cohorts (fixed): {list(C.TEST_COHORTS)}")

    return train_idx, val_idx, test_idx


# ────────────────────────────────────────────────────────────
# 5) Scaling
# ────────────────────────────────────────────────────────────
def scale_features(
    X_df: pd.DataFrame,
    train_idx: np.ndarray,
    mode: str = "standard",
    use_quantile: bool = False,
) -> np.ndarray:
    """
    Fit scaler on train_idx, transform all.  Returns Xs_gene (samples × genes) as float32.
    """
    X_np = X_df.values.astype(np.float32)

    if use_quantile:
        qt = QuantileTransformer(output_distribution="normal", random_state=0)
        qt.fit(X_np[train_idx])
        X_np = qt.transform(X_np).astype(np.float32)

    if mode == "standard":
        scaler = StandardScaler()
    elif mode == "minmax_-1_1":
        scaler = MinMaxScaler(feature_range=(-1, 1))
    elif mode == "none":
        return X_np
    else:
        raise ValueError(f"Unknown scale_mode: {mode}")

    scaler.fit(X_np[train_idx])
    Xs = scaler.transform(X_np).astype(np.float32)
    return Xs


# ────────────────────────────────────────────────────────────
# 6) Connected-only filter
# ────────────────────────────────────────────────────────────
def apply_connected_only(
    Xs_gene: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    edge_type: np.ndarray,
    genes_kegg: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Remove isolated nodes (degree 0) and reindex."""
    G = len(genes_kegg)
    deg = np.bincount(edge_index.reshape(-1), minlength=G)
    keep = np.where(deg > 0)[0]

    old2new = np.full(G, -1, dtype=np.int64)
    old2new[keep] = np.arange(len(keep))

    mask = np.isin(edge_index[0], keep) & np.isin(edge_index[1], keep)
    ei_new = old2new[edge_index[:, mask]]
    ew_new = edge_weight[mask]
    et_new = edge_type[mask]

    Xs_new = Xs_gene[:, keep]
    genes_new = [genes_kegg[i] for i in keep]

    print(f"[CONNECTED_ONLY] {G} -> {len(keep)} genes, {edge_index.shape[1]} -> {ei_new.shape[1]} edges")
    return Xs_new, ei_new, ew_new, et_new, genes_new


# ────────────────────────────────────────────────────────────
# 7) Dataset and DataLoaders
# ────────────────────────────────────────────────────────────
class ExpressionDataset(Dataset):
    """Indexed dataset – avoids copying X[train_idx]."""

    def __init__(self, X_gene_full: np.ndarray, X_graph_full: np.ndarray,
                 y_full: np.ndarray, idx: np.ndarray):
        self.X_gene_full = X_gene_full
        self.X_graph_full = X_graph_full
        self.y_full = y_full
        self.idx = np.asarray(idx, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return self.X_gene_full[j], self.X_graph_full[j], self.y_full[j], j


def make_dataloaders(
    Xs_gene: np.ndarray,
    Xs_graph: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = 20,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    pin = torch.cuda.is_available()
    kw = dict(num_workers=num_workers, pin_memory=pin, drop_last=False)

    dl_tr = DataLoader(
        ExpressionDataset(Xs_gene, Xs_graph, y, train_idx),
        batch_size=batch_size, shuffle=True, **kw
    )
    dl_va = DataLoader(
        ExpressionDataset(Xs_gene, Xs_graph, y, val_idx),
        batch_size=batch_size, shuffle=False, **kw
    )
    dl_te = DataLoader(
        ExpressionDataset(Xs_gene, Xs_graph, y, test_idx),
        batch_size=batch_size, shuffle=False, **kw
    )
    print(f"Batches train/val/test: {len(dl_tr)}/{len(dl_va)}/{len(dl_te)}")
    return dl_tr, dl_va, dl_te


# ────────────────────────────────────────────────────────────
# 8) Regulator features (X_graph / Xs_graph)
# ────────────────────────────────────────────────────────────

def build_regulator_features(
    Xs_gene: np.ndarray,
    genes: List[str],
    edge_index: np.ndarray,
    stats: Tuple[str, ...] = ("mean", "std", "max"),
    min_targets: int = 5,
    max_regulators: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    G = len(genes)
    idx2sym = np.array([str(g) for g in genes], dtype=object)

    reg2tgt: Dict[str, List[int]] = {}
    for s, t in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if 0 <= s < G and 0 <= t < G:
            reg2tgt.setdefault(idx2sym[s], []).append(t)

    items = [
        (reg, np.unique(np.array(tgts, dtype=np.int64)))
        for reg, tgts in reg2tgt.items()
        if len(np.unique(tgts)) >= min_targets
    ]
    items.sort(key=lambda x: x[1].size, reverse=True)
    if max_regulators is not None:
        items = items[:max_regulators]

    print(f"[data] Regulators used for X_graph: {len(items)} (min_targets={min_targets})")
    if not items:
        return np.zeros((Xs_gene.shape[0], 0), dtype=np.float32), []

    feats, names = [], []
    for reg, tgt_ids in items:
        Xsub = Xs_gene[:, tgt_ids]
        if "mean" in stats:
            feats.append(Xsub.mean(axis=1, keepdims=True))
            names.append(f"OP_{reg}__mean")
        if "std" in stats:
            feats.append(Xsub.std(axis=1, keepdims=True))
            names.append(f"OP_{reg}__std")
        if "max" in stats:
            feats.append(Xsub.max(axis=1, keepdims=True))
            names.append(f"OP_{reg}__max")

    Xs_graph = np.hstack(feats).astype(np.float32)
    print(f"[data] X_graph shape: {Xs_graph.shape}")
    return Xs_graph, names
