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


# ────────────────────────────────────────────────────────────
# 1) Load expression + metadata
# ────────────────────────────────────────────────────────────
def load_expression_and_metadata(
    expr_csv: Path,
    meta_csv: Path,
    sample_col: str = "sample",
    label_col: str = "label",
    cohort_col: str = "batch",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X_df : pd.DataFrame   (samples × genes, numeric)
    y_str : np.ndarray    string labels
    cohort : np.ndarray   string cohort/batch ids
    """
    expr_csv = Path(expr_csv)
    meta_csv = Path(meta_csv)
    missing = [str(p) for p in [expr_csv, meta_csv] if not p.exists()]
    if missing:
        msg = (
            "No encuentro los datos de entrada esperados:\n  - "
            + "\n  - ".join(missing)
            + "\n\nEjecuta primero: python scripts/download_zenodo_data.py --extract\n"
            + "o define INTERGATE_DATA_DIR / INTERGATE_EXPR_CSV / INTERGATE_META_CSV."
        )
        raise FileNotFoundError(msg)

    # Expression: genes × samples
    df_expr = pd.read_csv(expr_csv, index_col=0)
    df_expr.index = df_expr.index.astype(str).str.strip()
    df_expr = df_expr.apply(pd.to_numeric, errors="coerce")

    # Aggregate duplicated gene rows by mean
    dup_mask = df_expr.index.duplicated(keep=False)
    if dup_mask.any():
        n_genes_dup = int(df_expr.index[dup_mask].nunique())
        print(f"[INFO] Genes duplicados: {n_genes_dup}. Agrupando por media...")
        df_expr = df_expr.groupby(level=0, sort=False).mean()

    df_expr = df_expr.dropna(how="all")

    # Metadata
    meta = pd.read_csv(meta_csv, index_col=0).reset_index(drop=True)
    meta = meta[[sample_col, label_col, cohort_col]]

    # Transpose to samples × genes
    df_samples = df_expr.T.copy()
    df_samples.index.name = sample_col
    df_samples.reset_index(inplace=True)

    # Merge
    df_merged = meta.merge(df_samples, on=sample_col, how="inner")
    df_merged = df_merged.dropna(subset=[label_col]).reset_index(drop=True)

    for c in [sample_col, cohort_col, label_col]:
        if c not in df_merged.columns:
            raise ValueError(f"Falta la columna '{c}' en df_merged.")

    y_str = df_merged[label_col].astype(str).values
    cohort = df_merged[cohort_col].astype(str).values

    X_df = df_merged.drop(columns=[sample_col, label_col, cohort_col], errors="ignore")
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

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
    labels = y_str.astype(str)
    classes = sorted(pd.unique(labels).tolist())
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[c] for c in labels], dtype=np.int64)
    print(f"y: {y.shape}, n_classes: {len(classes)}, classes: {classes}")
    return y, classes, label_map


# ────────────────────────────────────────────────────────────
# 4) Cohort-based split
# ────────────────────────────────────────────────────────────
def cohort_split(
    cohort: np.ndarray,
    y: np.ndarray,
    train_cohort_frac: float = 0.80,
    val_size: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by cohort (80/20) then random val from train pool.

    Returns  train_idx, val_idx, test_idx
    """
    cohort_s = pd.Series(cohort).astype(str)
    cohort_up = cohort_s.str.upper().str.strip()
    all_cohorts = sorted(pd.unique(cohort_up))
    print(f"Total cohorts: {len(all_cohorts)}")

    if len(all_cohorts) <= 1:
        train_cohorts = set(all_cohorts)
        test_cohorts = set()
    else:
        train_list, test_list = train_test_split(
            all_cohorts, train_size=train_cohort_frac, random_state=1234, shuffle=True
        )
        train_cohorts = set(train_list)
        test_cohorts = set(test_list)

    train_pool_mask = cohort_up.isin(train_cohorts)
    test_mask = cohort_up.isin(test_cohorts)

    assert not (train_pool_mask & test_mask).any(), "Overlap TRAIN/TEST"

    train_pool_idx = np.where(train_pool_mask.values)[0]
    test_idx = np.where(test_mask.values)[0]

    train_idx, val_idx = train_test_split(
        train_pool_idx, test_size=val_size, random_state=seed + 1, shuffle=True
    )

    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(set(train_idx) & set(val_idx)) == 0

    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Cohorts TRAIN: {sorted(train_cohorts)}")
    print(f"Cohorts TEST: {sorted(test_cohorts)}")

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
