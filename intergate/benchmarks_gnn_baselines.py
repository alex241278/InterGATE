"""
Fixed-prior GNN baselines: SimpleGIN, SimpleGraphTransformer.

These complement SimpleGraphSAGE in benchmarks.py by providing experimental GIN and
GraphTransformer fixed-prior baselines. Use them only when the same registered
runtime, split, preprocessing and hyperparameter protocol are used.

Drop this file into the same package folder as benchmarks.py:

    intergate/benchmarks_gnn_baselines.py

and then in your notebook:

    from intergate.benchmarks_gnn_baselines import (
        run_gin_bioinfo,
        run_graphtransformer_bioinfo,
        run_all_fixed_prior_gnn_baselines,
    )

The protocol mirrors run_graphsage_bioinfo() exactly:
  - scalar gene expression as the only node feature (in_channels=1)
  - the integrated HuRI + OmniPath prior held FIXED and UNTRAINED
  - 2 message-passing layers, hidden dimension 128
  - global mean pooling
  - 2-layer MLP classification head
  - AdamW, weighted multiclass cross-entropy with class weights from training
  - validation-driven early stopping on macro-F1, patience 20
  - 3 independent seeds: [1234, 42, 369]

NOTE on the conceptual contrast with backbone_ablation.py:
The blocks in intergate.backbone_blocks (WeightedGINBlock,
LocalGraphTransformerBlock) are drop-in replacements INSIDE the proposed
framework. They keep the learned gates, Top-K pruning, X_h conditioning,
attention pooling, hybrid head, and two-phase training active. They are
appropriate for an internal backbone analysis, NOT for fixed-prior baselines.
The classes here below ARE the fixed-prior baselines.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────
#  Optional PyG imports (mirrors benchmarks.py)
# ───────────────────────────────────────────────────────────────────────────

try:
    from torch_geometric.nn import GINConv, TransformerConv, global_mean_pool
    try:
        from torch_geometric.loader import DataLoader as PygDL
    except Exception:
        from torch_geometric.data import DataLoader as PygDL
    HAS_PYG = True
except Exception:
    HAS_PYG = False


# Private helpers reused from benchmarks.py.  These names are private but the
# user owns both files; this avoids ~200 lines of duplication across
# GraphSAGE / GIN / GraphTransformer baselines.
from intergate.benchmarks import (
    _require,
    _rt,
    _build_pyg_dataset,
    _metrics_from_pred_and_scores,
)
from intergate.utils import set_all_seeds


# ───────────────────────────────────────────────────────────────────────────
#  Hyperparameters (shared across all three fixed-prior GNN baselines)
# ───────────────────────────────────────────────────────────────────────────

FIXED_PRIOR_GNN_SEEDS    = [1234, 42, 369]
FIXED_PRIOR_GNN_EPOCHS   = 120
FIXED_PRIOR_GNN_LR       = 3e-3
FIXED_PRIOR_GNN_HIDDEN   = 128
FIXED_PRIOR_GNN_DROPOUT  = 0.30
FIXED_PRIOR_GNN_PATIENCE = 20
FIXED_PRIOR_GTR_HEADS    = 4   # heads for TransformerConv


# ───────────────────────────────────────────────────────────────────────────
#  Models
# ───────────────────────────────────────────────────────────────────────────

class SimpleGIN(nn.Module):
    """
    Fixed-prior GIN baseline.

    Mirrors SimpleGraphSAGE structure:
      - in_channels=1 (scalar gene expression)
      - 2 GINConv layers, each wrapping an internal 2-layer MLP
      - global mean pooling
      - 2-layer MLP classification head
    """

    def __init__(self, in_channels: int, hidden: int, n_classes: int, dropout: float = 0.30):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for SimpleGIN")

        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        mlp2 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.conv1 = GINConv(mlp1, train_eps=True)
        self.conv2 = GINConv(mlp2, train_eps=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
        self.dropout = float(dropout)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        hg = global_mean_pool(h, batch)
        return self.head(hg)


class SimpleGraphTransformer(nn.Module):
    """
    Fixed-prior GraphTransformer baseline.

    Mirrors SimpleGraphSAGE structure:
      - in_channels=1 (scalar gene expression)
      - 2 TransformerConv layers (multi-head, concatenated → hidden)
      - global mean pooling
      - 2-layer MLP classification head
    """

    def __init__(self, in_channels: int, hidden: int, n_classes: int,
                 dropout: float = 0.30, heads: int = 4):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for SimpleGraphTransformer")
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}")

        head_dim = hidden // heads
        # concat=True (default) → output dim = heads * head_dim = hidden
        self.conv1 = TransformerConv(in_channels, head_dim, heads=heads, dropout=0.0, beta=False)
        self.conv2 = TransformerConv(hidden,       head_dim, heads=heads, dropout=0.0, beta=False)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
        self.dropout = float(dropout)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        hg = global_mean_pool(h, batch)
        return self.head(hg)


# ───────────────────────────────────────────────────────────────────────────
#  Generic train+eval loop (mirrors run_graphsage_bioinfo exactly)
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _predict_pyg_baseline(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        all_logits.append(logits.detach().cpu())
        all_y.append(batch.y.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0).numpy().astype(np.int64)
    proba  = logits.softmax(dim=1).numpy()
    y_pred = proba.argmax(axis=1)
    return y_true, y_pred, proba


def _run_one_fixed_prior_pyg_seed(
    *,
    family_name: str,
    model_factory,           # callable(in_channels, hidden, n_classes, dropout) -> nn.Module
    seed: int,
    hidden: int,
    lr: float,
    epochs: int,
    dropout: float,
    patience: int,
) -> dict:
    """Run one seed of a fixed-prior PyG baseline.  Returns a dict row."""
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for fixed-prior PyG baselines")

    _require(["Xs_gene", "train_idx", "val_idx", "test_idx",
              "y", "label_map", "edge_index", "n_classes", "DEVICE"])

    X_node    = np.asarray(_rt("Xs_gene"), dtype=np.float32)
    y_arr     = np.asarray(_rt("y"),       dtype=np.int64)
    ei_t      = torch.tensor(_rt("edge_index"), dtype=torch.long)
    tr        = np.asarray(_rt("train_idx"), dtype=np.int64)
    va        = np.asarray(_rt("val_idx"),   dtype=np.int64)
    te        = np.asarray(_rt("test_idx"),  dtype=np.int64)
    label_map = _rt("label_map")
    n_classes = int(_rt("n_classes"))
    device    = _rt("DEVICE")

    set_all_seeds(seed)

    ds_tr = _build_pyg_dataset(X_node, ei_t, y_arr, tr)
    ds_va = _build_pyg_dataset(X_node, ei_t, y_arr, va)
    ds_te = _build_pyg_dataset(X_node, ei_t, y_arr, te)
    dl_tr = PygDL(ds_tr, batch_size=16, shuffle=True)
    dl_va = PygDL(ds_va, batch_size=32, shuffle=False)
    dl_te = PygDL(ds_te, batch_size=32, shuffle=False)

    counts = np.bincount(y_arr[tr], minlength=n_classes).astype(np.float32)
    cw = counts.sum() / np.maximum(counts, 1.0)
    cw /= max(cw.mean(), 1e-8)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)

    model = model_factory(in_channels=1, hidden=hidden,
                          n_classes=n_classes, dropout=dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=cw_t)

    best_state, best_val_f1, best_epoch, pat = None, -np.inf, -1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        for batch in dl_tr:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * int(batch.y.shape[0])
            n_seen  += int(batch.y.shape[0])

        y_va, pred_va, proba_va = _predict_pyg_baseline(model, dl_va, device)
        m_va = _metrics_from_pred_and_scores(y_va, pred_va, proba_va, label_map)

        if m_va["macro_f1"] > best_val_f1 + 1e-4:
            best_val_f1 = m_va["macro_f1"]
            best_epoch  = epoch
            best_state  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"  [{family_name}] epoch={epoch:03d} | loss={running/max(n_seen,1):.4f} | val_F1={m_va['macro_f1']:.4f}")

        if pat >= patience:
            print(f"  [{family_name}] early stop @ epoch={epoch} | best_epoch={best_epoch} | best_val_F1={best_val_f1:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_va, pred_va, proba_va = _predict_pyg_baseline(model, dl_va, device)
    y_te, pred_te, proba_te = _predict_pyg_baseline(model, dl_te, device)
    m_va = _metrics_from_pred_and_scores(y_va, pred_va, proba_va, label_map)
    m_te = _metrics_from_pred_and_scores(y_te, pred_te, proba_te, label_map)

    row = {
        "family": family_name,
        "seed":   int(seed),
        "val_macro_f1":      float(m_va["macro_f1"]),
        "val_acc":           float(m_va["acc"]),
        "val_auc_ovr_macro": float(m_va["auc_ovr_macro"]),
        "test_macro_f1":     float(m_te["macro_f1"]),
        "test_acc":          float(m_te["acc"]),
        "test_auc_ovr_macro":float(m_te["auc_ovr_macro"]),
        **{f"test_{k}": float(v) for k, v in m_te.items() if str(k).startswith("f1_")},
    }

    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass

    return row


def _summarize_baseline_rows(rows: List[dict], family_name: str, out_csv: Optional[str]) -> pd.DataFrame:
    df_raw = pd.DataFrame(rows)
    metric_cols = ["val_macro_f1", "val_acc", "val_auc_ovr_macro",
                   "test_macro_f1", "test_acc", "test_auc_ovr_macro"]
    agg = df_raw.groupby("family")[metric_cols].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    print(f"\n=== {family_name} summary (mean ± std across seeds) ===")
    print(agg.to_string(index=False))

    if out_csv:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_raw.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")

    return df_raw


# ───────────────────────────────────────────────────────────────────────────
#  Public entry points
# ───────────────────────────────────────────────────────────────────────────

def run_gin_bioinfo(*, seeds=None, out_csv: str = "./bioinfo_gin.csv") -> pd.DataFrame:
    """
    Fixed-prior GIN baseline.  Same protocol as run_graphsage_bioinfo().

    Returns a DataFrame with one row per seed.  The row carries
    family="gin_fixed_prior" so it concatenates cleanly with the existing
    GraphSAGE / fixed-prior controls panel for Table 1 of the paper.
    """
    if not HAS_PYG:
        print("torch_geometric no disponible. pip install torch_geometric")
        return pd.DataFrame()

    if seeds is None:
        seeds = FIXED_PRIOR_GNN_SEEDS

    rows = []
    for seed in seeds:
        print(f"\n[GIN] seed={seed}")
        row = _run_one_fixed_prior_pyg_seed(
            family_name="gin_fixed_prior",
            model_factory=lambda **kw: SimpleGIN(**kw),
            seed=seed,
            hidden=FIXED_PRIOR_GNN_HIDDEN,
            lr=FIXED_PRIOR_GNN_LR,
            epochs=FIXED_PRIOR_GNN_EPOCHS,
            dropout=FIXED_PRIOR_GNN_DROPOUT,
            patience=FIXED_PRIOR_GNN_PATIENCE,
        )
        rows.append(row)

    return _summarize_baseline_rows(rows, "GIN", out_csv)


def run_graphtransformer_bioinfo(*, seeds=None, out_csv: str = "./bioinfo_graphtransformer.csv") -> pd.DataFrame:
    """
    Fixed-prior GraphTransformer baseline.  Same protocol as run_graphsage_bioinfo().

    Returns a DataFrame with one row per seed (family="gtransformer_fixed_prior").
    """
    if not HAS_PYG:
        print("torch_geometric no disponible. pip install torch_geometric")
        return pd.DataFrame()

    if seeds is None:
        seeds = FIXED_PRIOR_GNN_SEEDS

    heads = FIXED_PRIOR_GTR_HEADS

    rows = []
    for seed in seeds:
        print(f"\n[GraphTransformer] seed={seed}")
        row = _run_one_fixed_prior_pyg_seed(
            family_name="gtransformer_fixed_prior",
            model_factory=lambda **kw: SimpleGraphTransformer(heads=heads, **kw),
            seed=seed,
            hidden=FIXED_PRIOR_GNN_HIDDEN,
            lr=FIXED_PRIOR_GNN_LR,
            epochs=FIXED_PRIOR_GNN_EPOCHS,
            dropout=FIXED_PRIOR_GNN_DROPOUT,
            patience=FIXED_PRIOR_GNN_PATIENCE,
        )
        rows.append(row)

    return _summarize_baseline_rows(rows, "GraphTransformer", out_csv)


def run_all_fixed_prior_gnn_baselines(
    *,
    seeds=None,
    out_dir: str = ".",
    include_graphsage: bool = True,
) -> pd.DataFrame:
    """
    Convenience: run GraphSAGE + GIN + GraphTransformer fixed-prior baselines
    and concat the per-seed results into a single DataFrame, plus save a
    combined CSV.  Use this once and you get all three rows for Table 1.
    """
    if not HAS_PYG:
        print("torch_geometric no disponible. pip install torch_geometric")
        return pd.DataFrame()

    pieces = []
    if include_graphsage:
        from intergate.benchmarks import run_graphsage_bioinfo
        df = run_graphsage_bioinfo(
            seeds=seeds,
            out_csv=os.path.join(out_dir, "bioinfo_graphsage.csv"),
        )
        if df is not None and not df.empty:
            pieces.append(df)

    df_gin = run_gin_bioinfo(
        seeds=seeds,
        out_csv=os.path.join(out_dir, "bioinfo_gin.csv"),
    )
    if df_gin is not None and not df_gin.empty:
        pieces.append(df_gin)

    df_gtr = run_graphtransformer_bioinfo(
        seeds=seeds,
        out_csv=os.path.join(out_dir, "bioinfo_graphtransformer.csv"),
    )
    if df_gtr is not None and not df_gtr.empty:
        pieces.append(df_gtr)

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, axis=0, ignore_index=True)
    final_csv = os.path.join(out_dir, "bioinfo_fixed_prior_gnn_baselines.csv")
    out.to_csv(final_csv, index=False)
    print(f"\n[SAVE] {final_csv}")
    return out
