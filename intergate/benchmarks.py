"""
Benchmarks: tabular baselines, fixed-prior graph controls,
PAM50 centroid, LightGBM, CatBoost, GCN, GraphSAGE, SOTA comparison.

USAGE
-----
    from intergate.benchmarks import register_runtime_benchmarks

    register_runtime_benchmarks(
        Xs_gene=Xs_gene, train_idx=train_idx, val_idx=val_idx,
        test_idx=test_idx, y=y, label_map=label_map, genes_kegg=genes_kegg,
        n_classes=n_classes, edge_index=edge_index, X_h=Xs_graph,
        DEVICE=DEVICE, BATCH_SIZE=CFG.BATCH_SIZE,
        # ... any other objects that benchmark functions need
    )

    from intergate.benchmarks import (
        run_tabular_baselines_bioinfo,
        run_fixed_prior_graph_controls_bioinfo,
        run_centroid_baselines_bioinfo,
        run_sota_tabular_bioinfo,
        run_simple_gcn_bioinfo,
        run_graphsage_bioinfo,
        run_signature291_tabular_benchmarks_bioinfo,
        build_sota_summary_table,
        build_requested_benchmarks_summary,
    )
"""

from __future__ import annotations

import os
import json
from copy import deepcopy
from dataclasses import dataclass
from itertools import product as iprod
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Lazy sklearn imports (used in model grids)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.base import clone

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# PyG imports (optional – only needed for GCN/GraphSAGE)
try:
    from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
    from torch_geometric.data import Data
    try:
        from torch_geometric.loader import DataLoader as PygDL
    except ImportError:
        from torch_geometric.data import DataLoader as PygDL
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from .utils import set_all_seeds


# ════════════════════════════════════════════════════════════════════════════
#  RUNTIME REGISTRY  (replaces globals()-based approach)
# ════════════════════════════════════════════════════════════════════════════

_RUNTIME: Dict[str, Any] = {}


def _default_runtime_objects() -> Dict[str, Any]:
    """Objects that can be inferred from the installed InterGATE package.

    Notebooks still register dataset-specific objects explicitly.  This helper
    prevents fixed-prior benchmark cells from failing because package-level
    constructors or configuration constants were not manually passed.
    """
    try:
        from intergate.config import CFG
        from intergate.ablation import (
            AblationConfig, build_model_from_cfg, make_prior_for_cfg,
            get_full_gene_matrix_and_genes,
        )
        from intergate.model import build_pruned_model_from
        from intergate.data import ExpressionDataset
        from intergate.training import (
            finetune_pruned, predict_proba_xh_mode, pretrain_masked_gene_model,
        )
        from intergate.losses import compute_metrics_full
    except Exception:
        return {}

    return {
        "AblationConfig": AblationConfig,
        "build_model_from_cfg": build_model_from_cfg,
        "make_prior_for_cfg": make_prior_for_cfg,
        "get_full_gene_matrix_and_genes": get_full_gene_matrix_and_genes,
        "build_pruned_model_from": build_pruned_model_from,
        "ExpressionDataset": ExpressionDataset,
        "finetune_pruned": finetune_pruned,
        "predict_proba_xh_mode": predict_proba_xh_mode,
        "pretrain_masked_gene_model": pretrain_masked_gene_model,
        "compute_metrics_full": compute_metrics_full,
        "DEVICE": CFG.DEVICE,
        "HIDDEN": CFG.HIDDEN,
        "NUM_LAYERS": CFG.NUM_LAYERS,
        "DROPOUT": CFG.DROPOUT,
        "POOL_HEADS": CFG.POOL_HEADS,
        "XGRAPH_DROPOUT": CFG.XGRAPH_DROPOUT,
        "BLOCK_USE_SELF": CFG.BLOCK_USE_SELF,
        "BLOCK_RESIDUAL": CFG.BLOCK_RESIDUAL,
        "HYBRID_TAB_LAYERS": CFG.HYBRID_TAB_LAYERS,
        "HYBRID_TAB_DROPOUT": CFG.HYBRID_TAB_DROPOUT,
        "HYBRID_FUSION_DROPOUT": CFG.HYBRID_FUSION_DROPOUT,
        "HYBRID_BLEND_LOGITS": CFG.HYBRID_BLEND_LOGITS,
        "HYBRID_BLEND_INIT": CFG.HYBRID_BLEND_INIT,
        "LR": CFG.LR,
        "WEIGHT_DECAY": CFG.WEIGHT_DECAY,
        "ACCUM_STEPS": CFG.ACCUM_STEPS,
        "USE_MIXUP": CFG.USE_MIXUP,
        "MIXUP_ALPHA": CFG.MIXUP_ALPHA,
        "MIXUP_P": CFG.MIXUP_P,
        "FT_EPOCHS_2A": CFG.FT_EPOCHS_A,
        "FT_EPOCHS_2B": CFG.FT_EPOCHS_B,
        "PHASE2_BEST_METRIC": "macro_f1",
        "PHASE2_PATIENCE": 20,
        "PHASE2_MIN_DELTA": 1e-4,
        "PHASE2_LR_TAB_MULT": 2.0,
        "PHASE2_LR_FUSION_MULT": 1.0,
    }


def register_runtime_benchmarks(**kwargs) -> None:
    """
    Register notebook objects so benchmark functions can access them.

    Dataset-specific objects must be passed from the notebook.  Package-level
    constructors, metrics and configuration constants are filled automatically.
    """
    defaults = _default_runtime_objects()
    defaults.update(kwargs)
    _RUNTIME.update(defaults)


def _rt(name: str):
    """Shortcut to retrieve a registered runtime object."""
    return _RUNTIME[name]


def _require(names: List[str]) -> None:
    """Raise if any of *names* are missing from the runtime registry."""
    missing = [n for n in names if n not in _RUNTIME]
    if missing:
        raise RuntimeError(
            "Faltan estos objetos en el registro de benchmarks: "
            + ", ".join(missing)
            + ". Llama a register_runtime_benchmarks() primero."
        )


# ════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _id_to_name_map(label_map) -> Dict[int, str]:
    if isinstance(list(label_map.keys())[0], str):
        return {v: k for k, v in label_map.items()}
    return dict(label_map)


def _safe_auc(y_true: np.ndarray, score_or_proba: Optional[np.ndarray]) -> float:
    try:
        if score_or_proba is None or score_or_proba.ndim == 1:
            return float("nan")
        return float(roc_auc_score(y_true, score_or_proba,
                                   multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _metrics_from_pred_and_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_or_proba: Optional[np.ndarray],
    label_map,
) -> Dict[str, float]:
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc_ovr_macro": _safe_auc(y_true, score_or_proba),
    }
    id_to_name = _id_to_name_map(label_map)
    per_class = f1_score(
        y_true, y_pred, average=None,
        labels=np.arange(len(id_to_name)),
        zero_division=0,
    )
    for cid, val in enumerate(per_class):
        out[f"f1_{id_to_name.get(cid, str(cid))}"] = float(val)
    return out


def _predict_scores(model, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Returns (y_pred, score_or_proba)."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        pred = np.asarray(proba).argmax(axis=1)
        return pred, np.asarray(proba)
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X))
        if score.ndim == 1:
            pred = (score > 0).astype(int)
            return pred, score
        pred = score.argmax(axis=1)
        return pred, score
    pred = model.predict(X)
    return np.asarray(pred), None


def _build_xy_from_runtime():
    _require(["Xs_gene", "train_idx", "val_idx", "test_idx", "y", "label_map"])
    X = np.asarray(_rt("Xs_gene"), dtype=np.float32)
    y = np.asarray(_rt("y"), dtype=np.int64)
    train_idx = np.asarray(_rt("train_idx"), dtype=np.int64)
    val_idx   = np.asarray(_rt("val_idx"),   dtype=np.int64)
    test_idx  = np.asarray(_rt("test_idx"),  dtype=np.int64)
    label_map = _rt("label_map")
    return {
        "X_tr": X[train_idx], "X_va": X[val_idx], "X_te": X[test_idx],
        "y_tr": y[train_idx], "y_va": y[val_idx], "y_te": y[test_idx],
        "label_map": label_map,
    }


def _fit_tabular_model(model, X: np.ndarray, y: np.ndarray):
    if HAS_XGBOOST and isinstance(model, XGBClassifier):
        sw = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, sample_weight=sw)
    else:
        model.fit(X, y)
    return model


# ════════════════════════════════════════════════════════════════════════════
#  1) STRONG TABULAR BASELINES
# ════════════════════════════════════════════════════════════════════════════

def _tabular_model_grid(random_state: int = 1234):
    grids = {
        "elasticnet_logreg": [
            Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(
                    solver="saga", l1_ratio=l1_ratio, C=C_val,
                    max_iter=4000, class_weight="balanced",
                    random_state=random_state,
                ))
            ])
            for C_val, l1_ratio in iprod([0.05], [0.5])
        ],
        "linear_svm_calibrated": [
            Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", CalibratedClassifierCV(
                    estimator=LinearSVC(
                        C=C_val, class_weight="balanced",
                        random_state=random_state, dual="auto", max_iter=5000,
                    ),
                    method="sigmoid", cv=3,
                ))
            ])
            for C_val in [0.01, 0.1, 1.0, 5.0]
        ],
        "extratrees": [
            ExtraTreesClassifier(
                n_estimators=ne, max_depth=md, min_samples_leaf=msl,
                class_weight="balanced_subsample",
                random_state=random_state, n_jobs=-1,
            )
            for ne, md, msl in iprod([400, 800], [None, 30], [1, 3])
        ],
        "random_forest": [
            RandomForestClassifier(
                n_estimators=ne, max_depth=md, min_samples_leaf=msl,
                class_weight="balanced_subsample",
                random_state=random_state, n_jobs=-1,
            )
            for ne, md, msl in iprod([400, 800], [None, 30], [1, 3])
        ],
        "mlp_graph_free": [
            Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=hls, alpha=alpha, batch_size=64,
                    learning_rate_init=lr, max_iter=200,
                    early_stopping=True, n_iter_no_change=15,
                    random_state=random_state,
                ))
            ])
            for hls, alpha, lr in iprod(
                [(64, 32), (256, 128)], [1e-4, 1e-3], [1e-3, 5e-4]
            )
        ],
    }

    if HAS_XGBOOST:
        grids["xgboost"] = [
            XGBClassifier(
                objective="multi:softprob",
                n_estimators=ne, max_depth=md, learning_rate=lr,
                subsample=sub, colsample_bytree=csb,
                tree_method="hist", n_jobs=-1,
                random_state=random_state, eval_metric="mlogloss", verbosity=0,
            )
            for ne, md, lr, sub, csb in iprod(
                [200, 400], [3, 5], [0.05, 0.1], [0.8, 1.0], [0.8, 1.0]
            )
        ]

    return grids


def run_tabular_baselines_bioinfo(
    *,
    random_state: int = 1234,
    out_csv: str = "./bioinfo_tabular_benchmarks.csv",
) -> pd.DataFrame:
    """
    Validation-driven model selection on the SAME split already used by the notebook.
    """
    data = _build_xy_from_runtime()
    X_tr, X_va, X_te = data["X_tr"], data["X_va"], data["X_te"]
    y_tr, y_va, y_te = data["y_tr"], data["y_va"], data["y_te"]
    label_map = data["label_map"]

    rows = []
    grids = _tabular_model_grid(random_state=random_state)

    for family, candidates in grids.items():
        best = None
        best_val = -np.inf
        best_idx = -1

        print(f"\n[TABULAR] {family}: {len(candidates)} candidates")
        for i, model in enumerate(candidates):
            _fit_tabular_model(model, X_tr, y_tr)
            pred_va, score_va = _predict_scores(model, X_va)
            m_va = _metrics_from_pred_and_scores(y_va, pred_va, score_va, label_map)
            val = float(m_va["macro_f1"])
            print(f"  cand {i:02d} | val_macro_f1={val:.4f} acc={m_va['acc']:.4f} auc={m_va['auc_ovr_macro']:.4f}")
            if val > best_val:
                best, best_val, best_idx = model, val, i

        pred_te, score_te = _predict_scores(best, X_te)
        m_te = _metrics_from_pred_and_scores(y_te, pred_te, score_te, label_map)

        rows.append({
            "family": family,
            "selected_candidate": int(best_idx),
            "val_macro_f1": float(best_val),
            **{f"test_{k}": v for k, v in m_te.items()},
        })

    df = pd.DataFrame(rows).sort_values(
        ["test_macro_f1", "test_acc"], ascending=[False, False]
    ).reset_index(drop=True)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")
    return df


# ════════════════════════════════════════════════════════════════════════════
#  2) FIXED-PRIOR GRAPH CONTROLS
# ════════════════════════════════════════════════════════════════════════════

FIXED_PRIOR_SEEDS       = [1234, 42, 369]
FIXED_PRIOR_DO_PRETRAIN = True
FIXED_PRIOR_TOPK_EDGES  = 5000


def _class_weights_from_runtime():
    _require(["y", "train_idx", "n_classes"])
    y_ = np.asarray(_rt("y"), dtype=np.int64)
    tr = np.asarray(_rt("train_idx"), dtype=np.int64)
    nc = int(_rt("n_classes"))
    counts = np.bincount(y_[tr], minlength=nc).astype(np.float32)
    w = counts.sum() / np.maximum(counts, 1.0)
    return w / max(w.mean(), 1e-8)


def _freeze_gating_for_fixed_graph(model):
    """Neutralize and freeze all edge-gating parameters."""
    core = model.gnn if hasattr(model, "gnn") else model

    if hasattr(core, "sample_cond_gating"):
        core.sample_cond_gating = False

    if hasattr(core, "edge_logit"):
        with torch.no_grad():
            core.edge_logit.fill_(8.0)
        core.edge_logit.requires_grad_(False)

    for attr in ("type_scale", "type_bias"):
        if hasattr(core, attr):
            param = getattr(core, attr)
            with torch.no_grad():
                try:
                    param.fill_(1.0 if attr == "type_scale" else 0.0)
                except Exception:
                    pass
            if hasattr(param, "requires_grad_"):
                param.requires_grad_(False)

    if hasattr(core, "cond_mlp") and core.cond_mlp is not None:
        for p in core.cond_mlp.parameters():
            p.requires_grad = False

    return model


def _filter_prior_topk(ei, ew, et, k: int):
    if ew is None or ei.shape[1] <= k:
        return ei, ew, et
    w_abs = ew.abs()
    topk_idx = torch.topk(w_abs, k=min(k, w_abs.shape[0]), largest=True).indices
    return ei[:, topk_idx], ew[topk_idx], (et[topk_idx] if et is not None else None)


def _make_fixed_prior_model(use_hybrid: bool, ei_prior, ew_prior, et_prior):
    _require([
        "AblationConfig", "build_pruned_model_from", "get_full_gene_matrix_and_genes",
        "DEVICE", "HIDDEN", "n_classes", "X_h", "NUM_LAYERS", "DROPOUT",
        "POOL_HEADS", "XGRAPH_DROPOUT", "BLOCK_USE_SELF", "BLOCK_RESIDUAL",
        "HYBRID_TAB_LAYERS", "HYBRID_TAB_DROPOUT", "HYBRID_FUSION_DROPOUT",
        "HYBRID_BLEND_LOGITS", "HYBRID_BLEND_INIT",
    ])
    X_full, _ = _rt("get_full_gene_matrix_and_genes")()

    cfg_seed = _rt("AblationConfig")(
        name="FIXED_PRIOR_CTRL",
        edge_type_gating=True,
        sample_cond_gating=False,
        sample_cond_mode="global",
        use_signed_prior=True,
        signed_channels_mode="type_only",
        connectivity_penalty=False,
        do_pretrain=False,
        do_stability=False,
    )
    cw = _class_weights_from_runtime()
    cw_t = torch.tensor(cw, dtype=torch.float32, device=_rt("DEVICE"))

    model_seed = _rt("build_model_from_cfg")(cfg_seed, cw_t)

    model = _rt("build_pruned_model_from")(
        model_gated=model_seed,
        edge_index_p=ei_prior,
        edge_weight_p=ew_prior,
        edge_type_p=et_prior,
        num_nodes=int(X_full.shape[1]),
        hidden=int(_rt("HIDDEN")),
        n_classes=int(_rt("n_classes")),
        graph_feat_dim=int(_rt("X_h").shape[1]),
        num_layers=int(_rt("NUM_LAYERS")),
        dropout=float(_rt("DROPOUT")),
        num_heads=int(_rt("POOL_HEADS")),
        xgraph_dropout=float(_rt("XGRAPH_DROPOUT")),
        block_use_self=bool(_rt("BLOCK_USE_SELF")),
        block_residual=bool(_rt("BLOCK_RESIDUAL")),
        pool_level="gene",
        device=_rt("DEVICE"),
        use_hybrid=bool(use_hybrid),
        tab_layers=int(_rt("HYBRID_TAB_LAYERS")),
        tab_dropout=float(_rt("HYBRID_TAB_DROPOUT")),
        fusion_dropout=float(_rt("HYBRID_FUSION_DROPOUT")),
        blend_logits=bool(_rt("HYBRID_BLEND_LOGITS")),
        blend_init=float(_rt("HYBRID_BLEND_INIT")),
    )
    return _freeze_gating_for_fixed_graph(model)


def _make_fixed_prior_loaders():
    _require(["ExpressionDataset", "Xs_gene", "X_h", "y",
              "train_idx", "val_idx", "test_idx", "BATCH_SIZE"])
    X   = np.asarray(_rt("Xs_gene"), dtype=np.float32)
    Xh  = np.asarray(_rt("X_h"),    dtype=np.float32)
    y_  = np.asarray(_rt("y"),      dtype=np.int64)
    pin = bool(torch.cuda.is_available())
    num_workers = int(_RUNTIME.get("NUM_WORKERS", 0))

    def _dl(idx, shuffle):
        ds = _rt("ExpressionDataset")(X, Xh, y_, idx)
        return torch.utils.data.DataLoader(
            ds, batch_size=_rt("BATCH_SIZE"), shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin,
        )
    return (
        _dl(_rt("train_idx"), True),
        _dl(_rt("val_idx"), False),
        _dl(_rt("test_idx"), False),
    )


def _run_one_fixed_prior_seed(name, use_hybrid, ei_prior, ew_prior, et_prior, seed):
    set_all_seeds(seed)

    cw = _class_weights_from_runtime()
    cw_t = torch.tensor(cw, dtype=torch.float32, device=_rt("DEVICE"))
    loss_fn_fp = torch.nn.CrossEntropyLoss(weight=cw_t)

    model = _make_fixed_prior_model(use_hybrid, ei_prior, ew_prior, et_prior)

    if FIXED_PRIOR_DO_PRETRAIN and "pretrain_masked_gene_model" in _RUNTIME:
        dl_tr_pt, _, _ = _make_fixed_prior_loaders()
        model = _rt("pretrain_masked_gene_model")(model, dl_tr_pt, _rt("DEVICE"))
        _freeze_gating_for_fixed_graph(model)

    dl_tr, dl_va, dl_te = _make_fixed_prior_loaders()

    base_lr = float(_rt("LR")) * float(_RUNTIME.get("PHASE2_LR_MULT", 0.20))

    model = _rt("finetune_pruned")(
        model, None, dl_tr, dl_va, _rt("DEVICE"), _rt("label_map"), loss_fn_fp,
        lr=base_lr,
        weight_decay=float(_rt("WEIGHT_DECAY")),
        epochs_A=int(_rt("FT_EPOCHS_2A")),
        epochs_B=int(_rt("FT_EPOCHS_2B")),
        accum_steps=int(_rt("ACCUM_STEPS")),
        use_mixup_B=bool(_rt("USE_MIXUP")),
        mixup_alpha=float(_rt("MIXUP_ALPHA")),
        mixup_prob=float(_rt("MIXUP_P")),
        best_metric=str(_rt("PHASE2_BEST_METRIC")),
        patience_B=int(_rt("PHASE2_PATIENCE")),
        min_delta=float(_rt("PHASE2_MIN_DELTA")),
        lr_tab_mult=float(_rt("PHASE2_LR_TAB_MULT")),
        lr_fusion_mult=float(_rt("PHASE2_LR_FUSION_MULT")),
    )

    proba_va, y_va = _rt("predict_proba_xh_mode")(model, None, dl_va, _rt("DEVICE"), xh_mode="orig")
    m_va = _rt("compute_metrics_full")(y_va, proba_va, _rt("label_map"))

    proba_te, y_te = _rt("predict_proba_xh_mode")(model, None, dl_te, _rt("DEVICE"), xh_mode="orig")
    m_te = _rt("compute_metrics_full")(y_te, proba_te, _rt("label_map"))

    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "family": name,
        "seed": seed,
        "val_macro_f1":      float(m_va.get("macro_f1",       float("nan"))),
        "val_acc":           float(m_va.get("acc",            float("nan"))),
        "val_auc_ovr_macro": float(m_va.get("auc_macro_ovr",  float("nan"))),
        "test_macro_f1":     float(m_te.get("macro_f1",       float("nan"))),
        "test_acc":          float(m_te.get("acc",            float("nan"))),
        "test_auc_ovr_macro":float(m_te.get("auc_macro_ovr",  float("nan"))),
        **{f"test_{k}": float(v) for k, v in m_te.items() if str(k).startswith("f1_")},
    }


def run_fixed_prior_graph_controls_bioinfo(
    *,
    out_csv: str = "./bioinfo_fixed_prior_controls.csv",
    seeds: list = None,
) -> pd.DataFrame:
    _require(["make_prior_for_cfg", "AblationConfig"])

    if seeds is None:
        seeds = FIXED_PRIOR_SEEDS

    cfg_dummy = _rt("AblationConfig")(
        name="__prior__", edge_type_gating=True, sample_cond_gating=False,
        sample_cond_mode="global", use_signed_prior=True,
        signed_channels_mode="type_only", connectivity_penalty=False,
        do_pretrain=False, do_stability=False,
    )
    ei_full, ew_full, et_full = _rt("make_prior_for_cfg")(cfg_dummy)
    ei_filt, ew_filt, et_filt = _filter_prior_topk(
        ei_full, ew_full, et_full, k=FIXED_PRIOR_TOPK_EDGES
    )

    controls = [
        ("fixed_prior_gnn",      False, ei_full, ew_full, et_full),
        ("fixed_prior_hybrid",   True,  ei_full, ew_full, et_full),
        ("fixed_prior_filtered", True,  ei_filt, ew_filt, et_filt),
    ]

    all_rows = []
    for name, use_hybrid, ei, ew, et in controls:
        n_edges = ei.shape[1]
        print(f"\n[FIXED-PRIOR] {name}  |  edges={n_edges}  |  seeds={seeds}")
        for seed in seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)
            row = _run_one_fixed_prior_seed(name, use_hybrid, ei, ew, et, seed)
            print(f"val_F1={row['val_macro_f1']:.3f}  test_F1={row['test_macro_f1']:.3f}")
            all_rows.append(row)

    df_raw = pd.DataFrame(all_rows)

    metric_cols = ["val_macro_f1", "val_acc", "val_auc_ovr_macro",
                   "test_macro_f1", "test_acc", "test_auc_ovr_macro"]
    agg = (
        df_raw.groupby("family")[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index().sort_values("test_macro_f1_mean", ascending=False)

    print("\n=== Fixed-prior summary (mean ± std across seeds) ===")
    print(agg[["family", "val_macro_f1_mean", "val_macro_f1_std",
                "test_macro_f1_mean", "test_macro_f1_std",
                "test_acc_mean", "test_auc_ovr_macro_mean"]].to_string(index=False))

    if out_csv:
        df_raw.to_csv(out_csv.replace(".csv", "_raw.csv"), index=False)
        agg.to_csv(out_csv, index=False)
        print(f"\n[SAVE] {out_csv}  (raw: {out_csv.replace('.csv','_raw.csv')})")

    return agg


# ════════════════════════════════════════════════════════════════════════════
#  3) PAM50 CENTROID CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════

PAM50_GENES = [
    "ACTR3B","ANLN","BAG1","BCL2","BIRC5","BLVRA","CCNB1","CCNE1",
    "CDC20","CDC6","CDH3","CENPF","CEP55","CXXC5","EGFR","ERBB2",
    "ESR1","EXO1","FGFR4","FOXA1","FOXC1","GPR160","GRB7","KIF2C",
    "KRT14","KRT17","KRT5","MAPT","MDM2","MELK","MIA","MKI67","MLPH",
    "MMP11","MYBL2","MYC","NAT1","NDC80","NUF2","ORC6","PGR",
    "PHGDH","PTTG1","RRM2","SFRP1","SLC39A6","TMEM45B","TYMS",
    "UBE2C","UBE2T",
]


def run_centroid_baselines_bioinfo(out_csv="./bioinfo_centroid_benchmarks.csv"):
    """Fit NearestCentroid classifiers and evaluate on the same split."""
    _require(["Xs_gene", "train_idx", "val_idx", "test_idx",
              "y", "label_map", "genes_kegg"])

    X_all = np.asarray(_rt("Xs_gene"), dtype=np.float32)
    y_all = np.asarray(_rt("y"),       dtype=np.int64)
    genes = [str(x).upper() for x in _rt("genes_kegg")]
    gene_idx = {g_: i for i, g_ in enumerate(genes)}

    tr = np.asarray(_rt("train_idx"), dtype=np.int64)
    va = np.asarray(_rt("val_idx"),   dtype=np.int64)
    te = np.asarray(_rt("test_idx"),  dtype=np.int64)
    label_map = _rt("label_map")

    pam50_present = [gg for gg in PAM50_GENES if gg in gene_idx]
    pam50_cols    = [gene_idx[gg] for gg in pam50_present]
    print(f"PAM50 genes found in matrix: {len(pam50_present)}/50")
    if len(pam50_present) < 10:
        print("  WARNING: fewer than 10 PAM50 genes found; "
              "check that gene_idx uses UPPER-CASE symbols.")

    variants = {
        "pam50_centroid_50":   pam50_cols if pam50_cols else None,
        "pam50_centroid_full": None,
    }

    rows = []
    for name, col_idx in variants.items():
        if col_idx is not None and len(col_idx) == 0:
            print(f"  SKIP {name}: no PAM50 genes found.")
            continue

        Xv = X_all[:, col_idx] if col_idx is not None else X_all

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    NearestCentroid(metric="euclidean")),
        ])
        pipe.fit(Xv[tr], y_all[tr])

        def _centroid_predict(pipe_, X_):
            scaler = pipe_.named_steps["scaler"]
            clf    = pipe_.named_steps["clf"]
            Xs = scaler.transform(X_)
            pred = clf.predict(Xs)
            centroids = clf.centroids_
            dists = np.linalg.norm(Xs[:, None, :] - centroids[None, :, :], axis=2)
            scores = -dists
            proba  = np.exp(scores - scores.max(axis=1, keepdims=True))
            proba /= proba.sum(axis=1, keepdims=True)
            return pred, proba

        pred_va, proba_va = _centroid_predict(pipe, Xv[va])
        m_va = _metrics_from_pred_and_scores(y_all[va], pred_va, proba_va, label_map)

        pred_te, proba_te = _centroid_predict(pipe, Xv[te])
        m_te = _metrics_from_pred_and_scores(y_all[te], pred_te, proba_te, label_map)

        rows.append({
            "family":       name,
            "n_genes_used": len(col_idx) if col_idx is not None else Xv.shape[1],
            "pam50_found":  len(pam50_present),
            "val_macro_f1": float(m_va["macro_f1"]),
            **{f"test_{k}": v for k, v in m_te.items()},
        })
        print(f"  {name:30s}  val_F1={m_va['macro_f1']:.4f}  "
              f"test_F1={m_te['macro_f1']:.4f}  "
              f"test_acc={m_te['acc']:.4f}")

    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")
    return df


# ════════════════════════════════════════════════════════════════════════════
#  4) LightGBM & CatBoost (SOTA tabular)
# ════════════════════════════════════════════════════════════════════════════

def _lgbm_grid(rs=1234):
    if not HAS_LGBM:
        return []
    return [
        lgb.LGBMClassifier(
            objective="multiclass",
            num_leaves=nl, n_estimators=ne,
            learning_rate=lr, colsample_bytree=csb,
            subsample=sub, reg_alpha=0.1,
            class_weight="balanced",
            random_state=rs, n_jobs=-1, verbosity=-1,
        )
        for nl, ne, lr, csb, sub in iprod(
            [63, 127], [300, 600], [0.05, 0.1], [0.5, 0.8], [0.8, 1.0],
        )
    ]


def _catboost_grid(rs=1234):
    if not HAS_CATBOOST:
        return []
    return [
        cb.CatBoostClassifier(
            iterations=ni, depth=d, learning_rate=lr,
            l2_leaf_reg=3.0, auto_class_weights="Balanced",
            random_seed=rs, verbose=0, thread_count=-1,
        )
        for ni, d, lr in iprod([400, 800], [6, 8], [0.05, 0.1])
    ]


def run_sota_tabular_bioinfo(random_state=1234,
                              out_csv="./bioinfo_sota_tabular.csv"):
    _require(["Xs_gene", "train_idx", "val_idx", "test_idx", "y", "label_map"])
    data = _build_xy_from_runtime()
    X_tr, X_va, X_te = data["X_tr"], data["X_va"], data["X_te"]
    y_tr, y_va, y_te = data["y_tr"], data["y_va"], data["y_te"]
    label_map = data["label_map"]

    families = {}
    if HAS_LGBM:     families["lightgbm"] = _lgbm_grid(random_state)
    if HAS_CATBOOST:  families["catboost"] = _catboost_grid(random_state)
    if not families:
        print("Instala lightgbm o catboost para ejecutar esta celda.")
        return pd.DataFrame()

    rows = []
    for family, candidates in families.items():
        best, best_val, best_idx = None, -np.inf, -1
        print(f"\n[SOTA-TABULAR] {family}: {len(candidates)} candidatos")

        for i, model in enumerate(candidates):
            try:
                if HAS_LGBM and isinstance(model, lgb.LGBMClassifier):
                    model.fit(X_tr, y_tr)
                elif HAS_CATBOOST and isinstance(model, cb.CatBoostClassifier):
                    model.fit(X_tr, y_tr)
                else:
                    sw = compute_sample_weight("balanced", y_tr)
                    model.fit(X_tr, y_tr, sample_weight=sw)

                pred_va, proba_va = _predict_scores(model, X_va)
                m_va = _metrics_from_pred_and_scores(y_va, pred_va, proba_va, label_map)
                val = float(m_va["macro_f1"])
                print(f"  cand {i:03d} | val_F1={val:.4f}  "
                      f"acc={m_va['acc']:.4f}  auc={m_va['auc_ovr_macro']:.4f}")
                if val > best_val:
                    best, best_val, best_idx = model, val, i
            except Exception as exc:
                print(f"  cand {i:03d} FAILED: {exc}")

        if best is None:
            continue

        pred_te, proba_te = _predict_scores(best, X_te)
        m_te = _metrics_from_pred_and_scores(y_te, pred_te, proba_te, label_map)
        rows.append({
            "family": family,
            "selected_candidate": int(best_idx),
            "val_macro_f1": float(best_val),
            **{f"test_{k}": v for k, v in m_te.items()},
        })
        print(f"  ► SELECCIONADO cand {best_idx} | "
              f"test_F1={m_te['macro_f1']:.4f}  "
              f"test_acc={m_te['acc']:.4f}  "
              f"test_auc={m_te['auc_ovr_macro']:.4f}")

    df = pd.DataFrame(rows)
    if out_csv and len(df):
        df.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")
    return df


# ════════════════════════════════════════════════════════════════════════════
#  5) SIMPLE GCN BASELINE (Kipf & Welling, ICLR 2017)
# ════════════════════════════════════════════════════════════════════════════

SIMPLE_GCN_SEEDS    = [1234, 42, 369]
SIMPLE_GCN_EPOCHS   = 120
SIMPLE_GCN_LR       = 3e-3
SIMPLE_GCN_HIDDEN   = 128
SIMPLE_GCN_DROPOUT  = 0.3
SIMPLE_GCN_PATIENCE = 20


class SimpleGCN(nn.Module):
    """2-layer GCN with global mean pooling and a 2-layer MLP head."""
    def __init__(self, in_channels, hidden, n_classes, dropout=0.3):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for SimpleGCN")
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head  = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = global_mean_pool(h, batch)
        return self.head(h)


def _build_pyg_dataset(X_node, edge_index_t, y_arr, idx):
    data_list = []
    for i in idx:
        x = torch.tensor(X_node[i], dtype=torch.float32).unsqueeze(1)
        d = Data(
            x=x, edge_index=edge_index_t,
            y=torch.tensor(int(y_arr[i]), dtype=torch.long),
        )
        data_list.append(d)
    return data_list


def run_simple_gcn_bioinfo(seeds=None, out_csv="./bioinfo_simple_gcn.csv"):
    if not HAS_PYG:
        print("torch_geometric no disponible. pip install torch_geometric")
        return pd.DataFrame()

    _require(["Xs_gene", "train_idx", "val_idx", "test_idx",
              "y", "label_map", "edge_index", "n_classes", "DEVICE"])

    if seeds is None:
        seeds = SIMPLE_GCN_SEEDS

    X_node    = np.asarray(_rt("Xs_gene"), dtype=np.float32)
    y_arr     = np.asarray(_rt("y"),       dtype=np.int64)
    ei_t      = torch.tensor(_rt("edge_index"), dtype=torch.long)
    tr        = np.asarray(_rt("train_idx"), dtype=np.int64)
    va        = np.asarray(_rt("val_idx"),   dtype=np.int64)
    te        = np.asarray(_rt("test_idx"),  dtype=np.int64)
    label_map = _rt("label_map")
    n_classes = int(_rt("n_classes"))
    device    = _rt("DEVICE")

    counts = np.bincount(y_arr[tr], minlength=n_classes).astype(np.float32)
    cw     = counts.sum() / np.maximum(counts, 1.0)
    cw    /= cw.mean()
    cw_t   = torch.tensor(cw, dtype=torch.float32, device=device)

    print("Building PyG datasets ...")
    ds_tr = _build_pyg_dataset(X_node, ei_t, y_arr, tr)
    ds_va = _build_pyg_dataset(X_node, ei_t, y_arr, va)
    ds_te = _build_pyg_dataset(X_node, ei_t, y_arr, te)
    print(f"  train={len(ds_tr)}  val={len(ds_va)}  test={len(ds_te)}")

    bs = min(32, len(ds_tr))
    dl_tr = PygDL(ds_tr, batch_size=bs, shuffle=True)
    dl_va = PygDL(ds_va, batch_size=bs, shuffle=False)
    dl_te = PygDL(ds_te, batch_size=bs, shuffle=False)

    all_rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = SimpleGCN(
            in_channels=1, hidden=SIMPLE_GCN_HIDDEN,
            n_classes=n_classes, dropout=SIMPLE_GCN_DROPOUT,
        ).to(device)

        opt     = torch.optim.AdamW(model.parameters(), lr=SIMPLE_GCN_LR, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=cw_t)
        best_val, best_state, no_improve = -np.inf, None, 0

        print(f"\n[SimpleGCN] seed={seed}")
        for epoch in range(1, SIMPLE_GCN_EPOCHS + 1):
            model.train()
            for batch in dl_tr:
                batch = batch.to(device)
                opt.zero_grad()
                logits = model(batch.x, batch.edge_index, batch.batch)
                loss   = loss_fn(logits, batch.y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in dl_va:
                    batch = batch.to(device)
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    all_preds.append(logits.argmax(1).cpu().numpy())
                    all_labels.append(batch.y.cpu().numpy())
            preds_va = np.concatenate(all_preds)
            labs_va  = np.concatenate(all_labels)
            val_f1   = float(f1_score(labs_va, preds_va, average="macro", zero_division=0))

            if val_f1 > best_val:
                best_val   = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 20 == 0:
                print(f"  epoch {epoch:3d} | val_F1={val_f1:.4f} (best={best_val:.4f})")
            if no_improve >= SIMPLE_GCN_PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        all_preds, all_labels, all_probas = [], [], []
        with torch.no_grad():
            for batch in dl_te:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                proba  = torch.softmax(logits, dim=1).cpu().numpy()
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())
                all_probas.append(proba)
        preds_te  = np.concatenate(all_preds)
        labs_te   = np.concatenate(all_labels)
        probas_te = np.concatenate(all_probas)

        m_te = _metrics_from_pred_and_scores(labs_te, preds_te, probas_te, label_map)
        print(f"  ► test_F1={m_te['macro_f1']:.4f}  "
              f"test_acc={m_te['acc']:.4f}  "
              f"test_auc={m_te['auc_ovr_macro']:.4f}")

        all_rows.append({
            "family": "simple_gcn_kipf",
            "seed": seed,
            "val_macro_f1": float(best_val),
            **{f"test_{k}": float(v) for k, v in m_te.items()},
        })
        del model
        torch.cuda.empty_cache()

    df_raw = pd.DataFrame(all_rows)
    metric_cols = ["val_macro_f1", "test_macro_f1", "test_acc", "test_auc_ovr_macro"]
    agg = df_raw.groupby("family")[metric_cols].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    print("\n=== SimpleGCN summary ===")
    print(agg.to_string(index=False))

    if out_csv:
        df_raw.to_csv(out_csv.replace(".csv", "_raw.csv"), index=False)
        agg.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")

    return agg


# ════════════════════════════════════════════════════════════════════════════
#  6) GraphSAGE BASELINE
# ════════════════════════════════════════════════════════════════════════════

GRAPHSAGE_SEEDS    = [1234, 42, 369]
GRAPHSAGE_EPOCHS   = 120
GRAPHSAGE_LR       = 3e-3
GRAPHSAGE_HIDDEN   = 128
GRAPHSAGE_DROPOUT  = 0.30
GRAPHSAGE_PATIENCE = 20


class SimpleGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden, n_classes, dropout=0.30):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for SimpleGraphSAGE")
        self.conv1 = SAGEConv(in_channels, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, hidden, aggr="mean")
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


@torch.no_grad()
def _predict_graphsage(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        all_logits.append(logits.detach().cpu())
        all_y.append(batch.y.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0).numpy().astype(np.int64)
    proba = logits.softmax(dim=1).numpy()
    y_pred = proba.argmax(axis=1)
    return y_true, y_pred, proba


def run_graphsage_bioinfo(*, seeds=None, out_csv="./bioinfo_graphsage.csv"):
    if not HAS_PYG:
        print("torch_geometric no disponible. pip install torch_geometric")
        return pd.DataFrame()

    _require(["Xs_gene", "train_idx", "val_idx", "test_idx",
              "y", "label_map", "edge_index", "n_classes", "DEVICE"])

    if seeds is None:
        seeds = GRAPHSAGE_SEEDS

    X_node    = np.asarray(_rt("Xs_gene"), dtype=np.float32)
    y_arr     = np.asarray(_rt("y"),       dtype=np.int64)
    ei_t      = torch.tensor(_rt("edge_index"), dtype=torch.long)
    tr        = np.asarray(_rt("train_idx"), dtype=np.int64)
    va        = np.asarray(_rt("val_idx"),   dtype=np.int64)
    te        = np.asarray(_rt("test_idx"),  dtype=np.int64)
    label_map = _rt("label_map")
    n_classes = int(_rt("n_classes"))
    device    = _rt("DEVICE")

    ds_tr = _build_pyg_dataset(X_node, ei_t, y_arr, tr)
    ds_va = _build_pyg_dataset(X_node, ei_t, y_arr, va)
    ds_te = _build_pyg_dataset(X_node, ei_t, y_arr, te)

    rows = []
    for seed in seeds:
        print(f"\n[GraphSAGE] seed={seed}")
        set_all_seeds(seed)

        dl_tr = PygDL(ds_tr, batch_size=16, shuffle=True)
        dl_va = PygDL(ds_va, batch_size=32, shuffle=False)
        dl_te = PygDL(ds_te, batch_size=32, shuffle=False)

        counts = np.bincount(y_arr[tr], minlength=n_classes).astype(np.float32)
        cw = counts.sum() / np.maximum(counts, 1.0)
        cw /= max(cw.mean(), 1e-8)
        cw_t = torch.tensor(cw, dtype=torch.float32, device=device)

        model = SimpleGraphSAGE(
            in_channels=1, hidden=GRAPHSAGE_HIDDEN,
            n_classes=n_classes, dropout=GRAPHSAGE_DROPOUT,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=GRAPHSAGE_LR, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=cw_t)

        best_state, best_val_f1, best_epoch, patience = None, -np.inf, -1, 0

        for epoch in range(1, GRAPHSAGE_EPOCHS + 1):
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
                n_seen += int(batch.y.shape[0])

            y_va, pred_va, proba_va = _predict_graphsage(model, dl_va, device)
            m_va = _metrics_from_pred_and_scores(y_va, pred_va, proba_va, label_map)

            if m_va["macro_f1"] > best_val_f1 + 1e-4:
                best_val_f1 = m_va["macro_f1"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if epoch == 1 or epoch % 10 == 0:
                print(f"  epoch={epoch:03d} | loss={running/max(n_seen,1):.4f} | val_F1={m_va['macro_f1']:.4f}")

            if patience >= GRAPHSAGE_PATIENCE:
                print(f"  early stop @ epoch={epoch} | best_epoch={best_epoch} | best_val_F1={best_val_f1:.4f}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        y_va, pred_va, proba_va = _predict_graphsage(model, dl_va, device)
        y_te, pred_te, proba_te = _predict_graphsage(model, dl_te, device)
        m_va = _metrics_from_pred_and_scores(y_va, pred_va, proba_va, label_map)
        m_te = _metrics_from_pred_and_scores(y_te, pred_te, proba_te, label_map)

        rows.append({
            "family": "graphsage_fixed_prior",
            "seed": int(seed),
            "val_macro_f1": float(m_va["macro_f1"]),
            "val_acc": float(m_va["acc"]),
            "val_auc_ovr_macro": float(m_va["auc_ovr_macro"]),
            "test_macro_f1": float(m_te["macro_f1"]),
            "test_acc": float(m_te["acc"]),
            "test_auc_ovr_macro": float(m_te["auc_ovr_macro"]),
            **{f"test_{k}": float(v) for k, v in m_te.items() if str(k).startswith("f1_")},
        })

        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

    df_raw = pd.DataFrame(rows)
    metric_cols = ["val_macro_f1", "val_acc", "val_auc_ovr_macro",
                   "test_macro_f1", "test_acc", "test_auc_ovr_macro"]
    agg = df_raw.groupby("family")[metric_cols].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    print("\n=== GraphSAGE summary (mean ± std across seeds) ===")
    print(agg.to_string(index=False))

    if out_csv:
        df_raw.to_csv(out_csv.replace(".csv", "_raw.csv"), index=False)
        agg.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}  (raw: {out_csv.replace('.csv','_raw.csv')})")

    return agg


# ════════════════════════════════════════════════════════════════════════════
#  7) 291-GENE SIGNATURE BASELINES
# ════════════════════════════════════════════════════════════════════════════

SIGNATURE_291_MODELS = ("xgboost", "elasticnet_logreg", "mlp_graph_free")


def _subset_by_gene_list(X_all, genes_ref, gene_list):
    """Subset columns of X_all to those matching gene_list."""
    gene_upper = {str(g).upper(): i for i, g in enumerate(genes_ref)}
    keep_idx, genes_sig, missing = [], [], []
    for g in gene_list:
        gu = str(g).upper()
        if gu in gene_upper:
            keep_idx.append(gene_upper[gu])
            genes_sig.append(gu)
        else:
            missing.append(g)
    X_sig = X_all[:, keep_idx] if keep_idx else X_all[:, :0]
    return X_sig, keep_idx, genes_sig, missing


# NOTE: GENES_291_UP should be registered via register_runtime_benchmarks
# or defined here if you have the canonical list.

def _build_xy_from_signature291():
    _require(["Xs_gene", "train_idx", "val_idx", "test_idx", "y",
              "label_map", "genes_kegg", "GENES_291_UP"])
    X_all = np.asarray(_rt("Xs_gene"), dtype=np.float32)
    y_all = np.asarray(_rt("y"),       dtype=np.int64)
    genes_ref = list(_rt("genes_kegg"))

    X_sig, keep_idx, genes_sig, missing = _subset_by_gene_list(
        X_all, genes_ref, _rt("GENES_291_UP")
    )

    tr = np.asarray(_rt("train_idx"), dtype=np.int64)
    va = np.asarray(_rt("val_idx"),   dtype=np.int64)
    te = np.asarray(_rt("test_idx"),  dtype=np.int64)

    print(f"[291-GENE] matched genes: {len(genes_sig)} | missing: {len(missing)}")
    if missing:
        print("  Missing (first 20):", missing[:20])

    return {
        "X_tr": X_sig[tr], "X_va": X_sig[va], "X_te": X_sig[te],
        "y_tr": y_all[tr], "y_va": y_all[va], "y_te": y_all[te],
        "label_map": _rt("label_map"),
        "genes_sig": genes_sig, "keep_idx": keep_idx, "missing": missing,
    }


def run_signature291_tabular_benchmarks_bioinfo(
    *,
    random_state: int = 1234,
    families=SIGNATURE_291_MODELS,
    out_csv: str = "./bioinfo_signature291_tabular.csv",
) -> pd.DataFrame:
    data = _build_xy_from_signature291()
    X_tr, X_va, X_te = data["X_tr"], data["X_va"], data["X_te"]
    y_tr, y_va, y_te = data["y_tr"], data["y_va"], data["y_te"]
    label_map = data["label_map"]

    full_grid = _tabular_model_grid(random_state=random_state)
    rows = []

    for family in families:
        if family not in full_grid:
            print(f"[WARN] Family no encontrada en _tabular_model_grid(): {family}")
            continue

        candidates = full_grid[family]
        best, best_val, best_idx = None, -np.inf, -1

        print(f"\n[291-GENE TABULAR] {family}: {len(candidates)} candidates")
        for i, model in enumerate(candidates):
            mdl = clone(model)
            mdl = _fit_tabular_model(mdl, X_tr, y_tr)

            y_pred_va, score_va = _predict_scores(mdl, X_va)
            m_va = _metrics_from_pred_and_scores(y_va, y_pred_va, score_va, label_map)

            if m_va["macro_f1"] > best_val:
                best_val, best, best_idx = m_va["macro_f1"], mdl, i

        y_pred_va, score_va = _predict_scores(best, X_va)
        y_pred_te, score_te = _predict_scores(best, X_te)

        m_va = _metrics_from_pred_and_scores(y_va, y_pred_va, score_va, label_map)
        m_te = _metrics_from_pred_and_scores(y_te, y_pred_te, score_te, label_map)

        rows.append({
            "family": f"{family}_291",
            "base_family": family,
            "feature_set": "signature_291",
            "n_features": int(X_tr.shape[1]),
            "selected_candidate_idx": int(best_idx),
            "val_macro_f1": m_va["macro_f1"],
            "val_acc": m_va["acc"],
            "val_auc_ovr_macro": m_va["auc_ovr_macro"],
            "test_macro_f1": m_te["macro_f1"],
            "test_acc": m_te["acc"],
            "test_auc_ovr_macro": m_te["auc_ovr_macro"],
            **{f"test_{k}": v for k, v in m_te.items() if str(k).startswith("f1_")},
        })

        print(f"  best candidate={best_idx} | "
              f"val_F1={m_va['macro_f1']:.3f} | "
              f"test_F1={m_te['macro_f1']:.3f} | "
              f"test_acc={m_te['acc']:.3f}")

    df = pd.DataFrame(rows).sort_values(["test_macro_f1", "test_acc"], ascending=False)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")
    return df


# ════════════════════════════════════════════════════════════════════════════
#  8) CONSOLIDATED SOTA COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════

def build_sota_summary_table(
    out_csv="./bioinfo_sota_summary.csv",
    *,
    df_tab=None,
    df_sota_tab=None,
    df_centroid=None,
    df_fix=None,
    df_gcn=None,
    proposed_metrics=None,
):
    """
    Merge all benchmark families into one manuscript-ready table.

    Pass each benchmark DataFrame explicitly instead of relying on globals.

    Parameters
    ----------
    proposed_metrics : dict, optional
        Keys: "Val macro-F1", "Test macro-F1", "Test accuracy", "Test OvR AUC"
    """
    rows = []

    def _row(family, df, val_col="val_macro_f1",
             test_f1_col="test_macro_f1",
             test_acc_col="test_acc",
             test_auc_col="test_auc_ovr_macro",
             family_label=None):
        if df is None or len(df) == 0:
            return
        sub = df[df["family"] == family] if "family" in df.columns else df
        if len(sub) == 0:
            sub = df
        row_dict = sub.iloc[0].to_dict()
        val_f1 = row_dict.get(val_col, row_dict.get(val_col + "_mean", float("nan")))
        te_f1  = row_dict.get(test_f1_col, row_dict.get(test_f1_col + "_mean", float("nan")))
        te_acc = row_dict.get(test_acc_col, row_dict.get(test_acc_col + "_mean", float("nan")))
        te_auc = row_dict.get(test_auc_col, row_dict.get(test_auc_col + "_mean", float("nan")))
        rows.append({
            "Family":        family_label or family,
            "Model":         family,
            "Val macro-F1":  round(float(val_f1), 4) if val_f1 == val_f1 else float("nan"),
            "Test macro-F1": round(float(te_f1),  4) if te_f1  == te_f1  else float("nan"),
            "Test accuracy": round(float(te_acc), 4) if te_acc == te_acc  else float("nan"),
            "Test OvR AUC":  round(float(te_auc), 4) if te_auc == te_auc  else float("nan"),
        })

    # Proposed model
    if proposed_metrics is None:
        proposed_metrics = {
            "Val macro-F1": 0.922, "Test macro-F1": 0.924,
            "Test accuracy": 0.948, "Test OvR AUC": 0.991,
        }
    rows.append({
        "Family": "Proposed",
        "Model": "FULL (graph-learning GNN)",
        **proposed_metrics,
    })

    if df_tab is not None:
        for fam in ["xgboost", "random_forest", "extratrees",
                    "elasticnet_logreg", "linear_svm_calibrated", "mlp_graph_free"]:
            _row(fam, df_tab, family_label="Tabular baseline")

    if df_sota_tab is not None:
        for fam in ["lightgbm", "catboost"]:
            _row(fam, df_sota_tab, family_label="SOTA tabular")

    if df_centroid is not None:
        for fam in ["pam50_centroid_50", "pam50_centroid_full"]:
            _row(fam, df_centroid, family_label="Clinical centroid (PAM50-style)")

    if df_fix is not None:
        for fam in ["fixed_prior_hybrid", "fixed_prior_gnn", "fixed_prior_filtered"]:
            _row(fam, df_fix,
                 val_col="val_macro_f1_mean",
                 test_f1_col="test_macro_f1_mean",
                 test_acc_col="test_acc_mean",
                 test_auc_col="test_auc_ovr_macro_mean",
                 family_label="Fixed-prior GNN control")

    if df_gcn is not None:
        _row("simple_gcn_kipf", df_gcn,
             val_col="val_macro_f1_mean",
             test_f1_col="test_macro_f1_mean",
             test_acc_col="test_acc_mean",
             test_auc_col="test_auc_ovr_macro_mean",
             family_label="Simple GCN (Kipf 2017)")

    df_summary = (
        pd.DataFrame(rows)
        .sort_values("Test macro-F1", ascending=False)
        .reset_index(drop=True)
    )

    if out_csv:
        df_summary.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")

    return df_summary


def build_requested_benchmarks_summary(
    out_csv="./bioinfo_requested_extra_benchmarks_summary.csv",
    *,
    df_sig291=None,
    df_rewired=None,
    df_graphsage=None,
):
    rows = []

    def _push(df, family, label, val_col="val_macro_f1", test_f1_col="test_macro_f1",
              test_acc_col="test_acc", test_auc_col="test_auc_ovr_macro"):
        if df is None or len(df) == 0:
            return
        sub = df[df["family"] == family] if "family" in df.columns else df
        if len(sub) == 0:
            return
        row = sub.iloc[0].to_dict()
        rows.append({
            "Benchmark": label,
            "Val macro-F1": row.get(val_col, row.get(val_col + "_mean", np.nan)),
            "Test macro-F1": row.get(test_f1_col, row.get(test_f1_col + "_mean", np.nan)),
            "Test accuracy": row.get(test_acc_col, row.get(test_acc_col + "_mean", np.nan)),
            "Test OvR AUC": row.get(test_auc_col, row.get(test_auc_col + "_mean", np.nan)),
        })

    if df_sig291 is not None:
        _push(df_sig291, "xgboost_291", "XGBoost (291 genes)")
        _push(df_sig291, "elasticnet_logreg_291", "ElasticNet (291 genes)")
        _push(df_sig291, "mlp_graph_free_291", "MLP (291 genes)")

    if df_rewired is not None:
        _push(df_rewired, "rewired_prior_hybrid", "Rewired prior hybrid")
        _push(df_rewired, "rewired_prior_gnn", "Rewired prior GNN")

    if df_graphsage is not None:
        _push(df_graphsage, "graphsage_fixed_prior", "GraphSAGE fixed prior")

    out = pd.DataFrame(rows).sort_values("Test macro-F1", ascending=False)
    if out_csv:
        out.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")
    return out
