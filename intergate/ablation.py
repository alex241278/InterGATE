"""
Ablation framework: AblationConfig, model building helpers,
artifact save/load, run_single_seed, run_ablation.

Usage from notebook::

    from intergate.ablation import register_runtime, AblationConfig, ...

    # After loading data, register runtime objects:
    register_runtime(
        Xs_gene=Xs_gene, genes_kegg=genes_kegg,
        X_h=X_h, y=y, n_classes=n_classes, label_map=label_map,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        edge_index_t=edge_index_t, edge_weight_t=edge_weight_t,
        edge_type_t=edge_type_t, reg_groups_t=reg_groups_t,
    )
"""

import os
import gc
import json
import time
import hashlib
import shutil
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Any, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import CFG
from .utils import set_all_seeds, cleanup_memory


# ── Runtime registry ──────────────────────────────
# Notebook-scope objects (data matrices, splits, etc.) that cannot
# live in CFG.  Call register_runtime(**kw) once from the notebook.

_RT: Dict[str, Any] = {}


def register_runtime(**kwargs):
    """Register notebook-scope runtime objects so ablation functions can use them.

    Typical call::

        register_runtime(
            Xs_gene=Xs_gene, genes_kegg=genes_kegg,
            X_h=X_h, y=y, n_classes=n_classes, label_map=label_map,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            edge_index_t=edge_index_t, edge_weight_t=edge_weight_t,
            edge_type_t=edge_type_t, reg_groups_t=reg_groups_t,
        )
    """
    _RT.update(kwargs)


def _rt(name, default=None):
    """Get a runtime-registered variable (with optional default)."""
    if name in _RT:
        return _RT[name]
    if default is not None:
        return default
    raise KeyError(
        f"Runtime variable '{name}' not registered. "
        f"Call register_runtime({name}=...) from the notebook first."
    )


# ── Lazy imports (avoid circular) ─────────────────

def _get_model_classes():
    from .model import ImprovedSharedGraphGNN, HybridGNNTabular, build_pruned_model_from
    return ImprovedSharedGraphGNN, HybridGNNTabular, build_pruned_model_from


def _get_data_class():
    from .data import ExpressionDataset
    return ExpressionDataset


def _get_training_fns():
    from .training import (
        train_graph_learning, pretrain_masked_gene_model,
        finetune_pruned, predict_proba_xh_mode,
    )
    return train_graph_learning, pretrain_masked_gene_model, finetune_pruned, predict_proba_xh_mode


def _get_pruning_fns():
    from .pruning import export_pruned_graph
    return (export_pruned_graph,)


def _get_stability_fns():
    from .stability import stability_edge_sets, edge_set_from_edge_index, jaccard
    return stability_edge_sets, edge_set_from_edge_index, jaccard


def _get_metrics_fn():
    from .stability import compute_metrics
    return compute_metrics


# ── AblationConfig ────────────────────────────────


@dataclass
class AblationConfig:
    name: str
    edge_type_gating: bool
    sample_cond_gating: bool
    sample_cond_mode: str        # "per_type" o "global"
    use_signed_prior: bool       # si False: colapsa OP- a OP+ y usa |w|
    signed_channels_mode: str    # "type_only" o "dual_backbone"
    connectivity_penalty: bool
    do_pretrain: bool
    do_stability: bool

    # constantes (normalmente no se ablaten)
    edge_type_mode: str = "huri_op_signed"
    pool_level: str = "gene"


# ── Default configs (use CFG values) ──────────────

CFG_FULL = AblationConfig(
    name="FULL",
    edge_type_gating=True,
    sample_cond_gating=True,
    sample_cond_mode="per_type",
    use_signed_prior=True,
    signed_channels_mode="type_only",
    connectivity_penalty=True,
    do_pretrain=CFG.DO_PRETRAIN,
    do_stability=CFG.DO_STABILITY_SELECTION,
)

ABLA_CONFIGS: List[AblationConfig] = [CFG_FULL]

# Seeds / epochs (from CFG, overridable at module level)
ABLA_SEEDS = list(range(CFG.STAB_RUNS))
MAIN_EPOCHS = CFG.EPOCHS1
FT_EPOCHS_2A = CFG.FT_EPOCHS_A
FT_EPOCHS_2B = CFG.FT_EPOCHS_B

FAST_DEV = False

NUM_WORKERS = 0   # default; override from notebook if desired


def apply_fast_dev():
    """Reduce epochs/seeds for quick debugging."""
    global MAIN_EPOCHS, FT_EPOCHS_2A, FT_EPOCHS_2B, ABLA_SEEDS, FAST_DEV
    FAST_DEV = True
    MAIN_EPOCHS = 1
    FT_EPOCHS_2A = 5
    FT_EPOCHS_2B = 20
    ABLA_SEEDS = ABLA_SEEDS[:1]
    print("FAST_DEV=True -> epochs/seeds reducidos")


# ── Prior / model building helpers ────────────────


def get_full_gene_matrix_and_genes():
    """Return (X_full, genes_full) from runtime registry."""
    X_full = _RT.get("_Xs_gene_full", _RT.get("Xs_gene", None))
    G_full = _RT.get("_genes_full", None)
    if G_full is None:
        G_full = list(_rt("genes_kegg"))
    if X_full is None:
        raise KeyError("Xs_gene not registered. Call register_runtime(Xs_gene=...).")
    return X_full, G_full


def make_prior_for_cfg(cfg: AblationConfig):
    """
    Devuelve (edge_index, edge_weight_base, edge_type) en TORCH.

    - Si cfg.use_signed_prior=False: toma abs(w) y colapsa tipo 2->1.
    - Si cfg.edge_type_gating=False: fuerza edge_type=0.
    """
    device = CFG.DEVICE

    if "edge_index_t" in _RT:
        ei = _RT["edge_index_t"]
        ew = _RT["edge_weight_t"]
        et = _RT.get("edge_type_t", None)
    else:
        ei = torch.as_tensor(_rt("edge_index"), dtype=torch.long, device=device)
        ew = torch.as_tensor(_rt("edge_weight"), dtype=torch.float32, device=device)
        et_np = _RT.get("edge_type", None)
        et = torch.as_tensor(et_np, dtype=torch.long, device=device) if et_np is not None else None

    ei = ei.clone()
    ew = ew.clone().float()
    et = None if et is None else et.clone().long()

    if not cfg.use_signed_prior:
        ew = ew.abs()
        if et is not None:
            et = torch.where(et == 2, torch.ones_like(et), et)

    if not cfg.edge_type_gating:
        if et is None:
            et = torch.zeros(ei.size(1), dtype=torch.long, device=device)
        else:
            et = torch.zeros_like(et)

    if et is None:
        et = torch.zeros(ei.size(1), dtype=torch.long, device=device)

    return ei, ew, et


def build_model_from_cfg(cfg: AblationConfig, class_weights_t: Optional[torch.Tensor]):
    ImprovedSharedGraphGNN, _, _ = _get_model_classes()

    ei, ew, et = make_prior_for_cfg(cfg)
    X_h = _rt("X_h")
    n_classes = _rt("n_classes")
    reg_groups_t = _RT.get("reg_groups_t", None)

    model = ImprovedSharedGraphGNN(
        num_nodes=get_full_gene_matrix_and_genes()[0].shape[1],
        hidden=CFG.HIDDEN,
        n_classes=n_classes,
        graph_feat_dim=X_h.shape[1],
        num_layers=CFG.NUM_LAYERS,
        dropout=CFG.DROPOUT,
        num_heads=CFG.POOL_HEADS,
        gat_heads=CFG.NUM_HEADS,
        edge_index=ei,
        edge_weight=ew,
        edge_type=et,

        # toggles ablation
        edge_type_gating=cfg.edge_type_gating,
        edge_type_mode=cfg.edge_type_mode,
        signed_channels=CFG.SIGNED_CHANNELS if hasattr(CFG, "SIGNED_CHANNELS") else False,
        signed_channels_mode=cfg.signed_channels_mode,
        sample_cond_gating=cfg.sample_cond_gating,
        sample_cond_mode=cfg.sample_cond_mode,

        # conectividad
        add_connectivity_penalty=cfg.connectivity_penalty,
        connectivity_min_deg=CFG.CONNECTIVITY_MIN_DEG,
        connectivity_use_abs=CFG.CONNECTIVITY_USE_ABS,

        # misc
        xgraph_dropout=CFG.XGRAPH_DROPOUT,
        keep_self_loops=CFG.KEEP_SELF_LOOPS,
        block_use_self=CFG.BLOCK_USE_SELF,
        block_residual=CFG.BLOCK_RESIDUAL,
        use_regulator_pool=(cfg.pool_level == "regulator"),
        regulator_groups=(reg_groups_t if cfg.pool_level == "regulator" else None),
    ).to(CFG.DEVICE)
    return model


def make_loaders(X_gene_full: np.ndarray):
    ExpressionDataset = _get_data_class()

    X_h = _rt("X_h")
    y = _rt("y")
    train_idx = _rt("train_idx")
    val_idx = _rt("val_idx")
    test_idx = _rt("test_idx")

    tr_ds = ExpressionDataset(X_gene_full, X_h, y, train_idx)
    va_ds = ExpressionDataset(X_gene_full, X_h, y, val_idx)
    te_ds = ExpressionDataset(X_gene_full, X_h, y, test_idx)

    pin = bool(torch.cuda.is_available())
    nw = NUM_WORKERS
    dl_tr = torch.utils.data.DataLoader(tr_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,  num_workers=nw, pin_memory=pin)
    dl_va = torch.utils.data.DataLoader(va_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=pin)
    dl_te = torch.utils.data.DataLoader(te_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=pin)
    return dl_tr, dl_va, dl_te


# ── Artifacts save/load ──────────────────────────


ARTIFACTS_ROOT = os.path.abspath(CFG.ARTIFACTS_ROOT)

# --- Qué guardar ---
SAVE_ALL_CONFIGS = True
SAVE_CFGS: Set[str] = set()
SAVE_BEST_ONLY_CFGS: Set[str] = {"no_conn"}
BEST_ONLY_METRIC = "macro_f1"
_BEST_REGISTRY: Dict[str, dict] = {}

SAVE_STABILITY_DETAILS = True
SAVE_STABILITY_CFGS: Set[str] = {"no_conn"}
SAVE_STAB_RUN_GRAPHS = False
STAB_RETURN_DETAILS = False

RUN_FINAL_AFTER_ABLATION = True
FINAL_SELECT_METRIC = "macro_f1"


def _mkdir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def _reset_dir(p: str):
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    os.makedirs(p, exist_ok=True)
    return p


def _to_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def cfg_to_dict(cfg: "AblationConfig") -> dict:
    """Serializa cfg + hiperparámetros globales y splits para reproducibilidad."""
    d = asdict(cfg)

    d["globals"] = {
        "HIDDEN": int(CFG.HIDDEN),
        "NUM_LAYERS": int(CFG.NUM_LAYERS),
        "DROPOUT": float(CFG.DROPOUT),
        "NUM_HEADS": int(CFG.NUM_HEADS),
        "POOL_HEADS": int(CFG.POOL_HEADS),
        "BATCH_SIZE": int(CFG.BATCH_SIZE),
        "LR": float(CFG.LR),
        "WEIGHT_DECAY": float(CFG.WEIGHT_DECAY),
        "EDGE_L1_PER_EDGE": float(CFG.EDGE_L1_PER_EDGE),
        "AUX_LAMBDA": float(CFG.AUX_LAMBDA),
        "KEEP_MIN": float(CFG.KEEP_MIN),
        "MAIN_EPOCHS": int(MAIN_EPOCHS),
        "FT_EPOCHS_2A": int(FT_EPOCHS_2A),
        "FT_EPOCHS_2B": int(FT_EPOCHS_2B),
        "STAB_KEEP_FINAL": float(CFG.STAB_KEEP_FINAL),
        "STAB_FREQ_THR": float(CFG.STAB_FREQ_THR),
        "STAB_EPOCHS": int(CFG.STAB_EPOCHS),
        "ABLA_SEEDS": _to_list(ABLA_SEEDS),
        "USE_MIXUP": bool(CFG.USE_MIXUP),
        "MIXUP_ALPHA": float(CFG.MIXUP_ALPHA),
        "MIXUP_P": float(CFG.MIXUP_P),
        "ACCUM_STEPS": int(CFG.ACCUM_STEPS),
        "KEEP_SELF_LOOPS": bool(CFG.KEEP_SELF_LOOPS),
        "BLOCK_USE_SELF": bool(CFG.BLOCK_USE_SELF),
        "BLOCK_RESIDUAL": bool(CFG.BLOCK_RESIDUAL),
        "USE_HYBRID": bool(CFG.USE_HYBRID_MODEL),
        "HYBRID_TAB_LAYERS": int(CFG.HYBRID_TAB_LAYERS),
        "HYBRID_TAB_DROPOUT": float(CFG.HYBRID_TAB_DROPOUT),
        "HYBRID_FUSION_DROPOUT": float(CFG.HYBRID_FUSION_DROPOUT),
        "HYBRID_BLEND_LOGITS": bool(CFG.HYBRID_BLEND_LOGITS),
        "HYBRID_BLEND_INIT": float(CFG.HYBRID_BLEND_INIT),
        "XGRAPH_DROPOUT": float(CFG.XGRAPH_DROPOUT),
        "DEVICE": str(CFG.DEVICE),
    }

    # Splits
    train_idx = _RT.get("train_idx", None)
    val_idx = _RT.get("val_idx", None)
    test_idx = _RT.get("test_idx", None)
    d["splits"] = {
        "train_idx": _to_list(train_idx),
        "val_idx": _to_list(val_idx),
        "test_idx": _to_list(test_idx),
    }

    # Firma del dataset
    try:
        X_full, genes_full = get_full_gene_matrix_and_genes()
        sig = hashlib.md5(("|".join(map(str, genes_full))).encode("utf-8")).hexdigest()
        d["dataset_signature"] = {
            "n_samples": int(X_full.shape[0]),
            "n_genes_full": int(X_full.shape[1]),
            "genes_md5": sig,
        }
    except Exception:
        d["dataset_signature"] = None

    try:
        d["n_classes"] = int(_rt("n_classes"))
    except Exception:
        pass

    return d


def save_bundle(
    out_dir: str,
    *,
    cfg: "AblationConfig",
    seed: int,
    genes_full: list,
    nodes_used_full: "np.ndarray",
    genes_comp: list,
    edge_index_full: torch.Tensor,
    edge_weight_full,
    edge_type_full,
    edge_index_comp: torch.Tensor,
    edge_weight_comp,
    edge_type_comp,
    reg_groups_comp,
    model_pruned: torch.nn.Module,
    metrics: dict,
    stab: dict = None,
    phase1_best_state: dict = None,
):
    out_dir = _mkdir(out_dir)

    meta = cfg_to_dict(cfg)
    meta.update({
        "cfg_name": cfg.name,
        "seed": int(seed),
        "metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()},
        "n_edges_final_full": int(edge_index_full.shape[1]),
        "n_nodes_compact": int(len(genes_comp)),
    })
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    graph = {
        "genes_full": list(genes_full),
        "nodes_used_full": nodes_used_full,
        "genes_comp": list(genes_comp),
        "edge_index_full": edge_index_full.detach().cpu(),
        "edge_weight_full": (edge_weight_full.detach().cpu() if isinstance(edge_weight_full, torch.Tensor) else None),
        "edge_type_full": (edge_type_full.detach().cpu() if isinstance(edge_type_full, torch.Tensor) else None),
        "edge_index_comp": edge_index_comp.detach().cpu(),
        "edge_weight_comp": (edge_weight_comp.detach().cpu() if isinstance(edge_weight_comp, torch.Tensor) else None),
        "edge_type_comp": (edge_type_comp.detach().cpu() if isinstance(edge_type_comp, torch.Tensor) else None),
        "reg_groups_comp": (reg_groups_comp.detach().cpu() if isinstance(reg_groups_comp, torch.Tensor) else None),
    }
    torch.save(graph, os.path.join(out_dir, "graph_bundle.pt"))

    sd = {k: v.detach().cpu() for k, v in model_pruned.state_dict().items()}
    core_model = model_pruned.gnn if bool(getattr(model_pruned, "is_hybrid", False)) and hasattr(model_pruned, "gnn") else model_pruned

    model_flags = {
        "edge_type_gating": bool(getattr(core_model, "edge_type_gating", True)),
        "edge_type_mode": str(getattr(core_model, "edge_type_mode", getattr(cfg, "edge_type_mode", "huri_op_signed"))),
        "sample_cond_gating": bool(getattr(core_model, "sample_cond_gating", False)),
        "sample_cond_mode": str(getattr(core_model, "sample_cond_mode", "per_type")),
        "signed_channels_mode": str(getattr(core_model, "signed_channels_mode", "type_only")),
        "use_regulator_pool": bool(getattr(core_model, "use_regulator_pool", False)),
        "is_hybrid": bool(getattr(model_pruned, "is_hybrid", False)),
        "tab_layers": int(getattr(model_pruned, "tab_layers", 2)),
        "tab_dropout": float(getattr(model_pruned, "tab_dropout", 0.20)),
        "fusion_dropout": float(getattr(model_pruned, "fusion_dropout", 0.20)),
        "blend_logits": bool(getattr(model_pruned, "blend_logits", False)),
        "blend_init": float(getattr(model_pruned, "blend_init", 0.70)),
    }

    torch.save({"state_dict": sd, "model_flags": model_flags}, os.path.join(out_dir, "model_pruned.pt"))

    if phase1_best_state is not None:
        sd1 = {k: v.detach().cpu() for k, v in phase1_best_state.items()}
        torch.save(sd1, os.path.join(out_dir, "phase1_best_state.pt"))

    if (stab is not None) and SAVE_STABILITY_DETAILS:
        try:
            if "runs_df" in stab and isinstance(stab["runs_df"], pd.DataFrame):
                stab["runs_df"].to_csv(os.path.join(out_dir, "stability_runs.csv"), index=False)
        except Exception:
            pass
        try:
            if "jaccard_df" in stab and isinstance(stab["jaccard_df"], pd.DataFrame):
                stab["jaccard_df"].to_csv(os.path.join(out_dir, "stability_jaccard.csv"))
        except Exception:
            pass


def load_bundle(bundle_dir: str) -> dict:
    meta = json.load(open(os.path.join(bundle_dir, "meta.json"), "r", encoding="utf-8"))
    graph = torch.load(os.path.join(bundle_dir, "graph_bundle.pt"), map_location="cpu", weights_only=False)
    model = torch.load(os.path.join(bundle_dir, "model_pruned.pt"), map_location="cpu", weights_only=False)
    return {"meta": meta, "graph": graph, "model": model}


def build_pruned_model_from_bundle(bundle: dict, device=None):
    ImprovedSharedGraphGNN, HybridGNNTabular, _ = _get_model_classes()

    if device is None:
        device = CFG.DEVICE

    meta = bundle["meta"]
    g = bundle["graph"]
    m = bundle["model"]
    flags = m.get("model_flags", {}) if isinstance(m, dict) else {}

    num_nodes = int(len(g["genes_comp"]))
    hidden = int(meta["globals"]["HIDDEN"])
    n_classes = int(meta.get("n_classes", _rt("n_classes")))
    X_h = _rt("X_h")

    sd = m["state_dict"]
    is_hybrid = bool(flags.get("is_hybrid", False)) or any(str(k).startswith("gnn.") for k in sd.keys())

    base = ImprovedSharedGraphGNN(
        num_nodes=num_nodes,
        hidden=hidden,
        n_classes=n_classes,
        graph_feat_dim=int(X_h.shape[1]),
        num_layers=int(meta["globals"]["NUM_LAYERS"]),
        dropout=float(meta["globals"]["DROPOUT"]),
        num_heads=int(meta["globals"].get("POOL_HEADS", CFG.POOL_HEADS)),
        gat_heads=int(meta["globals"].get("NUM_HEADS", CFG.NUM_HEADS)),
        edge_index=g["edge_index_comp"].to(device),
        edge_weight=(g["edge_weight_comp"].to(device) if g["edge_weight_comp"] is not None else None),
        edge_type=(g["edge_type_comp"].to(device) if g["edge_type_comp"] is not None else None),
        keep_self_loops=False,
        edge_type_gating=bool(flags.get("edge_type_gating", True)),
        edge_type_mode=str(flags.get("edge_type_mode", getattr(CFG_FULL, "edge_type_mode", "huri_op_signed"))),
        sample_cond_gating=bool(flags.get("sample_cond_gating", False)),
        sample_cond_mode=str(flags.get("sample_cond_mode", "per_type")),
        signed_channels=CFG.SIGNED_CHANNELS if hasattr(CFG, "SIGNED_CHANNELS") else False,
        signed_channels_mode=str(flags.get("signed_channels_mode", "type_only")),
        use_regulator_pool=bool(flags.get("use_regulator_pool", False)),
        regulator_groups=(g["reg_groups_comp"].to(device) if (flags.get("use_regulator_pool", False) and g.get("reg_groups_comp") is not None) else None),
        add_connectivity_penalty=False,
        xgraph_dropout=float(meta["globals"].get("XGRAPH_DROPOUT", CFG.XGRAPH_DROPOUT)),
        block_use_self=bool(meta["globals"].get("BLOCK_USE_SELF", CFG.BLOCK_USE_SELF)),
        block_residual=bool(meta["globals"].get("BLOCK_RESIDUAL", CFG.BLOCK_RESIDUAL)),
    ).to(device)

    if is_hybrid:
        model_p = HybridGNNTabular(
            gnn=base, tab_in_dim=num_nodes, hidden=hidden, n_classes=n_classes,
            tab_layers=int(flags.get("tab_layers", 2)),
            tab_dropout=float(flags.get("tab_dropout", 0.20)),
            fusion_dropout=float(flags.get("fusion_dropout", 0.20)),
            use_fusion_head=True,
            blend_logits=bool(flags.get("blend_logits", False)),
            blend_init=float(flags.get("blend_init", 0.70)),
        ).to(device)
        model_p.load_state_dict(sd, strict=False)
        if hasattr(model_p.gnn, "edge_logit"):
            model_p.gnn.edge_logit.requires_grad_(False)
            with torch.no_grad():
                model_p.gnn.edge_logit.data.fill_(20.0)
            model_p.gnn.set_hard_keep_ratio(None)
        model_p.eval()
        return model_p

    base.load_state_dict(sd, strict=False)
    if hasattr(base, "edge_logit"):
        base.edge_logit.requires_grad_(False)
        with torch.no_grad():
            base.edge_logit.data.fill_(20.0)
        base.set_hard_keep_ratio(None)
    base.eval()
    return base


def evaluate_bundle_on_test(bundle_dir: str, device=None) -> dict:
    """Carga bundle y evalúa directamente en TEST."""
    ExpressionDataset = _get_data_class()
    _, _, _, predict_proba_xh_mode = _get_training_fns()
    compute_metrics = _get_metrics_fn()

    if device is None:
        device = CFG.DEVICE

    bundle = load_bundle(bundle_dir)
    meta = bundle.get("meta", {})
    splits = meta.get("splits", {}) if isinstance(meta, dict) else {}
    g = bundle["graph"]

    nodes_used = np.asarray(g["nodes_used_full"], dtype=np.int64)
    X_full, genes_full = get_full_gene_matrix_and_genes()
    X_comp = X_full[:, nodes_used]
    X_h = _rt("X_h")
    y = _rt("y")

    te_idx = splits.get("test_idx", None)
    te_idx = np.asarray(te_idx, dtype=np.int64) if te_idx is not None else np.asarray(_rt("test_idx"), dtype=np.int64)

    te_ds = ExpressionDataset(X_comp[te_idx], X_h[te_idx], y[te_idx])
    dl_te = torch.utils.data.DataLoader(
        te_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    model_p = build_pruned_model_from_bundle(bundle, device=device)
    proba_te, y_te = predict_proba_xh_mode(model_p, None, dl_te, device, xh_mode="orig")
    return compute_metrics(y_te, proba_te)


# ── Run single seed ──────────────────────────────


STAB_CACHE: Dict[str, Any] = {}


def run_single_seed(
    cfg: AblationConfig,
    seed: int,
    *,
    save_root: Optional[str] = None,
    force_save: bool = False,
) -> Dict[str, Any]:
    _, _, build_pruned_model_from = _get_model_classes()
    ExpressionDataset = _get_data_class()
    train_graph_learning, pretrain_masked_gene_model, finetune_pruned, predict_proba_xh_mode = _get_training_fns()
    (export_pruned_graph,) = _get_pruning_fns()
    stability_edge_sets, edge_set_from_edge_index, jaccard = _get_stability_fns()
    compute_metrics = _get_metrics_fn()

    set_all_seeds(seed)

    X_full, genes_full = get_full_gene_matrix_and_genes()
    dl_tr, dl_va, dl_te = make_loaders(X_full)
    cleanup_memory('after_make_loaders')

    X_h = _rt("X_h")
    y = _rt("y")
    n_classes = _rt("n_classes")
    label_map = _rt("label_map")
    train_idx = _rt("train_idx")
    val_idx = _rt("val_idx")
    test_idx = _rt("test_idx")

    if save_root is None:
        save_root = ARTIFACTS_ROOT

    do_save_seed = bool(force_save or SAVE_ALL_CONFIGS or (cfg.name in SAVE_CFGS))
    is_best_only = bool((not force_save) and (not SAVE_ALL_CONFIGS) and (cfg.name in SAVE_BEST_ONLY_CFGS))

    run_dir = None
    if do_save_seed:
        run_dir = os.path.join(save_root, cfg.name, f"seed_{seed}")
        _mkdir(run_dir)

    # class_weights
    class_weights = _RT.get("class_weights", None)
    if class_weights is None:
        counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float32)
        w = counts.sum() / np.maximum(counts, 1.0)
        class_weights = (w / w.mean()).tolist()
        _RT["class_weights"] = class_weights

    if isinstance(class_weights, torch.Tensor):
        class_weights_t = class_weights.detach().to(device=CFG.DEVICE, dtype=torch.float32)
        cw_list = class_weights_t.detach().cpu().tolist()
    else:
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=CFG.DEVICE)
        cw_list = list(class_weights)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_t)

    # Modelo fase 1
    model = build_model_from_cfg(cfg, class_weights_t)

    if cfg.do_pretrain:
        model = pretrain_masked_gene_model(model, dl_tr, CFG.DEVICE)

    model, best_state, best_macro_f1 = train_graph_learning(
        model, adj=None, dl_tr=dl_tr, dl_va=dl_va,
        device=CFG.DEVICE, label_map=label_map,
        epochs=MAIN_EPOCHS, lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY,
        accum_steps=CFG.ACCUM_STEPS,
        class_weights=cw_list, class_weights_t=class_weights_t,
        edge_l1_per_edge=CFG.EDGE_L1_PER_EDGE,
        aux_lambda=CFG.AUX_LAMBDA, loss_fn=loss_fn,
        keep_min=CFG.KEEP_MIN,
    )
    model.load_state_dict(best_state)
    cleanup_memory('after_load_best_state')

    ei_p_full, ew_p_full, et_p_full, thr_logit, kept, E_total = export_pruned_graph(
        model, keep_ratio=CFG.KEEP_MIN, keep_self_loops=False
    )
    edge_single = edge_set_from_edge_index(ei_p_full)

    # ---- Stability selection ----
    stab = None
    ei_use, ew_use, et_use = ei_p_full, ew_p_full, et_p_full

    if cfg.do_stability:
        global STAB_CACHE
        if cfg.name in STAB_CACHE:
            stab = STAB_CACHE[cfg.name]
        else:
            stab_dir = None
            if (cfg.name in SAVE_STABILITY_CFGS) or (cfg.name in SAVE_BEST_ONLY_CFGS) or (cfg.name in SAVE_CFGS) or SAVE_ALL_CONFIGS:
                stab_dir = os.path.join(ARTIFACTS_ROOT, cfg.name, "_stability")
            stab = stability_edge_sets(
                cfg=cfg, seeds=ABLA_SEEDS, keep_ratio=CFG.STAB_KEEP_FINAL,
                epochs=CFG.STAB_EPOCHS, class_weights_t=class_weights_t,
                verbose=False, save_dir=stab_dir,
            )
            STAB_CACHE[cfg.name] = stab

        cons = stab["consensus_edges"]
        if len(cons) > 0:
            ei0, ew0, et0 = make_prior_for_cfg(cfg)
            ei0c = ei0.detach().cpu().numpy()
            ew0c = ew0.detach().cpu().numpy() if ew0 is not None else None
            et0c = et0.detach().cpu().numpy() if et0 is not None else None
            if ew0c is None:
                ew0c = np.ones(ei0c.shape[1], dtype=np.float32)
            if et0c is None:
                et0c = np.zeros(ei0c.shape[1], dtype=np.int64)

            prior_map = {(int(s), int(d)): (float(w), int(t))
                         for s, d, w, t in zip(ei0c[0], ei0c[1], ew0c, et0c)}
            sel = [(e[0], e[1], prior_map[e][0], prior_map[e][1]) for e in cons if e in prior_map]

            if len(sel) > 0:
                src = [s for s, _, _, _ in sel]
                dst = [d for _, d, _, _ in sel]
                ww  = [w for *_, w, _ in sel]
                tt  = [t for *_, t in sel]
                ei_use = torch.tensor([src, dst], dtype=torch.long, device=CFG.DEVICE)
                ew_use = torch.tensor(ww, dtype=torch.float32, device=CFG.DEVICE)
                et_use = torch.tensor(tt, dtype=torch.long, device=CFG.DEVICE)

    # ---- Compactación ----
    nodes_used = torch.unique(ei_use.reshape(-1)).detach().cpu().numpy().astype(np.int64)
    mask = np.zeros(X_full.shape[1], dtype=bool)
    mask[nodes_used] = True

    X_comp = X_full[:, mask]
    genes_comp = [g for g, m in zip(genes_full, mask) if m]

    old2new = -np.ones(X_full.shape[1], dtype=np.int64)
    old2new[nodes_used] = np.arange(len(nodes_used), dtype=np.int64)

    ei_cpu = ei_use.detach().cpu().numpy()
    ei_comp_t = torch.tensor(np.vstack([old2new[ei_cpu[0]], old2new[ei_cpu[1]]]), dtype=torch.long, device=CFG.DEVICE)

    ew_comp_t = ew_use if isinstance(ew_use, torch.Tensor) else (torch.tensor(ew_use, dtype=torch.float32, device=CFG.DEVICE) if ew_use is not None else None)
    et_comp_t = et_use if isinstance(et_use, torch.Tensor) else (torch.tensor(et_use, dtype=torch.long, device=CFG.DEVICE) if et_use is not None else None)
    if isinstance(ew_comp_t, torch.Tensor):
        ew_comp_t = ew_comp_t.to(device=CFG.DEVICE)
    if isinstance(et_comp_t, torch.Tensor):
        et_comp_t = et_comp_t.to(device=CFG.DEVICE)

    # Pooling por regulator
    reg_groups_comp_t = None
    if cfg.pool_level == "regulator":
        reg_groups_full = _RT.get("reg_groups_full", None)
        if reg_groups_full is not None:
            rg = reg_groups_full.detach().cpu().numpy() if isinstance(reg_groups_full, torch.Tensor) else np.asarray(reg_groups_full)
            reg_groups_comp_t = torch.tensor(rg[mask], dtype=torch.long, device=CFG.DEVICE)

    # Modelo fase 2
    model_p = build_pruned_model_from(
        model_gated=model,
        edge_index_p=ei_comp_t,
        edge_weight_p=ew_comp_t,
        edge_type_p=et_comp_t,
        num_nodes=len(genes_comp),
        hidden=CFG.HIDDEN,
        n_classes=n_classes,
        graph_feat_dim=X_h.shape[1],
        num_layers=CFG.NUM_LAYERS,
        dropout=CFG.DROPOUT,
        num_heads=CFG.POOL_HEADS,
        xgraph_dropout=CFG.XGRAPH_DROPOUT,
        block_use_self=CFG.BLOCK_USE_SELF,
        block_residual=CFG.BLOCK_RESIDUAL,
        pool_level=cfg.pool_level,
        reg_groups_t=reg_groups_comp_t if cfg.pool_level == "regulator" else None,
        reg_groups=reg_groups_comp_t if cfg.pool_level == "regulator" else None,
        reg_pool_min_targets=int(_RT.get("REG_POOL_MIN_TARGETS", 3)),
        device=CFG.DEVICE,
        use_hybrid=CFG.USE_HYBRID_MODEL,
        tab_layers=CFG.HYBRID_TAB_LAYERS,
        tab_dropout=CFG.HYBRID_TAB_DROPOUT,
        fusion_dropout=CFG.HYBRID_FUSION_DROPOUT,
        blend_logits=CFG.HYBRID_BLEND_LOGITS,
        blend_init=CFG.HYBRID_BLEND_INIT,
    )

    # Loaders compactos
    tr_ds2 = ExpressionDataset(X_comp, X_h, y, train_idx)
    va_ds2 = ExpressionDataset(X_comp, X_h, y, val_idx)
    te_ds2 = ExpressionDataset(X_comp, X_h, y, test_idx)
    nw = NUM_WORKERS
    dl_tr2 = torch.utils.data.DataLoader(tr_ds2, batch_size=CFG.BATCH_SIZE, shuffle=True,  num_workers=nw, pin_memory=True)
    dl_va2 = torch.utils.data.DataLoader(va_ds2, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=True)
    dl_te2 = torch.utils.data.DataLoader(te_ds2, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=True)

    phase2_lr_mult = float(_RT.get("PHASE2_LR_MULT", 0.20))
    phase2_best_metric = _RT.get("PHASE2_BEST_METRIC", "macro_f1")
    phase2_patience = int(_RT.get("PHASE2_PATIENCE", 20))
    phase2_min_delta = float(_RT.get("PHASE2_MIN_DELTA", 1e-4))
    phase2_lr_tab_mult = float(_RT.get("PHASE2_LR_TAB_MULT", 2.0))
    phase2_lr_fusion_mult = float(_RT.get("PHASE2_LR_FUSION_MULT", 2.0))
    phase2_best_path = os.path.join(run_dir, "phase2_best.pt") if run_dir is not None else None

    loss_fn_phase2 = torch.nn.CrossEntropyLoss(weight=class_weights_t)

    model_p = finetune_pruned(
        model_pruned=model_p, adj=None,
        dl_tr=dl_tr2, dl_va=dl_va2,
        device=CFG.DEVICE, label_map=label_map,
        loss_fn=loss_fn_phase2,
        lr=CFG.LR * phase2_lr_mult,
        weight_decay=CFG.WEIGHT_DECAY,
        epochs_A=FT_EPOCHS_2A, epochs_B=FT_EPOCHS_2B,
        accum_steps=CFG.ACCUM_STEPS,
        use_mixup_B=CFG.USE_MIXUP,
        mixup_alpha=CFG.MIXUP_ALPHA,
        mixup_prob=CFG.MIXUP_P,
        best_metric=phase2_best_metric,
        patience_B=phase2_patience,
        min_delta=phase2_min_delta,
        save_best_path=phase2_best_path,
        lr_tab_mult=phase2_lr_tab_mult,
        lr_fusion_mult=phase2_lr_fusion_mult,
    )

    # Test
    proba_te, y_te = predict_proba_xh_mode(model_p, None, dl_te2, CFG.DEVICE, xh_mode="orig")
    cleanup_memory('after_test_predict')
    met = compute_metrics(y_te, proba_te)

    out = {
        "cfg": cfg.name, "seed": seed,
        "val_macro_f1_phase1": float(best_macro_f1),
        "n_edges_final": int(ei_use.shape[1]),
        "n_edges_single": int(ei_p_full.shape[1]),
        "n_nodes_compact": int(X_comp.shape[1]),
        "use_signed_prior": cfg.use_signed_prior,
        "jaccard_single_vs_final": float(jaccard(edge_single, edge_set_from_edge_index(ei_use))) if cfg.do_stability else 1.0,
        **met,
    }

    if stab is not None:
        out["stab_mean_jaccard"] = float(stab.get("jaccard_mean", float("nan")))
        out["stab_std_jaccard"]  = float(stab.get("jaccard_std", float("nan")))
        out["stab_min_jaccard"]  = float(stab.get("jaccard_min", float("nan")))
        out["stab_max_jaccard"]  = float(stab.get("jaccard_max", float("nan")))
        out["stab_n_pairs"]      = int(stab.get("jaccard_n_pairs", 0))
        out["stab_consensus_size"] = int(stab.get("consensus_size", 0))
        tcs = stab.get("to_consensus_stats", None)
        out["stab_to_cons_mean"] = float(tcs.get("to_consensus_mean", float("nan"))) if isinstance(tcs, dict) else float("nan")
        out["stab_to_cons_std"]  = float(tcs.get("to_consensus_std", float("nan"))) if isinstance(tcs, dict) else float("nan")
    else:
        for k in ["stab_mean_jaccard","stab_std_jaccard","stab_min_jaccard","stab_max_jaccard","stab_to_cons_mean","stab_to_cons_std"]:
            out[k] = float("nan")
        out["stab_n_pairs"] = 0
        out["stab_consensus_size"] = float("nan")

    # ---- Guardado ----
    if do_save_seed and run_dir is not None:
        try:
            save_bundle(
                run_dir, cfg=cfg, seed=seed,
                genes_full=genes_full, nodes_used_full=nodes_used, genes_comp=genes_comp,
                edge_index_full=ei_use, edge_weight_full=ew_use, edge_type_full=et_use,
                edge_index_comp=ei_comp_t, edge_weight_comp=ew_comp_t, edge_type_comp=et_comp_t,
                reg_groups_comp=reg_groups_comp_t,
                model_pruned=model_p, metrics=out, stab=stab, phase1_best_state=best_state,
            )
            print(f"[SAVE] Bundle guardado en: {run_dir}")
        except Exception as e:
            print(f"[SAVE-WARN] No pude guardar bundle en {run_dir}: {e}")
    elif is_best_only:
        try:
            score = float(out.get(BEST_ONLY_METRIC, out.get("macro_f1", float("-inf"))))
        except Exception:
            score = float("-inf")
        best = _BEST_REGISTRY.get(cfg.name, {"score": float("-inf"), "seed": None})
        if score > float(best.get("score", float("-inf"))):
            _BEST_REGISTRY[cfg.name] = {"score": score, "seed": int(seed)}
            best_dir = os.path.join(save_root, cfg.name, "best")
            _reset_dir(best_dir)
            try:
                save_bundle(
                    best_dir, cfg=cfg, seed=seed,
                    genes_full=genes_full, nodes_used_full=nodes_used, genes_comp=genes_comp,
                    edge_index_full=ei_use, edge_weight_full=ew_use, edge_type_full=et_use,
                    edge_index_comp=ei_comp_t, edge_weight_comp=ew_comp_t, edge_type_comp=et_comp_t,
                    reg_groups_comp=reg_groups_comp_t,
                    model_pruned=model_p, metrics=out, stab=stab, phase1_best_state=best_state,
                )
                print(f"[SAVE] BEST bundle actualizado: {best_dir} (seed={seed}, {BEST_ONLY_METRIC}={score:.4f})")
            except Exception as e:
                print(f"[SAVE-WARN] No pude guardar BEST bundle en {best_dir}: {e}")

    try:
        del model, model_p
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    return out


# ── Run ablation ─────────────────────────────────


def run_ablation(configs: List[AblationConfig], seeds: List[int], *, save_root: Optional[str] = None) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    if save_root is None:
        save_root = ARTIFACTS_ROOT
    _mkdir(save_root)

    for cfg in configs:
        print(f"\n=== {cfg.name} ===")
        for seed in seeds:
            print(f"  - seed {seed}")
            r = run_single_seed(cfg, seed, save_root=save_root, force_save=False)
            rows.append(r)
            cleanup_memory('after_seed')
            print(f"    macroF1={r['macro_f1']:.4f}  auc={r['auc_ovr_macro']:.4f}  edges={r['n_edges_final']}  nodes={r['n_nodes_compact']}")

    df = pd.DataFrame(rows)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    return df
