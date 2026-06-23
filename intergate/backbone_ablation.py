"""
Backbone-ablation launcher for the InterGATE pipeline.

Drop this file into the same package folder as ablation.py, for example:

    intergate/backbone_ablation.py

This wrapper keeps your current ablation/training/stability code intact.  It
only changes the message-passing block used inside ImprovedSharedGraphGNN by
patching model.ResGATBlock during model construction.

Typical notebook usage
----------------------

    from intergate.ablation import register_runtime
    from intergate.backbone_ablation import (
        make_backbone_configs,
        run_backbone_ablation,
        summarize_backbone_results,
    )

    register_runtime(... same objects as before ...)

    configs = make_backbone_configs(
        backbones=("gat", "sage", "gin", "graph_transformer"),
        do_stability=False,   # first pass: fast validation/test screen
        do_pretrain=False,    # optional for speed
    )

    df = run_backbone_ablation(configs, seeds=[1234], save_root="./artifacts_backbones_dev")
    summary = summarize_backbone_results(df)
    display(summary)

After deciding which backbones are competitive, rerun selected configs with
more seeds and do_stability=True.
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from typing import Iterable, List, Optional, Dict, Any

import numpy as np
import pandas as pd

from .config import CFG
from .ablation import (
    AblationConfig,
    CFG_FULL,
    register_runtime,
    run_single_seed,
    evaluate_bundle_on_test,
    _mkdir,
    cleanup_memory,
)
from .backbone_blocks import patch_backbone, normalize_backbone_name


# -------------------------------------------------------------------------
# Config creation
# -------------------------------------------------------------------------


def _clone_full_cfg(name: str, *, do_stability: Optional[bool], do_pretrain: Optional[bool]) -> AblationConfig:
    cfg = deepcopy(CFG_FULL)
    cfg.name = str(name)
    if do_stability is not None:
        cfg.do_stability = bool(do_stability)
    if do_pretrain is not None:
        cfg.do_pretrain = bool(do_pretrain)
    return cfg


def make_backbone_configs(
    backbones: Iterable[str] = ("gat", "sage", "gin", "graph_transformer"),
    *,
    prefix: str = "FULL",
    do_stability: Optional[bool] = None,
    do_pretrain: Optional[bool] = None,
) -> List[AblationConfig]:
    """
    Create FULL-like configs that differ only in the message-passing backbone.

    The returned configs are standard AblationConfig objects, with an extra
    dynamic attribute cfg.backbone. Your original ablation.py is not modified.
    """
    out = []
    for bb in backbones:
        bb_norm = normalize_backbone_name(bb)
        suffix = {
            "gat": "GAT",
            "sage": "GRAPHSAGE",
            "gin": "GIN",
            "graph_transformer": "GRAPH_TRANSFORMER",
        }[bb_norm]
        cfg = _clone_full_cfg(f"{prefix}_{suffix}", do_stability=do_stability, do_pretrain=do_pretrain)
        cfg.backbone = bb_norm
        out.append(cfg)
    return out


def infer_backbone_from_cfg_or_name(cfg_or_name) -> str:
    """Infer backbone from cfg.backbone or from a config/bundle name."""
    if hasattr(cfg_or_name, "backbone"):
        return normalize_backbone_name(getattr(cfg_or_name, "backbone"))

    name = str(cfg_or_name).upper()
    if "GRAPH_TRANSFORMER" in name or "GRAPHTRANSFORMER" in name or "TRANSFORMER" in name:
        return "graph_transformer"
    if "GRAPHSAGE" in name or "GRAPH_SAGE" in name or name.endswith("SAGE"):
        return "sage"
    if "GIN" in name:
        return "gin"
    return "gat"


# -------------------------------------------------------------------------
# Running
# -------------------------------------------------------------------------


def run_single_backbone_seed(
    cfg: AblationConfig,
    seed: int,
    *,
    save_root: Optional[str] = None,
    force_save: bool = False,
) -> Dict[str, Any]:
    """Run one config/seed under the selected backbone patch."""
    backbone = infer_backbone_from_cfg_or_name(cfg)
    with patch_backbone(backbone):
        row = run_single_seed(cfg, seed, save_root=save_root, force_save=force_save)
    row["backbone"] = backbone
    row["cfg"] = cfg.name
    return row


def run_backbone_ablation(
    configs: List[AblationConfig],
    seeds: Iterable[int],
    *,
    save_root: Optional[str] = None,
    save_csv: bool = True,
    force_save: bool = False,
) -> pd.DataFrame:
    """
    Run backbone ablation with the same data split, graph, gates and training code.

    Parameters
    ----------
    configs:
        Output from make_backbone_configs().
    seeds:
        Seeds to run. For a fast screen, use one seed. For paper-level ablation,
        use several seeds; if do_stability=True, each config also performs its
        internal stability-selection runs.
    save_root:
        Artifact directory.
    save_csv:
        Whether to write backbone_ablation_results.csv and summary CSV.
    force_save:
        If True, forces bundle saving for every config/seed.
    """
    if save_root is None:
        save_root = "./artifacts_backbone_ablation"
    _mkdir(save_root)

    rows = []
    t0 = time.time()

    for cfg in configs:
        backbone = infer_backbone_from_cfg_or_name(cfg)
        print(f"\n=== {cfg.name} | backbone={backbone} ===")
        for seed in list(seeds):
            print(f"  - seed {seed}")
            r = run_single_backbone_seed(cfg, int(seed), save_root=save_root, force_save=force_save)
            rows.append(r)
            cleanup_memory("after_backbone_seed")
            mf1 = r.get("macro_f1", np.nan)
            auc = r.get("auc_ovr_macro", np.nan)
            nodes = r.get("n_nodes_compact", np.nan)
            edges = r.get("n_edges_final", np.nan)
            print(f"    macroF1={mf1:.4f}  auc={auc:.4f}  edges={edges}  nodes={nodes}")

    df = pd.DataFrame(rows)

    if save_csv:
        out_csv = os.path.join(save_root, "backbone_ablation_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")
        try:
            summary = summarize_backbone_results(df)
            out_sum = os.path.join(save_root, "backbone_ablation_summary.csv")
            summary.to_csv(out_sum, index=False)
            print(f"[SAVE] {out_sum}")
        except Exception as e:
            print(f"[WARN] Could not save summary: {e}")

        meta = {
            "configs": [getattr(c, "name", str(c)) for c in configs],
            "backbones": [infer_backbone_from_cfg_or_name(c) for c in configs],
            "seeds": [int(s) for s in list(seeds)],
            "elapsed_min": (time.time() - t0) / 60.0,
            "cfg_globals": {
                "HIDDEN": int(CFG.HIDDEN),
                "NUM_LAYERS": int(CFG.NUM_LAYERS),
                "NUM_HEADS": int(CFG.NUM_HEADS),
                "POOL_HEADS": int(CFG.POOL_HEADS),
                "KEEP_MIN": float(CFG.KEEP_MIN),
                "STAB_KEEP_FINAL": float(CFG.STAB_KEEP_FINAL),
                "STAB_FREQ_THR": float(CFG.STAB_FREQ_THR),
                "USE_HYBRID_MODEL": bool(CFG.USE_HYBRID_MODEL),
            },
        }
        with open(os.path.join(save_root, "backbone_ablation_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nDONE in {(time.time() - t0) / 60.0:.1f} min")
    return df


# -------------------------------------------------------------------------
# Summaries and loading saved bundles
# -------------------------------------------------------------------------


def summarize_backbone_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compact mean/std summary by backbone and config."""
    if df is None or df.empty:
        return pd.DataFrame()

    metric_cols = [
        c for c in [
            "acc", "macro_f1", "weighted_f1", "auc_ovr_macro",
            "val_macro_f1_phase1", "n_nodes_compact", "n_edges_final",
            "stab_mean_jaccard", "stab_consensus_size",
        ]
        if c in df.columns
    ]

    group_cols = [c for c in ["cfg", "backbone"] if c in df.columns]
    if not group_cols:
        group_cols = ["cfg"]

    pieces = []
    for metric in metric_cols:
        tmp = df.groupby(group_cols, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
        tmp = tmp.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": f"{metric}_n"})
        pieces.append(tmp)

    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on=group_cols, how="outer")

    sort_col = "macro_f1_mean" if "macro_f1_mean" in out.columns else None
    if sort_col:
        out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return out


def evaluate_backbone_bundle_on_test(bundle_dir: str, *, backbone: Optional[str] = None, device=None) -> Dict[str, Any]:
    """
    Evaluate a saved bundle under the correct backbone patch.

    If backbone is None, it is inferred from the bundle path/name. This is useful
    because the original ablation.py metadata does not know about the dynamic
    cfg.backbone attribute.
    """
    if backbone is None:
        backbone = infer_backbone_from_cfg_or_name(bundle_dir)
    backbone = normalize_backbone_name(backbone)
    with patch_backbone(backbone):
        out = evaluate_bundle_on_test(bundle_dir, device=device)
    out["backbone"] = backbone
    return out
