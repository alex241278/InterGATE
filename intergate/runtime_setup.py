"""Single, auditable setup used by every InterGATE model/reviewer notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import CFG
from .prepared_data import load_development_bundle, load_prepared_bundle
from .data import (
    load_expression_and_metadata, prepare_task, prepare_genes, encode_labels,
    cohort_split, scale_features, apply_connected_only,
)
from .graph_cache import get_or_build_backbone, get_or_build_Xh
from .ablation import register_runtime, resolve_class_keep_ratio
from .biology import (
    DifferentialEvidenceConfig,
    evidence_fingerprint,
    get_or_build_differential_evidence,
    load_evidence_cache,
)


def assert_v6_protocol(cfg=CFG) -> None:
    """Fail closed when a biology-first V6 invariant was changed."""
    expected = {
        "PER_LABEL_GRAPHS": True,
        "GRAPH_LEARNING_XH_MODE": "zero",
        "CLASS_GRAPH_XH_MODE": "orig",
        "PHASE2_A_MODE": "graph_only",
        "PHASE2_FIXED_GRAPH_BYPASS_GATES": True,
        "PHASE2_FREEZE_STRUCTURAL_PARAMETERS": True,
        "USE_HYBRID_MODEL": True,
        "SPARSITY_MODE": "target",
        "CLASS_GRAPH_STABILITY_FIXED_BUDGET": True,
        "SELECT_METRIC": "macro_f1",
        "OUTER_RUNS": 5,
        "STAB_RUNS": 0,
        "STAB_SEED_OFFSETS": (),
        "CENTER_CLASS_DELTAS": True,
        "BIOLOGY_FIRST": True,
        "BIOLOGY_EVIDENCE_METHOD": (
            "within_cohort_differential_correlation"
        ),
        "ADD_CONNECTIVITY_PENALTY": False,
        "CONNECTIVITY_LAMBDA": 0.0,
        "DO_STABILITY_SELECTION": False,
    }
    bad = {
        key: (getattr(cfg, key), value)
        for key, value in expected.items()
        if getattr(cfg, key) != value
    }
    if bad:
        raise ValueError(f"Overrides incompatibles con el protocolo V6: {bad}")
    if abs(float(cfg.KEEP_MIN) - 0.01) > 1e-12:
        raise ValueError("El análisis principal fija kappa=0.01.")
    if tuple(cfg.STABILITY_SEEDS) != (1234, 2025, 7, 99, 31337):
        raise ValueError("Las semillas externas no son las preespecificadas.")
    if (int(cfg.FT_EPOCHS_A), int(cfg.FT_EPOCHS_B)) != (5, 80):
        raise ValueError(
            "Phase 2 V6 requiere cinco epochs graph-only y hasta 80 "
            "epochs híbridos."
        )


# Alias de importación para módulos históricos. Los notebooks entregados en
# V6 llaman explícitamente a assert_v6_protocol.
assert_v4_protocol = assert_v6_protocol


def assert_transport_protocol(cfg=CFG) -> None:
    """Fail closed if a prespecified InterGATE V6 setting was modified."""
    assert_v6_protocol(cfg)
    expected = {
        "TRANSPORT_PROTOCOL_VERSION": "intergate-v6-biology-first-v1",
        "TRANSPORT_EPISODES": 60,
        "TRANSPORT_INNER_LR": 5e-4,
        "TRANSPORT_FIRST_ORDER": True,
        "TRANSPORT_META_TRAIN_WEIGHT": 0.25,
        "TRANSPORT_META_VALIDATION_WEIGHT": 1.0,
        "TRANSPORT_TOPOLOGY_LAMBDA": 0.10,
        "TRANSPORT_SOFT_TOPK_TAU": 0.10,
        "TRANSPORT_QUOTA_END_FRACTION": 0.65,
        "TRANSPORT_PATIENCE": 15,
        "TRANSPORT_BALANCED_SAMPLING": True,
        "TRANSPORT_STEPS_PER_EPISODE": 96,
        "TRANSPORT_CHECKPOINT_METRIC": "cohort_macro_f1",
        "TRANSPORT_SUPPORT_STEPS": 5,
        "TRANSPORT_SUPPORT_BATCHES": 32,
        "TRANSPORT_MIN_COHORT_SAMPLES": 2,
        "BIOLOGY_CONTEXT_LAMBDA": 0.10,
        "BIOLOGY_CONTEXT_WARMUP_EPISODES": 8,
        "BIOLOGY_SCORE_CONTEXT_WEIGHT": 0.35,
        "BIOLOGY_SCORE_SUPPORT_WEIGHT": 0.25,
        "DEFINITIVE_BIOLOGY_SCORE_SUPPORT_WEIGHT": 0.075,
    }
    bad = {}
    for key, wanted in expected.items():
        observed = getattr(cfg, key)
        equal = (
            abs(float(observed) - float(wanted)) <= 1e-12
            if isinstance(wanted, float)
            else observed == wanted
        )
        if not equal:
            bad[key] = (observed, wanted)
    if bad:
        raise ValueError(
            f"Overrides incompatibles con InterGATE V6: {bad}"
        )


def prepare_runtime(cfg=CFG, *, force_rebuild_backbone: bool = False,
                    force_rebuild_xh: bool = False,
                    force_rebuild_biology: bool = False,
                    build_biology_evidence: bool = True,
                    validate: bool = True,
                    include_external: bool = False):
    """Load fixed data, fit train-only scaling, build prior and register state.

    By default only development is read and ``test_idx`` is empty. The
    external matrices and labels are opened only with ``include_external=True``
    in notebook 07, after verification of the protocol lock.
    """
    cfg.refresh_options()
    if validate:
        assert_v6_protocol(cfg)
    bundle = (
        load_prepared_bundle(cfg, validate=True)
        if include_external
        else load_development_bundle(cfg, validate=True)
    )
    X_df, y_str, cohort = load_expression_and_metadata(
        sample_col=cfg.SAMPLE_COL,
        label_col=cfg.LABEL_COL,
        cohort_col=cfg.COHORT_COL,
        include_external=bool(include_external),
    )
    X_df, y_str, cohort, task_info = prepare_task(
        X_df, y_str, cohort, cfg,
    )
    X_df_genes, genes = prepare_genes(X_df)
    y, classes, label_map = encode_labels(y_str)
    n_classes = len(classes)
    if tuple(classes) != tuple(cfg.MULTICLASS_LABELS):
        raise ValueError(
            "Orden de clases cargado distinto del contrato: "
            f"{classes} != {cfg.MULTICLASS_LABELS}"
        )

    edge_index, edge_weight, edge_type = get_or_build_backbone(
        genes,
        cache_dir=cfg.PIPELINE_CACHE_DIR,
        force_rebuild=bool(force_rebuild_backbone),
        graph_source=cfg.GRAPH_SOURCE,
    )
    train_idx, val_idx, test_idx = cohort_split(
        cohort, y, val_size=cfg.VAL_SIZE, seed=cfg.SEED,
    )
    X_scaled = scale_features(
        X_df_genes, train_idx, mode=cfg.SCALE_MODE,
        use_quantile=cfg.USE_QUANTILE,
    )
    if cfg.CONNECTED_ONLY:
        X_scaled, edge_index, edge_weight, edge_type, genes = (
            apply_connected_only(
                X_scaled, edge_index, edge_weight, edge_type, genes,
            )
        )

    biology = None
    if bool(build_biology_evidence):
        evidence_cache = Path(cfg.PIPELINE_CACHE_DIR) / "v6_biology_evidence.npz"
        biology_cfg = DifferentialEvidenceConfig(
            min_samples_per_group=int(cfg.BIOLOGY_MIN_SAMPLES_PER_GROUP),
            min_cohorts=int(cfg.BIOLOGY_MIN_COHORTS),
            edge_batch_size=int(cfg.BIOLOGY_EDGE_BATCH_SIZE),
            meta_z_clip=float(cfg.BIOLOGY_META_Z_CLIP),
            heterogeneity_penalty=float(cfg.BIOLOGY_HETEROGENEITY_PENALTY),
        )
        if bool(include_external):
            # Once external labels are visible, even a train-only rebuild is
            # forbidden. The cache key intentionally depends only on TRAIN,
            # so the exact pre-lock object must already exist and match.
            fingerprint = evidence_fingerprint(
                X=X_scaled, genes=genes, edge_index=edge_index, y=y,
                cohort=cohort, train_idx=train_idx, class_names=classes,
                config=biology_cfg,
            )
            biology = load_evidence_cache(
                evidence_cache, fingerprint=fingerprint
            )
            if biology is None:
                raise RuntimeError(
                    "El cache biológico V6 no existe o no coincide. No se "
                    "permite reconstruirlo después de abrir el externo. "
                    "Vuelva al notebook 01 antes de congelar el protocolo."
                )
        else:
            biology = get_or_build_differential_evidence(
                cache_path=evidence_cache,
                X=X_scaled,
                genes=genes,
                y=y,
                cohort=cohort,
                train_idx=train_idx,
                edge_index=edge_index,
                class_names=classes,
                cohort_order=cfg.TRAIN_COHORTS,
                config=biology_cfg,
                force_rebuild=bool(force_rebuild_biology),
            )

    X_h, graph_feature_names = get_or_build_Xh(
        X_scaled, genes, edge_index,
        cache_dir=cfg.PIPELINE_CACHE_DIR,
        force_rebuild=bool(force_rebuild_xh),
        stats=cfg.REG_STATS,
        min_targets=cfg.REG_MIN_GENES,
        max_regulators=cfg.REG_MAX_REGULATORS,
    )
    device = cfg.DEVICE
    edge_index_t = torch.as_tensor(
        edge_index, dtype=torch.long, device=device
    )
    edge_weight_t = torch.as_tensor(
        edge_weight, dtype=torch.float32, device=device
    )
    edge_type_t = torch.as_tensor(
        edge_type, dtype=torch.long, device=device
    )
    register_runtime(
        Xs_gene=X_scaled, genes_kegg=genes,
        X_h=X_h, y=y, n_classes=n_classes, label_map=label_map,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        edge_index_t=edge_index_t, edge_weight_t=edge_weight_t,
        edge_type_t=edge_type_t,
        sample_weights=None, task_info=task_info,
        cohort=cohort, y_str=y_str,
        biology_evidence=(
            None if biology is None else biology["context_score"]
        ),
        biology_evidence_bundle=biology,
        PHASE2_BEST_METRIC="macro_f1",
        reg_groups_t=None,
    )

    candidate_edges = int((edge_index[0] != edge_index[1]).sum())
    kappa = resolve_class_keep_ratio(edge_index_t, cfg.KEEP_MIN)
    k_nominal = max(1, int(round(float(kappa) * candidate_edges)))
    return {
        "bundle": bundle,
        "X_df": X_df_genes,
        "X_scaled": X_scaled,
        "X_h": X_h,
        "graph_feature_names": graph_feature_names,
        "genes": genes,
        "y": y,
        "y_str": y_str,
        "classes": classes,
        "label_map": label_map,
        "cohort": cohort,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "edge_type": edge_type,
        "edge_index_t": edge_index_t,
        "edge_weight_t": edge_weight_t,
        "edge_type_t": edge_type_t,
        "candidate_edges": candidate_edges,
        "kappa": float(kappa),
        "k_nominal": k_nominal,
        "biology_evidence": biology,
        "external_opened": bool(include_external),
    }
