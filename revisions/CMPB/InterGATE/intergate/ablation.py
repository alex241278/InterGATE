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
from dataclasses import dataclass, asdict, field, replace
from typing import Optional, Dict, List, Any, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import (
    CFG, implementation_fingerprint, model_config_fingerprint,
    model_config_snapshot,
)
from .utils import set_all_seeds, cleanup_memory


# ── Runtime registry ──────────────────────────────
# Notebook-scope objects (data matrices, splits, etc.) that cannot
# live in CFG.  Call register_runtime(**kw) once from the notebook.

_RT: Dict[str, Any] = {}


def _update_content_digest(digest, name: str, values) -> None:
    """Hash an array-like value without changing its numerical representation."""
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii", errors="replace"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    if array.dtype.kind in {"O", "U", "S"}:
        digest.update("\x1f".join(map(str, array.reshape(-1))).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(b"\0")


def _refresh_dataset_content_fingerprint() -> None:
    required = (
        "Xs_gene", "X_h", "y", "cohort", "train_idx", "val_idx",
        "test_idx", "genes_kegg", "edge_index_t", "edge_weight_t",
        "edge_type_t",
    )
    if not all(name in _RT for name in required):
        return
    digest = hashlib.sha256()
    for name in required:
        _update_content_digest(digest, name, _RT[name])
    _RT["dataset_content_sha256"] = digest.hexdigest()


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
    fingerprint_inputs = {
        "Xs_gene", "X_h", "y", "cohort", "train_idx", "val_idx",
        "test_idx", "genes_kegg", "edge_index_t", "edge_weight_t",
        "edge_type_t",
    }
    if fingerprint_inputs.intersection(kwargs):
        _refresh_dataset_content_fingerprint()


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
    from .pruning import export_pruned_graph, export_pruned_graphs_by_class
    return export_pruned_graph, export_pruned_graphs_by_class


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
    use_signed_prior: bool       # si False: elimina solo el signo usando |w|
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
    use_signed_prior=True,
    signed_channels_mode=CFG.SIGNED_CHANNELS_MODE,
    # Se conserva el atributo para compatibilidad de arquitectura; en V8 el
    # coeficiente efectivo es exactamente cero y la coherencia es modular.
    connectivity_penalty=True,
    do_pretrain=CFG.DO_PRETRAIN,
    do_stability=CFG.DO_STABILITY_SELECTION,
)

CFG_NO_TYPE = replace(
    CFG_FULL, name="NO_TYPE", edge_type_gating=False,
)
CFG_NO_SIGN = replace(
    CFG_FULL, name="NO_SIGN", use_signed_prior=False,
)
CFG_NO_CONNECTIVITY = replace(
    CFG_FULL, name="NO_CONNECTIVITY", connectivity_penalty=False,
)

# Invalidates progress rows produced before NO_SIGN was made orthogonal to
# NO_TYPE. All four arms are rerun together to preserve matched comparisons.
PHASE1_COMPONENT_ABLATION_PROTOCOL = "v4.1-no-sign-abs-only-20260815"


def component_ablation_configs() -> List[AblationConfig]:
    """Return the matched Phase-1-only component-ablation arms.

    FULL, NO_TYPE, NO_SIGN and NO_CONNECTIVITY share every setting except the
    single structural component named by each ablation. Stability is disabled
    because the component analysis is intentionally Phase 1 only.
    """
    full = replace(CFG_FULL, name="FULL", do_stability=False)
    configs = [
        full,
        replace(full, name="NO_TYPE", edge_type_gating=False),
        replace(full, name="NO_SIGN", use_signed_prior=False),
        replace(full, name="NO_CONNECTIVITY", connectivity_penalty=False),
    ]
    validate_component_ablation_configs(configs)
    return configs


def validate_component_ablation_configs(
    configs: List[AblationConfig],
) -> None:
    """Fail if an arm changes more than its named network component."""
    by_name = {str(cfg.name).upper(): cfg for cfg in configs}
    required = {"FULL", "NO_TYPE", "NO_SIGN", "NO_CONNECTIVITY"}
    if set(by_name) != required:
        raise ValueError(
            "La ablación debe contener exactamente FULL, NO_TYPE, NO_SIGN y "
            f"NO_CONNECTIVITY; recibido={sorted(by_name)}."
        )
    full = by_name["FULL"]
    if bool(full.do_stability):
        raise ValueError("La ablación de componentes es Phase1-only, sin estabilidad.")
    fields = ("edge_type_gating", "use_signed_prior", "connectivity_penalty")
    expected = {
        "NO_TYPE": "edge_type_gating",
        "NO_SIGN": "use_signed_prior",
        "NO_CONNECTIVITY": "connectivity_penalty",
    }
    for name, changed_field in expected.items():
        arm = by_name[name]
        changed = {
            field_name for field_name in fields
            if getattr(arm, field_name) != getattr(full, field_name)
        }
        if changed != {changed_field}:
            raise ValueError(
                f"{name} debe diferir de FULL solo en {changed_field}; "
                f"diferencias observadas={sorted(changed)}."
            )
        for invariant in (
            "signed_channels_mode", "do_pretrain", "do_stability",
            "edge_type_mode", "pool_level",
        ):
            if getattr(arm, invariant) != getattr(full, invariant):
                raise ValueError(
                    f"{name} cambia además {invariant}; la comparación deja "
                    "de ser una ablación de un único componente."
                )


# Protocolo definitivo: solo los tres componentes de red solicitados. La
# ablación de la fuente del backbone se ejecuta en un notebook separado.
ABLA_CONFIGS: List[AblationConfig] = component_ablation_configs()

# Seeds / epochs (from CFG, overridable at module level)
ABLA_SEEDS = list(CFG.STABILITY_SEEDS)
# Offsets internos históricos. Para una semilla externa ``s`` se ejecutan
# las ocho semillas ``s*1000 + offset``; no se comparte el consenso entre
# ejecuciones externas.
STAB_SEEDS = list(CFG.STAB_SEED_OFFSETS)
MAIN_EPOCHS = CFG.EPOCHS1
FT_EPOCHS_2A = CFG.FT_EPOCHS_A
FT_EPOCHS_2B = CFG.FT_EPOCHS_B

FAST_DEV = False

NUM_WORKERS = 0   # default; override from notebook if desired


def apply_fast_dev():
    """Reduce epochs/seeds for quick debugging."""
    global MAIN_EPOCHS, FT_EPOCHS_2A, FT_EPOCHS_2B
    global ABLA_SEEDS, STAB_SEEDS, FAST_DEV
    FAST_DEV = True
    MAIN_EPOCHS = 1
    FT_EPOCHS_2A = 5
    FT_EPOCHS_2B = 20
    ABLA_SEEDS = ABLA_SEEDS[:1]
    STAB_SEEDS = STAB_SEEDS[:1]
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

    - Si cfg.use_signed_prior=False: toma abs(w), sin modificar edge_type.
      De este modo NO_SIGN elimina exclusivamente el signo y no queda
      confundida con NO_TYPE.
    - Si cfg.edge_type_gating=False: desactiva solo la calibración tipada del
      gate; conserva edge_type para el signo, la procedencia y la auditoría.
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

    if et is None:
        et = torch.zeros(ei.size(1), dtype=torch.long, device=device)

    return ei, ew, et


def build_model_from_cfg(cfg: AblationConfig, class_weights_t: Optional[torch.Tensor]):
    ImprovedSharedGraphGNN, _, _ = _get_model_classes()

    ei, ew, et = make_prior_for_cfg(cfg)
    X_h = _rt("X_h")
    n_classes = _rt("n_classes")
    label_map = _rt("label_map")
    class_names = [None] * int(n_classes)
    for name, class_id in label_map.items():
        class_names[int(class_id)] = str(name)
    if any(name is None for name in class_names):
        raise ValueError(f"label_map incompleto para {n_classes} clases: {label_map}")
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
        per_label_graphs=bool(CFG.PER_LABEL_GRAPHS),
        class_names=class_names,

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
        use_node_identity=bool(getattr(CFG, "USE_NODE_IDENTITY", False)),
    ).to(CFG.DEVICE)
    biology_evidence = _RT.get("biology_evidence", None)
    if bool(getattr(CFG, "BIOLOGY_FIRST", False)):
        if biology_evidence is None:
            raise RuntimeError(
                "V8 requiere biology_evidence en el runtime. Use "
                "prepare_runtime(..., build_biology_evidence=True)."
            )
        model.set_biology_evidence(biology_evidence)
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
    batch_size = (
        int(CFG.GRAPH_BATCH_SIZE)
        if bool(CFG.PER_LABEL_GRAPHS)
        else int(CFG.BATCH_SIZE)
    )
    dl_tr = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=pin)
    dl_va = torch.utils.data.DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin)
    dl_te = torch.utils.data.DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin)
    return dl_tr, dl_va, dl_te


def class_names_from_label_map(label_map: dict, n_classes: int) -> List[str]:
    """Ordena los nombres de las clases por su identificador numérico."""
    names: List[Optional[str]] = [None] * int(n_classes)
    for name, class_id in label_map.items():
        names[int(class_id)] = str(name)
    if any(name is None for name in names):
        raise ValueError(f"label_map incompleto para {n_classes} clases: {label_map}")
    return [str(name) for name in names]


def resolve_class_keep_ratio(
    edge_index: torch.Tensor,
    fallback_ratio: Optional[float] = None,
) -> float:
    """Resuelve kappa para los grafos de clase.

    ``CLASS_GRAPH_TARGET_EDGES=None`` mantiene el presupuesto proporcional
    ``fallback_ratio`` (por defecto ``KEEP_MIN``). Un entero actúa únicamente
    como override y se convierte a ratio sobre las aristas no triviales del
    backbone actual.
    """
    if fallback_ratio is None:
        fallback_ratio = float(CFG.KEEP_MIN)
    target = getattr(CFG, "CLASS_GRAPH_TARGET_EDGES", None)
    if not bool(CFG.PER_LABEL_GRAPHS) or target is None:
        return float(fallback_ratio)

    edge_index = torch.as_tensor(edge_index)
    candidates = int((edge_index[0] != edge_index[1]).sum().item())
    if candidates <= 0:
        raise ValueError("El backbone no contiene aristas no triviales.")
    target = int(target)
    if target <= 0:
        raise ValueError("CLASS_GRAPH_TARGET_EDGES debe ser positivo o None.")
    if target > candidates:
        raise ValueError(
            f"CLASS_GRAPH_TARGET_EDGES={target} supera las {candidates} "
            "aristas candidatas."
        )
    return float(target / candidates)


def class_graphs_from_edge_sets(
    edge_sets_by_class: Dict[int, set],
    cfg: AblationConfig,
    class_names: List[str],
    selection_frequency_by_class: Optional[
        Dict[int, Dict[Tuple[int, int], float]]
    ] = None,
    selection_rank_by_class: Optional[
        Dict[int, Dict[Tuple[int, int], float]]
    ] = None,
    selection_score_by_class: Optional[
        Dict[int, Dict[Tuple[int, int], float]]
    ] = None,
) -> Dict[int, Dict[str, Any]]:
    """Materializa conjuntos (src,dst) con pesos/tipos del prior original."""
    ei0, ew0, et0 = make_prior_for_cfg(cfg)
    ei_np = ei0.detach().cpu().numpy()
    ew_np = ew0.detach().cpu().numpy()
    et_np = et0.detach().cpu().numpy()
    prior = {
        (int(src), int(dst)): (
            float(weight), int(edge_type), int(edge_id)
        )
        for edge_id, (src, dst, weight, edge_type) in enumerate(zip(
            ei_np[0], ei_np[1], ew_np, et_np
        ))
    }

    out: Dict[int, Dict[str, Any]] = {}
    for class_id, edges in edge_sets_by_class.items():
        pairs = sorted(edge for edge in edges if edge in prior)
        if not pairs:
            raise ValueError(
                f"El grafo consenso de {class_names[class_id]} está vacío."
            )
        src = [pair[0] for pair in pairs]
        dst = [pair[1] for pair in pairs]
        weights = [prior[pair][0] for pair in pairs]
        types = [prior[pair][1] for pair in pairs]
        edge_ids = [prior[pair][2] for pair in pairs]
        frequencies = (selection_frequency_by_class or {}).get(
            int(class_id), {}
        )
        median_ranks = (selection_rank_by_class or {}).get(
            int(class_id), {}
        )
        median_scores = (selection_score_by_class or {}).get(
            int(class_id), {}
        )
        ranked_pairs = sorted(
            pairs,
            key=lambda pair: (
                -float(frequencies.get(pair, 0.0)),
                float(median_ranks.get(pair, float("inf"))),
                (
                    -float(median_scores[pair])
                    if pair in median_scores
                    and np.isfinite(float(median_scores[pair]))
                    else float("inf")
                ),
                pair[0], pair[1],
            ),
        )
        consensus_rank = {
            pair: rank for rank, pair in enumerate(ranked_pairs, start=1)
        }
        out[int(class_id)] = {
            "class_id": int(class_id),
            "class_name": class_names[class_id],
            "edge_index": torch.tensor([src, dst], dtype=torch.long),
            "edge_weight": torch.tensor(weights, dtype=torch.float32),
            "edge_type": torch.tensor(types, dtype=torch.long),
            "candidate_edge_ids": torch.tensor(
                edge_ids, dtype=torch.long
            ),
            "selection_frequency": torch.tensor(
                [frequencies.get(pair, float("nan")) for pair in pairs],
                dtype=torch.float32,
            ),
            "consensus_rank": torch.tensor(
                [consensus_rank[pair] for pair in pairs], dtype=torch.long
            ),
            "inner_median_rank": torch.tensor(
                [median_ranks.get(pair, float("nan")) for pair in pairs],
                dtype=torch.float32,
            ),
            "inner_median_logit": torch.tensor(
                [median_scores.get(pair, float("nan")) for pair in pairs],
                dtype=torch.float32,
            ),
            # Compatibilidad con lectores anteriores: en un grafo consenso,
            # learned_logit representa el logit mediano de las corridas
            # internas en las que apareció la arista.
            "learned_logit": torch.tensor(
                [median_scores.get(pair, float("nan")) for pair in pairs],
                dtype=torch.float32,
            ),
            "learned_gate": torch.full(
                (len(pairs),), float("nan"), dtype=torch.float32
            ),
            "kept": int(len(pairs)),
            "total": int(ei0.shape[1]),
            "thr_logit": float("nan"),
        }
    return out


def build_union_class_graph(
    class_graphs: Dict[int, Dict[str, Any]],
    n_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Construye una unión física y conserva una máscara por experto.

    La unión solo evita duplicar nodos/aristas en memoria. No se usa como un
    tercer grafo predictivo compartido.
    """
    pair_payload: Dict[Tuple[int, int], Tuple[float, int]] = {}
    pairs_by_class: Dict[int, set] = {}
    for class_id in range(int(n_classes)):
        if class_id not in class_graphs:
            raise KeyError(f"Falta el grafo de la clase {class_id}.")
        graph = class_graphs[class_id]
        ei = graph["edge_index"].detach().cpu().numpy()
        ew = graph["edge_weight"].detach().cpu().numpy()
        et = graph["edge_type"].detach().cpu().numpy()
        pairs = set()
        for src, dst, weight, edge_type in zip(ei[0], ei[1], ew, et):
            pair = (int(src), int(dst))
            pairs.add(pair)
            payload = (float(weight), int(edge_type))
            if pair in pair_payload and pair_payload[pair] != payload:
                raise ValueError(f"Prior inconsistente para arista {pair}.")
            pair_payload[pair] = payload
        pairs_by_class[class_id] = pairs

    union_pairs = sorted(pair_payload)
    if not union_pairs:
        raise ValueError("La unión de grafos por clase está vacía.")
    pair_to_union = {pair: idx for idx, pair in enumerate(union_pairs)}
    edge_index = torch.tensor(
        [
            [pair[0] for pair in union_pairs],
            [pair[1] for pair in union_pairs],
        ],
        dtype=torch.long,
    )
    edge_weight = torch.tensor(
        [pair_payload[pair][0] for pair in union_pairs],
        dtype=torch.float32,
    )
    edge_type = torch.tensor(
        [pair_payload[pair][1] for pair in union_pairs],
        dtype=torch.long,
    )
    ids_by_class = [
        torch.tensor(
            [pair_to_union[pair] for pair in sorted(pairs_by_class[class_id])],
            dtype=torch.long,
        )
        for class_id in range(int(n_classes))
    ]
    return edge_index, edge_weight, edge_type, ids_by_class


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
        "GRAPH_BATCH_SIZE": int(CFG.GRAPH_BATCH_SIZE),
        "LR": float(CFG.LR),
        "WEIGHT_DECAY": float(CFG.WEIGHT_DECAY),
        "EDGE_L1_PER_EDGE": float(CFG.EDGE_L1_PER_EDGE),
        "SPARSITY_MODE": str(CFG.SPARSITY_MODE),
        "SPARSITY_LAMBDA": float(CFG.SPARSITY_LAMBDA),
        "SPARSITY_BINARIZE": float(CFG.SPARSITY_BINARIZE),
        "AUX_LAMBDA": float(CFG.AUX_LAMBDA),
        "KEEP_MIN": float(CFG.KEEP_MIN),
        "MAIN_EPOCHS": int(MAIN_EPOCHS),
        "FT_EPOCHS_2A": int(FT_EPOCHS_2A),
        "FT_EPOCHS_2B": int(FT_EPOCHS_2B),
        "STAB_KEEP_FINAL": float(CFG.STAB_KEEP_FINAL),
        "OUTER_RUNS": int(CFG.OUTER_RUNS),
        "STABILITY_SEEDS": _to_list(CFG.STABILITY_SEEDS),
        "STAB_RUNS": int(CFG.STAB_RUNS),
        "STAB_SEED_OFFSETS": _to_list(CFG.STAB_SEED_OFFSETS),
        "STAB_FREQ_THR": float(CFG.STAB_FREQ_THR),
        "STAB_STRICT_FREQ_THR": float(CFG.STAB_STRICT_FREQ_THR),
        "OUTER_GENE_CORE_THR": float(CFG.OUTER_GENE_CORE_THR),
        "OUTER_GENE_SENSITIVITY_THR": float(
            CFG.OUTER_GENE_SENSITIVITY_THR
        ),
        "STAB_EPOCHS": int(CFG.STAB_EPOCHS),
        "STABILITY_SCOPE": str(CFG.STABILITY_SCOPE),
        "STABILITY_DATA_RESAMPLING": bool(CFG.STABILITY_DATA_RESAMPLING),
        "TRANSPORT_PROTOCOL_VERSION": str(CFG.TRANSPORT_PROTOCOL_VERSION),
        "TRANSPORT_EPISODES": int(CFG.TRANSPORT_EPISODES),
        "TRANSPORT_INNER_LR": float(CFG.TRANSPORT_INNER_LR),
        "TRANSPORT_FIRST_ORDER": bool(CFG.TRANSPORT_FIRST_ORDER),
        "TRANSPORT_META_TRAIN_WEIGHT": float(
            CFG.TRANSPORT_META_TRAIN_WEIGHT
        ),
        "TRANSPORT_META_VALIDATION_WEIGHT": float(
            CFG.TRANSPORT_META_VALIDATION_WEIGHT
        ),
        "TRANSPORT_TOPOLOGY_LAMBDA": float(
            CFG.TRANSPORT_TOPOLOGY_LAMBDA
        ),
        "TRANSPORT_SOFT_TOPK_TAU": float(CFG.TRANSPORT_SOFT_TOPK_TAU),
        "TRANSPORT_QUOTA_END_FRACTION": float(
            CFG.TRANSPORT_QUOTA_END_FRACTION
        ),
        "TRANSPORT_PATIENCE": int(CFG.TRANSPORT_PATIENCE),
        "TRANSPORT_SUPPORT_BATCHES": int(CFG.TRANSPORT_SUPPORT_BATCHES),
        "ABLA_SEEDS": _to_list(ABLA_SEEDS),
        "STAB_SEEDS": _to_list(STAB_SEEDS),
        "USE_MIXUP": bool(CFG.USE_MIXUP),
        "MIXUP_ALPHA": float(CFG.MIXUP_ALPHA),
        "MIXUP_P": float(CFG.MIXUP_P),
        "ACCUM_STEPS": int(CFG.ACCUM_STEPS),
        "GRAPH_ACCUM_STEPS": int(CFG.GRAPH_ACCUM_STEPS),
        "PER_LABEL_GRAPHS": bool(CFG.PER_LABEL_GRAPHS),
        "PER_LABEL_DELTA_L1": float(CFG.PER_LABEL_DELTA_L1),
        "PER_LABEL_DELTA_DIVERSITY_LAMBDA": float(
            CFG.PER_LABEL_DELTA_DIVERSITY_LAMBDA
        ),
        "PER_LABEL_DELTA_LR_MULT": float(CFG.PER_LABEL_DELTA_LR_MULT),
        "CLASS_GRAPH_TARGET_EDGES": (
            None
            if CFG.CLASS_GRAPH_TARGET_EDGES is None
            else int(CFG.CLASS_GRAPH_TARGET_EDGES)
        ),
        "CLASS_GRAPH_STABILITY_FIXED_BUDGET": bool(
            CFG.CLASS_GRAPH_STABILITY_FIXED_BUDGET
        ),
        "TRAIN_REPORT_LABEL": str(CFG.TRAIN_REPORT_LABEL),
        "TRAIN_LOG_FULL_PRUNING": bool(CFG.TRAIN_LOG_FULL_PRUNING),
        "GRAPH_LEARNING_XH_MODE": str(CFG.GRAPH_LEARNING_XH_MODE),
        "CLASS_GRAPH_XH_MODE": str(CFG.CLASS_GRAPH_XH_MODE),
        "PHASE2_A_MODE": str(CFG.PHASE2_A_MODE),
        "PHASE2_FIXED_GRAPH_BYPASS_GATES": bool(
            CFG.PHASE2_FIXED_GRAPH_BYPASS_GATES
        ),
        "PHASE2_FREEZE_STRUCTURAL_PARAMETERS": bool(
            CFG.PHASE2_FREEZE_STRUCTURAL_PARAMETERS
        ),
        "ADD_CONNECTIVITY_PENALTY": bool(CFG.ADD_CONNECTIVITY_PENALTY),
        "CONNECTIVITY_LAMBDA": float(CFG.CONNECTIVITY_LAMBDA),
        "CONNECTIVITY_MIN_DEG": float(CFG.CONNECTIVITY_MIN_DEG),
        "CONNECTIVITY_USE_ABS": bool(CFG.CONNECTIVITY_USE_ABS),
        "GRAPH_SOURCE": str(CFG.GRAPH_SOURCE),
        "USE_HURI": bool(CFG.USE_HURI),
        "USE_OMNIPATH": bool(CFG.USE_OMNIPATH),
        "USE_STRING": bool(CFG.USE_STRING),
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
        "MODEL_VERSION": str(CFG.MODEL_VERSION),
        "USE_NODE_IDENTITY": bool(CFG.USE_NODE_IDENTITY),
        "EXPERT_DEEP_SUPERVISION_LAMBDA": float(
            CFG.EXPERT_DEEP_SUPERVISION_LAMBDA
        ),
        "TAB_FULL_GENE_UNIVERSE": bool(CFG.TAB_FULL_GENE_UNIVERSE),
        "HYBRID_TAB_HIDDEN": int(CFG.HYBRID_TAB_HIDDEN),
        "BRANCH_DROPOUT_TAB": float(CFG.BRANCH_DROPOUT_TAB),
        "BRANCH_DROPOUT_GNN": float(CFG.BRANCH_DROPOUT_GNN),
        "GNN_AUX_LOSS_LAMBDA": float(CFG.GNN_AUX_LOSS_LAMBDA),
        "BIOLOGY_MATCHED_NULL_DRAWS": int(CFG.BIOLOGY_MATCHED_NULL_DRAWS),
        "BIOLOGY_REWIRING_NULL_DRAWS": int(
            CFG.BIOLOGY_REWIRING_NULL_DRAWS
        ),
        "DEVICE": str(CFG.DEVICE),
    }
    d["model_config"] = model_config_snapshot(CFG)
    d["model_config_sha256"] = model_config_fingerprint(CFG)
    d["implementation_sha256"] = implementation_fingerprint()

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
        y_full = np.asarray(_RT.get("y", []), dtype=np.int64)
        cohort_full = np.asarray(_RT.get("cohort", [])).astype(str)

        def _sha256_text(values) -> str:
            return hashlib.sha256(
                "\x1f".join(map(str, values)).encode("utf-8")
            ).hexdigest()

        def _sha256_int(values) -> str:
            array = np.ascontiguousarray(
                np.asarray(values, dtype=np.int64)
            )
            return hashlib.sha256(array.tobytes()).hexdigest()

        d["dataset_signature"] = {
            "n_samples": int(X_full.shape[0]),
            "n_genes_full": int(X_full.shape[1]),
            "genes_sha256": _sha256_text(genes_full),
            "labels_sha256": _sha256_int(y_full),
            "cohorts_sha256": _sha256_text(cohort_full),
            "train_idx_sha256": _sha256_int(_RT.get("train_idx", [])),
            "val_idx_sha256": _sha256_int(_RT.get("val_idx", [])),
            "test_idx_sha256": _sha256_int(_RT.get("test_idx", [])),
            "content_sha256": str(_RT.get("dataset_content_sha256", "")),
        }
    except Exception:
        d["dataset_signature"] = None

    try:
        d["n_classes"] = int(_rt("n_classes"))
        d["label_map"] = {
            str(k): int(v) for k, v in _rt("label_map").items()
        }
    except Exception:
        pass
    if "task_info" in _RT:
        d["task_info"] = dict(_RT["task_info"])

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
    class_graphs_full: Optional[Dict[int, Dict[str, Any]]] = None,
    class_edge_ids_comp: Optional[List[torch.Tensor]] = None,
    class_names: Optional[List[str]] = None,
    model_pruned: torch.nn.Module,
    metrics: dict,
    stab: dict = None,
    phase1_best_state: dict = None,
    proba_te=None,
    y_te=None,
):
    out_dir = _mkdir(out_dir)

    meta = cfg_to_dict(cfg)
    meta.update({
        "cfg_name": cfg.name,
        "seed": int(seed),
        "metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()},
        "n_edges_final_full": int(edge_index_full.shape[1]),
        "n_nodes_compact": int(len(genes_comp)),
        "phase2_protocol": {
            "phase2a": "graph_only",
            "phase2a_xh": "zero",
            "phase2a_mixup": False,
            "phase2b": "hybrid" if bool(CFG.USE_HYBRID_MODEL) else "graph_only",
            "phase2b_xh": str(CFG.CLASS_GRAPH_XH_MODE),
            "phase2b_mixup": bool(CFG.USE_MIXUP),
            "fixed_graph": True,
            "gates_bypassed": bool(CFG.PHASE2_FIXED_GRAPH_BYPASS_GATES),
            "structural_parameters_frozen": bool(
                CFG.PHASE2_FREEZE_STRUCTURAL_PARAMETERS
            ),
            "phase2a_max_epochs": int(FT_EPOCHS_2A),
            "phase2b_max_epochs": int(FT_EPOCHS_2B),
            "phase2b_checkpoint_metric": str(
                _RT.get("PHASE2_BEST_METRIC", "macro_f1")
            ),
        },
    })
    if stab is not None:
        inner_seeds = [int(value) for value in stab.get("seeds", [])]
        meta["stability_protocol"] = {
            "level": "inner_within_outer_seed",
            "outer_seed": int(seed),
            "carrier_phase1_seed": int(seed),
            "phase2_weight_source": "outer_carrier_phase1_checkpoint",
            "inner_seeds": inner_seeds,
            "inner_runs": int(stab.get("n_runs", len(inner_seeds))),
            "inner_exact_k_used_by_phase2": True,
            "outer_primary_gene_signature": "gene_core100_across_5_outer_consensus_graphs",
            "outer_sensitivity_gene_signature": "gene_core80_across_5_outer_consensus_graphs",
            "data_resampling": bool(CFG.STABILITY_DATA_RESAMPLING),
        }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    class_graphs_cpu = {}
    for class_id, class_graph in (class_graphs_full or {}).items():
        saved_graph = {
            "class_id": int(class_id),
            "class_name": str(class_graph["class_name"]),
            "edge_index": class_graph["edge_index"].detach().cpu(),
            "edge_weight": class_graph["edge_weight"].detach().cpu(),
            "edge_type": class_graph["edge_type"].detach().cpu(),
        }
        for field_name in (
            "candidate_edge_ids",
            "learned_logit",
            "learned_gate",
            "selection_frequency",
            "consensus_rank",
            "structural_unit_id",
            "inner_median_rank",
            "inner_median_logit",
            "mean_adapted_logit",
            "differential_logit",
            "biology_score",
        ):
            value = class_graph.get(field_name, None)
            if isinstance(value, torch.Tensor):
                saved_graph[field_name] = value.detach().cpu()
        class_graphs_cpu[int(class_id)] = saved_graph

    edge_index_comp_cpu = edge_index_comp.detach().cpu()
    edge_weight_comp_cpu = (
        edge_weight_comp.detach().cpu()
        if isinstance(edge_weight_comp, torch.Tensor) else None
    )
    edge_type_comp_cpu = (
        edge_type_comp.detach().cpu()
        if isinstance(edge_type_comp, torch.Tensor) else None
    )
    class_graphs_comp = {}
    for class_id, ids in enumerate(class_edge_ids_comp or []):
        ids_cpu = ids.detach().cpu().long()
        name = (
            class_names[class_id]
            if class_names and class_id < len(class_names)
            else f"class_{class_id}"
        )
        class_graphs_comp[int(class_id)] = {
            "class_id": int(class_id),
            "class_name": str(name),
            "edge_index": edge_index_comp_cpu[:, ids_cpu],
            "edge_weight": (
                edge_weight_comp_cpu[ids_cpu]
                if edge_weight_comp_cpu is not None else None
            ),
            "edge_type": (
                edge_type_comp_cpu[ids_cpu]
                if edge_type_comp_cpu is not None else None
            ),
        }

    graph = {
        "genes_full": list(genes_full),
        "nodes_used_full": nodes_used_full,
        "genes_comp": list(genes_comp),
        "edge_index_full": edge_index_full.detach().cpu(),
        "edge_weight_full": (edge_weight_full.detach().cpu() if isinstance(edge_weight_full, torch.Tensor) else None),
        "edge_type_full": (edge_type_full.detach().cpu() if isinstance(edge_type_full, torch.Tensor) else None),
        "edge_index_comp": edge_index_comp_cpu,
        "edge_weight_comp": edge_weight_comp_cpu,
        "edge_type_comp": edge_type_comp_cpu,
        "reg_groups_comp": (reg_groups_comp.detach().cpu() if isinstance(reg_groups_comp, torch.Tensor) else None),
        "class_names": list(class_names or []),
        "class_graphs_full": class_graphs_cpu,
        "class_graphs_comp": class_graphs_comp,
        "class_edge_ids_comp": [
            ids.detach().cpu() for ids in (class_edge_ids_comp or [])
        ],
    }
    torch.save(graph, os.path.join(out_dir, "graph_bundle.pt"))

    edge_source_name = {
        0: "HuRI",
        1: "OmniPath_positive",
        2: "OmniPath_negative",
        3: "STRING",
    }
    for class_id, class_graph in class_graphs_cpu.items():
        class_name = str(class_graph["class_name"])
        safe_name = "".join(
            ch if ch.isalnum() or ch in "-_" else "_"
            for ch in class_name
        )
        torch.save(
            class_graph,
            os.path.join(out_dir, f"graph_{safe_name}.pt"),
        )
        ei_np = class_graph["edge_index"].numpy()
        ew_np = class_graph["edge_weight"].numpy()
        et_np = class_graph["edge_type"].numpy()
        n_edges = int(ei_np.shape[1])
        candidate_ids = class_graph.get(
            "candidate_edge_ids",
            torch.full((n_edges,), -1, dtype=torch.long),
        ).numpy()
        learned_logit = class_graph.get(
            "learned_logit",
            torch.full((n_edges,), float("nan")),
        ).numpy()
        learned_gate = class_graph.get(
            "learned_gate",
            torch.full((n_edges,), float("nan")),
        ).numpy()
        selection_frequency = class_graph.get(
            "selection_frequency",
            torch.full((n_edges,), float("nan")),
        ).numpy()
        consensus_rank = class_graph.get(
            "consensus_rank",
            torch.full((n_edges,), -1, dtype=torch.long),
        ).numpy()
        structural_unit_id = class_graph.get(
            "structural_unit_id",
            torch.full((n_edges,), -1, dtype=torch.long),
        ).numpy()
        inner_median_rank = class_graph.get(
            "inner_median_rank",
            torch.full((n_edges,), float("nan")),
        ).numpy()
        inner_median_logit = class_graph.get(
            "inner_median_logit",
            torch.full((n_edges,), float("nan")),
        ).numpy()
        mean_adapted_logit = class_graph.get(
            "mean_adapted_logit",
            torch.full((n_edges,), float("nan")),
        ).numpy()
        differential_logit = class_graph.get(
            "differential_logit",
            torch.full((n_edges,), float("nan")),
        ).numpy()
        biology_score = class_graph.get(
            "biology_score",
            torch.full((n_edges,), float("nan")),
        ).numpy()
        edge_table = pd.DataFrame({
            "class_id": int(class_id),
            "class_name": class_name,
            "source_idx_full": ei_np[0].astype(np.int64),
            "target_idx_full": ei_np[1].astype(np.int64),
            "source_gene": [genes_full[int(i)] for i in ei_np[0]],
            "target_gene": [genes_full[int(i)] for i in ei_np[1]],
            "prior_weight": ew_np.astype(float),
            "edge_type": et_np.astype(np.int64),
            "candidate_edge_id": candidate_ids.astype(np.int64),
            "learned_logit": learned_logit.astype(float),
            "learned_gate": learned_gate.astype(float),
            "selection_frequency": selection_frequency.astype(float),
            "consensus_rank": consensus_rank.astype(np.int64),
            "structural_unit_id": structural_unit_id.astype(np.int64),
            "inner_median_rank": inner_median_rank.astype(float),
            "inner_median_logit": inner_median_logit.astype(float),
            "mean_adapted_logit": mean_adapted_logit.astype(float),
            "differential_logit": differential_logit.astype(float),
            "biology_score": biology_score.astype(float),
            "edge_source": [
                edge_source_name.get(int(t), f"type_{int(t)}")
                for t in et_np
            ],
        })
        edge_table.to_csv(
            os.path.join(out_dir, f"edges_{safe_name}.csv"),
            index=False,
        )

    sd = {k: v.detach().cpu() for k, v in model_pruned.state_dict().items()}
    core_model = model_pruned.gnn if bool(getattr(model_pruned, "is_hybrid", False)) and hasattr(model_pruned, "gnn") else model_pruned

    model_flags = {
        "edge_type_gating": bool(getattr(core_model, "edge_type_gating", True)),
        "edge_type_mode": str(getattr(core_model, "edge_type_mode", getattr(cfg, "edge_type_mode", "huri_op_signed"))),
        "signed_channels_mode": str(getattr(core_model, "signed_channels_mode", "type_only")),
        "use_regulator_pool": bool(getattr(core_model, "use_regulator_pool", False)),
        "is_hybrid": bool(getattr(model_pruned, "is_hybrid", False)),
        "tab_layers": int(getattr(model_pruned, "tab_layers", 2)),
        "tab_dropout": float(getattr(model_pruned, "tab_dropout", 0.20)),
        "fusion_dropout": float(getattr(model_pruned, "fusion_dropout", 0.20)),
        "blend_logits": bool(getattr(model_pruned, "blend_logits", False)),
        "blend_init": float(getattr(model_pruned, "blend_init", 0.70)),
        "use_node_identity": bool(
            getattr(core_model, "use_node_identity", False)
        ),
        "tab_in_dim": int(getattr(model_pruned, "tab_in_dim", 0)),
        "tab_hidden": int(getattr(model_pruned, "tab_hidden", 0)),
        "branch_dropout_tab": float(
            getattr(model_pruned, "branch_dropout_tab", 0.0)
        ),
        "branch_dropout_gnn": float(
            getattr(model_pruned, "branch_dropout_gnn", 0.0)
        ),
        "gnn_aux_loss_lambda": float(
            getattr(CFG, "GNN_AUX_LOSS_LAMBDA", 0.0)
        ),
        "tab_full_gene_universe": bool(
            getattr(CFG, "TAB_FULL_GENE_UNIVERSE", False)
        ),
        "per_label_graphs": bool(
            getattr(core_model, "per_label_graphs", False)
        ),
        "class_names": list(
            getattr(core_model, "class_names", class_names or [])
        ),
        "fixed_graph_mode": bool(
            getattr(core_model, "fixed_graph_mode", False)
        ),
        "fixed_graph_bypass_gates": bool(
            getattr(core_model, "fixed_graph_bypass_gates", False)
        ),
        "structural_parameters_trainable": [
            name for name in (
                "edge_logit", "edge_logit_delta", "type_scale", "type_bias"
            )
            if isinstance(getattr(core_model, name, None), torch.nn.Parameter)
            and bool(getattr(core_model, name).requires_grad)
        ],
    }
    if not model_flags["fixed_graph_mode"]:
        raise RuntimeError("El bundle final no está marcado como grafo fijo.")
    if not model_flags["fixed_graph_bypass_gates"]:
        raise RuntimeError("El bundle final todavía aplica gates en Phase 2.")
    if model_flags["structural_parameters_trainable"]:
        raise RuntimeError(
            "El bundle contiene parámetros estructurales entrenables: "
            f"{model_flags['structural_parameters_trainable']}"
        )

    torch.save({"state_dict": sd, "model_flags": model_flags}, os.path.join(out_dir, "model_pruned.pt"))

    # Predicciones de test: sin esto la matriz de confusion exige re-inferir,
    # que obliga a reconstruir la matriz compacta y el dataloader a mano.
    if proba_te is not None and y_te is not None:
        try:
            np.savez_compressed(
                os.path.join(out_dir, "test_predictions.npz"),
                proba=np.asarray(proba_te, dtype=np.float32),
                y_true=np.asarray(y_te).reshape(-1).astype(np.int64),
                class_names=np.asarray(
                    [str(c) for c in (class_names or [])], dtype=object
                ),
            )
        except Exception as exc:
            print(f"[SAVE-WARN] no pude guardar test_predictions.npz: {exc}")

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
            frames = stab.get("jaccard_df_by_class", {})
            for class_id, frame in frames.items():
                if not isinstance(frame, pd.DataFrame):
                    continue
                name = (
                    class_names[int(class_id)]
                    if class_names else f"class_{class_id}"
                )
                safe_name = "".join(
                    ch if ch.isalnum() or ch in "-_" else "_"
                    for ch in str(name)
                )
                frame.to_csv(
                    os.path.join(
                        out_dir, f"stability_jaccard_{safe_name}.csv"
                    )
                )
        except Exception:
            pass


def load_bundle(bundle_dir: str) -> dict:
    meta = json.load(open(os.path.join(bundle_dir, "meta.json"), "r", encoding="utf-8"))
    graph = torch.load(os.path.join(bundle_dir, "graph_bundle.pt"), map_location="cpu", weights_only=False)
    model = torch.load(os.path.join(bundle_dir, "model_pruned.pt"), map_location="cpu", weights_only=False)
    return {"meta": meta, "graph": graph, "model": model}


def get_class_graph_from_bundle(
    bundle: dict,
    class_name: Any = "Basal-like",
    *,
    compact: bool = True,
) -> dict:
    """Devuelve un grafo de clase, evitando usar accidentalmente la unión."""
    graph_bundle = bundle.get("graph", bundle)
    key = "class_graphs_comp" if compact else "class_graphs_full"
    class_graphs = graph_bundle.get(key, None)
    if not class_graphs:
        raise KeyError(
            f"El bundle no contiene {key}; probablemente es anterior a los "
            "grafos predictivos por clase."
        )

    if isinstance(class_name, (int, np.integer)):
        class_id = int(class_name)
    else:
        wanted = str(class_name).casefold()
        matches = [
            int(class_id)
            for class_id, class_graph in class_graphs.items()
            if str(class_graph.get("class_name", "")).casefold() == wanted
        ]
        if len(matches) != 1:
            available = [
                str(class_graph.get("class_name", class_id))
                for class_id, class_graph in class_graphs.items()
            ]
            raise KeyError(
                f"Clase {class_name!r} no encontrada; disponibles: "
                f"{available}"
            )
        class_id = matches[0]

    class_graph = class_graphs.get(
        class_id, class_graphs.get(str(class_id), None)
    )
    if class_graph is None:
        raise KeyError(f"No existe el grafo de class_id={class_id}.")
    out = dict(class_graph)
    out["genes"] = list(
        graph_bundle["genes_comp" if compact else "genes_full"]
    )
    return out


def bundle_view_for_class_graph(bundle: dict, class_name: Any = "Basal-like") -> dict:
    """Vista no destructiva para visualizadores antiguos que esperan un grafo."""
    class_graph = get_class_graph_from_bundle(
        bundle, class_name=class_name, compact=True
    )
    view = dict(bundle)
    graph_view = dict(bundle["graph"])
    graph_view["edge_index_comp"] = class_graph["edge_index"]
    graph_view["edge_weight_comp"] = class_graph.get("edge_weight", None)
    graph_view["edge_type_comp"] = class_graph.get("edge_type", None)
    graph_view["selected_class_graph"] = str(class_graph["class_name"])
    view["graph"] = graph_view
    return view


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
    per_label_graphs = bool(flags.get("per_label_graphs", False))
    class_names = list(
        flags.get(
            "class_names",
            g.get("class_names", [f"class_{i}" for i in range(n_classes)]),
        )
    )

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
        per_label_graphs=per_label_graphs,
        class_names=class_names,
        signed_channels=CFG.SIGNED_CHANNELS if hasattr(CFG, "SIGNED_CHANNELS") else False,
        signed_channels_mode=str(flags.get("signed_channels_mode", "type_only")),
        use_regulator_pool=bool(flags.get("use_regulator_pool", False)),
        regulator_groups=(g["reg_groups_comp"].to(device) if (flags.get("use_regulator_pool", False) and g.get("reg_groups_comp") is not None) else None),
        add_connectivity_penalty=False,
        xgraph_dropout=float(meta["globals"].get("XGRAPH_DROPOUT", CFG.XGRAPH_DROPOUT)),
        block_use_self=bool(meta["globals"].get("BLOCK_USE_SELF", CFG.BLOCK_USE_SELF)),
        block_residual=bool(meta["globals"].get("BLOCK_RESIDUAL", CFG.BLOCK_RESIDUAL)),
        use_node_identity=bool(flags.get("use_node_identity", False)),
    ).to(device)

    if per_label_graphs:
        ids_by_class = g.get("class_edge_ids_comp", None)
        if not ids_by_class or len(ids_by_class) != n_classes:
            raise ValueError(
                "Bundle por clase sin class_edge_ids_comp completos."
            )
        base.set_fixed_class_edge_ids([
            ids.to(device=device, dtype=torch.long)
            for ids in ids_by_class
        ])

    if is_hybrid:
        model_p = HybridGNNTabular(
            gnn=base,
            # V8-P3: si el bundle se entreno con el universo completo, la rama
            # tabular tiene 7.131 entradas y no |nodes_used|. Sin esto,
            # load_state_dict(strict=False) descartaria la rama entera en
            # silencio y se estaria evaluando V8.
            tab_in_dim=int(flags.get("tab_in_dim", 0) or num_nodes),
            hidden=hidden, n_classes=n_classes,
            tab_hidden=int(flags.get("tab_hidden", 0)) or None,
            tab_layers=int(flags.get("tab_layers", 2)),
            tab_dropout=float(flags.get("tab_dropout", 0.20)),
            fusion_dropout=float(flags.get("fusion_dropout", 0.20)),
            use_fusion_head=True,
            blend_logits=bool(flags.get("blend_logits", False)),
            blend_init=float(flags.get("blend_init", 0.70)),
        ).to(device)
        model_p.load_state_dict(sd, strict=False)
        if hasattr(model_p.gnn, "edge_logit"):
            with torch.no_grad():
                model_p.gnn.edge_logit.data.fill_(20.0)
                if model_p.gnn.edge_logit_delta is not None:
                    model_p.gnn.edge_logit_delta.zero_()
            model_p.gnn.set_hard_keep_ratio(None)
            model_p.gnn.configure_fixed_graph_phase2(
                bypass_gates=bool(flags.get(
                    "fixed_graph_bypass_gates",
                    CFG.PHASE2_FIXED_GRAPH_BYPASS_GATES,
                )),
                freeze_structure=bool(
                    CFG.PHASE2_FREEZE_STRUCTURAL_PARAMETERS
                ),
            )
            model_p.gnn.assert_fixed_graph_phase2()
        model_p.eval()
        return model_p

    base.load_state_dict(sd, strict=False)
    if hasattr(base, "edge_logit"):
        with torch.no_grad():
            base.edge_logit.data.fill_(20.0)
            if base.edge_logit_delta is not None:
                base.edge_logit_delta.zero_()
        base.set_hard_keep_ratio(None)
        base.configure_fixed_graph_phase2(
            bypass_gates=bool(flags.get(
                "fixed_graph_bypass_gates",
                CFG.PHASE2_FIXED_GRAPH_BYPASS_GATES,
            )),
            freeze_structure=bool(CFG.PHASE2_FREEZE_STRUCTURAL_PARAMETERS),
        )
        base.assert_fixed_graph_phase2()
    base.eval()
    return base


def _predict_bundle_on_indices(
    bundle_dir: str,
    indices,
    *,
    device=None,
) -> dict:
    """Run a frozen bundle on an explicitly supplied registered split.

    This private helper deliberately contains no logic that unlocks the
    external block.  Public callers below obtain either ``val_idx`` or
    ``test_idx`` from the already registered runtime; the external caller
    additionally verifies that the locked test indices are non-empty.
    """
    ExpressionDataset = _get_data_class()
    _, _, _, predict_proba_xh_mode = _get_training_fns()
    if device is None:
        device = CFG.DEVICE

    bundle = load_bundle(bundle_dir)
    meta = bundle.get("meta", {})
    g = bundle["graph"]

    nodes_used = np.asarray(g["nodes_used_full"], dtype=np.int64)
    X_full, genes_full = get_full_gene_matrix_and_genes()
    if list(map(str, genes_full)) != list(map(str, g["genes_full"])):
        raise RuntimeError(
            "El orden/universo de genes actual no coincide con el bundle congelado."
        )
    # V8-3. Debe coincidir exactamente con transport.py o la inferencia lee
    # columnas distintas de las del entrenamiento sin lanzar ningún error.
    X_comp = (
        np.hstack([X_full[:, nodes_used], X_full]).astype(np.float32)
        if bool(bundle.get("model", {}).get("model_flags", {}).get(
            "tab_full_gene_universe",
            getattr(CFG, "TAB_FULL_GENE_UNIVERSE", False),
        ))
        else X_full[:, nodes_used]
    )
    X_h = _rt("X_h")
    y = _rt("y")

    selected_idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if selected_idx.size == 0:
        raise ValueError("La partición solicitada está vacía.")
    if np.any(selected_idx < 0) or np.any(selected_idx >= len(y)):
        raise IndexError("La partición solicitada contiene índices inválidos.")

    selected_ds = ExpressionDataset(X_comp, X_h, y, selected_idx)
    selected_loader = torch.utils.data.DataLoader(
        selected_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model_p = build_pruned_model_from_bundle(bundle, device=device)
    xh_mode = str(
        meta.get("globals", {}).get(
            "CLASS_GRAPH_XH_MODE", CFG.CLASS_GRAPH_XH_MODE
        )
    )
    probability, y_true = predict_proba_xh_mode(
        model_p, None, selected_loader, device, xh_mode=xh_mode
    )
    cohort = np.asarray(_rt("cohort")).astype(str)
    return {
        "y_true": np.asarray(y_true, dtype=np.int64),
        "proba": np.asarray(probability, dtype=np.float32),
        "sample_idx": selected_idx,
        "cohort": cohort[selected_idx],
        "class_names": list(g.get("class_names", [])),
    }


def predict_bundle_on_validation(bundle_dir: str, device=None) -> dict:
    """Predict the fixed development-validation split with a frozen bundle."""
    val_idx = np.asarray(_rt("val_idx"), dtype=np.int64)
    prediction = _predict_bundle_on_indices(
        bundle_dir, val_idx, device=device
    )
    prediction["val_idx"] = prediction["sample_idx"]
    return prediction


def predict_bundle_on_test(bundle_dir: str, device=None) -> dict:
    """Predict the currently unlocked external block with a frozen bundle."""
    # Bundles are fitted with a runtime that never opened test, so the only
    # valid indices are those registered after notebook 07 verifies the lock.
    test_idx = np.asarray(_rt("test_idx"), dtype=np.int64)
    if test_idx.size == 0:
        raise RuntimeError(
            "El test externo sigue bloqueado. Ejecuta prepare_runtime("
            "include_external=True) para la evaluación predictiva del notebook 07."
        )
    prediction = _predict_bundle_on_indices(
        bundle_dir, test_idx, device=device
    )
    # Preserve the public key used by the locked notebook and older callers.
    prediction["test_idx"] = prediction["sample_idx"]
    return prediction


def evaluate_bundle_on_test(bundle_dir: str, device=None) -> dict:
    """Load a frozen bundle and score the currently unlocked external block."""
    compute_metrics = _get_metrics_fn()
    pred = predict_bundle_on_test(bundle_dir, device=device)
    return compute_metrics(pred["y_true"], pred["proba"])


# ── Run single seed ──────────────────────────────


STAB_CACHE: Dict[str, Any] = {}


def run_single_seed(
    cfg: AblationConfig,
    seed: int,
    *,
    save_root: Optional[str] = None,
    force_save: bool = False,
    evaluate_external: bool = False,
) -> Dict[str, Any]:
    _, _, build_pruned_model_from = _get_model_classes()
    ExpressionDataset = _get_data_class()
    train_graph_learning, pretrain_masked_gene_model, finetune_pruned, predict_proba_xh_mode = _get_training_fns()
    export_pruned_graph, export_pruned_graphs_by_class = _get_pruning_fns()
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
    class_names = class_names_from_label_map(label_map, n_classes)
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

    sample_weights_full = _RT.get("sample_weights", None)
    if sample_weights_full is not None:
        if len(sample_weights_full) != len(y):
            raise ValueError(
                "sample_weights debe tener un peso por muestra: "
                f"{len(sample_weights_full)} != {len(y)}"
            )
        class_weights_t = torch.ones(
            int(n_classes), dtype=torch.float32, device=CFG.DEVICE
        )
        cw_list = None
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        class_weights = _RT.get("class_weights", None)
        if class_weights is None:
            counts = np.bincount(
                y[train_idx], minlength=n_classes
            ).astype(np.float32)
            weights = counts.sum() / np.maximum(counts, 1.0)
            class_weights = (weights / weights.mean()).tolist()
            _RT["class_weights"] = class_weights

        if isinstance(class_weights, torch.Tensor):
            class_weights_t = class_weights.detach().to(
                device=CFG.DEVICE, dtype=torch.float32
            )
            cw_list = class_weights_t.detach().cpu().tolist()
        else:
            class_weights_t = torch.tensor(
                class_weights, dtype=torch.float32, device=CFG.DEVICE
            )
            cw_list = list(class_weights)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_t)

    # Modelo fase 1
    model = build_model_from_cfg(cfg, class_weights_t)
    core = getattr(model, "gnn", model)
    class_keep_ratio = resolve_class_keep_ratio(
        core.edge_index, fallback_ratio=CFG.KEEP_MIN
    )
    candidate_edges = int(
        (core.edge_index[0] != core.edge_index[1]).sum().item()
    )
    expected_edges = max(
        1, int(round(class_keep_ratio * candidate_edges))
    )
    budget_source = (
        "kappa"
        if CFG.CLASS_GRAPH_TARGET_EDGES is None
        else "override"
    )
    graph_accum_steps = (
        int(CFG.GRAPH_ACCUM_STEPS)
        if bool(getattr(core, "per_label_graphs", False))
        else int(CFG.ACCUM_STEPS)
    )
    print(
        "[class-graphs] "
        f"classes={class_names} | budget={budget_source} "
        f"| kappa={class_keep_ratio:.8f} "
        f"| candidates={candidate_edges} "
        f"| expected_edges_per_class≈{expected_edges} "
        f"| microbatch={CFG.GRAPH_BATCH_SIZE} "
        f"| accum={graph_accum_steps}"
    )

    if cfg.do_pretrain:
        model = pretrain_masked_gene_model(model, dl_tr, CFG.DEVICE)

    model, best_state, best_macro_f1 = train_graph_learning(
        model, adj=None, dl_tr=dl_tr, dl_va=dl_va,
        device=CFG.DEVICE, label_map=label_map,
        epochs=MAIN_EPOCHS, lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY,
        accum_steps=graph_accum_steps,
        class_weights=cw_list, class_weights_t=class_weights_t,
        edge_l1_per_edge=CFG.EDGE_L1_PER_EDGE,
        aux_lambda=CFG.AUX_LAMBDA, loss_fn=loss_fn,
        keep_min=class_keep_ratio,
        sample_weights_full=sample_weights_full,
    )
    model.load_state_dict(best_state)
    cleanup_memory('after_load_best_state')

    class_graphs_single = export_pruned_graphs_by_class(
        model,
        keep_ratio=class_keep_ratio,
        class_names=class_names,
        keep_self_loops=False,
    )
    selected_edges = {
        class_names[class_id]: int(graph["kept"])
        for class_id, graph in class_graphs_single.items()
    }
    if CFG.CLASS_GRAPH_TARGET_EDGES is not None:
        absolute_target = int(CFG.CLASS_GRAPH_TARGET_EDGES)
        if any(size != absolute_target for size in selected_edges.values()):
            raise RuntimeError(
                f"El override K={absolute_target} no coincide con la poda "
                f"exportada {selected_edges}. Si el backbone está en pares "
                "bidireccionales, usa un K par compatible."
            )
    print(f"[class-graphs] selected_edges={selected_edges}")
    edge_sets_single = {
        class_id: edge_set_from_edge_index(graph["edge_index"])
        for class_id, graph in class_graphs_single.items()
    }

    # ---- Stability selection ----
    stab = None
    class_graphs_final = class_graphs_single

    if cfg.do_stability:
        global STAB_CACHE
        # Comportamiento histórico: cada semilla externa construye SU propio
        # consenso Exact-K con ocho Phase 1 internas. Los cinco consensos no
        # se comparten ni se recuperan de una caché común.
        stab_seeds_run = [
            int(seed) * 1000 + int(offset) for offset in STAB_SEEDS
        ]
        if len(stab_seeds_run) != int(CFG.STAB_RUNS) or (
            len(set(stab_seeds_run)) != int(CFG.STAB_RUNS)
        ):
            raise RuntimeError(
                "Cada semilla externa requiere exactamente ocho semillas "
                f"internas distintas; recibidas {stab_seeds_run}."
            )
        cache_key = (
            cfg.name,
            int(seed),
            tuple(stab_seeds_run),
            round(class_keep_ratio, 12),
            int(CFG.STAB_EPOCHS),
            float(CFG.STAB_FREQ_THR),
            float(CFG.STAB_STRICT_FREQ_THR),
            bool(CFG.CLASS_GRAPH_STABILITY_FIXED_BUDGET),
            (
                None
                if CFG.CLASS_GRAPH_TARGET_EDGES is None
                else int(CFG.CLASS_GRAPH_TARGET_EDGES)
            ),
        )
        if cache_key in STAB_CACHE:
            stab = STAB_CACHE[cache_key]
        else:
            stab_dir = None
            if (cfg.name in SAVE_STABILITY_CFGS) or (cfg.name in SAVE_BEST_ONLY_CFGS) or (cfg.name in SAVE_CFGS) or SAVE_ALL_CONFIGS:
                stab_dir = os.path.join(
                    save_root, cfg.name, f"seed_{seed}", "stability_inner"
                )
            stab = stability_edge_sets(
                cfg=cfg, seeds=stab_seeds_run, keep_ratio=class_keep_ratio,
                epochs=CFG.STAB_EPOCHS, class_weights_t=class_weights_t,
                verbose=False, save_dir=stab_dir,
            )
            STAB_CACHE[cache_key] = stab

        consensus_by_class = {}
        for class_id in range(int(n_classes)):
            consensus = set(
                stab.get("consensus_edges_by_class", {}).get(class_id, set())
            )
            if not consensus:
                raise RuntimeError(
                    "El consenso Exact-K interno está vacío para la clase "
                    f"{class_names[class_id]}; se aborta en lugar de sustituirlo "
                    "por el TopK de la corrida portadora."
                )
            consensus_by_class[class_id] = consensus
        class_graphs_final = class_graphs_from_edge_sets(
            consensus_by_class,
            cfg,
            class_names,
            selection_frequency_by_class={
                int(class_id): details.get("selection_frequency", {})
                for class_id, details in stab.get("per_class", {}).items()
            },
            selection_rank_by_class={
                int(class_id): details.get("selection_median_rank", {})
                for class_id, details in stab.get("per_class", {}).items()
            },
            selection_score_by_class={
                int(class_id): details.get("selection_median_score", {})
                for class_id, details in stab.get("per_class", {}).items()
            },
        )

    ei_use, ew_use, et_use, class_edge_ids_union = (
        build_union_class_graph(class_graphs_final, n_classes)
    )

    # ---- Compactación ----
    nodes_used = torch.unique(ei_use.reshape(-1)).detach().cpu().numpy().astype(np.int64)
    mask = np.zeros(X_full.shape[1], dtype=bool)
    mask[nodes_used] = True

    X_comp = X_full[:, mask]
    genes_comp = [g for g, m in zip(genes_full, mask) if m]

    old2new = -np.ones(X_full.shape[1], dtype=np.int64)
    old2new[nodes_used] = np.arange(len(nodes_used), dtype=np.int64)

    ei_cpu = ei_use.detach().cpu().numpy()
    ei_comp_t = torch.tensor(
        np.vstack([old2new[ei_cpu[0]], old2new[ei_cpu[1]]]),
        dtype=torch.long,
        device=CFG.DEVICE,
    )

    ew_comp_t = ew_use if isinstance(ew_use, torch.Tensor) else (torch.tensor(ew_use, dtype=torch.float32, device=CFG.DEVICE) if ew_use is not None else None)
    et_comp_t = et_use if isinstance(et_use, torch.Tensor) else (torch.tensor(et_use, dtype=torch.long, device=CFG.DEVICE) if et_use is not None else None)
    if isinstance(ew_comp_t, torch.Tensor):
        ew_comp_t = ew_comp_t.to(device=CFG.DEVICE)
    if isinstance(et_comp_t, torch.Tensor):
        et_comp_t = et_comp_t.to(device=CFG.DEVICE)
    class_edge_ids_comp = [
        ids.to(device=CFG.DEVICE, dtype=torch.long)
        for ids in class_edge_ids_union
    ]

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
        class_edge_ids=class_edge_ids_comp,
        class_names=class_names,
        per_label_graphs=bool(CFG.PER_LABEL_GRAPHS),
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

    loss_fn_phase2 = (
        torch.nn.CrossEntropyLoss()
        if sample_weights_full is not None
        else torch.nn.CrossEntropyLoss(weight=class_weights_t)
    )

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
        sample_weights_full=sample_weights_full,
        xh_mode_B=CFG.CLASS_GRAPH_XH_MODE,
    )

    # El test externo permanece bloqueado salvo petición explícita del último
    # notebook. La ejecución ordinaria informa la validación de desarrollo.
    if evaluate_external:
        proba_eval, y_eval = predict_proba_xh_mode(
            model_p, None, dl_te2, CFG.DEVICE,
            xh_mode=CFG.CLASS_GRAPH_XH_MODE,
        )
        evaluation_split = "external_test"
        cleanup_memory('after_external_test_predict')
    else:
        proba_eval, y_eval = predict_proba_xh_mode(
            model_p, None, dl_va2, CFG.DEVICE,
            xh_mode=CFG.CLASS_GRAPH_XH_MODE,
        )
        evaluation_split = "development_validation"
    met = compute_metrics(y_eval, proba_eval)
    from .training import reported_class_metrics
    report_label = str(CFG.TRAIN_REPORT_LABEL)
    report_met = reported_class_metrics(
        y_eval, proba_eval, label_map, report_label
    )
    safe_report_label = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in report_label
    )
    met.update({
        f"precision_{safe_report_label}": float(report_met["precision"]),
        f"recall_{safe_report_label}": float(report_met["recall"]),
        f"f1_{safe_report_label}": float(report_met["f1"]),
        f"auc_{safe_report_label}": float(report_met["auc"]),
        f"support_{safe_report_label}": int(report_met["support"]),
    })

    edge_sets_final = {
        class_id: edge_set_from_edge_index(graph["edge_index"])
        for class_id, graph in class_graphs_final.items()
    }
    single_union = set().union(*edge_sets_single.values())
    final_union = set().union(*edge_sets_final.values())
    single_vs_final = {
        class_id: jaccard(
            edge_sets_single[class_id], edge_sets_final[class_id]
        )
        for class_id in range(int(n_classes))
    }
    out = {
        "cfg": cfg.name, "seed": seed,
        "evaluation_split": evaluation_split,
        "val_macro_f1_phase1": float(best_macro_f1),
        "n_edges_final": int(ei_use.shape[1]),
        "n_edges_single": int(len(single_union)),
        "n_nodes_compact": int(X_comp.shape[1]),
        "use_signed_prior": cfg.use_signed_prior,
        "class_graph_keep_ratio": float(class_keep_ratio),
        "class_graph_budget_source": budget_source,
        "class_graph_candidate_edges": int(candidate_edges),
        "class_graph_expected_edges_per_class": int(expected_edges),
        "jaccard_single_vs_final": float(
            np.mean(list(single_vs_final.values()))
        ),
        **met,
    }
    for class_id, class_name in enumerate(class_names):
        safe_name = "".join(
            ch if ch.isalnum() or ch in "-_" else "_"
            for ch in class_name
        )
        out[f"n_edges_{safe_name}"] = int(
            len(edge_sets_final[class_id])
        )
        out[f"n_edges_single_{safe_name}"] = int(
            len(edge_sets_single[class_id])
        )
        out[f"jaccard_single_vs_final_{safe_name}"] = float(
            single_vs_final[class_id]
        )

    if stab is not None:
        out["stab_inner_runs"] = int(stab.get("n_runs", 0))
        out["stab_inner_seeds"] = [
            int(value) for value in stab.get("seeds", [])
        ]
        out["stab_mean_jaccard"] = float(stab.get("jaccard_mean", float("nan")))
        out["stab_std_jaccard"]  = float(stab.get("jaccard_std", float("nan")))
        out["stab_min_jaccard"]  = float(stab.get("jaccard_min", float("nan")))
        out["stab_max_jaccard"]  = float(stab.get("jaccard_max", float("nan")))
        out["stab_n_pairs"]      = int(stab.get("jaccard_n_pairs", 0))
        out["stab_consensus_size"] = int(stab.get("consensus_size", 0))
        tcs = stab.get("to_consensus_stats", None)
        out["stab_to_cons_mean"] = float(tcs.get("to_consensus_mean", float("nan"))) if isinstance(tcs, dict) else float("nan")
        out["stab_to_cons_std"]  = float(tcs.get("to_consensus_std", float("nan"))) if isinstance(tcs, dict) else float("nan")
        for class_id, details in stab.get("per_class", {}).items():
            class_name = class_names[int(class_id)]
            safe_name = "".join(
                ch if ch.isalnum() or ch in "-_" else "_"
                for ch in class_name
            )
            js = details.get("jaccard_stats", {})
            out[f"stab_mean_jaccard_{safe_name}"] = float(
                js.get("jaccard_mean", float("nan"))
            )
            out[f"stab_consensus_size_{safe_name}"] = int(
                details.get("consensus_size", 0)
            )
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
                class_graphs_full=class_graphs_final,
                class_edge_ids_comp=class_edge_ids_comp,
                class_names=class_names,
                model_pruned=model_p, metrics=out, stab=stab, phase1_best_state=best_state,
                proba_te=(proba_eval if evaluate_external else None),
                y_te=(y_eval if evaluate_external else None),
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
                    class_graphs_full=class_graphs_final,
                    class_edge_ids_comp=class_edge_ids_comp,
                    class_names=class_names,
                    model_pruned=model_p, metrics=out, stab=stab, phase1_best_state=best_state,
                    proba_te=(proba_eval if evaluate_external else None),
                    y_te=(y_eval if evaluate_external else None),
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


# ── Phase 1 only: component/backbone ablations ──────────────────────

def _json_scalar(value):
    """Convierte escalares NumPy para los metadatos JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_phase1_bundle(
    out_dir: str,
    *,
    cfg: "AblationConfig",
    seed: int,
    genes_full: list,
    class_graphs: Dict[int, Dict[str, Any]],
    class_names: List[str],
    best_state: dict,
    metrics: dict,
    proba_te=None,
    y_te=None,
):
    """Guarda exclusivamente el checkpoint y los grafos de Phase 1.

    No crea ``model_pruned.pt`` ni ningún artefacto de Phase 2. Las aristas se
    conservan en coordenadas del universo completo de genes.
    """
    out_dir = _mkdir(out_dir)
    edge_sets = {
        int(class_id): set(zip(
            graph["edge_index"][0].detach().cpu().tolist(),
            graph["edge_index"][1].detach().cpu().tolist(),
        ))
        for class_id, graph in class_graphs.items()
    }
    union_edges = set().union(*edge_sets.values()) if edge_sets else set()
    nodes_used = sorted({node for edge in union_edges for node in edge})

    meta = cfg_to_dict(cfg)
    meta.update({
        "cfg_name": str(cfg.name),
        "seed": int(seed),
        "evaluation_phase": "phase1_only",
        "phase2_used": False,
        "hybrid_tabular_used": False,
        "stability_used": False,
        "n_edges_phase1_union": int(len(union_edges)),
        "n_nodes_phase1_graph": int(len(nodes_used)),
        "metrics": {
            str(k): _json_scalar(v) for k, v in metrics.items()
        },
    })
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    state_cpu = {
        k: v.detach().cpu().clone() for k, v in best_state.items()
    }
    torch.save(state_cpu, os.path.join(out_dir, "phase1_best_state.pt"))

    class_graphs_cpu: Dict[int, Dict[str, Any]] = {}
    edge_source_name = {
        0: "HuRI",
        1: "OmniPath_positive",
        2: "OmniPath_negative",
        3: "STRING",
    }
    for class_id, graph in class_graphs.items():
        saved_graph = {
            "class_id": int(class_id),
            "class_name": str(graph["class_name"]),
            "edge_index": graph["edge_index"].detach().cpu(),
            "edge_weight": graph["edge_weight"].detach().cpu(),
            "edge_type": graph["edge_type"].detach().cpu(),
        }
        for field_name in (
            "candidate_edge_ids", "learned_logit", "learned_gate",
            "selection_frequency",
        ):
            value = graph.get(field_name, None)
            if isinstance(value, torch.Tensor):
                saved_graph[field_name] = value.detach().cpu()
        class_graphs_cpu[int(class_id)] = saved_graph

        safe_name = "".join(
            ch if ch.isalnum() or ch in "-_" else "_"
            for ch in str(saved_graph["class_name"])
        )
        torch.save(
            saved_graph, os.path.join(out_dir, f"graph_{safe_name}.pt")
        )
        ei_np = saved_graph["edge_index"].numpy()
        ew_np = saved_graph["edge_weight"].numpy()
        et_np = saved_graph["edge_type"].numpy()
        n_edges = int(ei_np.shape[1])
        candidate_ids = saved_graph.get(
            "candidate_edge_ids",
            torch.full((n_edges,), -1, dtype=torch.long),
        ).numpy()
        learned_logit = saved_graph.get(
            "learned_logit", torch.full((n_edges,), float("nan"))
        ).numpy()
        learned_gate = saved_graph.get(
            "learned_gate", torch.full((n_edges,), float("nan"))
        ).numpy()
        edge_table = pd.DataFrame({
            "class_id": int(class_id),
            "class_name": str(saved_graph["class_name"]),
            "source_idx_full": ei_np[0].astype(np.int64),
            "target_idx_full": ei_np[1].astype(np.int64),
            "source_gene": [genes_full[int(i)] for i in ei_np[0]],
            "target_gene": [genes_full[int(i)] for i in ei_np[1]],
            "prior_weight": ew_np.astype(float),
            "edge_type": et_np.astype(np.int64),
            "candidate_edge_id": candidate_ids.astype(np.int64),
            "learned_logit": learned_logit.astype(float),
            "learned_gate": learned_gate.astype(float),
            "edge_source": [
                edge_source_name.get(int(t), f"type_{int(t)}")
                for t in et_np
            ],
        })
        edge_table.to_csv(
            os.path.join(out_dir, f"edges_{safe_name}.csv"), index=False
        )

    ei_union, ew_union, et_union, class_edge_ids = (
        build_union_class_graph(class_graphs_cpu, len(class_names))
    )
    torch.save({
        "genes_full": list(genes_full),
        "nodes_used_full": torch.tensor(nodes_used, dtype=torch.long),
        "class_names": list(class_names),
        "class_graphs_full": class_graphs_cpu,
        "edge_index_union_full": ei_union.detach().cpu(),
        "edge_weight_union_full": ew_union.detach().cpu(),
        "edge_type_union_full": et_union.detach().cpu(),
        "class_edge_ids_union": [ids.detach().cpu() for ids in class_edge_ids],
    }, os.path.join(out_dir, "phase1_graph_bundle.pt"))

    if proba_te is not None and y_te is not None:
        np.savez_compressed(
            os.path.join(out_dir, "phase1_test_predictions.npz"),
            proba=np.asarray(proba_te, dtype=np.float32),
            y_true=np.asarray(y_te).reshape(-1).astype(np.int64),
            class_names=np.asarray(class_names, dtype=object),
        )


def run_single_seed_phase1_only(
    cfg: AblationConfig,
    seed: int,
    *,
    save_root: Optional[str] = None,
    force_save: bool = True,
    evaluate_external: bool = False,
) -> Dict[str, Any]:
    """Entrena y evalúa directamente Phase 1; nunca construye Phase 2.

    El checkpoint se selecciona solo con validación. Después de restaurarlo se
    recalcula el TopK objetivo, porque la máscara dura es estado Python y no
    forma parte de ``state_dict``. Por defecto informa solo validación de
    desarrollo; el test externo permanece bloqueado hasta el notebook final.
    """
    if bool(cfg.do_stability):
        raise ValueError(
            "run_single_seed_phase1_only requiere do_stability=False. "
            "La estabilidad es un procedimiento posterior al entrenamiento "
            "principal y no pertenece a esta comparación directa."
        )

    from .training import (
        train_graph_learning, predict_proba_xh_mode,
        reported_class_metrics,
    )
    from .pruning import export_pruned_graphs_by_class
    from .stability import compute_metrics, edge_set_from_edge_index, jaccard

    set_all_seeds(int(seed))
    X_full, genes_full = get_full_gene_matrix_and_genes()
    dl_tr, dl_va, dl_te = make_loaders(X_full)
    cleanup_memory("after_make_loaders_phase1_only")

    y = _rt("y")
    n_classes = int(_rt("n_classes"))
    label_map = _rt("label_map")
    class_names = class_names_from_label_map(label_map, n_classes)
    train_idx = np.asarray(_rt("train_idx"), dtype=np.int64)

    if save_root is None:
        save_root = ARTIFACTS_ROOT
    do_save_seed = bool(
        force_save or SAVE_ALL_CONFIGS or (cfg.name in SAVE_CFGS)
    )
    run_dir = (
        os.path.join(save_root, cfg.name, f"seed_{int(seed)}")
        if do_save_seed else None
    )
    if run_dir is not None:
        _mkdir(run_dir)

    sample_weights_full = _RT.get("sample_weights", None)
    if sample_weights_full is not None:
        if len(sample_weights_full) != len(y):
            raise ValueError(
                "sample_weights debe tener un peso por muestra: "
                f"{len(sample_weights_full)} != {len(y)}"
            )
        class_weights_t = torch.ones(
            n_classes, dtype=torch.float32, device=CFG.DEVICE
        )
        class_weights_list = None
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        counts = np.bincount(
            y[train_idx], minlength=n_classes
        ).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        class_weights_list = weights.tolist()
        class_weights_t = torch.tensor(
            class_weights_list, dtype=torch.float32, device=CFG.DEVICE
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_t)

    model = build_model_from_cfg(cfg, class_weights_t)
    core = getattr(model, "gnn", model)
    actual_connectivity = bool(getattr(
        core, "add_connectivity_penalty", CFG.ADD_CONNECTIVITY_PENALTY
    ))
    if actual_connectivity != bool(cfg.connectivity_penalty):
        raise RuntimeError(
            f"El modelo no respetó connectivity_penalty para {cfg.name}: "
            f"cfg={cfg.connectivity_penalty}, modelo={actual_connectivity}."
        )

    class_keep_ratio = resolve_class_keep_ratio(
        core.edge_index, fallback_ratio=CFG.KEEP_MIN
    )
    candidate_edges = int(
        (core.edge_index[0] != core.edge_index[1]).sum().item()
    )
    expected_edges = max(
        1, int(round(class_keep_ratio * candidate_edges))
    )
    graph_accum_steps = (
        int(CFG.GRAPH_ACCUM_STEPS)
        if bool(getattr(core, "per_label_graphs", False))
        else int(CFG.ACCUM_STEPS)
    )
    print(
        "[Phase1-only] "
        f"classes={class_names} | kappa={class_keep_ratio:.8f} | "
        f"expected_edges_per_class≈{expected_edges} | "
        f"connectivity={actual_connectivity}"
    )

    if bool(cfg.do_pretrain):
        from .training import pretrain_masked_gene_model
        model = pretrain_masked_gene_model(model, dl_tr, CFG.DEVICE)

    model, best_state, best_val_during_schedule = train_graph_learning(
        model, adj=None, dl_tr=dl_tr, dl_va=dl_va,
        device=CFG.DEVICE, label_map=label_map,
        epochs=MAIN_EPOCHS, lr=CFG.LR,
        weight_decay=CFG.WEIGHT_DECAY,
        accum_steps=graph_accum_steps,
        class_weights=class_weights_list,
        class_weights_t=class_weights_t,
        edge_l1_per_edge=CFG.EDGE_L1_PER_EDGE,
        aux_lambda=CFG.AUX_LAMBDA,
        loss_fn=loss_fn,
        keep_min=class_keep_ratio,
        sample_weights_full=sample_weights_full,
    )

    model.load_state_dict(best_state)
    model.set_hard_keep_ratio(class_keep_ratio)
    core = getattr(model, "gnn", model)
    graph_xh_mode = str(CFG.GRAPH_LEARNING_XH_MODE).lower()
    forward_xh_mode = "zero" if graph_xh_mode == "zero" else "orig"
    core.use_xgraph_in_head = (graph_xh_mode == "orig")
    if graph_xh_mode == "gate_only" and bool(
        getattr(core, "use_xgraph_in_head", True)
    ):
        raise RuntimeError("gate_only no debe introducir X_h en el clasificador.")

    proba_va, y_va = predict_proba_xh_mode(
        model, None, dl_va, CFG.DEVICE, xh_mode=forward_xh_mode
    )
    val_metrics = compute_metrics(y_va, proba_va)
    if evaluate_external:
        proba_eval, y_eval = predict_proba_xh_mode(
            model, None, dl_te, CFG.DEVICE, xh_mode=forward_xh_mode
        )
        evaluation_split = "external_test"
    else:
        proba_eval, y_eval = proba_va, y_va
        evaluation_split = "development_validation"
    eval_metrics = compute_metrics(y_eval, proba_eval)

    class_graphs = export_pruned_graphs_by_class(
        model,
        keep_ratio=class_keep_ratio,
        class_names=class_names,
        keep_self_loops=False,
    )
    edge_sets = {
        class_id: edge_set_from_edge_index(graph["edge_index"])
        for class_id, graph in class_graphs.items()
    }
    union_edges = set().union(*edge_sets.values())
    nodes_used = {node for edge in union_edges for node in edge}
    report_label = str(CFG.TRAIN_REPORT_LABEL)
    report = reported_class_metrics(
        y_eval, proba_eval, label_map, report_label
    )
    safe_report_label = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in report_label
    )
    row: Dict[str, Any] = {
        "cfg": str(cfg.name),
        "seed": int(seed),
        "protocol_version": PHASE1_COMPONENT_ABLATION_PROTOCOL,
        "evaluation_phase": "phase1_only",
        "evaluation_split": evaluation_split,
        "phase2_used": False,
        "hybrid_tabular_used": False,
        "stability_used": False,
        "connectivity_penalty": bool(cfg.connectivity_penalty),
        "connectivity_lambda_effective": (
            float(CFG.CONNECTIVITY_LAMBDA)
            if cfg.connectivity_penalty else 0.0
        ),
        "edge_type_gating": bool(cfg.edge_type_gating),
        "use_signed_prior": bool(cfg.use_signed_prior),
        "signed_channels_mode": str(cfg.signed_channels_mode),
        "graph_learning_xh_mode": graph_xh_mode,
        "class_graph_keep_ratio": float(class_keep_ratio),
        "class_graph_candidate_edges": int(candidate_edges),
        "class_graph_expected_edges_per_class": int(expected_edges),
        "best_val_select_metric_during_schedule": float(
            best_val_during_schedule
        ),
        "val_macro_f1_phase1": float(val_metrics["macro_f1"]),
        "val_acc_phase1": float(val_metrics["acc"]),
        "val_auc_ovr_macro_phase1": float(
            val_metrics["auc_ovr_macro"]
        ),
        "n_edges_final": int(len(union_edges)),
        "n_edges_phase1_union": int(len(union_edges)),
        "n_nodes_compact": int(len(nodes_used)),
        "n_nodes_phase1_graph": int(len(nodes_used)),
        **{k: float(v) for k, v in eval_metrics.items()},
        f"precision_{safe_report_label}": float(report["precision"]),
        f"recall_{safe_report_label}": float(report["recall"]),
        f"f1_{safe_report_label}": float(report["f1"]),
        f"auc_{safe_report_label}": float(report["auc"]),
        f"support_{safe_report_label}": int(report["support"]),
    }
    for class_id, class_name in enumerate(class_names):
        safe_name = "".join(
            ch if ch.isalnum() or ch in "-_" else "_"
            for ch in class_name
        )
        row[f"n_edges_{safe_name}"] = int(len(edge_sets[class_id]))

    if run_dir is not None:
        save_phase1_bundle(
            run_dir,
            cfg=cfg,
            seed=int(seed),
            genes_full=genes_full,
            class_graphs=class_graphs,
            class_names=class_names,
            best_state=best_state,
            metrics=row,
            proba_te=(proba_eval if evaluate_external else None),
            y_te=(y_eval if evaluate_external else None),
        )
        print(f"[SAVE Phase1-only] {run_dir}")

    del model
    cleanup_memory("after_phase1_only_seed")
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    return row


def run_ablation_phase1_only(
    configs: List[AblationConfig],
    seeds: List[int],
    *,
    save_root: Optional[str] = None,
    resume: bool = True,
    force_save: bool = True,
    evaluate_external: bool = False,
) -> pd.DataFrame:
    """Ejecuta una ablación completa sin ninguna ruta de Phase 2."""
    if any(bool(cfg.do_stability) for cfg in configs):
        raise ValueError(
            "Todas las configuraciones deben usar do_stability=False en "
            "run_ablation_phase1_only."
        )
    if save_root is None:
        save_root = os.path.join(ARTIFACTS_ROOT, "phase1_only")
    _mkdir(save_root)
    progress_path = os.path.join(
        save_root, "component_ablation_phase1_only_results.csv"
    )

    config_names = [str(cfg.name) for cfg in configs]
    seed_values = [int(seed) for seed in seeds]
    rows: List[Dict[str, Any]] = []
    if bool(resume) and os.path.exists(progress_path):
        previous = pd.read_csv(progress_path)
        required = {"cfg", "seed", "evaluation_phase"}
        if not required.issubset(previous.columns):
            raise ValueError(
                f"El CSV de progreso no es Phase1-only: {progress_path}"
            )
        if "protocol_version" not in previous.columns:
            print(
                "[resume] Se ignora el progreso antiguo: no identifica la "
                "definición ortogonal de NO_SIGN."
            )
            previous = previous.iloc[0:0].copy()
        else:
            previous = previous[
                previous["protocol_version"].astype(str).eq(
                    PHASE1_COMPONENT_ABLATION_PROTOCOL
                )
            ].copy()
        previous = previous[
            previous["cfg"].astype(str).isin(config_names)
            & previous["seed"].astype(int).isin(seed_values)
        ].copy()
        if not previous.empty and not previous[
            "evaluation_phase"
        ].astype(str).eq("phase1_only").all():
            raise ValueError(
                "El CSV de progreso mezcla resultados de otra fase. "
                "Usa un directorio limpio."
            )
        previous = previous.drop_duplicates(
            subset=["cfg", "seed"], keep="last"
        )
        rows = previous.to_dict("records")

    completed = {
        (str(row["cfg"]), int(row["seed"])) for row in rows
    }
    t0 = time.time()
    for cfg in configs:
        print(f"\n=== {cfg.name} | PHASE 1 ONLY ===")
        for seed in seed_values:
            key = (str(cfg.name), int(seed))
            if key in completed:
                print(f"  - seed {seed}: ya completada")
                continue
            print(f"  - seed {seed}")
            result = run_single_seed_phase1_only(
                cfg,
                int(seed),
                save_root=save_root,
                force_save=force_save,
                evaluate_external=evaluate_external,
            )
            rows.append(result)
            completed.add(key)
            pd.DataFrame(rows).to_csv(progress_path, index=False)
            print(
                f"    {result['evaluation_split']} Phase1 "
                f"macroF1={result['macro_f1']:.4f} "
                f"AUC={result['auc_ovr_macro']:.4f} "
                f"edges={result['n_edges_final']} "
                f"nodes={result['n_nodes_phase1_graph']}"
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        cfg_order = {name: i for i, name in enumerate(config_names)}
        seed_order = {seed: i for i, seed in enumerate(seed_values)}
        df["_cfg_order"] = df["cfg"].map(cfg_order)
        df["_seed_order"] = df["seed"].astype(int).map(seed_order)
        df = (
            df.sort_values(["_cfg_order", "_seed_order"])
            .drop(columns=["_cfg_order", "_seed_order"])
            .reset_index(drop=True)
        )
        df.to_csv(progress_path, index=False)

    protocol = {
        "evaluation_phase": "phase1_only",
        "protocol_version": PHASE1_COMPONENT_ABLATION_PROTOCOL,
        "phase2_used": False,
        "hybrid_tabular_used": False,
        "stability_used": False,
        "evaluate_external": bool(evaluate_external),
        "configs": [asdict(cfg) for cfg in configs],
        "seeds": seed_values,
        "results_csv": os.path.basename(progress_path),
    }
    with open(
        os.path.join(save_root, "phase1_only_protocol.json"),
        "w", encoding="utf-8",
    ) as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    print(f"\nDONE Phase1-only in {(time.time() - t0) / 60:.1f} min")
    return df


# Alias cortos para notebooks que prefieran el nombre sin sufijo.
run_single_seed_phase1 = run_single_seed_phase1_only
run_ablation_phase1 = run_ablation_phase1_only


# ── Run ablation ─────────────────────────────────


def run_ablation(configs: List[AblationConfig], seeds: List[int], *,
                 save_root: Optional[str] = None,
                 evaluate_external: bool = False) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    if save_root is None:
        save_root = ARTIFACTS_ROOT
    _mkdir(save_root)

    for cfg in configs:
        print(f"\n=== {cfg.name} ===")
        for seed in seeds:
            print(f"  - seed {seed}")
            r = run_single_seed(
                cfg, seed, save_root=save_root, force_save=False,
                evaluate_external=evaluate_external,
            )
            rows.append(r)
            cleanup_memory('after_seed')
            report_label = str(CFG.TRAIN_REPORT_LABEL)
            safe_label = "".join(
                ch if ch.isalnum() or ch in "-_" else "_"
                for ch in report_label
            )
            precision_key = f"precision_{safe_label}"
            recall_key = f"recall_{safe_label}"
            f1_key = f"f1_{safe_label}"
            auc_key = f"auc_{safe_label}"
            print(
                f"    {report_label}: "
                f"P={r[precision_key]:.4f} "
                f"R={r[recall_key]:.4f} "
                f"F1={r[f1_key]:.4f} "
                f"AUC={r[auc_key]:.4f}  "
                f"edges={r['n_edges_final']} "
                f"nodes={r['n_nodes_compact']}"
            )

    df = pd.DataFrame(rows)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    return df
