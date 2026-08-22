"""Biology-first cohort-transport learning for InterGATE V6.

Every meta-episode holds out one development cohort and uses the same fixed
number of class/cohort-balanced updates. Structural parameters are adapted on
the remaining cohorts and evaluated on the held-out cohort; the external
block is never loaded. A train-only differential-correlation meta-analysis
provides weak contextual supervision without consuming PAM50, curated subtype
signatures, DepMap or external labels.

The selected checkpoint is audited with multiple leave-one-development-
cohort influence steps. The audit exports parameter deltas, logit changes,
rank correlations and TopK turnover, so an unchanged support result cannot be
misreported as evidence of transport. V6 exports two separate Exact-K objects:
the total-score graph used by the classifier and a centered-differential,
module-aware association graph used for biological interpretation. Neither is
claimed to be a de-novo or causal network.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:  # PyTorch >= 2.0
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover - compatibility with older PyTorch
    from torch.nn.utils.stateless import functional_call as _functional_call

from .config import CFG
from .data import ExpressionDataset
from .losses import compute_metrics_full
from .training import (
    _connectivity_options,
    keep_schedule,
    per_label_param_groups,
    predict_proba_xh_mode,
)
from .utils import cleanup_memory, set_all_seeds
from .biology import (
    combine_biology_scores,
    module_aware_exact_k,
    weak_topology_summary,
)
from .ablation import (
    CFG_FULL,
    _RT,
    _rt,
    build_model_from_cfg,
    build_union_class_graph,
    class_names_from_label_map,
    resolve_class_keep_ratio,
    save_bundle,
)


TRANSPORT_ARTIFACT_NAME = "INTERGATE_V6"


@dataclass(frozen=True)
class TransportConfig:
    """Immutable settings recorded in every InterGATE V6 artifact."""

    protocol_version: str
    episodes: int
    inner_lr: float
    first_order: bool
    meta_train_weight: float
    meta_validation_weight: float
    topology_lambda: float
    soft_topk_tau: float
    quota_end_fraction: float
    patience: int
    balanced_sampling: bool
    steps_per_episode: int
    support_batches: int
    support_steps: int
    min_cohort_samples: int
    checkpoint_metric: str
    biology_context_lambda: float
    biology_context_warmup_episodes: int
    biology_score_context_weight: float
    biology_score_support_weight: float

    @classmethod
    def from_global(cls, cfg=CFG) -> "TransportConfig":
        return cls(
            protocol_version=str(cfg.TRANSPORT_PROTOCOL_VERSION),
            episodes=int(cfg.TRANSPORT_EPISODES),
            inner_lr=float(cfg.TRANSPORT_INNER_LR),
            first_order=bool(cfg.TRANSPORT_FIRST_ORDER),
            meta_train_weight=float(cfg.TRANSPORT_META_TRAIN_WEIGHT),
            meta_validation_weight=float(
                cfg.TRANSPORT_META_VALIDATION_WEIGHT
            ),
            topology_lambda=float(cfg.TRANSPORT_TOPOLOGY_LAMBDA),
            soft_topk_tau=float(cfg.TRANSPORT_SOFT_TOPK_TAU),
            quota_end_fraction=float(cfg.TRANSPORT_QUOTA_END_FRACTION),
            patience=int(cfg.TRANSPORT_PATIENCE),
            balanced_sampling=bool(cfg.TRANSPORT_BALANCED_SAMPLING),
            steps_per_episode=int(cfg.TRANSPORT_STEPS_PER_EPISODE),
            support_batches=int(cfg.TRANSPORT_SUPPORT_BATCHES),
            support_steps=int(cfg.TRANSPORT_SUPPORT_STEPS),
            min_cohort_samples=int(cfg.TRANSPORT_MIN_COHORT_SAMPLES),
            checkpoint_metric=str(cfg.TRANSPORT_CHECKPOINT_METRIC),
            biology_context_lambda=float(cfg.BIOLOGY_CONTEXT_LAMBDA),
            biology_context_warmup_episodes=int(
                cfg.BIOLOGY_CONTEXT_WARMUP_EPISODES
            ),
            biology_score_context_weight=float(
                cfg.BIOLOGY_SCORE_CONTEXT_WEIGHT
            ),
            biology_score_support_weight=float(
                cfg.BIOLOGY_SCORE_SUPPORT_WEIGHT
            ),
        )


@dataclass(frozen=True)
class StructuralBudget:
    """Exact units, with reciprocal directions paired only when exact.

    The legacy InterGATE pruning path recognizes the special layout in which
    the second half of the candidate list is exactly the reverse of the first
    half. InterGATE V6 preserves that layout atomically when it is genuinely
    present. A directed or differently ordered interactome instead uses one
    directed edge per structural unit; near-matches are never paired.
    """

    first_ids: torch.Tensor
    second_ids: Optional[torch.Tensor]
    candidate_ids: torch.Tensor
    k_units: int
    paired: bool

    @classmethod
    def from_core(cls, core: nn.Module, keep_ratio: float) -> "StructuralBudget":
        edge_index = core.edge_index
        loop_mask = edge_index[0].eq(edge_index[1])
        candidate_ids = torch.where(~loop_mask)[0]
        if candidate_ids.numel() == 0:
            raise ValueError("El interactoma no contiene aristas elegibles.")

        paired = False
        second_ids: Optional[torch.Tensor] = None
        first_ids = candidate_ids
        if int(candidate_ids.numel()) % 2 == 0:
            half = int(candidate_ids.numel()) // 2
            a = candidate_ids[:half]
            b = candidate_ids[half:]
            reciprocal = (
                edge_index[0, a].eq(edge_index[1, b])
                & edge_index[1, a].eq(edge_index[0, b])
            )
            paired = bool(torch.all(reciprocal).item())
            if paired:
                first_ids, second_ids = a, b

        n_units = int(first_ids.numel())
        k_units = max(1, int(round(float(keep_ratio) * n_units)))
        if k_units > n_units:
            raise ValueError(
                f"K={k_units} supera las {n_units} unidades estructurales."
            )
        return cls(
            first_ids=first_ids,
            second_ids=second_ids,
            candidate_ids=candidate_ids,
            k_units=k_units,
            paired=paired,
        )

    def unit_scores(self, edge_scores: torch.Tensor) -> torch.Tensor:
        """Map ``(..., E)`` edge scores to ``(..., U)`` structural units."""
        first_ids = self.first_ids.to(device=edge_scores.device)
        first = edge_scores.index_select(-1, first_ids)
        if self.second_ids is None:
            return first
        second_ids = self.second_ids.to(device=edge_scores.device)
        second = edge_scores.index_select(-1, second_ids)
        return torch.maximum(first, second)

    def expand_units(self, unit_ids: torch.Tensor) -> torch.Tensor:
        first_ids = self.first_ids.to(device=unit_ids.device)
        first = first_ids.index_select(0, unit_ids)
        if self.second_ids is None:
            return first
        second_ids = self.second_ids.to(device=unit_ids.device)
        second = second_ids.index_select(0, unit_ids)
        return torch.cat([first, second], dim=0)

    def hard_topk_units(self, unit_scores: torch.Tensor) -> torch.Tensor:
        """Stable deterministic TopK; candidate order breaks exact ties."""
        try:
            order = torch.argsort(
                unit_scores, descending=True, stable=True
            )
        except TypeError:  # pragma: no cover - old PyTorch fallback
            eps = torch.finfo(unit_scores.dtype).eps
            tie = torch.arange(
                unit_scores.numel(), device=unit_scores.device,
                dtype=unit_scores.dtype,
            )
            order = torch.argsort(
                unit_scores - eps * tie, descending=True
            )
        return order[: self.k_units]


def _core(model: nn.Module) -> nn.Module:
    return getattr(model, "gnn", model)


def _structural_named_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    core = _core(model)
    wanted = {
        id(value)
        for name in ("edge_logit", "edge_logit_delta", "type_scale", "type_bias")
        for value in (getattr(core, name, None),)
        if isinstance(value, nn.Parameter) and value.requires_grad
    }
    out = {
        name: parameter
        for name, parameter in model.named_parameters()
        if id(parameter) in wanted
    }
    if "edge_logit" not in {name.rsplit(".", 1)[-1] for name in out}:
        raise RuntimeError("No se encontró el parámetro estructural edge_logit.")
    return out


def _override_value(
    overrides: Mapping[str, torch.Tensor], core: nn.Module, attribute: str
) -> torch.Tensor:
    matches = [
        value for name, value in overrides.items()
        if name.rsplit(".", 1)[-1] == attribute
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Parámetro estructural ambiguo: {attribute}.")
    if matches:
        return matches[0]
    value = getattr(core, attribute)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{attribute} no es tensor.")
    return value


def class_edge_logits_from_overrides(
    model: nn.Module,
    overrides: Optional[Mapping[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Return class-by-edge structural logits for base or adapted parameters."""
    core = _core(model)
    values: Mapping[str, torch.Tensor] = overrides or {}
    edge_logit = _override_value(values, core, "edge_logit")
    if bool(getattr(core, "edge_type_gating", False)):
        edge_type = core.edge_type.to(edge_logit.device)
        type_scale = _override_value(values, core, "type_scale")
        type_bias = _override_value(values, core, "type_bias")
        base = edge_logit * type_scale[edge_type] + type_bias[edge_type]
    else:
        base = edge_logit

    delta = getattr(core, "edge_logit_delta", None)
    if bool(getattr(core, "per_label_graphs", False)) and delta is not None:
        delta = _override_value(values, core, "edge_logit_delta")
        if bool(getattr(CFG, "CENTER_CLASS_DELTAS", False)):
            delta = delta - delta.mean(dim=0, keepdim=True)
        return base.unsqueeze(0) + delta
    return base.unsqueeze(0)


def _soft_exact_k_membership(
    unit_scores: torch.Tensor,
    k_units: int,
    temperature: float,
) -> torch.Tensor:
    """Differentiable approximation to membership in the exact TopK set."""
    if unit_scores.ndim != 2:
        raise ValueError("unit_scores debe tener forma (clase, unidad).")
    k_units = int(k_units)
    if not 1 <= k_units <= unit_scores.shape[1]:
        raise ValueError("K fuera del universo estructural.")
    with torch.no_grad():
        threshold = torch.topk(
            unit_scores.detach(), k_units, dim=1, largest=True,
        ).values[:, -1]
    return torch.sigmoid(
        (unit_scores - threshold.unsqueeze(1)) / float(temperature)
    )


def soft_topk_jaccard_distance(
    base_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    budget: StructuralBudget,
    temperature: float,
) -> torch.Tensor:
    """Mean soft-Jaccard distance over class-specific structural units."""
    base = _soft_exact_k_membership(
        budget.unit_scores(base_logits), budget.k_units, temperature
    )
    adapted = _soft_exact_k_membership(
        budget.unit_scores(adapted_logits), budget.k_units, temperature
    )
    # Fuzzy-set Jaccard with min/max is exactly zero for identical soft sets;
    # the product t-norm would penalize even an unchanged non-binary ranking.
    intersection = torch.minimum(base, adapted).sum(dim=1)
    union = torch.maximum(base, adapted).sum(dim=1).clamp_min(1e-8)
    return (1.0 - intersection / union).mean()


def _functional_forward(
    model: nn.Module,
    parameter_overrides: Mapping[str, torch.Tensor],
    args: Tuple[Any, ...],
):
    """Functional forward with private copies of mutable BatchNorm buffers."""
    buffer_overrides = {
        name: value.detach().clone()
        for name, value in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    try:
        return _functional_call(
            model,
            (dict(parameter_overrides), buffer_overrides),
            args,
            strict=False,
        )
    except TypeError:  # pragma: no cover - stateless compatibility path
        merged = dict(parameter_overrides)
        merged.update(buffer_overrides)
        return _functional_call(model, merged, args)


def _move_batch(batch, device: torch.device):
    Xg, Xh, yb, sample_ids = batch
    Xg = torch.as_tensor(Xg, dtype=torch.float32, device=device)
    # The temporal contract is enforced here, rather than relying on the
    # caller to have zeroed X_h.
    Xh = torch.zeros_like(
        torch.as_tensor(Xh, dtype=torch.float32, device=device)
    )
    yb = torch.as_tensor(yb, dtype=torch.long, device=device)
    return Xg, Xh, yb, sample_ids


def _make_loader(
    X_gene: np.ndarray,
    X_h: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        ExpressionDataset(X_gene, X_h, y, np.asarray(indices, dtype=np.int64)),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=bool(torch.cuda.is_available()),
        drop_last=False,
        generator=generator,
    )


def _deterministic_stratified_prefix(
    indices: np.ndarray,
    cohort: Sequence[str],
    y: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """Round-robin prefix over cohort-by-class strata for support audits."""
    indices = np.asarray(indices, dtype=np.int64)
    cohort = np.asarray(cohort).astype(str)
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    strata: Dict[Tuple[str, int], List[int]] = {}
    for index in indices:
        key = (cohort[int(index)].strip().upper(), int(y[int(index)]))
        strata.setdefault(key, []).append(int(index))
    queues: Dict[Tuple[str, int], List[int]] = {}
    for key, values in strata.items():
        queues[key] = rng.permutation(values).astype(np.int64).tolist()
    ordered: List[int] = []
    keys = sorted(queues)
    while len(ordered) < int(max_samples):
        added = False
        for key in keys:
            if queues[key]:
                ordered.append(queues[key].pop())
                added = True
                if len(ordered) >= int(max_samples):
                    break
        if not added:
            break
    return np.asarray(ordered, dtype=np.int64)


def _balanced_stratified_stream(
    indices: np.ndarray,
    cohort: Sequence[str],
    y: np.ndarray,
    *,
    n_samples: int,
    seed: int,
    include_cohort: bool,
) -> np.ndarray:
    """Fixed-length round-robin stream with deterministic stratum recycling.

    ``include_cohort=True`` balances classes first and rotates the available
    cohorts within every class. In the held-out cohort,
    ``include_cohort=False`` balances only the classes actually present. Small
    strata are reshuffled and recycled; every meta-episode therefore has
    exactly the same number of updates.
    """
    indices = np.asarray(indices, dtype=np.int64)
    cohort = np.asarray(cohort).astype(str)
    y = np.asarray(y, dtype=np.int64)
    n_samples = int(n_samples)
    if indices.size == 0 or n_samples <= 0:
        raise ValueError("El stream balanceado requiere índices y tamaño positivos.")
    rng = np.random.default_rng(int(seed))
    strata: Dict[Tuple[Any, ...], np.ndarray] = {}
    for index in indices:
        key = (
            (cohort[int(index)].strip().upper(), int(y[int(index)]))
            if include_cohort else (int(y[int(index)]),)
        )
        strata.setdefault(key, []).append(int(index))
    strata = {
        key: np.asarray(values, dtype=np.int64)
        for key, values in strata.items()
        if len(values)
    }
    if not strata:
        raise ValueError("No se pudieron construir estratos balanceados.")
    keys = sorted(strata)
    queues: Dict[Tuple[Any, ...], list[int]] = {}

    def refill(key):
        queues[key] = rng.permutation(strata[key]).astype(np.int64).tolist()

    for key in keys:
        refill(key)
    # Class is the primary balance level. Within each class, cohorts rotate.
    # A class missing from one cohort therefore does not receive fewer draws
    # than a class represented in every cohort.
    keys_by_class: Dict[int, list[Tuple[Any, ...]]] = {}
    for key in keys:
        class_id = int(key[-1])
        keys_by_class.setdefault(class_id, []).append(key)
    class_ids = sorted(keys_by_class)
    cohort_cursor = {class_id: 0 for class_id in class_ids}
    ordered: List[int] = []
    while len(ordered) < n_samples:
        for class_id in class_ids:
            class_keys = keys_by_class[class_id]
            position = cohort_cursor[class_id] % len(class_keys)
            key = class_keys[position]
            cohort_cursor[class_id] += 1
            if not queues[key]:
                refill(key)
            ordered.append(queues[key].pop())
            if len(ordered) >= n_samples:
                break
    return np.asarray(ordered, dtype=np.int64)


def _class_weight_loss(y: np.ndarray, train_idx: np.ndarray, n_classes: int, device):
    counts = np.bincount(
        np.asarray(y)[np.asarray(train_idx, dtype=np.int64)],
        minlength=int(n_classes),
    ).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"El TRAIN meta no contiene todas las clases: {counts}.")
    weights = counts.sum() / counts
    weights /= weights.mean()
    tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=tensor), tensor


def assert_transport_runtime(
    cohort: Sequence[str],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    cfg=CFG,
) -> List[str]:
    """Fail closed if external data or an undeclared cohort is visible."""
    if np.asarray(test_idx).size:
        raise RuntimeError(
            "InterGATE V6 exige un runtime development-only: test_idx debe "
            "estar vacío."
        )
    if str(cfg.GRAPH_LEARNING_XH_MODE).lower() != "zero":
        raise RuntimeError("InterGATE V6 exige X_h=0 durante toda Phase 1.")
    cohort = np.asarray(cohort).astype(str)
    dev_idx = np.unique(np.concatenate([
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
    ]))
    observed = {value.upper().strip() for value in cohort[dev_idx]}
    declared = {str(value).upper().strip() for value in cfg.TRAIN_COHORTS}
    external = {str(value).upper().strip() for value in cfg.TEST_COHORTS}
    if observed & external:
        raise RuntimeError(
            f"Cohortes externas visibles en InterGATE V6: {sorted(observed & external)}"
        )
    if observed != declared:
        raise RuntimeError(
            "Las cohortes de desarrollo no coinciden con el contrato: "
            f"observadas={sorted(observed)}, declaradas={sorted(declared)}."
        )
    # Preserve the declared order for round-robin episodes.
    return [str(value) for value in cfg.TRAIN_COHORTS]


def _cohort_equal_macro_f1(
    y_true: np.ndarray,
    probability: np.ndarray,
    cohort_values: Sequence[str],
    label_map: Mapping[str, int],
) -> Tuple[float, Dict[str, float]]:
    """Average validation macro-F1 with one vote per observed cohort."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probability)
    cohort_values = np.asarray(cohort_values).astype(str)
    if not (len(y_true) == len(probability) == len(cohort_values)):
        raise ValueError("Métricas y cohortes de validación no están alineadas.")
    rows: Dict[str, float] = {}
    normalized = np.char.upper(np.char.strip(cohort_values))
    for cohort_name in sorted(np.unique(normalized)):
        mask = normalized == cohort_name
        if not np.any(mask):
            continue
        rows[str(cohort_name)] = float(
            compute_metrics_full(
                y_true[mask], probability[mask], dict(label_map)
            )["macro_f1"]
        )
    if not rows:
        raise RuntimeError("No hay cohortes en la validación de desarrollo.")
    return float(np.mean(list(rows.values()))), rows


def _structural_regularization(
    model: nn.Module,
    *,
    sparsity_target: float,
    episode: int,
    biology_context_lambda: float,
    biology_context_warmup_episodes: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    core = _core(model)
    if str(CFG.SPARSITY_MODE).lower() != "target":
        raise RuntimeError("InterGATE V6 implementa el protocolo target-sparsity.")
    sparsity = float(CFG.SPARSITY_LAMBDA) * model.sparsity_penalty(
        target=float(sparsity_target),
        entropy_beta=float(CFG.SPARSITY_BINARIZE),
    )
    loss = sparsity
    connectivity = torch.zeros((), device=loss.device)
    conn_lambda, min_deg, use_abs = _connectivity_options(model)
    if conn_lambda > 0:
        connectivity = float(conn_lambda) * model.connectivity_penalty(
            min_deg=float(min_deg), use_abs=bool(use_abs)
        )
        loss = loss + connectivity
    delta_l1 = torch.zeros((), device=loss.device)
    if bool(getattr(core, "per_label_graphs", False)):
        delta_l1 = float(CFG.PER_LABEL_DELTA_L1) * core.per_label_delta_l1()
        loss = loss + delta_l1
    context = torch.zeros((), device=loss.device)
    context_lambda = float(biology_context_lambda)
    if context_lambda > 0 and hasattr(core, "biology_alignment_penalty"):
        warmup = max(1, int(biology_context_warmup_episodes))
        ramp = min(1.0, max(0.0, float(episode) / float(warmup)))
        context = context_lambda * ramp * core.biology_alignment_penalty()
        loss = loss + context
    if not torch.isfinite(loss):
        raise FloatingPointError("Regularización estructural no finita.")
    return loss, {
        "sparsity": sparsity,
        "connectivity": connectivity,
        "delta_l1": delta_l1,
        "biology_context": context,
    }


def _gate_tau_for_episode(episode: int) -> float:
    for last_episode, value in CFG.GATE_TAU_SCHEDULE:
        if int(episode) <= int(last_episode):
            return float(value)
    return float(CFG.GATE_TAU_SCHEDULE[-1][1])


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_transport_graph_learning(
    model: nn.Module,
    *,
    X_gene: np.ndarray,
    X_h: np.ndarray,
    y: np.ndarray,
    cohort: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    label_map: Mapping[str, int],
    keep_ratio: float,
    seed: int,
    transport: Optional[TransportConfig] = None,
    checkpoint_path: Optional[Path] = None,
    resume: bool = True,
) -> Tuple[nn.Module, Dict[str, torch.Tensor], float, pd.DataFrame]:
    """Train one V6 carrier with first-order structural meta-updates."""
    transport = transport or TransportConfig.from_global(CFG)
    device = CFG.DEVICE
    cohort_order = assert_transport_runtime(
        cohort, train_idx, val_idx, test_idx, CFG
    )
    cohort_array = np.asarray(cohort).astype(str)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    if _RT.get("sample_weights", None) is not None:
        raise NotImplementedError(
            "InterGATE V6 no mezcla pesos muestrales con el metaobjetivo; use "
            "la ponderación de clases prespecificada."
        )

    n_classes = int(_rt("n_classes"))
    outer_loss_fn = (
        nn.CrossEntropyLoss()
        if bool(transport.balanced_sampling)
        else _class_weight_loss(y, train_idx, n_classes, device)[0]
    )
    core = _core(model)
    core.use_xgraph_in_head = False
    budget = StructuralBudget.from_core(core, keep_ratio)
    structural = _structural_named_parameters(model)
    optimizer = torch.optim.AdamW(
        per_label_param_groups(model, float(CFG.LR)),
        weight_decay=float(CFG.WEIGHT_DECAY),
    )
    batch_size = (
        int(CFG.GRAPH_BATCH_SIZE)
        if bool(getattr(core, "per_label_graphs", False))
        else int(CFG.BATCH_SIZE)
    )
    accum_steps = (
        int(CFG.GRAPH_ACCUM_STEPS)
        if bool(getattr(core, "per_label_graphs", False))
        else int(CFG.ACCUM_STEPS)
    )
    val_loader = _make_loader(
        X_gene, X_h, y, val_idx,
        batch_size=batch_size, shuffle=False, seed=seed + 90_000,
    )

    quota_end = max(
        2,
        min(
            int(transport.episodes) - 1,
            int(math.ceil(
                float(transport.quota_end_fraction) * transport.episodes
            )),
        ),
    )
    start_episode = 1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = -float("inf")
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    if checkpoint_path is not None and resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("protocol_version") != transport.protocol_version:
            raise RuntimeError("El checkpoint pertenece a otro protocolo transport.")
        if saved.get("transport_config") != asdict(transport):
            raise RuntimeError("La configuración cambió desde el checkpoint.")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        start_episode = int(saved["episode"]) + 1
        best_state = saved.get("best_state")
        best_score = float(saved.get("best_score", -float("inf")))
        patience_counter = int(saved.get("patience_counter", 0))
        history = list(saved.get("history", []))
        _restore_rng_state(saved["rng_state"])
        print(
            f"[InterGATE-V6] reanudado en episodio {start_episode} "
            f"desde {checkpoint_path}",
            flush=True,
        )

    t_start = time.time()
    for episode in range(start_episode, int(transport.episodes) + 1):
        heldout = cohort_order[(episode - 1) % len(cohort_order)]
        normalized_cohort = np.char.strip(np.char.upper(cohort_array.astype(str)))
        heldout_mask = normalized_cohort == heldout.upper().strip()
        inner_idx = train_idx[~heldout_mask[train_idx]]
        outer_idx = train_idx[heldout_mask[train_idx]]
        if len(outer_idx) < int(transport.min_cohort_samples):
            raise RuntimeError(
                f"{heldout}: solo {len(outer_idx)} muestras TRAIN para el "
                "episodio exterior."
            )
        if len(np.unique(y[inner_idx])) != n_classes:
            raise RuntimeError(
                f"Al retener {heldout}, el meta-train pierde alguna clase."
            )
        inner_loss_fn = (
            nn.CrossEntropyLoss()
            if bool(transport.balanced_sampling)
            else _class_weight_loss(y, inner_idx, n_classes, device)[0]
        )

        if bool(transport.balanced_sampling):
            n_episode_samples = int(transport.steps_per_episode) * batch_size
            inner_stream = _balanced_stratified_stream(
                inner_idx, cohort_array, y,
                n_samples=n_episode_samples,
                seed=seed * 100_000 + episode * 2,
                include_cohort=True,
            )
            outer_stream = _balanced_stratified_stream(
                outer_idx, cohort_array, y,
                n_samples=n_episode_samples,
                seed=seed * 100_000 + episode * 2 + 1,
                include_cohort=False,
            )
            inner_loader = _make_loader(
                X_gene, X_h, y, inner_stream,
                batch_size=batch_size, shuffle=False,
                seed=seed * 100_000 + episode * 2,
            )
            outer_loader = _make_loader(
                X_gene, X_h, y, outer_stream,
                batch_size=batch_size, shuffle=False,
                seed=seed * 100_000 + episode * 2 + 1,
            )
        else:
            inner_loader = _make_loader(
                X_gene, X_h, y, inner_idx,
                batch_size=batch_size, shuffle=True,
                seed=seed * 100_000 + episode * 2,
            )
            outer_loader = _make_loader(
                X_gene, X_h, y, outer_idx,
                batch_size=batch_size, shuffle=True,
                seed=seed * 100_000 + episode * 2 + 1,
            )
        outer_iter = iter(outer_loader)

        model.train()
        gate_tau = _gate_tau_for_episode(episode)
        model.gate_tau = gate_tau
        keep_train = keep_schedule(
            episode,
            start_ep=1,
            end_ep=quota_end,
            start_keep=1.0,
            end_keep=float(keep_ratio),
        )
        model.set_hard_keep_ratio(keep_train)
        optimizer.zero_grad(set_to_none=True)
        sums = {
            "total": 0.0,
            "inner": 0.0,
            "outer": 0.0,
            "topology": 0.0,
            "biology_context": 0.0,
        }
        seen = 0
        last_step = 0

        for step, inner_batch in enumerate(inner_loader, start=1):
            last_step = step
            X_inner, Xh_inner, y_inner, _ = _move_batch(inner_batch, device)
            try:
                outer_batch = next(outer_iter)
            except StopIteration:
                outer_iter = iter(outer_loader)
                outer_batch = next(outer_iter)
            X_outer, Xh_outer, y_outer, _ = _move_batch(
                outer_batch, device
            )

            logits_inner = model(X_inner, None, Xh_inner)[0]
            supervised_inner = inner_loss_fn(logits_inner, y_inner)
            regularization, regularization_parts = _structural_regularization(
                model,
                sparsity_target=float(keep_train),
                episode=int(episode),
                biology_context_lambda=float(
                    transport.biology_context_lambda
                ),
                biology_context_warmup_episodes=int(
                    transport.biology_context_warmup_episodes
                ),
            )
            inner_loss = supervised_inner + regularization

            gate_gradients = torch.autograd.grad(
                inner_loss,
                tuple(structural.values()),
                create_graph=not bool(transport.first_order),
                retain_graph=True,
                allow_unused=True,
            )
            adapted: Dict[str, torch.Tensor] = {}
            for (name, parameter), gradient in zip(
                structural.items(), gate_gradients
            ):
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                if transport.first_order:
                    gradient = gradient.detach()
                adapted[name] = parameter - float(transport.inner_lr) * gradient

            logits_outer = _functional_forward(
                model, adapted, (X_outer, None, Xh_outer)
            )[0]
            outer_loss = outer_loss_fn(logits_outer, y_outer)
            base_logits = class_edge_logits_from_overrides(model)
            adapted_logits = class_edge_logits_from_overrides(model, adapted)
            topology_loss = soft_topk_jaccard_distance(
                base_logits,
                adapted_logits,
                budget,
                temperature=float(transport.soft_topk_tau),
            )
            total_loss = (
                float(transport.meta_train_weight) * inner_loss
                + float(transport.meta_validation_weight) * outer_loss
                + float(transport.topology_lambda) * topology_loss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Pérdida no finita en episodio {episode}, paso {step}."
                )
            (total_loss / max(1, accum_steps)).backward()
            if step % max(1, accum_steps) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_n = int(X_inner.shape[0])
            seen += batch_n
            sums["total"] += float(total_loss.detach().item()) * batch_n
            sums["inner"] += float(inner_loss.detach().item()) * batch_n
            sums["outer"] += float(outer_loss.detach().item()) * batch_n
            sums["topology"] += float(topology_loss.detach().item()) * batch_n
            sums["biology_context"] += float(
                regularization_parts["biology_context"].detach().item()
            ) * batch_n

        if last_step and last_step % max(1, accum_steps) != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if (
            bool(transport.balanced_sampling)
            and last_step != int(transport.steps_per_episode)
        ):
            raise RuntimeError(
                f"Episodio {episode}: {last_step} pasos, se esperaban "
                f"{transport.steps_per_episode}."
            )

        # Keep one discrete mask fixed during all differentiable updates in an
        # episode, then rebuild it from the updated scores. Checkpoint metrics
        # therefore evaluate the same topology that can later be restored.
        model.set_hard_keep_ratio(keep_train)
        probability, y_val = predict_proba_xh_mode(
            model, None, val_loader, device, xh_mode="zero"
        )
        metrics = compute_metrics_full(y_val, probability, dict(label_map))
        cohort_macro_f1, cohort_scores = _cohort_equal_macro_f1(
            y_val,
            probability,
            cohort_array[val_idx],
            label_map,
        )
        metrics["cohort_macro_f1"] = float(cohort_macro_f1)
        if transport.checkpoint_metric not in metrics:
            raise KeyError(
                f"Métrica de checkpoint inexistente: {transport.checkpoint_metric}."
            )
        score = float(metrics[transport.checkpoint_metric])
        at_target = episode >= quota_end
        if not at_target:
            patience_counter = 0
        elif np.isfinite(score) and score > best_score:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        row = {
            "episode": int(episode),
            "heldout_development_cohort": str(heldout),
            "n_meta_train": int(len(inner_idx)),
            "n_meta_validation": int(len(outer_idx)),
            "n_meta_train_draws": int(seen),
            "n_meta_validation_draws": int(last_step * batch_size),
            "steps_completed": int(last_step),
            "keep_ratio": float(keep_train),
            "target_budget": bool(at_target),
            "gate_tau": float(gate_tau),
            "loss_total": sums["total"] / max(1, seen),
            "loss_meta_train": sums["inner"] / max(1, seen),
            "loss_meta_validation": sums["outer"] / max(1, seen),
            "loss_topology": sums["topology"] / max(1, seen),
            "loss_biology_context": (
                sums["biology_context"] / max(1, seen)
            ),
            "validation_score": score,
            "validation_macro_f1": float(metrics["macro_f1"]),
            "validation_cohort_macro_f1": float(cohort_macro_f1),
            "validation_macro_f1_by_cohort": json.dumps(
                cohort_scores, sort_keys=True
            ),
            "validation_balanced_accuracy": float(metrics["bal_acc"]),
            "optimizer_steps_expected": int(math.ceil(
                last_step / max(1, accum_steps)
            )),
            "patience": int(patience_counter),
            "elapsed_seconds": float(time.time() - t_start),
        }
        history.append(row)
        print(
            "[InterGATE-V6] "
            f"episode={episode:03d}/{transport.episodes} "
            f"heldout={heldout} keep={keep_train:.6f} "
            f"Lmeta={row['loss_meta_validation']:.4f} "
            f"Ltopo={row['loss_topology']:.4f} "
            f"Lbio={row['loss_biology_context']:.4f} "
            f"VAL-{transport.checkpoint_metric}={score:.4f} "
            f"target={at_target} pat={patience_counter}",
            flush=True,
        )

        if checkpoint_path is not None:
            _atomic_torch_save({
                "protocol_version": transport.protocol_version,
                "transport_config": asdict(transport),
                "episode": int(episode),
                "model_state": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_score": float(best_score),
                "patience_counter": int(patience_counter),
                "history": history,
                "rng_state": _rng_state(),
            }, checkpoint_path)

        if at_target and patience_counter >= int(transport.patience):
            print("[InterGATE-V6] early stopping.", flush=True)
            break

    if best_state is None:
        raise RuntimeError(
            "No se obtuvo checkpoint bajo la cuota final; aumente episodios o "
            "revise TRANSPORT_QUOTA_END_FRACTION."
        )
    model.load_state_dict(best_state)
    model.set_hard_keep_ratio(float(keep_ratio))
    return model, best_state, float(best_score), pd.DataFrame(history)


def _jaccard_ids(a: torch.Tensor, b: torch.Tensor) -> float:
    left = set(map(int, a.detach().cpu().tolist()))
    right = set(map(int, b.detach().cpu().tolist()))
    return len(left & right) / len(left | right) if left or right else float("nan")


def _spearman_stable(left: torch.Tensor, right: torch.Tensor) -> float:
    """Spearman correlation with deterministic candidate-order tie breaking."""
    a = left.detach().cpu().numpy().astype(np.float64, copy=False)
    b = right.detach().cpu().numpy().astype(np.float64, copy=False)
    ids = np.arange(a.size, dtype=np.int64)
    rank_a = np.empty(a.size, dtype=np.float64)
    rank_b = np.empty(b.size, dtype=np.float64)
    rank_a[np.lexsort((ids, a))] = np.arange(a.size, dtype=np.float64)
    rank_b[np.lexsort((ids, b))] = np.arange(b.size, dtype=np.float64)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denominator = np.sqrt(np.dot(rank_a, rank_a) * np.dot(rank_b, rank_b))
    return float(np.dot(rank_a, rank_b) / denominator) if denominator else 1.0


def estimate_cohort_transport_support(
    model: nn.Module,
    *,
    X_gene: np.ndarray,
    X_h: np.ndarray,
    y: np.ndarray,
    cohort: Sequence[str],
    train_idx: np.ndarray,
    keep_ratio: float,
    seed: int,
    transport: Optional[TransportConfig] = None,
) -> Dict[str, Any]:
    """Estimate deterministic leave-one-cohort structural support.

    This uses multiple first-order structural influence steps, with dropout and
    BatchNorm updates disabled. It remains an influence/support analysis, not
    a claim that complete independent neural models were refitted.
    """
    transport = transport or TransportConfig.from_global(CFG)
    device = CFG.DEVICE
    core = _core(model)
    structural = _structural_named_parameters(model)
    budget = StructuralBudget.from_core(core, keep_ratio)
    cohort_array = np.asarray(cohort).astype(str)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    cohort_order = [str(value) for value in CFG.TRAIN_COHORTS]
    n_classes = int(_rt("n_classes"))
    batch_size = (
        int(CFG.GRAPH_BATCH_SIZE)
        if bool(getattr(core, "per_label_graphs", False))
        else int(CFG.BATCH_SIZE)
    )

    model.eval()
    model.set_hard_keep_ratio(float(keep_ratio))
    edge_count = int(core.edge_index.shape[1])
    selection_count = torch.zeros(
        (n_classes, edge_count), dtype=torch.int16
    )
    sum_logits = torch.zeros(
        (n_classes, edge_count), dtype=torch.float64
    )
    fold_edge_ids: Dict[str, Dict[int, torch.Tensor]] = {}
    fold_base_logits: Dict[str, torch.Tensor] = {}
    fold_adapted_logits: Dict[str, torch.Tensor] = {}
    audit_rows: List[Dict[str, Any]] = []
    base_logits_reference = class_edge_logits_from_overrides(model).detach()
    if base_logits_reference.shape[0] == 1 and n_classes > 1:
        base_logits_reference = base_logits_reference.expand(n_classes, -1)

    for fold_id, heldout in enumerate(cohort_order):
        normalized_cohort = np.char.strip(np.char.upper(cohort_array))
        heldout_mask = normalized_cohort == heldout.upper().strip()
        inner_idx = train_idx[~heldout_mask[train_idx]]
        support_loss_fn = (
            nn.CrossEntropyLoss()
            if bool(transport.balanced_sampling)
            else _class_weight_loss(y, inner_idx, n_classes, device)[0]
        )
        support_idx = _balanced_stratified_stream(
            inner_idx,
            cohort_array,
            y,
            n_samples=int(transport.support_batches) * batch_size,
            seed=seed + 800_000 + fold_id,
            include_cohort=True,
        )
        loader = _make_loader(
            X_gene, X_h, y, support_idx,
            batch_size=batch_size, shuffle=False,
            seed=seed + 800_000 + fold_id,
        )
        base_parameters = {
            name: parameter.detach().clone()
            for name, parameter in structural.items()
        }
        adapted = {
            name: value.detach().clone().requires_grad_(True)
            for name, value in base_parameters.items()
        }
        cached_batches = [
            _move_batch(batch, device) for batch in loader
        ]
        if not cached_batches:
            raise RuntimeError(f"Sin batches para el soporte de {heldout}.")
        for influence_step in range(1, int(transport.support_steps) + 1):
            gradient_sums = {
                name: torch.zeros_like(value)
                for name, value in adapted.items()
            }
            used_counts = {name: 0 for name in adapted}
            loss_sum = 0.0
            for Xg, Xh_zero, yb, _ in cached_batches:
                logits = _functional_forward(
                    model, adapted, (Xg, None, Xh_zero)
                )[0]
                loss = support_loss_fn(logits, yb)
                gradients = torch.autograd.grad(
                    loss, tuple(adapted.values()),
                    create_graph=False, retain_graph=False, allow_unused=True,
                )
                loss_sum += float(loss.detach().item())
                for (name, _value), gradient in zip(adapted.items(), gradients):
                    if gradient is not None:
                        gradient_sums[name].add_(gradient.detach())
                        used_counts[name] += 1
            updated = {}
            for name, value in adapted.items():
                divisor = max(1, used_counts[name])
                gradient = gradient_sums[name] / float(divisor)
                updated[name] = (
                    value.detach() - float(transport.inner_lr) * gradient
                ).requires_grad_(True)
                audit_rows.append({
                    "heldout_cohort": str(heldout),
                    "influence_step": int(influence_step),
                    "parameter": str(name),
                    "n_batches_with_gradient": int(used_counts[name]),
                    "gradient_l2": float(torch.linalg.vector_norm(gradient).item()),
                    "step_delta_l2": float(
                        torch.linalg.vector_norm(updated[name] - value.detach()).item()
                    ),
                    "mean_support_loss": loss_sum / float(len(cached_batches)),
                })
            adapted = updated
        for name, value in adapted.items():
            final_delta = value.detach() - base_parameters[name]
            base_norm = float(
                torch.linalg.vector_norm(base_parameters[name]).item()
            )
            final_norm = float(torch.linalg.vector_norm(final_delta).item())
            audit_rows.append({
                "heldout_cohort": str(heldout),
                "influence_step": int(transport.support_steps),
                "parameter": f"__FINAL__:{name}",
                "n_batches_with_gradient": int(len(cached_batches)),
                "gradient_l2": float("nan"),
                "step_delta_l2": final_norm,
                "relative_parameter_delta_l2": (
                    final_norm / max(base_norm, 1e-12)
                ),
                "mean_support_loss": float("nan"),
            })
        adapted_logits = class_edge_logits_from_overrides(
            model, adapted
        ).detach()
        if adapted_logits.shape[0] == 1 and n_classes > 1:
            adapted_logits = adapted_logits.expand(n_classes, -1)
        sum_logits.add_(adapted_logits.cpu().double())
        base_logits = base_logits_reference.detach()
        fold_base_logits[str(heldout)] = base_logits.cpu().float()
        fold_adapted_logits[str(heldout)] = adapted_logits.cpu().float()
        unit_scores = budget.unit_scores(adapted_logits)
        base_unit_scores = budget.unit_scores(base_logits)
        fold_edge_ids[str(heldout)] = {}
        for class_id in range(n_classes):
            units = budget.hard_topk_units(unit_scores[class_id])
            base_units = budget.hard_topk_units(base_unit_scores[class_id])
            edge_ids = budget.expand_units(units).detach().cpu().long()
            base_edge_ids = budget.expand_units(base_units).detach().cpu().long()
            selection_count[class_id, edge_ids] += 1
            fold_edge_ids[str(heldout)][class_id] = edge_ids
            sorted_base = torch.sort(
                base_unit_scores[class_id], descending=True
            ).values
            cutoff_margin = (
                float((sorted_base[budget.k_units - 1] - sorted_base[budget.k_units]).item())
                if budget.k_units < sorted_base.numel() else float("nan")
            )
            delta = adapted_logits[class_id] - base_logits[class_id]
            audit_rows.append({
                "heldout_cohort": str(heldout),
                "influence_step": int(transport.support_steps),
                "parameter": "__LOGITS__",
                "class_id": int(class_id),
                "n_batches_with_gradient": int(len(cached_batches)),
                "gradient_l2": float("nan"),
                "step_delta_l2": float(torch.linalg.vector_norm(delta).item()),
                "mean_abs_logit_delta": float(delta.abs().mean().item()),
                "max_abs_logit_delta": float(delta.abs().max().item()),
                "rank_spearman": _spearman_stable(
                    base_unit_scores[class_id], unit_scores[class_id]
                ),
                "base_adapted_topk_jaccard": _jaccard_ids(
                    base_edge_ids, edge_ids
                ),
                "topk_turnover_edges": int(
                    len(set(base_edge_ids.tolist()) ^ set(edge_ids.tolist())) // 2
                ),
                "base_cutoff_margin_k_kplus1": cutoff_margin,
                "mean_support_loss": float("nan"),
            })

    n_folds = len(cohort_order)
    mean_logits = (sum_logits / float(n_folds)).float()
    support_frequency = selection_count.float() / float(n_folds)
    final_edge_ids: Dict[int, torch.Tensor] = {}
    consensus_rank = torch.full(
        (n_classes, edge_count), -1, dtype=torch.int32
    )
    unit_mean_scores = budget.unit_scores(mean_logits)

    for class_id in range(n_classes):
        first_cpu = budget.first_ids.detach().cpu().long()
        unit_support = support_frequency[class_id, first_cpu]
        score = unit_mean_scores[class_id]
        # np.lexsort uses the final key as primary: support, score, unit id.
        order_np = np.lexsort((
            np.arange(len(unit_support), dtype=np.int64),
            -score.detach().cpu().numpy(),
            -unit_support.detach().cpu().numpy(),
        ))
        units = torch.as_tensor(
            order_np[: budget.k_units], dtype=torch.long,
            device=first_cpu.device,
        )
        edges = budget.expand_units(units).detach().cpu().long()
        final_edge_ids[class_id] = edges
        unit_rank_np = np.empty(len(order_np), dtype=np.int32)
        unit_rank_np[order_np] = np.arange(
            1, len(order_np) + 1, dtype=np.int32
        )
        unit_rank = torch.from_numpy(unit_rank_np)
        consensus_rank[class_id, first_cpu] = unit_rank
        if budget.second_ids is not None:
            second_cpu = budget.second_ids.detach().cpu().long()
            consensus_rank[class_id, second_cpu] = unit_rank

    pairwise_rows = []
    for left, right in combinations(cohort_order, 2):
        for class_id in range(n_classes):
            pairwise_rows.append({
                "cohort_a": left,
                "cohort_b": right,
                "class_id": int(class_id),
                "jaccard": _jaccard_ids(
                    fold_edge_ids[left][class_id],
                    fold_edge_ids[right][class_id],
                ),
            })

    structural_unit_id = torch.full(
        (edge_count,), -1, dtype=torch.int64
    )
    unit_ids = torch.arange(
        int(budget.first_ids.numel()), dtype=torch.int64
    )
    structural_unit_id[budget.first_ids.detach().cpu().long()] = unit_ids
    if budget.second_ids is not None:
        structural_unit_id[budget.second_ids.detach().cpu().long()] = unit_ids

    biology_bundle = _RT.get("biology_evidence_bundle", None)
    if biology_bundle is None:
        raise RuntimeError(
            "V6 requiere biology_evidence_bundle para construir las redes "
            "diferenciales."
        )
    context_score = np.asarray(
        biology_bundle["context_score"], dtype=np.float32
    )
    differential_logits = (
        mean_logits - mean_logits.mean(dim=0, keepdim=True)
    ).cpu().numpy()
    biology_scores = combine_biology_scores(
        differential_logits,
        context_score,
        support_frequency.cpu().numpy(),
        context_weight=float(transport.biology_score_context_weight),
        support_weight=float(transport.biology_score_support_weight),
    )
    edge_index_cpu = core.edge_index.detach().cpu().numpy().astype(np.int64)
    first_ids_np = budget.first_ids.detach().cpu().numpy().astype(np.int64)
    unit_edge_index = edge_index_cpu[:, first_ids_np]
    biology_final_edge_ids: Dict[int, torch.Tensor] = {}
    biology_raw_edge_ids: Dict[int, torch.Tensor] = {}
    biology_topology: Dict[str, Dict[str, Any]] = {}
    for class_id in range(n_classes):
        unit_score = biology_scores[class_id, first_ids_np]
        raw_units = np.lexsort((
            np.arange(unit_score.size, dtype=np.int64), -unit_score
        ))[: budget.k_units]
        raw_edges = budget.expand_units(
            torch.as_tensor(raw_units, dtype=torch.long)
        ).detach().cpu().long()
        module_units = module_aware_exact_k(
            unit_edge_index,
            unit_score,
            k_edges=budget.k_units,
            pool_multiplier=int(CFG.BIOLOGY_MODULE_POOL_MULTIPLIER),
            target_components=int(CFG.BIOLOGY_MODULE_TARGET_COMPONENTS),
            min_component_edges=int(CFG.BIOLOGY_MODULE_MIN_EDGES),
        )
        module_edges = budget.expand_units(
            torch.as_tensor(module_units, dtype=torch.long)
        ).detach().cpu().long()
        biology_raw_edge_ids[class_id] = raw_edges
        biology_final_edge_ids[class_id] = module_edges
        biology_topology[str(class_id)] = {
            "raw_differential": weak_topology_summary(
                edge_index_cpu[:, raw_edges.numpy()]
            ),
            "module_aware": weak_topology_summary(
                edge_index_cpu[:, module_edges.numpy()]
            ),
        }

    shared_unit_score = mean_logits.mean(dim=0).cpu().numpy()[first_ids_np]
    shared_units = np.lexsort((
        np.arange(shared_unit_score.size, dtype=np.int64), -shared_unit_score
    ))[: budget.k_units]
    shared_edge_ids = budget.expand_units(
        torch.as_tensor(shared_units, dtype=torch.long)
    ).detach().cpu().long()

    return {
        "protocol_version": transport.protocol_version,
        "support_steps": int(transport.support_steps),
        "support_batches": int(transport.support_batches),
        "biology_score_context_weight": float(
            transport.biology_score_context_weight
        ),
        "biology_score_support_weight": float(
            transport.biology_score_support_weight
        ),
        "cohort_order": cohort_order,
        "n_folds": int(n_folds),
        "paired_structural_units": bool(budget.paired),
        "n_structural_units": int(budget.first_ids.numel()),
        "k_units": int(budget.k_units),
        "realized_edges_per_class": int(
            budget.k_units * (2 if budget.paired else 1)
        ),
        "selection_count": selection_count,
        "selection_frequency": support_frequency,
        "mean_adapted_logit": mean_logits,
        "consensus_rank": consensus_rank,
        "structural_unit_id": structural_unit_id,
        "final_edge_ids": final_edge_ids,
        "differential_logits": torch.from_numpy(
            differential_logits.astype(np.float32)
        ),
        "biology_score": torch.from_numpy(biology_scores.astype(np.float32)),
        "biology_raw_edge_ids": biology_raw_edge_ids,
        "biology_final_edge_ids": biology_final_edge_ids,
        "biology_topology": biology_topology,
        "shared_backbone_edge_ids": shared_edge_ids,
        "fold_edge_ids": fold_edge_ids,
        "fold_base_logits": fold_base_logits,
        "fold_adapted_logits": fold_adapted_logits,
        "adaptation_audit": pd.DataFrame(audit_rows),
        "pairwise_jaccard": pd.DataFrame(pairwise_rows),
    }


def class_graphs_from_transport_support(
    model: nn.Module,
    support: Mapping[str, Any],
    class_names: Sequence[str],
) -> Dict[int, Dict[str, Any]]:
    """Materialize the frequency-first Exact-K consensus on the original prior."""
    core = _core(model)
    checkpoint_logits = core.all_class_edge_logits().detach().cpu()
    fold_logits = torch.stack([
        support["fold_adapted_logits"][str(cohort_name)]
        for cohort_name in support["cohort_order"]
    ], dim=0)
    out: Dict[int, Dict[str, Any]] = {}
    for class_id, class_name in enumerate(class_names):
        edge_ids = support["final_edge_ids"][class_id].to(
            device=core.edge_index.device, dtype=torch.long
        )
        mean_logits = support["mean_adapted_logit"][class_id, edge_ids.cpu()]
        checkpoint_selected = checkpoint_logits[class_id, edge_ids.cpu()]
        fold_selected = fold_logits[:, class_id, edge_ids.cpu()]
        median_logits = fold_selected.median(dim=0).values
        fold_rank_rows = []
        for fold_id in range(fold_logits.shape[0]):
            values = fold_logits[fold_id, class_id].numpy()
            ids = np.arange(values.size, dtype=np.int64)
            order = np.lexsort((ids, -values))
            ranks_np = np.empty(values.size, dtype=np.float32)
            ranks_np[order] = np.arange(1, values.size + 1, dtype=np.float32)
            fold_rank_rows.append(
                torch.from_numpy(ranks_np)[edge_ids.cpu()]
            )
        median_ranks = torch.stack(fold_rank_rows, dim=0).median(dim=0).values
        frequency = support["selection_frequency"][class_id, edge_ids.cpu()]
        rank = support["consensus_rank"][class_id, edge_ids.cpu()].long()
        structural_unit_id = support["structural_unit_id"][edge_ids.cpu()].long()
        out[class_id] = {
            "class_id": int(class_id),
            "class_name": str(class_name),
            "edge_index": core.edge_index[:, edge_ids].detach().cpu(),
            "edge_weight": core.edge_weight_base[edge_ids].detach().cpu(),
            "edge_type": core.edge_type[edge_ids].detach().cpu(),
            "candidate_edge_ids": edge_ids.detach().cpu(),
            "selection_frequency": frequency.detach().cpu().float(),
            "consensus_rank": rank.detach().cpu(),
            "structural_unit_id": structural_unit_id.detach().cpu(),
            "inner_median_rank": median_ranks.detach().cpu().float(),
            "inner_median_logit": median_logits.detach().cpu().float(),
            "mean_adapted_logit": mean_logits.detach().cpu().float(),
            "learned_logit": checkpoint_selected.detach().cpu().float(),
            "differential_logit": support["differential_logits"][
                class_id, edge_ids.cpu()
            ].detach().cpu().float(),
            "biology_score": support["biology_score"][
                class_id, edge_ids.cpu()
            ].detach().cpu().float(),
            "learned_gate": torch.sigmoid(
                checkpoint_selected.detach().cpu().float()
                / float(core.gate_tau)
            ),
            "kept": int(edge_ids.numel()),
            "total": int(core.edge_index.shape[1]),
            "thr_logit": float("nan"),
        }
    return out


def biology_graphs_from_transport_support(
    model: nn.Module,
    support: Mapping[str, Any],
    class_names: Sequence[str],
) -> Dict[int, Dict[str, Any]]:
    """Materialise the V6 differential, module-aware biological networks."""
    core = _core(model)
    biology_bundle = _RT.get("biology_evidence_bundle", None)
    if biology_bundle is None:
        raise RuntimeError("Falta biology_evidence_bundle en el runtime V6.")
    context = torch.as_tensor(
        biology_bundle["context_score"], dtype=torch.float32
    )
    out: Dict[int, Dict[str, Any]] = {}
    for class_id, class_name in enumerate(class_names):
        edge_ids = support["biology_final_edge_ids"][class_id].long().cpu()
        raw_ids = support["biology_raw_edge_ids"][class_id].long().cpu()
        out[class_id] = {
            "class_id": int(class_id),
            "class_name": str(class_name),
            "edge_index": core.edge_index[:, edge_ids.to(core.edge_index.device)].detach().cpu(),
            "edge_weight": core.edge_weight_base[edge_ids.to(core.edge_weight_base.device)].detach().cpu(),
            "edge_type": core.edge_type[edge_ids.to(core.edge_type.device)].detach().cpu(),
            "candidate_edge_ids": edge_ids,
            "biology_score": support["biology_score"][class_id, edge_ids].float(),
            "differential_logit": support["differential_logits"][class_id, edge_ids].float(),
            "context_score": context[class_id, edge_ids].float(),
            "selection_frequency": support["selection_frequency"][class_id, edge_ids].float(),
            "raw_differential_edge_ids": raw_ids,
            "topology": support["biology_topology"][str(class_id)],
            "kept": int(edge_ids.numel()),
            "total": int(core.edge_index.shape[1]),
            "interpretation": (
                "class-contrastive association network; prior-constrained, "
                "not causal and not a de-novo directed network"
            ),
        }
    return out


def save_transport_support(
    output_dir: Path,
    support: Mapping[str, Any],
    *,
    model: nn.Module,
    genes: Sequence[str],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Write dense compressed support and human-readable non-zero tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    core = _core(model)
    selection_count = support["selection_count"].cpu().numpy().astype(np.int8)
    selection_frequency = support["selection_frequency"].cpu().numpy().astype(np.float32)
    mean_logit = support["mean_adapted_logit"].cpu().numpy().astype(np.float32)
    rank = support["consensus_rank"].cpu().numpy().astype(np.int32)
    structural_unit_id = (
        support["structural_unit_id"].cpu().numpy().astype(np.int64)
    )
    final_mask = np.zeros_like(selection_count, dtype=bool)
    biology_mask = np.zeros_like(selection_count, dtype=bool)
    biology_raw_mask = np.zeros_like(selection_count, dtype=bool)
    for class_id, edge_ids in support["final_edge_ids"].items():
        final_mask[int(class_id), edge_ids.cpu().numpy()] = True
    for class_id, edge_ids in support["biology_final_edge_ids"].items():
        biology_mask[int(class_id), edge_ids.cpu().numpy()] = True
    for class_id, edge_ids in support["biology_raw_edge_ids"].items():
        biology_raw_mask[int(class_id), edge_ids.cpu().numpy()] = True
    differential_logits = support["differential_logits"].cpu().numpy().astype(np.float32)
    biology_score = support["biology_score"].cpu().numpy().astype(np.float32)
    context_score = np.asarray(
        _RT["biology_evidence_bundle"]["context_score"], dtype=np.float32
    )
    np.savez_compressed(
        output_dir / "transport_support_dense.npz",
        selection_count=selection_count,
        selection_frequency=selection_frequency,
        mean_adapted_logit=mean_logit,
        consensus_rank=rank,
        structural_unit_id=structural_unit_id,
        final_selected=final_mask,
        biology_selected=biology_mask,
        biology_raw_topk=biology_raw_mask,
        differential_logits=differential_logits,
        context_score=context_score,
        biology_score=biology_score,
        shared_backbone_selected=np.isin(
            np.arange(selection_count.shape[1]),
            support["shared_backbone_edge_ids"].cpu().numpy(),
        ),
        cohort_order=np.asarray(support["cohort_order"], dtype=object),
        class_names=np.asarray(class_names, dtype=object),
    )

    edge_index = core.edge_index.detach().cpu().numpy()
    edge_weight = core.edge_weight_base.detach().cpu().numpy()
    edge_type = core.edge_type.detach().cpu().numpy()
    shared_ids = support["shared_backbone_edge_ids"].cpu().numpy().astype(np.int64)
    pd.DataFrame({
        "candidate_edge_id": shared_ids,
        "source_idx": edge_index[0, shared_ids],
        "target_idx": edge_index[1, shared_ids],
        "source_gene": [genes[int(value)] for value in edge_index[0, shared_ids]],
        "target_gene": [genes[int(value)] for value in edge_index[1, shared_ids]],
        "prior_weight": edge_weight[shared_ids],
        "edge_type": edge_type[shared_ids],
        "shared_score": mean_logit[:, shared_ids].mean(axis=0),
    }).sort_values(
        ["shared_score", "candidate_edge_id"], ascending=[False, True]
    ).to_csv(output_dir / "biology_shared_backbone.csv", index=False)
    summary_classes = {}
    for class_id, class_name in enumerate(class_names):
        visible = np.flatnonzero(selection_count[class_id] > 0)
        table = pd.DataFrame({
            "candidate_edge_id": visible.astype(np.int64),
            "source_idx": edge_index[0, visible].astype(np.int64),
            "target_idx": edge_index[1, visible].astype(np.int64),
            "source_gene": [genes[int(value)] for value in edge_index[0, visible]],
            "target_gene": [genes[int(value)] for value in edge_index[1, visible]],
            "prior_weight": edge_weight[visible].astype(float),
            "edge_type": edge_type[visible].astype(np.int64),
            "structural_unit_id": structural_unit_id[visible],
            "selection_count": selection_count[class_id, visible].astype(np.int64),
            "selection_frequency": selection_frequency[class_id, visible].astype(float),
            "mean_adapted_logit": mean_logit[class_id, visible].astype(float),
            "consensus_rank": rank[class_id, visible].astype(np.int64),
            "final_exact_k": final_mask[class_id, visible],
        }).sort_values(
            ["final_exact_k", "selection_frequency", "consensus_rank"],
            ascending=[False, False, True],
        )
        safe = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in str(class_name)
        )
        table.to_csv(output_dir / f"transport_support_{safe}.csv", index=False)
        biology_ids = np.flatnonzero(biology_mask[class_id])
        biology_table = pd.DataFrame({
            "class_id": int(class_id),
            "class_name": str(class_name),
            "candidate_edge_id": biology_ids.astype(np.int64),
            "source_idx": edge_index[0, biology_ids].astype(np.int64),
            "target_idx": edge_index[1, biology_ids].astype(np.int64),
            "source_gene": [genes[int(value)] for value in edge_index[0, biology_ids]],
            "target_gene": [genes[int(value)] for value in edge_index[1, biology_ids]],
            "prior_weight": edge_weight[biology_ids].astype(float),
            "edge_type": edge_type[biology_ids].astype(np.int64),
            "structural_unit_id": structural_unit_id[biology_ids],
            "differential_logit": differential_logits[class_id, biology_ids],
            "differential_direction": np.where(
                differential_logits[class_id, biology_ids] >= 0,
                "strengthened_for_class", "attenuated_for_class",
            ),
            "context_score": context_score[class_id, biology_ids],
            "context_agrees_with_learned_direction": (
                np.sign(differential_logits[class_id, biology_ids])
                == np.sign(context_score[class_id, biology_ids])
            ),
            "selection_frequency": selection_frequency[class_id, biology_ids],
            "biology_score": biology_score[class_id, biology_ids],
            "also_in_predictive_topk": final_mask[class_id, biology_ids],
            "module_aware_exact_k": True,
        }).sort_values(
            ["biology_score", "candidate_edge_id"],
            ascending=[False, True],
        )
        biology_table.insert(
            0, "biology_rank", np.arange(1, len(biology_table) + 1)
        )
        biology_table.to_csv(
            output_dir / f"biology_edges_{safe}.csv", index=False
        )
        selected = final_mask[class_id]
        summary_classes[str(class_name)] = {
            "final_edges": int(selected.sum()),
            "ever_selected_edges": int((selection_count[class_id] > 0).sum()),
            "strict_all_cohort_edges": int(
                (selection_count[class_id] == int(support["n_folds"])).sum()
            ),
            "mean_support_final": float(
                selection_frequency[class_id, selected].mean()
            ),
            "biology_edges": int(biology_mask[class_id].sum()),
            "biology_predictive_overlap": int(
                np.logical_and(biology_mask[class_id], selected).sum()
            ),
            "biology_topology": support["biology_topology"][str(class_id)],
        }

    pairwise = support["pairwise_jaccard"].copy()
    pairwise["class_name"] = pairwise["class_id"].map(
        {idx: name for idx, name in enumerate(class_names)}
    )
    pairwise.to_csv(output_dir / "transport_pairwise_jaccard.csv", index=False)
    audit = support["adaptation_audit"].copy()
    audit.to_csv(output_dir / "transport_adaptation_audit.csv", index=False)
    np.savez_compressed(
        output_dir / "transport_fold_logits.npz",
        cohort_order=np.asarray(support["cohort_order"], dtype=object),
        base_logits=np.stack([
            support["fold_base_logits"][str(name)].numpy()
            for name in support["cohort_order"]
        ]),
        adapted_logits=np.stack([
            support["fold_adapted_logits"][str(name)].numpy()
            for name in support["cohort_order"]
        ]),
    )
    logit_audit = audit[audit["parameter"].eq("__LOGITS__")]
    summary = {
        "protocol_version": support["protocol_version"],
        "cohort_order": list(support["cohort_order"]),
        "n_cohort_deletions": int(support["n_folds"]),
        "paired_structural_units": bool(support["paired_structural_units"]),
        "n_structural_units": int(support["n_structural_units"]),
        "k_units": int(support["k_units"]),
        "realized_edges_per_class": int(support["realized_edges_per_class"]),
        "support_sampling": (
            "fixed_length_balanced_round_robin_with_stratum_recycling"
        ),
        "support_steps": int(support["support_steps"]),
        "support_batches_per_step": int(support["support_batches"]),
        "biology_score_weights": {
            "differential_strength": 1.0,
            "signed_context_agreement": float(
                support["biology_score_context_weight"]
            ),
            "cohort_support": float(
                support["biology_score_support_weight"]
            ),
        },
        "pairwise_jaccard_mean": float(pairwise["jaccard"].mean()),
        "adaptation_audit": {
            "min_mean_abs_logit_delta": float(
                logit_audit["mean_abs_logit_delta"].min()
            ),
            "max_mean_abs_logit_delta": float(
                logit_audit["mean_abs_logit_delta"].max()
            ),
            "median_rank_spearman": float(
                logit_audit["rank_spearman"].median()
            ),
            "median_base_adapted_topk_jaccard": float(
                logit_audit["base_adapted_topk_jaccard"].median()
            ),
            "all_parameter_deltas_zero": bool(
                audit[audit["parameter"].astype(str).str.startswith("__FINAL__:")]
                ["step_delta_l2"].fillna(0).eq(0).all()
            ),
        },
        "classes": summary_classes,
    }
    (output_dir / "transport_support_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _completed_transport_result(
    run_dir: Path,
    *,
    transport: TransportConfig,
    artifact_name: str,
) -> Optional[Dict[str, Any]]:
    meta_path = run_dir / "meta.json"
    required = [
        meta_path,
        run_dir / "graph_bundle.pt",
        run_dir / "biology_graph_bundle.pt",
        run_dir / "model_pruned.pt",
        run_dir / "transport_support_summary.json",
    ]
    if not all(path.is_file() for path in required):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    protocol = meta.get("transport_protocol", {})
    if protocol.get("completed") is not True:
        return None
    if str(meta.get("cfg_name")) != str(artifact_name):
        return None
    if any(
        protocol.get(key) != value
        for key, value in asdict(transport).items()
    ):
        return None
    return dict(meta.get("metrics", {}))


def run_transport_seed(
    seed: int,
    *,
    save_root: Optional[str] = None,
    transport: Optional[TransportConfig] = None,
    resume: bool = True,
    force: bool = False,
    artifact_name: str = TRANSPORT_ARTIFACT_NAME,
) -> Dict[str, Any]:
    """Run InterGATE V6, cohort support, fixed Phase 2 and save both graphs."""
    from .model import build_pruned_model_from
    from .training import finetune_pruned

    transport = transport or TransportConfig.from_global(CFG)
    artifact_name = str(artifact_name).strip()
    if not artifact_name or any(
        char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
        for char in artifact_name
    ):
        raise ValueError(f"artifact_name no seguro: {artifact_name!r}")
    set_all_seeds(int(seed))
    X_full = np.asarray(_rt("Xs_gene"), dtype=np.float32)
    genes_full = list(map(str, _rt("genes_kegg")))
    X_h = np.asarray(_rt("X_h"), dtype=np.float32)
    y = np.asarray(_rt("y"), dtype=np.int64)
    cohort = np.asarray(_rt("cohort")).astype(str)
    train_idx = np.asarray(_rt("train_idx"), dtype=np.int64)
    val_idx = np.asarray(_rt("val_idx"), dtype=np.int64)
    test_idx = np.asarray(_rt("test_idx"), dtype=np.int64)
    label_map = dict(_rt("label_map"))
    n_classes = int(_rt("n_classes"))
    class_names = class_names_from_label_map(label_map, n_classes)
    assert_transport_runtime(cohort, train_idx, val_idx, test_idx, CFG)

    if save_root is None:
        save_root = str(Path(CFG.ARTIFACTS_ROOT))
    run_dir = Path(save_root) / artifact_name / f"seed_{int(seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        completed = _completed_transport_result(
            run_dir,
            transport=transport,
            artifact_name=artifact_name,
        )
        if completed is not None:
            print(f"[InterGATE-V6] reutilizando bundle completo: {run_dir}")
            return completed

    loss_fn, class_weights_t = _class_weight_loss(
        y, train_idx, n_classes, CFG.DEVICE
    )
    model = build_model_from_cfg(CFG_FULL, class_weights_t)
    core = _core(model)
    keep_ratio = resolve_class_keep_ratio(core.edge_index, CFG.KEEP_MIN)
    checkpoint_path = run_dir / "transport_last_checkpoint.pt"
    model, best_state, best_score, history = train_transport_graph_learning(
        model,
        X_gene=X_full,
        X_h=X_h,
        y=y,
        cohort=cohort,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        label_map=label_map,
        keep_ratio=keep_ratio,
        seed=int(seed),
        transport=transport,
        checkpoint_path=checkpoint_path,
        resume=bool(resume and not force),
    )
    history.to_csv(run_dir / "transport_training_history.csv", index=False)

    support = estimate_cohort_transport_support(
        model,
        X_gene=X_full,
        X_h=X_h,
        y=y,
        cohort=cohort,
        train_idx=train_idx,
        keep_ratio=keep_ratio,
        seed=int(seed),
        transport=transport,
    )
    class_graphs = class_graphs_from_transport_support(
        model, support, class_names
    )
    biology_graphs = biology_graphs_from_transport_support(
        model, support, class_names
    )
    _atomic_torch_save({
        "protocol_version": transport.protocol_version,
        "genes_full": genes_full,
        "class_names": class_names,
        "graphs": biology_graphs,
        "shared_backbone_edge_ids": support["shared_backbone_edge_ids"],
        "shared_backbone_edge_index": _core(model).edge_index[
            :, support["shared_backbone_edge_ids"].to(
                _core(model).edge_index.device
            )
        ].detach().cpu(),
        "interpretation": (
            "Class-contrastive, prior-constrained association networks. "
            "Direction/sign are inherited annotations; no causal claim."
        ),
    }, run_dir / "biology_graph_bundle.pt")
    support_summary = save_transport_support(
        run_dir, support, model=model, genes=genes_full,
        class_names=class_names,
    )

    # Evaluate the support-consensus topology before Phase 2.  Candidate IDs
    # refer to the original full prior and can therefore be fixed directly.
    core.set_fixed_class_edge_ids([
        support["final_edge_ids"][class_id].to(CFG.DEVICE)
        for class_id in range(n_classes)
    ])
    val_loader_full = _make_loader(
        X_full, X_h, y, val_idx,
        batch_size=int(CFG.GRAPH_BATCH_SIZE), shuffle=False,
        seed=int(seed) + 910_000,
    )
    phase1_probability, phase1_truth = predict_proba_xh_mode(
        model, None, val_loader_full, CFG.DEVICE, xh_mode="zero"
    )
    phase1_metrics = compute_metrics_full(
        phase1_truth, phase1_probability, label_map
    )

    edge_index_union, edge_weight_union, edge_type_union, class_ids_union = (
        build_union_class_graph(class_graphs, n_classes)
    )
    nodes_used = torch.unique(
        edge_index_union.reshape(-1)
    ).detach().cpu().numpy().astype(np.int64)
    nodes_used.sort()
    X_comp = X_full[:, nodes_used]
    genes_comp = [genes_full[index] for index in nodes_used]
    old_to_new = np.full(X_full.shape[1], -1, dtype=np.int64)
    old_to_new[nodes_used] = np.arange(len(nodes_used), dtype=np.int64)
    edge_index_np = edge_index_union.detach().cpu().numpy()
    edge_index_comp = torch.as_tensor(
        np.vstack([
            old_to_new[edge_index_np[0]],
            old_to_new[edge_index_np[1]],
        ]),
        dtype=torch.long,
        device=CFG.DEVICE,
    )
    edge_weight_comp = edge_weight_union.to(CFG.DEVICE)
    edge_type_comp = edge_type_union.to(CFG.DEVICE)
    class_ids_comp = [ids.to(CFG.DEVICE) for ids in class_ids_union]

    model_phase2 = build_pruned_model_from(
        model_gated=model,
        edge_index_p=edge_index_comp,
        edge_weight_p=edge_weight_comp,
        edge_type_p=edge_type_comp,
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
        pool_level=CFG_FULL.pool_level,
        class_edge_ids=class_ids_comp,
        class_names=class_names,
        per_label_graphs=True,
        device=CFG.DEVICE,
        use_hybrid=CFG.USE_HYBRID_MODEL,
        tab_layers=CFG.HYBRID_TAB_LAYERS,
        tab_dropout=CFG.HYBRID_TAB_DROPOUT,
        fusion_dropout=CFG.HYBRID_FUSION_DROPOUT,
        blend_logits=CFG.HYBRID_BLEND_LOGITS,
        blend_init=CFG.HYBRID_BLEND_INIT,
    )
    train_loader_comp = _make_loader(
        X_comp, X_h, y, train_idx,
        batch_size=int(CFG.BATCH_SIZE), shuffle=True,
        seed=int(seed) + 920_000,
    )
    val_loader_comp = _make_loader(
        X_comp, X_h, y, val_idx,
        batch_size=int(CFG.BATCH_SIZE), shuffle=False,
        seed=int(seed) + 930_000,
    )
    model_phase2 = finetune_pruned(
        model_pruned=model_phase2,
        adj=None,
        dl_tr=train_loader_comp,
        dl_va=val_loader_comp,
        device=CFG.DEVICE,
        label_map=label_map,
        loss_fn=loss_fn,
        lr=float(CFG.LR) * float(_RT.get("PHASE2_LR_MULT", 0.20)),
        weight_decay=float(CFG.WEIGHT_DECAY),
        epochs_A=int(CFG.FT_EPOCHS_A),
        epochs_B=int(CFG.FT_EPOCHS_B),
        accum_steps=int(CFG.ACCUM_STEPS),
        use_mixup_B=bool(CFG.USE_MIXUP),
        mixup_alpha=float(CFG.MIXUP_ALPHA),
        mixup_prob=float(CFG.MIXUP_P),
        best_metric=str(_RT.get("PHASE2_BEST_METRIC", "macro_f1")),
        patience_B=int(_RT.get("PHASE2_PATIENCE", 20)),
        min_delta=float(_RT.get("PHASE2_MIN_DELTA", 1e-4)),
        save_best_path=str(run_dir / "phase2_best.pt"),
        lr_tab_mult=float(_RT.get("PHASE2_LR_TAB_MULT", 2.0)),
        lr_fusion_mult=float(_RT.get("PHASE2_LR_FUSION_MULT", 2.0)),
        sample_weights_full=None,
        xh_mode_B=str(CFG.CLASS_GRAPH_XH_MODE),
    )
    probability, truth = predict_proba_xh_mode(
        model_phase2, None, val_loader_comp, CFG.DEVICE,
        xh_mode=str(CFG.CLASS_GRAPH_XH_MODE),
    )
    metrics = compute_metrics_full(truth, probability, label_map)
    metrics.update({
        "cfg": artifact_name,
        "seed": int(seed),
        "protocol_version": transport.protocol_version,
        "evaluation_split": "development_validation",
        "external_opened": False,
        "transport_episodes_completed": int(history["episode"].max()),
        "transport_best_validation_score_phase1": float(best_score),
        "val_macro_f1_phase1_transport_consensus": float(
            phase1_metrics["macro_f1"]
        ),
        "val_bal_acc_phase1_transport_consensus": float(
            phase1_metrics["bal_acc"]
        ),
        "transport_pairwise_jaccard_mean": float(
            support_summary["pairwise_jaccard_mean"]
        ),
        "transport_median_rank_spearman": float(
            support_summary["adaptation_audit"]["median_rank_spearman"]
        ),
        "transport_all_parameter_deltas_zero": bool(
            support_summary["adaptation_audit"]["all_parameter_deltas_zero"]
        ),
        "class_graph_keep_ratio": float(keep_ratio),
        "class_graph_edges_per_class": int(
            support["realized_edges_per_class"]
        ),
        "n_edges_final": int(edge_index_union.shape[1]),
        "n_nodes_compact": int(len(nodes_used)),
    })
    for class_name, details in support_summary["classes"].items():
        safe = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in class_name
        )
        metrics[f"transport_mean_support_{safe}"] = float(
            details["mean_support_final"]
        )
        metrics[f"transport_strict_edges_{safe}"] = int(
            details["strict_all_cohort_edges"]
        )

    save_bundle(
        str(run_dir),
        cfg=CFG_FULL,
        seed=int(seed),
        genes_full=genes_full,
        nodes_used_full=nodes_used,
        genes_comp=genes_comp,
        edge_index_full=edge_index_union,
        edge_weight_full=edge_weight_union,
        edge_type_full=edge_type_union,
        edge_index_comp=edge_index_comp,
        edge_weight_comp=edge_weight_comp,
        edge_type_comp=edge_type_comp,
        reg_groups_comp=None,
        class_graphs_full=class_graphs,
        class_edge_ids_comp=class_ids_comp,
        class_names=class_names,
        model_pruned=model_phase2,
        metrics=metrics,
        stab=None,
        phase1_best_state=best_state,
        proba_te=None,
        y_te=None,
    )
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["cfg_name"] = artifact_name
    meta["transport_protocol"] = {
        **asdict(transport),
        "artifact_name": artifact_name,
        "completed": True,
        "development_cohorts": list(support["cohort_order"]),
        "episode_outer_samples": "training_split_of_heldout_development_cohort",
        "checkpoint_samples": "disjoint_global_development_validation_split",
        "inner_adapted_parameters": [
            "edge_logit", "edge_logit_delta", "type_scale", "type_bias"
        ],
        "episode_sampling": (
            "fixed_steps_balanced_round_robin_with_stratum_recycling"
        ),
        "inner_class_weights": "none_because_sampling_is_class_balanced",
        "query_class_weights": "none_because_sampling_is_class_balanced",
        "hard_topk_refresh": "between_meta_episodes",
        "support_definition": (
            "frequency_across_deterministic_multistep_first_order_"
            "leave_one_development_cohort_out_structural_influence_refits"
        ),
        "support_sampling": (
            "fixed_length_round_robin_over_available_cohort_by_class_strata_"
            f"{int(transport.support_steps)}_steps_x_"
            f"{int(transport.support_batches)}_batches"
        ),
        "predictive_graph": "frequency_first_mean_total_logit_exact_k",
        "biological_graph": (
            "centered_class_delta_plus_within_cohort_context_plus_support_"
            "module_aware_exact_k"
        ),
        "known_biology_used_for_fitting": False,
        "xh_phase1": "zero",
        "external_expression_or_labels_opened": False,
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if checkpoint_path.exists():
        # Keep the resumable checkpoint until the complete bundle is safely
        # written; then mark it as completed instead of deleting evidence.
        (run_dir / "transport_checkpoint_COMPLETE.txt").write_text(
            "The resumable checkpoint is retained for provenance.\n",
            encoding="utf-8",
        )
    cleanup_memory("after_intergate_transport_seed")
    return metrics


def run_transport_experiment(
    seeds: Optional[Iterable[int]] = None,
    *,
    save_root: Optional[str] = None,
    transport: Optional[TransportConfig] = None,
    resume: bool = True,
    force: bool = False,
    artifact_name: str = TRANSPORT_ARTIFACT_NAME,
) -> pd.DataFrame:
    """Run/resume the prespecified outer seeds, saving after every seed."""
    seed_values = list(
        map(int, seeds if seeds is not None else CFG.STABILITY_SEEDS)
    )
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("Las semillas transport deben ser distintas.")
    rows = []
    for seed in seed_values:
        rows.append(run_transport_seed(
            seed,
            save_root=save_root,
            transport=transport,
            resume=resume,
            force=force,
            artifact_name=artifact_name,
        ))
        if save_root is not None:
            progress = Path(save_root) / str(artifact_name)
            progress.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(
                progress / "development_results.csv", index=False
            )
    return pd.DataFrame(rows)
