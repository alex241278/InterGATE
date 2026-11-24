"""Biology-first network evidence and post-selection for InterGATE V8.

The predictive graph and the biological graph answer different questions.
Prediction uses the total class score (shared backbone plus class deviation).
Biological interpretation uses the centered class deviation, transcriptomic
evidence reproduced within development cohorts, and cohort-deletion support.

No known subtype signature, PAM50 membership, C2:CGP result, DepMap value or
external-cohort label is consumed by this module while fitting a graph.
Direction and regulatory sign remain annotations inherited from the supplied
prior; correlation does not identify direction or causality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
import hashlib
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DifferentialEvidenceConfig:
    min_samples_per_group: int = 8
    min_cohorts: int = 2
    edge_batch_size: int = 4096
    meta_z_clip: float = 6.0
    heterogeneity_penalty: float = 0.50


def evidence_fingerprint(
    *,
    X: np.ndarray,
    genes: Sequence[str],
    edge_index: np.ndarray,
    y: np.ndarray,
    cohort: Sequence[str],
    train_idx: np.ndarray,
    class_names: Sequence[str],
    config: DifferentialEvidenceConfig,
) -> str:
    """Stable cache key without hashing the full expression matrix."""
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, genes)).encode("utf-8"))
    X = np.asarray(X)
    # The same TRAIN block must produce the same key before and after the
    # external block is appended. Hash only the fitted development subset,
    # never the full matrix shape.
    digest.update(
        str((int(len(train_idx)), int(X.shape[1]))).encode("ascii")
    )
    for start in range(0, len(train_idx), 256):
        block_idx = np.asarray(train_idx[start:start + 256], dtype=np.int64)
        digest.update(
            np.ascontiguousarray(X[block_idx], dtype=np.float32).tobytes()
        )
    digest.update(np.asarray(edge_index, dtype=np.int64).tobytes())
    digest.update(np.asarray(y, dtype=np.int64)[train_idx].tobytes())
    digest.update("\n".join(np.asarray(cohort).astype(str)[train_idx]).encode("utf-8"))
    digest.update("\n".join(map(str, class_names)).encode("utf-8"))
    digest.update(json.dumps(config.__dict__, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _edge_correlations(
    X: np.ndarray,
    sample_idx: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Pearson correlations for candidate edges, processed in edge batches."""
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    source = np.asarray(source, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    out = np.zeros(source.size, dtype=np.float64)
    if sample_idx.size < 4:
        return out
    X = np.asarray(X)
    for start in range(0, source.size, int(batch_size)):
        stop = min(source.size, start + int(batch_size))
        a = np.asarray(X[np.ix_(sample_idx, source[start:stop])], dtype=np.float64)
        b = np.asarray(X[np.ix_(sample_idx, target[start:stop])], dtype=np.float64)
        a -= a.mean(axis=0, keepdims=True)
        b -= b.mean(axis=0, keepdims=True)
        numerator = np.einsum("ij,ij->j", a, b, optimize=True)
        denominator = np.sqrt(
            np.einsum("ij,ij->j", a, a, optimize=True)
            * np.einsum("ij,ij->j", b, b, optimize=True)
        )
        values = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )
        out[start:stop] = np.clip(values, -0.999999, 0.999999)
    return out


def compute_differential_edge_evidence(
    X: np.ndarray,
    y: np.ndarray,
    cohort: Sequence[str],
    train_idx: np.ndarray,
    edge_index: np.ndarray,
    class_names: Sequence[str],
    cohort_order: Sequence[str],
    *,
    config: DifferentialEvidenceConfig,
) -> Dict[str, Any]:
    """Meta-analyse class-vs-rest differential correlation per prior edge.

    Correlations are estimated separately inside each development cohort.
    Fisher-z differences are then combined with inverse-variance weights. The
    returned ``context_score`` is bounded, penalised for sign inconsistency and
    heterogeneity, and centred between classes for every edge.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    cohort = np.asarray(cohort).astype(str)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index debe tener forma (2, E).")
    if X.shape[0] != y.size or cohort.size != y.size:
        raise ValueError("X, y y cohort no tienen el mismo número de muestras.")

    source, target = edge_index
    n_classes = len(class_names)
    n_cohorts = len(cohort_order)
    n_edges = edge_index.shape[1]
    effects = np.full((n_classes, n_cohorts, n_edges), np.nan, dtype=np.float32)
    variances = np.full((n_classes, n_cohorts), np.nan, dtype=np.float64)
    group_counts = np.zeros((n_classes, n_cohorts, 2), dtype=np.int64)
    normalized_cohort = np.char.upper(np.char.strip(cohort.astype(str)))

    for cohort_id, cohort_name in enumerate(cohort_order):
        cohort_idx = train_idx[
            normalized_cohort[train_idx] == str(cohort_name).strip().upper()
        ]
        for class_id in range(n_classes):
            case_idx = cohort_idx[y[cohort_idx] == class_id]
            rest_idx = cohort_idx[y[cohort_idx] != class_id]
            group_counts[class_id, cohort_id] = (case_idx.size, rest_idx.size)
            if (
                case_idx.size < int(config.min_samples_per_group)
                or rest_idx.size < int(config.min_samples_per_group)
            ):
                continue
            r_case = _edge_correlations(
                X, case_idx, source, target, batch_size=config.edge_batch_size
            )
            r_rest = _edge_correlations(
                X, rest_idx, source, target, batch_size=config.edge_batch_size
            )
            effects[class_id, cohort_id] = (
                np.arctanh(r_case) - np.arctanh(r_rest)
            ).astype(np.float32)
            variances[class_id, cohort_id] = (
                1.0 / (case_idx.size - 3.0) + 1.0 / (rest_idx.size - 3.0)
            )

    meta_effect = np.zeros((n_classes, n_edges), dtype=np.float32)
    meta_se = np.full((n_classes, n_edges), np.nan, dtype=np.float32)
    meta_z = np.zeros((n_classes, n_edges), dtype=np.float32)
    i2 = np.ones((n_classes, n_edges), dtype=np.float32)
    sign_consistency = np.zeros((n_classes, n_edges), dtype=np.float32)
    cohorts_used = np.zeros(n_classes, dtype=np.int64)

    for class_id in range(n_classes):
        valid_cohorts = np.flatnonzero(np.isfinite(variances[class_id]))
        cohorts_used[class_id] = valid_cohorts.size
        if valid_cohorts.size == 0:
            continue
        current = effects[class_id, valid_cohorts].astype(np.float64)
        weights = 1.0 / variances[class_id, valid_cohorts]
        total_weight = float(weights.sum())
        combined = np.sum(current * weights[:, None], axis=0) / total_weight
        se = np.sqrt(1.0 / total_weight)
        z_score = combined / se
        meta_effect[class_id] = combined.astype(np.float32)
        meta_se[class_id] = np.float32(se)
        meta_z[class_id] = z_score.astype(np.float32)
        concordant = np.sign(current) == np.sign(combined)[None, :]
        concordant[:, np.isclose(combined, 0.0)] = True
        sign_consistency[class_id] = concordant.mean(axis=0).astype(np.float32)
        if valid_cohorts.size > 1:
            q = np.sum(weights[:, None] * (current - combined[None, :]) ** 2, axis=0)
            df = float(valid_cohorts.size - 1)
            i2[class_id] = np.maximum(0.0, (q - df) / np.maximum(q, 1e-12)).astype(
                np.float32
            )
        else:
            i2[class_id] = 1.0

    sufficient = cohorts_used >= int(config.min_cohorts)
    if not np.all(sufficient):
        failed = {
            str(class_names[class_id]): int(cohorts_used[class_id])
            for class_id in np.flatnonzero(~sufficient)
        }
        raise ValueError(
            "Evidencia diferencial insuficiente para centrar todas las "
            f"clases (cohortes utilizables): {failed}."
        )
    bounded = np.tanh(
        np.clip(meta_z, -float(config.meta_z_clip), float(config.meta_z_clip)) / 3.0
    )
    reliability = sign_consistency * np.clip(
        1.0 - float(config.heterogeneity_penalty) * i2, 0.0, 1.0
    )
    context_score = bounded * reliability
    context_score -= context_score.mean(axis=0, keepdims=True)

    return {
        "context_score": context_score.astype(np.float32),
        "meta_effect": meta_effect,
        "meta_se": meta_se,
        "meta_z": meta_z,
        "i2": i2,
        "sign_consistency": sign_consistency,
        "cohorts_used": cohorts_used,
        "cohort_effects": effects,
        "group_counts": group_counts,
        "class_names": np.asarray(class_names, dtype=object),
        "cohort_order": np.asarray(cohort_order, dtype=object),
        "method": "within_cohort_differential_correlation",
    }


def save_evidence_cache(
    cache_path: Path,
    evidence: Mapping[str, Any],
    *,
    fingerprint: str,
    config: DifferentialEvidenceConfig,
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value for key, value in evidence.items()
        if isinstance(value, np.ndarray)
    }
    np.savez_compressed(cache_path, **arrays)
    manifest = {
        "fingerprint": str(fingerprint),
        "method": str(evidence["method"]),
        "config": config.__dict__,
    }
    cache_path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_evidence_cache(
    cache_path: Path,
    *,
    fingerprint: str,
) -> Optional[Dict[str, Any]]:
    cache_path = Path(cache_path)
    manifest_path = cache_path.with_suffix(".json")
    if not cache_path.exists() or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != str(fingerprint):
        return None
    with np.load(cache_path, allow_pickle=True) as loaded:
        out = {key: loaded[key] for key in loaded.files}
    out["method"] = str(manifest["method"])
    return out


def get_or_build_differential_evidence(
    *,
    cache_path: Path,
    X: np.ndarray,
    genes: Sequence[str],
    y: np.ndarray,
    cohort: Sequence[str],
    train_idx: np.ndarray,
    edge_index: np.ndarray,
    class_names: Sequence[str],
    cohort_order: Sequence[str],
    config: DifferentialEvidenceConfig,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    fingerprint = evidence_fingerprint(
        X=X, genes=genes, edge_index=edge_index, y=y, cohort=cohort,
        train_idx=train_idx, class_names=class_names, config=config,
    )
    if not force_rebuild:
        cached = load_evidence_cache(cache_path, fingerprint=fingerprint)
        if cached is not None:
            return cached
    evidence = compute_differential_edge_evidence(
        X, y, cohort, train_idx, edge_index, class_names, cohort_order,
        config=config,
    )
    save_evidence_cache(
        cache_path, evidence, fingerprint=fingerprint, config=config
    )
    return evidence


def _rank_scale(values: np.ndarray, *, preserve_zeros: bool = False) -> np.ndarray:
    """Average-tie rank transform to [-1, 1].

    Equal scores receive exactly the same value. This is important when the
    cohort support saturates: an all-one support vector must contribute zero,
    not an arbitrary candidate-order gradient.
    """
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values, dtype=np.float64)
    mask = np.isfinite(values)
    if preserve_zeros:
        mask &= values != 0
    ids = np.flatnonzero(mask)
    if ids.size < 2:
        return result.astype(np.float32)
    order = np.argsort(values[ids], kind="mergesort")
    ordered_values = values[ids][order]
    ranks_ordered = np.empty(ids.size, dtype=np.float64)
    start = 0
    while start < ids.size:
        stop = start + 1
        while stop < ids.size and ordered_values[stop] == ordered_values[start]:
            stop += 1
        ranks_ordered[start:stop] = 0.5 * (start + stop - 1)
        start = stop
    ranks = np.empty(ids.size, dtype=np.float64)
    ranks[order] = ranks_ordered
    if float(ranks.max() - ranks.min()) <= 0.0:
        return result.astype(np.float32)
    result[ids] = (
        2.0 * (ranks - ranks.min()) / float(ranks.max() - ranks.min()) - 1.0
    )
    return result.astype(np.float32)


def combine_biology_scores(
    differential_logits: np.ndarray,
    context_score: np.ndarray,
    support_frequency: np.ndarray,
    *,
    context_weight: float,
    support_weight: float,
) -> np.ndarray:
    """Combine differential strength, signed context agreement and support.

    Both strengthened and attenuated subtype relations can be biological. The
    primary term therefore ranks ``abs(differential_logits)``. Context enters
    positively only when its differential-correlation direction agrees with
    the learned deviation; support remains a direction-free stability term.
    The original signed quantities are exported for interpretation.
    """
    differential_logits = np.asarray(differential_logits)
    context_score = np.asarray(context_score)
    support_frequency = np.asarray(support_frequency)
    if not (
        differential_logits.shape == context_score.shape
        == support_frequency.shape
    ):
        raise ValueError("Las tres matrices de score deben tener la misma forma.")
    out = np.zeros_like(differential_logits, dtype=np.float32)
    for class_id in range(out.shape[0]):
        learned = differential_logits[class_id]
        context = context_score[class_id]
        signed_context_agreement = np.sign(learned) * context
        out[class_id] = (
            _rank_scale(np.abs(learned))
            + float(context_weight) * _rank_scale(
                signed_context_agreement, preserve_zeros=True
            )
            + float(support_weight) * _rank_scale(support_frequency[class_id])
        )
    return out


class _DisjointSet:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def find(self, value: int) -> int:
        value = int(value)
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def weak_topology_summary(edge_index: np.ndarray) -> Dict[str, Any]:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    dsu = _DisjointSet()
    degree: Dict[int, int] = {}
    for source, target in edge_index.T:
        source, target = int(source), int(target)
        dsu.union(source, target)
        degree[source] = degree.get(source, 0) + 1
        degree[target] = degree.get(target, 0) + 1
    nodes = sorted(degree)
    sizes: Dict[int, int] = {}
    edge_counts: Dict[int, int] = {}
    for node in nodes:
        root = dsu.find(node)
        sizes[root] = sizes.get(root, 0) + 1
    for source, _target in edge_index.T:
        root = dsu.find(int(source))
        edge_counts[root] = edge_counts.get(root, 0) + 1
    component_sizes = sorted(sizes.values(), reverse=True)
    component_edges = sorted(edge_counts.values(), reverse=True)
    return {
        "n_edges": int(edge_index.shape[1]),
        "n_nodes": int(len(nodes)),
        "n_components": int(len(component_sizes)),
        "giant_nodes": int(component_sizes[0]) if component_sizes else 0,
        "giant_fraction": (
            float(component_sizes[0] / len(nodes)) if nodes else float("nan")
        ),
        "dyad_components": int(sum(value == 1 for value in component_edges)),
        "degree_one_nodes": int(sum(value == 1 for value in degree.values())),
    }


def module_aware_exact_k(
    edge_index: np.ndarray,
    scores: np.ndarray,
    *,
    k_edges: int,
    pool_multiplier: int = 6,
    target_components: int = 12,
    min_component_edges: int = 3,
) -> np.ndarray:
    """Deterministic Exact-K forest heuristic for interpretable modules.

    Seeds are high-scoring, node-disjoint edges with local expansion support.
    Each seed is expanded before a global greedy pass that favours edges
    touching the selected forest and penalises creation of new components.
    This is an auditable heuristic, not an implementation of PCSF.
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if edge_index.shape != (2, scores.size):
        raise ValueError("edge_index y scores no están alineados.")
    eligible = np.flatnonzero(edge_index[0] != edge_index[1])
    k_edges = int(k_edges)
    if not 1 <= k_edges <= eligible.size:
        raise ValueError("k_edges fuera del universo elegible.")
    order = eligible[np.lexsort((eligible, -scores[eligible]))]
    pool_size = min(order.size, max(k_edges, int(pool_multiplier) * k_edges))
    pool = order[:pool_size]
    adjacency: Dict[int, list[int]] = {}
    for edge_id in pool:
        source, target = map(int, edge_index[:, edge_id])
        adjacency.setdefault(source, []).append(int(edge_id))
        adjacency.setdefault(target, []).append(int(edge_id))

    selected: list[int] = []
    selected_set: set[int] = set()
    seed_nodes: set[int] = set()
    seeds: list[int] = []
    for edge_id in pool:
        source, target = map(int, edge_index[:, edge_id])
        has_extension = (
            len(adjacency.get(source, ())) > 1
            or len(adjacency.get(target, ())) > 1
        )
        if source in seed_nodes or target in seed_nodes or not has_extension:
            continue
        seeds.append(int(edge_id))
        seed_nodes.update((source, target))
        if len(seeds) >= min(int(target_components), k_edges):
            break
    if not seeds:
        seeds = [int(pool[0])]
    for edge_id in seeds:
        if len(selected) >= k_edges:
            break
        selected.append(edge_id)
        selected_set.add(edge_id)

    # Give every seed the opportunity to form a module larger than a dyad.
    module_nodes = [set(map(int, edge_index[:, edge_id])) for edge_id in seeds]
    for module_id, nodes in enumerate(module_nodes):
        additions = 0
        while additions < max(0, int(min_component_edges) - 1) and len(selected) < k_edges:
            candidates = {
                candidate
                for node in nodes
                for candidate in adjacency.get(node, ())
                if candidate not in selected_set
            }
            if not candidates:
                break
            candidate = min(candidates, key=lambda idx: (-scores[idx], idx))
            selected.append(int(candidate))
            selected_set.add(int(candidate))
            nodes.update(map(int, edge_index[:, candidate]))
            additions += 1
        module_nodes[module_id] = nodes

    active_nodes = {node for edge_id in selected for node in map(int, edge_index[:, edge_id])}
    spread = float(np.nanstd(scores[pool]))
    if not np.isfinite(spread) or spread <= 1e-12:
        spread = 1.0
    while len(selected) < k_edges:
        remaining = np.asarray(
            [edge_id for edge_id in pool if int(edge_id) not in selected_set],
            dtype=np.int64,
        )
        if remaining.size == 0:
            break
        source_active = np.isin(edge_index[0, remaining], list(active_nodes))
        target_active = np.isin(edge_index[1, remaining], list(active_nodes))
        touch = source_active.astype(np.int8) + target_active.astype(np.int8)
        adjusted = scores[remaining].copy()
        adjusted += np.where(touch == 2, 0.30 * spread, 0.0)
        adjusted += np.where(touch == 1, 0.15 * spread, 0.0)
        adjusted -= np.where(touch == 0, 0.60 * spread, 0.0)
        best_order = np.lexsort((remaining, -adjusted))
        candidate = int(remaining[best_order[0]])
        selected.append(candidate)
        selected_set.add(candidate)
        active_nodes.update(map(int, edge_index[:, candidate]))

    if len(selected) < k_edges:
        for edge_id in order:
            if int(edge_id) not in selected_set:
                selected.append(int(edge_id))
                selected_set.add(int(edge_id))
                if len(selected) == k_edges:
                    break
    selected_array = np.asarray(selected[:k_edges], dtype=np.int64)
    selected_array = selected_array[
        np.lexsort((selected_array, -scores[selected_array]))
    ]
    if np.unique(selected_array).size != k_edges:
        raise RuntimeError("La selección modular no produjo K aristas únicas.")
    return selected_array


def evidence_table(
    evidence: Mapping[str, Any],
    *,
    edge_index: np.ndarray,
    genes: Sequence[str],
    edge_weight: np.ndarray,
    edge_type: np.ndarray,
) -> pd.DataFrame:
    """Long table for audit; it contains no validation annotations."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    rows = []
    for class_id, class_name in enumerate(evidence["class_names"]):
        frame = pd.DataFrame({
            "class_id": int(class_id),
            "class_name": str(class_name),
            "candidate_edge_id": np.arange(edge_index.shape[1], dtype=np.int64),
            "source_idx": edge_index[0],
            "target_idx": edge_index[1],
            "source_gene": [genes[int(value)] for value in edge_index[0]],
            "target_gene": [genes[int(value)] for value in edge_index[1]],
            "prior_weight": np.asarray(edge_weight),
            "edge_type": np.asarray(edge_type),
            "context_score": evidence["context_score"][class_id],
            "meta_effect": evidence["meta_effect"][class_id],
            "meta_z": evidence["meta_z"][class_id],
            "i2": evidence["i2"][class_id],
            "sign_consistency": evidence["sign_consistency"][class_id],
            "cohorts_used": int(evidence["cohorts_used"][class_id]),
        })
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def aggregate_biology_seed_tables(
    seed_root: str | Path,
    *,
    seeds: Sequence[int],
    class_names: Sequence[str],
    output_dir: str | Path,
    pool_multiplier: int = 6,
    target_components: int = 12,
    min_component_edges: int = 3,
) -> Dict[str, Any]:
    """Aggregate five module-aware biological graphs without pooling classes."""
    seed_root = Path(seed_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(map(int, seeds))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Se requieren al menos dos semillas outer distintas.")
    result: Dict[str, Any] = {}
    for class_name in class_names:
        safe = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in str(class_name)
        )
        frames = []
        for seed in seeds:
            path = seed_root / f"seed_{seed}" / f"biology_edges_{safe}.csv"
            if not path.is_file():
                raise FileNotFoundError(str(path))
            frame = pd.read_csv(path)
            required = {
                "source_gene", "target_gene", "biology_score",
                "candidate_edge_id", "class_name",
            }
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{path}: faltan {sorted(missing)}")
            if set(frame["class_name"].astype(str)) != {str(class_name)}:
                raise ValueError(f"{path}: clase incoherente.")
            frame = frame.copy()
            frame["seed"] = int(seed)
            frames.append(frame)
        sizes = [len(frame) for frame in frames]
        if len(set(sizes)) != 1 or sizes[0] <= 0:
            raise ValueError(f"Exact-K roto para {class_name}: {sizes}")
        k = sizes[0]
        pair_sets = [
            set(zip(frame.source_gene.astype(str), frame.target_gene.astype(str)))
            for frame in frames
        ]
        pairwise = []
        for left in range(len(seeds)):
            for right in range(left + 1, len(seeds)):
                union = pair_sets[left] | pair_sets[right]
                pairwise.append({
                    "class_name": str(class_name),
                    "seed_a": int(seeds[left]),
                    "seed_b": int(seeds[right]),
                    "edge_jaccard": (
                        len(pair_sets[left] & pair_sets[right]) / len(union)
                        if union else float("nan")
                    ),
                })
        all_edges = pd.concat(frames, ignore_index=True)
        grouped = (
            all_edges.groupby(
                ["source_gene", "target_gene"], as_index=False
            )
            .agg(
                selection_count=("seed", "nunique"),
                median_biology_score=("biology_score", "median"),
                mean_context_score=("context_score", "mean"),
                mean_selection_frequency=("selection_frequency", "mean"),
                candidate_edge_id=("candidate_edge_id", "min"),
                prior_weight=("prior_weight", "median"),
                edge_type=("edge_type", "first"),
            )
        )
        grouped["selection_frequency_seed"] = (
            grouped["selection_count"] / float(len(seeds))
        )
        # Frequency is primary; the learned biology score breaks ties. A final
        # module pass prevents aggregation from reverting to a list of dyads.
        frequency_rank = (
            2.0 * grouped["selection_frequency_seed"].rank(
                method="average", pct=True
            ).to_numpy(dtype=np.float64) - 1.0
        )
        score_rank = _rank_scale(
            grouped["median_biology_score"].to_numpy()
        )
        aggregate_score = 2.0 * frequency_rank + score_rank
        genes = sorted(set(grouped.source_gene) | set(grouped.target_gene))
        gene_id = {gene: index for index, gene in enumerate(genes)}
        local_edge_index = np.vstack([
            grouped.source_gene.map(gene_id).to_numpy(dtype=np.int64),
            grouped.target_gene.map(gene_id).to_numpy(dtype=np.int64),
        ])
        if len(grouped) < k:
            raise RuntimeError(
                f"{class_name}: la unión multisemilla tiene menos de K aristas."
            )
        selected_local = module_aware_exact_k(
            local_edge_index,
            aggregate_score,
            k_edges=k,
            pool_multiplier=pool_multiplier,
            target_components=target_components,
            min_component_edges=min_component_edges,
        )
        consensus = grouped.iloc[selected_local].copy()
        consensus["aggregate_biology_score"] = aggregate_score[selected_local]
        consensus = consensus.sort_values(
            ["aggregate_biology_score", "selection_count", "candidate_edge_id"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        consensus.insert(0, "consensus_rank", np.arange(1, k + 1))
        consensus.insert(0, "class_name", str(class_name))
        consensus.to_csv(
            output_dir / f"biology_consensus_{safe}.csv", index=False
        )

        gene_counts: Dict[str, int] = {}
        for edge_set in pair_sets:
            genes_seed = {gene for edge in edge_set for gene in edge}
            for gene in genes_seed:
                gene_counts[gene] = gene_counts.get(gene, 0) + 1
        gene_stability = pd.DataFrame({
            "class_name": str(class_name),
            "gene": sorted(gene_counts),
            "selection_count": [gene_counts[g] for g in sorted(gene_counts)],
        })
        gene_stability["selection_frequency"] = (
            gene_stability["selection_count"] / float(len(seeds))
        )
        gene_stability["strict_core"] = gene_stability["selection_count"].eq(
            len(seeds)
        )
        gene_stability.to_csv(
            output_dir / f"biology_gene_stability_{safe}.csv", index=False
        )
        result[str(class_name)] = {
            "k_edges": int(k),
            "n_union_edges": int(len(grouped)),
            "strict_edge_core": int(grouped.selection_count.eq(len(seeds)).sum()),
            "strict_gene_core": int(gene_stability.strict_core.sum()),
            "mean_pairwise_edge_jaccard": float(
                np.mean([row["edge_jaccard"] for row in pairwise])
            ),
            "topology": weak_topology_summary(
                local_edge_index[:, selected_local]
            ),
            "consensus_file": str(
                output_dir / f"biology_consensus_{safe}.csv"
            ),
            "gene_stability_file": str(
                output_dir / f"biology_gene_stability_{safe}.csv"
            ),
        }
        pd.DataFrame(pairwise).to_csv(
            output_dir / f"biology_pairwise_jaccard_{safe}.csv", index=False
        )
    (output_dir / "biology_stability_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
