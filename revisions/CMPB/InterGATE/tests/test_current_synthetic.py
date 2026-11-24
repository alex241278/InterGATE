"""Lightweight synthetic checks without importing PyTorch."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


biology = load_module("intergate_biology_test", ROOT / "intergate/biology.py")
validation = load_module(
    "intergate_biology_validation_test", ROOT / "intergate/biology_validation.py"
)

rng = np.random.default_rng(20260828)
n_samples = 160
n_genes = 8
cohort = np.repeat(np.array(["A", "B", "C", "D"]), n_samples // 4)
y = np.tile(np.repeat(np.arange(4), 10), 4)
X = rng.normal(size=(n_samples, n_genes)).astype(np.float32)
X[y == 0, 1] = X[y == 0, 0] + rng.normal(0.0, 0.05, np.sum(y == 0))

edge_index = np.asarray(
    [[0, 0, 1, 2, 3, 4, 5, 6], [1, 2, 2, 3, 4, 5, 6, 7]],
    dtype=np.int64,
)
cfg = biology.DifferentialEvidenceConfig(
    min_samples_per_group=8, min_cohorts=2, edge_batch_size=3
)
evidence = biology.compute_differential_edge_evidence(
    X,
    y,
    cohort,
    np.arange(n_samples),
    edge_index,
    ["c0", "c1", "c2", "c3"],
    ["A", "B", "C", "D"],
    config=cfg,
)
assert evidence["context_score"].shape == (4, edge_index.shape[1])
assert np.allclose(evidence["context_score"].mean(axis=0), 0.0, atol=1e-6)
assert evidence["context_score"][0, 0] > 0.5

genes = [f"G{i}" for i in range(n_genes)]
fingerprint_dev = biology.evidence_fingerprint(
    X=X,
    genes=genes,
    edge_index=edge_index,
    y=y,
    cohort=cohort,
    train_idx=np.arange(n_samples),
    class_names=["c0", "c1", "c2", "c3"],
    config=cfg,
)
X_external = np.vstack(
    [X, rng.normal(size=(20, n_genes)).astype(np.float32)]
)
y_external = np.concatenate([y, np.zeros(20, dtype=int)])
cohort_external = np.concatenate([cohort, np.repeat("EXTERNAL", 20)])
fingerprint_after_append = biology.evidence_fingerprint(
    X=X_external,
    genes=genes,
    edge_index=edge_index,
    y=y_external,
    cohort=cohort_external,
    train_idx=np.arange(n_samples),
    class_names=["c0", "c1", "c2", "c3"],
    config=cfg,
)
assert fingerprint_dev == fingerprint_after_append

scaled = biology._rank_scale(np.ones(20))
assert np.array_equal(scaled, np.zeros(20, dtype=np.float32))
chosen = biology.module_aware_exact_k(
    edge_index,
    np.arange(edge_index.shape[1], dtype=float),
    k_edges=4,
    target_components=2,
    min_component_edges=2,
)
assert len(chosen) == 4 and len(set(chosen.tolist())) == 4

candidates = validation.candidate_edge_table(
    edge_index,
    np.ones(edge_index.shape[1]),
    np.arange(edge_index.shape[1]) % 2,
    genes,
)
selected = candidates.iloc[[0, 2, 4]][["candidate_edge_id"]]
sampled = validation.sample_candidate_matched_edges(
    selected, candidates, rng=rng
)
assert len(sampled) == len(selected)
selected_full = candidates.set_index("candidate_edge_id").loc[
    selected.candidate_edge_id
]
for column in (
    "edge_type",
    "sign",
    "source_degree_bin",
    "target_degree_bin",
):
    assert sorted(sampled[column]) == sorted(selected_full[column])

selected_edges = selected.merge(candidates, on="candidate_edge_id")
summary, distribution = validation.matched_edge_signature_null(
    selected_edges,
    candidates,
    {"signature": {"G0", "G1"}},
    n_draws=5,
    seed=19,
)
assert len(summary) == 3
assert len(distribution) == 3 * (1 + 5)
print("Synthetic train-only and matched-null checks: OK")
