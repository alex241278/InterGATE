"""CPU-only checks for the V6 biology and null-selection primitives."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str):
    path = ROOT / "intergate" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"v6_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


biology = load_module("biology")
validation = load_module("biology_validation")

rng = np.random.default_rng(4)
n_samples, n_genes = 160, 8
X = rng.normal(size=(n_samples, n_genes)).astype(np.float32)
y = np.tile(np.arange(4), 40)
cohort = np.repeat(np.asarray(["A", "B", "C", "D"]), 40)
for cohort_name in ("A", "B", "C", "D"):
    ids = np.flatnonzero((cohort == cohort_name) & (y == 0))
    X[ids, 1] = X[ids, 0] + rng.normal(0.0, 0.05, len(ids))

edge_index = np.asarray([
    [0, 0, 1, 2, 3, 4, 5, 6],
    [1, 2, 2, 3, 4, 5, 6, 7],
], dtype=np.int64)
cfg = biology.DifferentialEvidenceConfig(
    min_samples_per_group=8,
    min_cohorts=2,
    edge_batch_size=3,
)
evidence = biology.compute_differential_edge_evidence(
    X, y, cohort, np.arange(n_samples), edge_index,
    ["c0", "c1", "c2", "c3"], ["A", "B", "C", "D"],
    config=cfg,
)
assert evidence["context_score"].shape == (4, edge_index.shape[1])
assert np.allclose(evidence["context_score"].mean(axis=0), 0.0, atol=1e-6)
assert evidence["context_score"][0, 0] > 0.5

# Appending a locked external block cannot change the train-only cache key.
fingerprint_dev = biology.evidence_fingerprint(
    X=X, genes=[f"G{i}" for i in range(n_genes)], edge_index=edge_index,
    y=y, cohort=cohort, train_idx=np.arange(n_samples),
    class_names=["c0", "c1", "c2", "c3"], config=cfg,
)
X_external = np.vstack([X, rng.normal(size=(20, n_genes)).astype(np.float32)])
y_external = np.concatenate([y, np.zeros(20, dtype=int)])
cohort_external = np.concatenate([cohort, np.repeat("EXTERNAL", 20)])
fingerprint_after_append = biology.evidence_fingerprint(
    X=X_external, genes=[f"G{i}" for i in range(n_genes)],
    edge_index=edge_index, y=y_external, cohort=cohort_external,
    train_idx=np.arange(n_samples), class_names=["c0", "c1", "c2", "c3"],
    config=cfg,
)
assert fingerprint_dev == fingerprint_after_append

# Saturated support must not create candidate-order information.
scaled = biology._rank_scale(np.ones(20))
assert np.array_equal(scaled, np.zeros(20, dtype=np.float32))

chosen = biology.module_aware_exact_k(
    edge_index, np.arange(edge_index.shape[1], dtype=float),
    k_edges=4, target_components=2, min_component_edges=2,
)
assert len(chosen) == 4 and len(set(chosen.tolist())) == 4

genes = [f"G{index}" for index in range(n_genes)]
candidates = validation.candidate_edge_table(
    edge_index, np.ones(edge_index.shape[1]),
    np.arange(edge_index.shape[1]) % 2, genes,
)
selected = candidates.iloc[[0, 2, 4]][["candidate_edge_id"]]
sampled = validation.sample_candidate_matched_edges(
    selected, candidates, rng=rng,
)
assert len(sampled) == len(selected)
selected_full = candidates.set_index("candidate_edge_id").loc[
    selected.candidate_edge_id
]
for column in (
    "edge_type", "sign", "source_degree_bin", "target_degree_bin"
):
    assert sorted(sampled[column]) == sorted(selected_full[column])

selected_edges = selected.merge(candidates, on="candidate_edge_id")
summary, distribution = validation.matched_edge_signature_null(
    selected_edges, candidates, {"signature": {"G0", "G1"}},
    n_draws=5, seed=19,
)
assert len(summary) == 3
assert len(distribution) == 3 * (1 + 5)
print("V6 synthetic checks: OK")
