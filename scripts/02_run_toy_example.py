#!/usr/bin/env python3
"""Run a tiny offline InterGATE data/graph check."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intergate.data import (
    load_expression_and_metadata, prepare_genes, encode_labels,
    cohort_split, scale_features, apply_connected_only,
)
from intergate.graph import load_edge_list_prior


def main() -> int:
    toy = PROJECT_ROOT / "examples" / "toy_data"
    X_df, y_str, cohort = load_expression_and_metadata(
        toy / "expr_toy.csv", toy / "metadata_toy.csv",
        sample_col="sample", label_col="label", cohort_col="batch",
    )
    X_df, genes = prepare_genes(X_df)
    y, classes, _ = encode_labels(y_str)
    train_idx, val_idx, test_idx = cohort_split(cohort, y, seed=42)
    Xs = scale_features(X_df, train_idx, mode="standard")
    edge_index, edge_weight, edge_type = load_edge_list_prior(toy / "prior_edges_toy.tsv", genes)
    Xs_conn, ei_conn, ew_conn, et_conn, genes_conn = apply_connected_only(
        Xs, edge_index, edge_weight, edge_type, genes
    )

    assert Xs_conn.shape[0] == len(y_str)
    assert ei_conn.shape[0] == 2
    assert ei_conn.shape[1] == len(ew_conn) == len(et_conn)
    assert len(classes) == 4
    assert len(train_idx) > 0 and len(val_idx) > 0

    print("[toy] classes:", classes)
    print("[toy] connected genes:", len(genes_conn), genes_conn)
    print("[toy] edge count:", ei_conn.shape[1])
    print("[ok] Toy example completed without external data or network access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
