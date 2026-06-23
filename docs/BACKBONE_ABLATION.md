# Backbone-replacement diagnostic

This note documents the backbone add-on integrated into the self-contained `intergate` package and clarifies how the results should be interpreted in the manuscript and supplementary material.

## What was integrated

The add-on was not copied verbatim because it was still named `breastgnn`, included executed notebooks and contained local absolute paths. The GitHub-ready integration ports the reusable components into the `intergate` namespace:

- `intergate.backbone_blocks`: weighted GraphSAGE, weighted GIN and graph-transformer-style message-passing blocks with the same forward API expected by the InterGATE model code.
- `intergate.backbone_ablation`: utilities for running a backbone-replacement diagnostic while keeping the rest of the FULL InterGATE protocol active.
- `notebooks/7_Backbone_Ablation.ipynb`: an output-free notebook for running the internal backbone-replacement analysis.
- `scripts/run_backbone_ablation.sh`: shell entry point for the optional diagnostic.

## Conceptual interpretation

The rows `FULL_GAT`, `FULL_GRAPHSAGE`, `FULL_GIN`, and `FULL_GRAPH_TRANSFORMER` correspond to an internal backbone-replacement diagnostic. They are not fixed-prior graph baselines.

In this diagnostic, the FULL InterGATE protocol remains active: typed edge calibration, sample-conditioned gates, hard TopK sparsification, graph compaction and the hybrid head are kept, while only the message-passing backbone is changed.

The results supplied for the manuscript used seed 1 only. Therefore, they should be interpreted as an architectural diagnostic that supports the primary ResGAT/GAT choice, not as a replacement for the full stability-selected 291-gene consensus model.

## Paper-aligned result summary

| Configuration | Backbone | Seed | Phase-1 val macro-F1 | Final edges | Compact genes | Test accuracy | Test macro-F1 | Test weighted-F1 | Test OvR macro-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `FULL_GAT` | GAT/ResGAT | 1 | 0.864 | 205 | 264 | 0.925 | 0.924 | 0.927 | 0.993 |
| `FULL_GRAPH_TRANSFORMER` | GraphTransformer | 1 | 0.686 | 205 | 304 | 0.857 | 0.841 | 0.862 | 0.976 |
| `FULL_GIN` | GIN | 1 | 0.682 | 205 | 343 | 0.857 | 0.834 | 0.862 | 0.973 |
| `FULL_GRAPHSAGE` | GraphSAGE | 1 | 0.708 | 205 | 271 | 0.848 | 0.830 | 0.851 | 0.974 |

The associated raw files are:

- `docs/backbone_ablation_meta.json`
- `docs/backbone_ablation_results.csv`
- `docs/backbone_ablation_summary.csv`

## Manuscript placement

The recommended placement is:

- Main manuscript: brief Methods and Discussion mention only.
- Supplementary material: one table reporting the backbone-replacement diagnostic.
- The primary model remains the stability-selected InterGATE/FULL ResGAT configuration with 291 retained genes.
