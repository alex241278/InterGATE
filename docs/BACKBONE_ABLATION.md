# Backbone ablation and fixed-prior graph controls

This note documents the integration of the backbone add-on into the self-contained `intergate` package.

## What was integrated

The add-on was not copied verbatim because it was still named `breastgnn`, included executed notebooks and contained local absolute paths. The GitHub-ready integration ports only the reusable components into the `intergate` namespace:

- `intergate.backbone_blocks`: weighted GraphSAGE, weighted GIN and local graph-transformer-style message-passing blocks with the same forward API as `ResGATBlock`.
- `intergate.backbone_ablation`: a wrapper that temporarily patches `intergate.model.ResGATBlock` during model construction, so the rest of the InterGATE protocol remains unchanged.
- `intergate.benchmarks_gnn_baselines`: optional PyG fixed-prior GIN and GraphTransformer controls that complement the existing GraphSAGE control.
- `notebooks/7_Backbone_Ablation.ipynb`: a clean, output-free notebook for running the internal backbone-replacement ablation.

## Conceptual separation

There are two different analyses and they should not be merged without labelling them clearly.

### 1. Internal backbone-replacement ablation

This analysis asks whether the ResGAT message-passing block is the best backbone inside the proposed sparse graph-learning framework. Gate learning, typed edge calibration, sample-conditioned gates, hard TopK pruning, stability/fine-tuning logic and the hybrid head remain active. Only the message-passing block changes.

This is the correct interpretation of rows such as `FULL_GAT`, `FULL_GRAPHSAGE`, `FULL_GIN` and `FULL_GRAPH_TRANSFORMER` produced by `intergate.backbone_ablation`.

### 2. Fixed-prior graph controls

This analysis asks whether ordinary message passing on the fixed biological prior is sufficient without supervised topology learning. In these controls, edge gates are not learned and the graph is held fixed. These rows belong with other benchmark controls, not with the internal architecture ablation.

## Coherence of the observed outputs supplied with the add-on

The saved executed add-on notebooks contained two different result families.

1. A `FULL_GAT` bundle evaluated with validation-optimized one-vs-rest thresholds produced a compact model with 264 genes and an external-test macro-F1 of approximately 0.923. This is coherent with the main InterGATE result because it is close to the reported primary performance, but it should not replace the final 291-gene consensus result unless the exact same consensus/stability and decision-rule protocol is used.

2. The simple fixed-prior GNN baseline cell produced much lower mean external-test macro-F1 values across seeds: GraphSAGE about 0.496, GIN about 0.302 and GraphTransformer about 0.536. These values should not replace the manuscript benchmark table if the manuscript table was produced with a different validated fixed-prior protocol. They are useful as a diagnostic showing that a simple fixed-prior PyG implementation is weak under this split, but they are not interchangeable with internal backbone-ablation results.

## Recommended reporting

For the manuscript/supplementary material, report the backbone-replacement analysis as an internal ablation. A suitable wording is:

> We further evaluated whether the observed performance depended on the residual graph-attention message-passing backbone. We repeated the InterGATE sparse graph-learning protocol while replacing the ResGAT block with weighted GraphSAGE, weighted GIN, or a local graph-transformer-style block. Because typed edge calibration, sample-conditioned gates, hard sparsification and the hybrid fine-tuning head remained active, this experiment should be interpreted as an internal backbone ablation rather than as a fixed-prior GNN baseline.

## Key references

- Veličković et al. introduced Graph Attention Networks, which motivate the attention-based backbone family used by ResGAT.
- Hamilton et al. introduced GraphSAGE, used here as a mean-aggregation backbone/control family.
- Xu et al. introduced GIN, used here as a sum-aggregation backbone/control family.
- Dwivedi and Bresson, and Shi et al., are commonly cited for graph-transformer-style message passing.
- Meinshausen and Bühlmann introduced stability selection, which motivates consensus graph selection.
