"""
InterGATE – Modular package for GNN-based breast cancer molecular subtype classification.

Modules:
    config       – Hyperparameters, paths, toggles
    utils        – Seeds, memory, formatting
    data         – Data loading, gene prep, splitting, scaling, Dataset/DataLoaders
    graph        – HuRI + OmniPath graph construction, regulator features
    model        – ResGATBlock, SignedResGATBlock, ImprovedSharedGraphGNN, HybridGNNTabular
    losses       – FocalLoss, metrics (AUC, F1, compute_metrics_full)
    training     – train_one_epoch, predict_proba, train_graph_learning, finetune_pruned
    pruning      – export_pruned_graph, evaluate_keep_ratios
    stability    – Edge-set utils, Jaccard, stability_edge_sets
    ablation     – AblationConfig, build_model_from_cfg, run_single_seed, run_ablation, artifacts
    axes         – Biological axes definitions, scores, OVR analysis, plotting
    enrichment   – gProfiler / Enrichr enrichment, KEGG map overlay
    visualization – Graph plotting, components grid, dotplot, barplot
    bootstrap    – Bootstrap CI for classification metrics
    benchmarks   – Tabular baselines, fixed-prior controls, PAM50, GCN, GraphSAGE, SOTA
    backbone_blocks – Drop-in weighted GraphSAGE, GIN and local GraphTransformer blocks
    backbone_ablation – Internal backbone-replacement ablation wrapper
    benchmarks_gnn_baselines – Optional PyG fixed-prior GIN/GraphTransformer baselines
    postprocessing – OVR thresholds, gene importance, post-hoc bundle loading
"""

__version__ = "1.0.0"
