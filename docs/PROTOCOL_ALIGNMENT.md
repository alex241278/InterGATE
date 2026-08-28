# Protocol alignment

This table is the compact human-readable counterpart of
`configs/protocol_v8.json` and the manuscript methods.

| Item | Locked value |
|---|---|
| Endpoint | LumA, LumB, HER2-enriched, Basal-like |
| Development cohorts | GSE96058, GSE25066, GSE162228, GSE20711 |
| External cohorts | TCGA-BRCA, GSE21653, GSE1456 |
| Development / external samples | 3,713 / 1,143 |
| Prepared / connected genes | 11,145 / 7,131 |
| Candidate rows / eligible structural units | 53,205 / 52,878 |
| Structural seeds | 1234, 2025, 7, 99, 31337 |
| Primary density and edge budget | \(\kappa=0.01\); Exact-\(K=529\) per class and seed |
| Structural meta-learning | 60 episodes; 192 microbatches × 2 samples; 2 optimizer updates per episode |
| Cohort-deletion support | 5 first-order steps; 64 microbatches × 2 samples per step |
| Structural context | train-only within-cohort class-versus-rest differential correlation |
| Phase 2A | 5 graph-only epochs; fixed topology; \(X_h=0\) |
| Phase 2B | ≤80 hybrid epochs; 7,131-gene tabular branch; width 256 |
| Hybrid safeguards | trainable gene identity; expert loss 0.20; tabular branch dropout 0.30; GNN auxiliary loss 0.50 |
| Training support coefficient | 0.25 |
| Definitive biological support coefficient | 0.075, fixed in development without retraining |
| Biological nulls | 1,000 candidate-matched panels; 1,000 directed degree/type/sign-preserving rewirings |
| Basal-like strict 5/5 core | 299 directed edges; 288 genes; mean edge Jaccard 0.607 |
| External structural result | meta-analytic Spearman 0.691; direction agreement 82.9% |
| External ensemble result | macro-F1 0.838; Basal-like F1 0.963 |
| Calibration | development-fitted temperature 1.17394; external ECE 0.0652→0.0466 |

The predictive graph and biological association network are separate,
count-matched outputs. The total structural score defines the fixed predictive
topology. Centred class deviation, train-only contextual agreement, and
cohort-deletion support define the biological network. Curated prior direction
and sign provide the interaction semantics.
