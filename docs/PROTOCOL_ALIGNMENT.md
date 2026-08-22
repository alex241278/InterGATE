# Protocol alignment

This file records the machine-readable contract used to keep the repository,
manuscript, and supplementary materials aligned.

| Item | Locked value |
|---|---|
| Endpoint | LumA, LumB, HER2-enriched, Basal-like |
| Development cohorts | GSE96058, GSE25066, GSE162228, GSE20711 |
| External cohorts | TCGA-BRCA, GSE21653, GSE1456 |
| Development samples | 3,713 |
| External samples | 1,143 |
| Prepared genes | 11,145 |
| HuRI--OmniPath-connected genes | 7,131 |
| Eligible directed interactions | 52,878 |
| Outer seeds | 1234, 2025, 7, 99, 31337 |
| Primary density | kappa = 0.01 |
| Exact edge budget | K = 529 per class and seed |
| Training support coefficient | 0.25 |
| Definitive biological support coefficient | 0.075 |
| Basal-like strict 5/5 core | 284 directed edges, 257 incident genes |

The 0.075 coefficient is selected using development-only structural
sensitivity after model fitting and before external opening. Reconstruction at
0.075 does not retrain the network. Both the training and definitive values are
represented in `configs/protocol_v6.json` so that neither is silently
overwritten by the other.

The predictive graph and biological association network have different
estimands. Direction and sign in the latter remain prior annotations combined
with class-contrastive evidence; they are not causal estimates.

