# Repository update audit

Audit date: 2026-08-28

This audit reconciles the source repository with the current manuscript,
supplementary methods, Supplementary Data 1--3, and the publication
graphical abstract.

## Scientific contract

| Layer | Audited contract | Status |
|---|---|---|
| Cohorts | 4 development cohorts (3,713 tumours) and 3 locked external cohorts (1,143 tumours) | Aligned |
| Feature universe | 11,145 prepared genes; 7,131 prior-connected genes | Aligned |
| Prior | 53,205 candidate rows; 52,878 eligible directed units; Exact-\(K=529\) | Aligned |
| Structural learning | 5 seeds, 60 episodes, 192 microbatches of 2 samples and 2 optimizer updates per episode | Aligned |
| Structural support | 5 first-order steps with 64 microbatches of 2 samples per step | Aligned |
| Phase 2 | 5 graph-only epochs followed by at most 80 fixed-graph hybrid epochs | Aligned |
| Hybrid model | 7,131-gene tabular branch, width 256, gene identity, expert loss 0.20, tabular dropout 0.30, GNN loss 0.50 | Aligned |
| Biological score | Training support weight 0.25; development-fixed definitive support weight 0.075 | Aligned |
| Biological nulls | 1,000 candidate-matched panels and 1,000 directed degree/type/sign-preserving rewirings | Aligned |
| External block | Code, data, models and development results locked before external evaluation | Aligned |

The implementation-to-method mapping is detailed in
`docs/MANUSCRIPT_CODE_ALIGNMENT.md`.

## Result reconciliation

| Result | Manuscript / supplementary data | Repository and graphical abstract |
|---|---:|---:|
| Basal-like strict core | 299 edges / 288 genes | 299 / 288 |
| Mean Basal-like edge Jaccard | 0.607 | 0.607 |
| Edge provenance | 199 HuRI / 67 activation / 33 inhibition | Same |
| Candidate-matched recovery folds | 2.22 / 5.94 / 2.31 | Same |
| Canonical global FDR | 0.0096 | 0.0096 |
| External meta-Spearman | 0.691 | 0.691 |
| External direction agreement | 82.9% | 82.9% |
| External ensemble macro-F1 | 0.838 | 0.838 |
| External Basal-like F1 | 0.963 | 0.963 |
| Temperature / external ECE | 1.17394; 0.0652 to 0.0466 | Same |
| Basal signature recovery, InterGATE / XGBoost | 71 / 63 genes; matched folds 2.787 / 1.886 | Same |
| Prespecified Basal-like programmes, InterGATE / XGBoost | 7 / 1 | Same |
| Basal specificity margin, InterGATE / XGBoost | +1.325 / −1.180 | Same |
| Direct panel contrast | one-sided \(P=0.01399\); BH FDR 0.05594 | Same |
| Frozen-XGBoost external attribution | Spearman 0.9897; sign agreement 0.9888 | Same |

Supplementary Data 1 was checked directly: 299 unique edge rows, 288 unique
genes, 75 weak components, and a 199/67/33 prior-provenance split.
Supplementary Data 2 was checked directly for ensemble, class-specific,
paired-comparator, and calibration values.
Supplementary Data 3 was checked directly for the frozen-model audit,
size-matched gene panels, common matched-null recovery, specificity,
prespecified programmes, and external attribution transportability.

## Reproducibility corrections included in this update

- Replaced the legacy protocol and notebook set with the current canonical
  workflow, including calibration, fixed-edge external concordance, atomic
  biological ablations, sensitivity controls, and the secondary XGBoost
  biological comparison.
- Removed tracked notebook outputs, execution counts, absolute paths, caches,
  checkpoints, and duplicate historical figure notebooks.
- Made the final-figure preflight validate against the active cryptographic
  protocol lock instead of a run-specific hard-coded SHA.
- Restored the full matched structural budget: 384 sampled observations and
  two optimizer updates per episode, plus 128 support observations per step.
- Added a dependency-aware runner, preflight checks, static protocol guards,
  synthetic train-only-evidence tests, and continuous integration.
- Made the direct matched-null panel contrast tie-safe by computing empirical
  statistics from integer overlap-count differences before scaling.
- Rebuilt the graphical abstract as editable SVG, PDF, and PNG using the
  current architecture and manuscript values.

## Verification performed

```text
Python source compilation:                 PASS
Internal module import inventory:          PASS (no missing modules)
Notebook JSON and source compilation:      PASS (23 notebooks, 292 cells / 185 code cells)
Notebook cleanliness and portable paths:   PASS
Protocol/config/document/graphic contract: PASS
Synthetic train-only evidence invariance:  PASS
Candidate-matched null synthetic check:    PASS
Tie-safe XGBoost panel contrast check:      PASS
Pipeline dependency-order dry run:         PASS
Main-manuscript LaTeX build:               PASS (25 pages)
Supplementary-file LaTeX build:            PASS (11 pages)
PDF visual inspection:                     PASS
```

The audit validates the supplied implementation and reconciles the reported
outputs; it does not replace the full GPU retraining already used to generate
the locked study results.
