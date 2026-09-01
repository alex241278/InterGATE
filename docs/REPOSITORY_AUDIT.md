# Repository update audit

Audit date: 2026-08-30

This audit reconciles the source repository with the current manuscript,
supplementary methods, Supplementary Data 1 and 2, and the publication
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
| Exact-\(K\) sensitivity | Five densities × five seeds; complete Phase 1 and Phase 2 retraining; only `KEEP_MIN` changed | Aligned |

The implementation-to-method mapping is detailed in
`docs/MANUSCRIPT_CODE_ALIGNMENT.md`.

## Result reconciliation

| Result | Manuscript / supplementary data | Repository and graphical abstract |
|---|---:|---:|
| Basal-like strict core | 299 edges / 288 genes | 299 / 288 |
| Mean Basal-like edge Jaccard | 0.607 | 0.607 |
| Edge provenance | 199 HuRI / 67 activation / 33 inhibition | Same |
| Candidate-matched recovery folds | 2.30 / 5.50 / 2.60 | Same |
| Three-statistic FDR | 0.0010 | 0.0010 |
| External meta-Spearman | 0.691 | 0.691 |
| External direction agreement | 82.9% | 82.9% |
| External ensemble macro-F1 | 0.838 | 0.838 |
| External Basal-like F1 | 0.963 | 0.963 |
| Temperature / external ECE | 1.17394; 0.0652 to 0.0466 | Same |
| Full-retraining predictive range | macro-F1 0.8877--0.8896 over \(\kappa=0.005\)--0.020 | Same |
| Basal-like stability transition | Jaccard 0.513→0.607 and Nogueira 0.674→0.752 from \(\kappa=0.007\) to 0.010 | Same |
| Graph reliance across density grid | GNN and message-passing neutralizations reduce macro-F1 at all five densities | Same |

Supplementary Data 1 was checked directly: 299 unique edge rows, 288 unique
genes, 75 weak components, and a 199/67/33 prior-provenance split.
Supplementary Data 2 was checked directly for ensemble, class-specific,
paired-comparator, and calibration values.

## Reproducibility corrections included in this update

- Replaced the legacy protocol and notebook set with the current canonical
  workflow, including calibration, fixed-edge external concordance, atomic
  biological ablations, sensitivity controls, and the optional XGBoost
  biological comparison.
- Removed tracked notebook outputs, execution counts, absolute paths, caches,
  checkpoints, and duplicate historical figure notebooks.
- Made the final-figure preflight validate against the active cryptographic
  protocol lock instead of a run-specific hard-coded SHA.
- Restored the full matched structural budget: 384 sampled observations and
  two optimizer updates per episode, plus 128 support observations per step.
- Added a dependency-aware runner, preflight checks, static protocol guards,
  synthetic train-only-evidence tests, and continuous integration.
- Rebuilt the graphical abstract as editable SVG, PDF, and PNG using the
  current architecture and manuscript values.
- Added the complete 12c density-retraining notebook, its integrity manifest,
  compact publication source tables, and the output-free 12d figure notebook.
- Verified that the implementation fingerprint and protocol-file SHA-256 in
  the 12c manifest exactly match this repository snapshot.

## Verification performed

```text
Python source compilation:                 PASS
Internal module import inventory:          PASS (no missing modules)
Notebook JSON and source compilation:      PASS (26 notebooks, 208 code cells)
Notebook cleanliness and portable paths:   PASS
Protocol/config/document/graphic contract: PASS
Synthetic train-only evidence invariance:  PASS
Candidate-matched null synthetic check:    PASS
Pipeline dependency-order dry run:         PASS
Main-manuscript LaTeX build:               PASS (26 pages)
Supplementary-file LaTeX build:            PASS (17 pages)
PDF visual inspection:                     PASS
```

The audit validates the supplied implementation and reconciles the reported
outputs; it does not replace the full GPU retraining already used to generate
the locked study results.
