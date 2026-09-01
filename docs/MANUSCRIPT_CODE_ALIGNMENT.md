# Manuscript–code alignment audit

This crosswalk maps the current manuscript claims to executable code,
machine-readable settings, and result-producing notebooks. It was checked
against the main text, supplementary methods, Supplementary Data 1 and 2, and
the source package supplied for this repository update.

| Manuscript item | Implementation | Executable source | Alignment |
|---|---|---|---|
| Fixed intrinsic-four endpoint and 4/3 cohort partition | Labels and cohort lists are immutable protocol fields; external loading is opt-in | `configs/protocol_v8.json`; `intergate/config.py`; `intergate/prepared_data.py` | Exact |
| Reference M-ComBat without external refitting | Reference fitted in development; each external cohort projected independently | `data_prepare/combat_ref.py`; preparation notebooks 03–04 | Exact |
| 7,131 connected genes and 52,878 eligible structural units | Connected-only filter is applied after the 53,205-row HuRI–OmniPath candidate prior is built | `intergate/graph.py`; `intergate/data.py`; notebook 01 | Exact |
| Shared edge backbone plus centred class deviations | `differential_edge_logits()` centres deviations across the four experts | `intergate/model.py` | Exact |
| Every sample traverses all four experts | Complete class-expert logits are computed; coordinate \(c\) is retained from expert \(c\) | `intergate/model.py` | Exact |
| Train-only differential-correlation context | Context is computed from `train_idx`; the cache fingerprint is invariant to an appended external block | `intergate/biology.py`; `intergate/runtime_setup.py` | Exact |
| 60 cohort-balanced meta-episodes per seed | Leave-one-cohort-out episodes, balanced cohort×class sampling, first-order updates | `intergate/transport.py` | Exact |
| Matched structural budget after microbatch reduction | 192 steps × 2 samples = 384 draws and 2 accumulated optimizer updates per episode | protocol JSON; `intergate/config.py`; `intergate/runtime_setup.py` | Exact |
| Five-step cohort-deletion support | 64 microbatches × 2 samples = 128 support samples per step | `intergate/transport.py` | Exact |
| Predictive Exact-\(K\) graph | Cohort support, adapted total score, and candidate ID determine a deterministic 529-edge graph | `intergate/transport.py` | Exact |
| Biological Exact-\(K\) network | Centred deviation + 0.35 context + 0.075 support; module-aware selection from a 6\(K\) pool | `intergate/biology.py`; notebooks 04–05 | Exact |
| Five graph-only warm-up epochs | Hybrid tabular/fusion branches are bypassed; \(X_h=0\) | `intergate/training.py` | Exact |
| Fixed-graph hybrid fine-tuning | Structural parameters and masks are frozen; gates are bypassed; signed prior weights are renormalized | `intergate/model.py`; `intergate/training.py`; `intergate/transport.py` | Exact |
| Gene identity and multimodal safeguards | Zero-initialized 64-d identity, expert loss 0.20, tabular branch dropout 0.30, GNN loss 0.50 | `intergate/model.py`; `intergate/training.py`; protocol JSON | Exact |
| Post-hoc branch/message-passing neutralizations | GNN, message passing, identity, tabular, and \(X_h\) interventions use the five frozen models on development validation | `intergate/ablation.py`; notebook 03 | Exact |
| 4×4 signature specificity family | Four networks × four locked directional annotations × three endpoint statistics; one 48-test BH family | `intergate/biology_validation.py`; notebook 05 | Exact |
| Directed functional-coherence null | Double-edge swaps preserve in/out degree, edge type, prior sign, and edge count | `intergate/revision_analyses.py`; notebook 05 | Exact |
| Atomic structural ablations | Context loss, cohort-query loss, type calibration, and signed weight are each changed and retrained separately | `intergate/notebook04_controls.py`; notebooks 04 and 09 | Exact |
| Locked external prediction | Notebook 06 hashes code, data, models, and development results; notebook 07 verifies hashes before opening external data | notebooks 06–07 | Exact |
| Development-fitted calibration | Temperature is fitted to development-validation probabilities and applied unchanged to saved external predictions | `intergate/calibration.py`; notebook 07b | Exact |
| Paired tabular benchmark | Same ordered 7,131 genes, samples, split, and five seeds; frozen models applied to external cohorts | `intergate/predictive_benchmark.py`; notebook 08a | Exact |
| Fixed-edge external concordance | The 299 locked core edges are evaluated without refitting; null panels exclude selected edges and match type/sign/degrees | `intergate/posthoc_external_concordance.py`; notebook 08b | Exact |
| Factorial and subset sensitivity | Score source × selector control and strict-core subset curves | notebook 12; notebook 13 | Exact |
| Full-retraining Exact-\(K\) sensitivity | Phase 1 and Phase 2 are retrained at \(\kappa\in\{0.005,0.007,0.010,0.015,0.020\}\); configuration isolation verifies that only `KEEP_MIN` changes | notebook 12c; `source_data/12c_kappa_retraining/KAPPA_RETRAINING_MANIFEST.json` | Exact |
| Density-sensitivity figure | The figure reads the audited 12c summary table and does not refit or open external data | notebook 12d; `source_data/12c_kappa_retraining/Supplementary_Table_Kappa_retraining_sensitivity.csv` | Exact |
| XGBoost--InterGATE biological comparison | Frozen XGBoost TreeSHAP panels are size-matched to the 288-gene InterGATE core and tested against the same locked programme annotations | `intergate/xgboost_intergate_biology.py`; notebook 08c | Exact |
| Final quantitative figures | Figure code reads current result tables, verifies result manifests against the active cryptographic protocol lock, and validates expected manuscript inputs before export | `intergate/final_figures.py`; notebooks 10, 12d, and 13 | Exact |

## Result consistency checks

| Reported result | Audited source | Status |
|---|---|---|
| Basal-like strict core: 299 edges / 288 genes | Supplementary Data 1 contains 299 edge rows and 288 unique genes | Verified |
| Edge provenance: 199 HuRI / 67 activation / 33 inhibition | Supplementary Data 1 edge inventory | Verified |
| Mean pairwise Basal-like Jaccard: 0.607 | main and supplementary structural tables | Consistent |
| Candidate-matched folds: 2.30 / 5.50 / 2.60; global FDR 0.0010 | selected-edge-excluding matched-null family in manuscript tables | Consistent |
| External meta-Spearman 0.691; direction agreement 82.9% | fixed-edge concordance tables | Consistent |
| External ensemble macro-F1 0.838; Basal-like F1 0.963 | Supplementary Data 2 and manuscript table | Consistent |
| Temperature 1.17394; ECE 0.0652→0.0466 | calibration tables in Supplementary Data 2 | Consistent |
| Full-retraining macro-F1 range 0.8877--0.8896 | 12c five-density summary; five seeds per density | Verified |
| Basal-like stability transition at \(\kappa=0.010\) | Jaccard 0.513→0.607; Nogueira 0.674→0.752; strict fraction 0.435→0.565 from 0.007 to 0.010 | Verified |
| Graph and message-passing contribution across all densities | Positive frozen-model macro-F1 decreases at each of the five densities | Verified |

## Deliberately separate analyses

- The 49 full-retraining label permutations are an extended reviewer control
  and are opt-in in the runner. They are not used to select the model or any
  reported network.
- The XGBoost–InterGATE gene-program comparison is a secondary manuscript
  analysis. It reuses frozen XGBoost models, does not infer an XGBoost edge
  network, and remains distinct from the paired predictive benchmark.

## Methodological references

- Veličković P et al. Graph Attention Networks. *ICLR* (2018).
- Xu K et al. Representation Learning on Graphs with Jumping Knowledge
  Networks. *ICML* (2018).
- Neverova N et al. ModDrop: adaptive multi-modal gesture recognition.
  *IEEE TPAMI* (2016).
- Guo C et al. On Calibration of Modern Neural Networks. *ICML* (2017).
- Luck K et al. A reference map of the human binary protein interactome.
  *Nature* (2020).
- Türei D et al. Integrated intra- and intercellular signaling knowledge for
  multicellular omics analysis. *Molecular Systems Biology* (2021).
