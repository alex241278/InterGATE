# Reviewer 2 cohort and harmonisation analyses

Run these analyses from the project root with the project Python environment. All outputs are written beside the scripts. The inputs from the submission and the trained models are read in place. The source, checksum, and GDC identifier of each TCGA covariate file are in `covariate_sources.json`.

```bash
python revision_CMPB/analysis/r2_cohorts/edge_contrasts.py
python revision_CMPB/analysis/r2_cohorts/mcombat_sensitivity.py
python revision_CMPB/analysis/r2_cohorts/frozen_ensembles.py
```

`edge_contrasts.py` verifies the published pooled-rest development and external effects, then writes the results by comparator, cohort, covariate adjustment, and analysis unit in `edge_concordance.csv`. `edge_effects.csv.gz` retains the individual effects. The companion files report denominators and joint covariate coverage. The original expression-based subtypes define both the within-cohort strata and the paired sensitivity contrasts. The covariate analysis is restricted to the TCGA complete cases. The 15-character sample-type identifier matches the resolution of the RNA identifiers. Eight matched TCGA samples had multiple leukocyte-fraction aliquots and their sample-level estimate is the median.

`mcombat_sensitivity.py` replays the archived cohort matrices and estimates the three nonreference development-batch transformations with validation samples excluded. `validation_classifier.csv` compares independently fitted, otherwise identical L2 logistic classifiers with a paired patient bootstrap conditional on the training runs. The script also writes the fixed-size external composition experiment. The output denominator is the number of matched index-patient evaluations, which includes repeated sampling across draws.

`frozen_ensembles.py` applies the original five XGBoost and five InterGATE predictors to the same index patients. It verifies trained feature order and replays the archived full-batch probabilities before writing one summary file and one probability file per model. `frozen_model_replay.csv` reports the numerical reproduction error for every seed. `harmonization_manifest.json` records the M-ComBat fitting population and versions. The manuscript text in `cohort_methods.tex` and `cohort_results.tex` describes the post-review design and results.
