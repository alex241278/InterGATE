# Data layout

This repository expects the processed input data under:

```text
data/processed/expr_combat_corrected.csv
data/processed/metadata_combined.csv
```

For the paper-aligned reproducibility run, the Zenodo download script also arranges the cache and model artifacts under:

```text
cache/pipeline_cache/backbone__362de41006ca05c1__e71ae7463734.npz
cache/pipeline_cache/backbone_global__n11907__sig5f9c12d26c9c90d5__op1__huri1__score1__min0p0.npz
cache/pipeline_cache/HuRI.filtered.with_score.min0.0.impute_median.npz
cache/pipeline_cache/Xh__0575558be8d8a450__c102b7893b38__2060915c8ab9.npz
artifacts_ablation/
artifacts_backbone_ablation/
```

If Zenodo stores the backbone-ablation archive using a variant such as
`artifact_back_bone_ablation`, `download_zenodo_data.py --extract` normalizes it
to the root-level directory `artifacts_backbone_ablation/`.

The external network resources used by the graph constructor are looked up under:

```text
data/external/omnipath_interactions.tsv
data/external/HuRI.tsv
data/external/HuRI.psi
```

To prepare the layout from Zenodo, run from the project root:

```bash
python scripts/download_zenodo_data.py --extract
python scripts/00_check_setup.py
```

The data DOI configured in `intergate/config.py` is:

```text
10.5281/zenodo.20815745
```

If Zenodo stores the CSV files with different names, copy or rename them manually to the two required names in `data/processed/`.
