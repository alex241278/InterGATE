# Data preparation

Run these notebooks before any model notebook. The supported order is:

1. `00a_download_microarrays.ipynb`
2. `00b_download_rnaseq.ipynb`
3. `01_union_datasets.ipynb`
4. `02_preparacion_y_suelo.ipynb`
5. `03_split_y_mcombat_ref.ipynb`
6. `04_qc_batch_y_fuga.ipynb`

The repository runner sets this directory as the working directory:

```bash
python scripts/run_pipeline.py data
```

The endpoint is restricted to four expression-defined intrinsic classes.
Clinical HER2 positivity is not substituted for HER2-enriched, and clinical
triple-negative status is not substituted for Basal-like. Expected cohort and
class counts are asserted in the download notebooks so upstream metadata
changes fail visibly.

