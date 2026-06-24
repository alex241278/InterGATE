# Data layout

The manuscript data are not committed to Git. Run:

```bash
python scripts/download_zenodo_data.py --extract
```

The downloader retrieves the Zenodo record `10.5281/zenodo.20817487`, verifies checksums when Zenodo exposes them, extracts archives and arranges the expected local layout:

```text
data/processed/expr_combat_corrected.csv
data/processed/metadata_combined.csv
cache/pipeline_cache/*.npz
artifacts_ablation/
artifacts_backbone_ablation/
```

For an offline package check that does not require the manuscript data, run:

```bash
python scripts/02_run_toy_example.py
```
