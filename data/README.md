# Data layout

This repository expects the processed input data under:

```text
data/processed/expr_combat_corrected.csv
data/processed/metadata_combined.csv
```

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
10.5281/zenodo.19476488
```

If Zenodo stores the CSV files with different names, copy or rename them manually to the two required names in `data/processed/`.
