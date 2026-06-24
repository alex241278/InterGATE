# InterGATE reproducibility and self-containment checklist

This release contains two reproducibility layers.

## 1. Offline self-contained checks

These commands must run from a clean clone without downloading manuscript data:

```bash
python scripts/01_smoke_test.py
python scripts/02_run_toy_example.py
```

They verify package imports, clean notebooks, absence of obvious absolute local paths, local toy data loading, cohort-aware splitting and parsing of a replaceable prior edge list.

## 2. Manuscript reproduction checks

After the Zenodo materials are downloaded and arranged:

```bash
python scripts/download_zenodo_data.py --extract
python scripts/00_check_setup.py
bash scripts/run_notebooks.sh
```

Expected local inputs after download are documented in `README.md` and `data/README.md`. The primary data DOI is `10.5281/zenodo.20817487`.

## Release hygiene

Before archiving or submitting the repository, run:

```bash
make clean
python scripts/01_smoke_test.py
python scripts/02_run_toy_example.py
```

Notebooks should be submitted without stored outputs or execution counts unless the journal explicitly requests executed notebooks.
