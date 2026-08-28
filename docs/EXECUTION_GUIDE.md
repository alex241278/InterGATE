# Execution guide

The source notebooks are the scientific record. `scripts/run_pipeline.py`
executes copies without writing outputs into the tracked notebooks and follows
the dependency order required by the protocol.

## 1. Create the environment

```bash
mamba env create -f environment.yml
conda activate intergate
python scripts/preflight.py --stage data
```

Python 3.11 is the supported runtime. Install a platform-specific PyTorch/CUDA
build before `pip install -e ".[full]"` when required by the local GPU.

## 2. Prepare the seven cohorts

```bash
python scripts/run_pipeline.py data
```

Run from the repository root. The data runner uses `data_prepare/` as its
working directory so the public-download caches remain colocated with the
preparation notebooks.

Expected prepared dimensions are:

| Block | Cohorts | Samples | Prepared genes |
|---|---:|---:|---:|
| Development | 4 | 3,713 | 11,145 |
| External | 3 | 1,143 | 11,145 |

Reference M-ComBat is fitted with GSE96058 as the development reference. Each
external cohort is projected independently with the stored reference and gene
order; no external parameter is refitted.

## 3. Supply licensed resources

For the complete DepMap analysis, follow
`resources_biological/README.md`. To reproduce all other biological analyses
without DepMap:

```bash
export INTERGATE_RUN_DEPMAP=0
```

The full paper lock in notebook 06 requires the reported DepMap tables.

## 4. Run the manuscript workflow

```bash
python scripts/preflight.py --stage analysis
python scripts/run_pipeline.py analysis
```

The main computational blocks are:

- notebook 02: five structural and hybrid model fits;
- notebook 04: development baselines and four atomic retraining ablations;
- notebook 05: matched biological nulls and directed rewiring;
- notebook 07: single-use locked external prediction;
- notebook 12: factorial and edge-budget sensitivity controls;
- notebooks 10 and 13: figures, executed after their source tables exist.

By default the runner sets `INTERGATE_RUN_LABEL_PERMUTATIONS=0` for notebook
12. This reproduces the analyses reported in the manuscript while avoiding an
additional 49×2 full-training control. To run that extended control:

```bash
python scripts/run_pipeline.py analysis --from-step 12 --through-step 12 \
  --include-label-permutations
```

Permutation batches can be distributed explicitly:

```bash
INTERGATE_PERM_START=0 INTERGATE_PERM_STOP=10 \
python scripts/run_pipeline.py analysis --from-step 12 --through-step 12 \
  --include-label-permutations
```

## 5. Resume safely

Inspect the exact sequence:

```bash
python scripts/run_pipeline.py list
python scripts/run_pipeline.py analysis --dry-run
```

Resume a bounded interval:

```bash
python scripts/run_pipeline.py data --from-step 02
python scripts/run_pipeline.py analysis --from-step 04 --through-step 06
```

Notebook 06 writes `artifacts/intrinsic4/protocol_lock_v8.json`. Notebook 07
verifies every frozen hash and every source-notebook hash before reading the
external expression matrices and labels. Its completion marker prevents a
routine rerun.

## 6. Optional XGBoost biology comparison

`08c_xgboost_vs_intergate_biology.ipynb` uses the five frozen XGBoost objects
from notebook 08a. It verifies model hashes and the exact ordered 7,131-gene
feature universe, ranks genes only on development validation with TreeSHAP,
and opens the external block only after the gene panels are locked.

If its ordered-gene preflight fails, rebuild notebook 08a in a new output
directory with the current prepared matrices. Reordering features after model
fitting would invalidate TreeSHAP attribution and is therefore not permitted.

## 7. Repository audit

```bash
make test
```

This lightweight audit does not train the model. It checks source compilation,
notebook cleanliness and portability, protocol constants, graphical-abstract
values, and synthetic invariance of train-only biological evidence.
