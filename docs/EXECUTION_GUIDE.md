# Execution guide

This guide expands the quick start in the repository root. The source
notebooks are the scientific record; `scripts/run_pipeline.py` executes clean
copies without writing outputs back into them.

## 1. Create the environment

```bash
mamba env create -f environment.yml
conda activate intergate
python scripts/preflight.py --stage data
```

The preflight requires Python 3.11 and verifies the packages used by the
download, endpoint, harmonization, and QC notebooks.

## 2. Prepare all public transcriptomic cohorts

```bash
python scripts/run_pipeline.py data
```

Run the command from the repository root. The runner intentionally sets the
working directory to `data_prepare/`, because notebooks 00a and 00b build their
download caches there. Public endpoints may be temporarily unavailable; the
notebooks cache successful downloads and fail closed if sample or class counts
change.

Expected final counts are:

| Block | Cohorts | Samples | Prepared genes |
|---|---:|---:|---:|
| Development | 4 | 3,713 | 11,145 |
| External | 3 | 1,143 | 11,145 |

Reference M-ComBat keeps GSE96058 unchanged, adjusts the development block,
and projects each external cohort separately. Notebook 03 verifies that adding
another external cohort cannot change development or an already projected
cohort beyond numerical precision.

## 3. Supply licensed optional resources

For the full DepMap analysis, follow
`resources_biological/README.md`. MSigDB GMT files are obtained automatically.
If DepMap is unavailable, set `INTERGATE_RUN_DEPMAP=0` and execute through
step 05. All other biological tests run, but the complete paper protocol cannot
be locked in step 06 without the reported DepMap outputs:

```bash
INTERGATE_RUN_DEPMAP=0 python scripts/run_pipeline.py analysis --through-step 05
```

## 4. Run the analysis

```bash
python scripts/preflight.py --stage analysis
python scripts/run_pipeline.py analysis
```

Notebook 02 is the computationally intensive five-seed fit and supports
resuming complete seed bundles. Notebook 07 is a locked, single-use external
evaluation. Its completion marker prevents accidental reopening on a routine
resume. Notebook 08 is a post-lock structural analysis, and notebook 11 applies
already frozen tabular comparators to the same external samples.

## 5. Resume safely

Inspect the sequence without running it:

```bash
python scripts/run_pipeline.py list
python scripts/run_pipeline.py analysis --dry-run
```

Resume a bounded interval:

```bash
python scripts/run_pipeline.py data --from-step 02
python scripts/run_pipeline.py analysis --from-step 04 --through-step 06
```

Do not delete or edit `artifacts/protocol_lock_v6.json` after notebook 06.
Notebook 07 checks hashes for the code, configuration, and source notebooks
before opening the external block.

## 6. Verify the repository

```bash
make test
```

This is a lightweight audit and does not train the model. It checks source
syntax, notebook cleanliness and order, portable paths, protocol constants,
and synthetic train-only invariance.
