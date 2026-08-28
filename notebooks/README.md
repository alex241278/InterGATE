# Analysis notebooks

The tracked notebooks are clean and output-free. `scripts/run_pipeline.py`
executes copies under `results/executed_notebooks/analysis/` in dependency
order:

```text
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 07b
   → 08a → 08b → 08c → 09 → 12 → 10 → 13
```

The external block remains closed through notebook 06. Notebook 07 performs
the locked predictive evaluation, 07b applies development-fitted calibration,
08a evaluates frozen paired tabular models, 08b estimates fixed-edge external
effects without refitting or graph selection, and 08c compares size-matched
Basal-like gene and programme recovery using development-locked XGBoost
TreeSHAP attribution.

Notebook 12 produces the manuscript factorial and edge-budget sensitivity
tables. Its 49 full-retraining label permutations are disabled by the standard
runner and activated only with `--include-label-permutations`.

Notebook 08c is a secondary manuscript analysis. It verifies the frozen model
hashes and gene order, performs no refitting, and opens the external block only
after the development gene panels are locked.
