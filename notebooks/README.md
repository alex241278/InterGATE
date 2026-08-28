# Analysis notebooks

The tracked notebooks are clean and output-free. `scripts/run_pipeline.py`
executes copies under `results/executed_notebooks/analysis/` in dependency
order:

```text
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 07b
   → 08a → 08b → 09 → 12 → 10 → 13
```

The external block remains closed through notebook 06. Notebook 07 performs
the locked predictive evaluation, 07b applies development-fitted calibration,
08a evaluates frozen paired tabular models, and 08b estimates fixed-edge
external effects without refitting or graph selection.

Notebook 12 produces the manuscript factorial and edge-budget sensitivity
tables. Its 49 full-retraining label permutations are disabled by the standard
runner and activated only with `--include-label-permutations`.

`08c_xgboost_vs_intergate_biology.ipynb` is an optional secondary analysis and
is not part of the manuscript execution sequence.
