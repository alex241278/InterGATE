# Analysis notebooks

The tracked notebooks are clean and output-free. `scripts/run_pipeline.py`
executes copies under `results/executed_notebooks/analysis/` in dependency
order:

```text
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 07b
   → 08a → 08b → 08c → 09 → 12 → 12c → 12d → 10 → 13
```

The external block remains closed through notebook 06. Notebook 07 performs
the locked predictive evaluation, 07b applies development-fitted calibration,
08a evaluates frozen paired tabular models, and 08b estimates fixed-edge
external effects without refitting or graph selection.

Notebook 12 produces the factorial and subset-sensitivity controls. Its 49
full-retraining label permutations are disabled by the standard runner and
activated only with `--include-label-permutations`.

Notebook 12c retrains Phase 1 and Phase 2 for five Exact-K densities and writes
an integrity manifest. Notebook 12d regenerates Supplementary Figure S8 from
its compact source table. Notebook 08c compares frozen XGBoost attribution with
the InterGATE strict core and is reported as a secondary manuscript analysis.
