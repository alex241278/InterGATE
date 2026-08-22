# Analysis notebooks

The eleven source notebooks are clean, output-free, and executed in numerical
order by `scripts/run_pipeline.py analysis`. The first six operate with the
external transcriptomic block closed. Notebook 06 freezes hashes and the
protocol; notebook 07 then performs the locked external evaluation. Notebook
08 estimates post-lock external edge effects without refitting or selection.
Notebook 11 produces the paired predictive benchmark and Supplementary Table
12 outputs.

Executed copies are written outside this directory under
`results/executed_notebooks/analysis/`.

