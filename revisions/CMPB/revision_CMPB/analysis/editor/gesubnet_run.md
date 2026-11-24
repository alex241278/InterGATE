# GeSubNet network comparison

The article, implementation and registry of the computational choices are identified in `gesubnet_protocol.json`. The matched-domain analysis uses the 146 gene symbols in the upstream BRCA expression file to identify eligible genes, and fits on the four original InterGATE development cohorts. Only 113 shared genes have an eligible within-domain edge. Checkpoint inference is inapplicable because the BRCA subgroup sample counts and the input widths in the five released NeoGNN checkpoints disagree, including in the public repository's initial data release. The notebook log matrices cannot be assigned to this BRCA expression data by padding, cropping or permuting samples.

Use a separate Python 3.11 environment with PyTorch 2.7.1, NumPy, pandas, SciPy, scikit-learn, IPython, Matplotlib, einops 0.4.0, torch-geometric 2.6.1 and CPU PyG wheels torch-scatter 2.1.2+pt27cpu and torch-sparse 0.6.18+pt27cpu. The scripts import the original release's SELM and NeoGNN classes. The comparator code is not included in the archive. From the root of the extracted revision archive, after regenerating the InterGATE training and external analysis outputs, run:

```bash
git clone https://github.com/chenzRG/GeSubNet upstream-GeSubNet
git -C upstream-GeSubNet checkout --detach a4fea879b2c56996672bb9e088a30f19c8c9ee41
python revision_CMPB/analysis/editor/gesubnet_comparison.py prepare --study InterGATE --upstream upstream-GeSubNet --work gesubnet-work
for seed in 1234 2025 7 99 31337
do
    python revision_CMPB/analysis/editor/gesubnet_comparison.py fit --study InterGATE --upstream upstream-GeSubNet --work gesubnet-work --seed "$seed"
done
python revision_CMPB/analysis/editor/evaluate_gesubnet.py --study InterGATE --work gesubnet-work --output gesubnet-results
```

The prepare step checks the training indices against the original validation labels and cohort IDs. The evaluation verifies the reconstructed biological score against all 529 originally exported edges in each seed immediately before ranking the common pairs. The training source never reads external expression or external edge effects. The evaluation attaches the frozen per-cohort and meta-analytic estimates after scoring. The full rankings use unthresholded logits and include the released decoder's single shared structural offset. See the methods fragment for objectives, fixed encodings, resource limits, adaptation choices and exact parameters.
