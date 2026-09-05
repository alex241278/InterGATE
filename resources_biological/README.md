# Biological resources

Notebook 05 downloads the MSigDB 2026.1.Hs Hallmark and C2:CGP GMT files into
`msigdb/` and records their release in the output manifest. The repository also
contains `PAM50_symbols.txt`, using current HGNC symbols for the original PAM50
genes, so the primary signature tests can exclude PAM50 members.

## DepMap

The orthogonal dependency analysis is locked to **DepMap Public 26Q1**. DepMap
files cannot be redistributed in this repository. Download the release and
place the following files under `depmap/26Q1/`:

```text
resources_biological/depmap/26Q1/
├── CRISPRGeneEffect.csv
├── Model.csv                         # `Models.csv` is also accepted
└── OmicsExpressionProteinCodingGenesTPMLogp1.csv
```

The alternative expression filename
`OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` is also accepted.

The expression file is required for the locked grouping rule. Among eligible
breast-lineage cell lines with Chronos scores and finite `ESR1`, `PGR`, and
`ERBB2` expression, Notebook 05 assigns `ERBB2-high` first
(`ERBB2` above the breast-lineage 75th percentile). Among the remaining
lines, `ESR1-or-PGR-high` requires `ESR1` or `PGR` above its corresponding
breast-lineage median; all remaining lines form the `ESR1/PGR-low` group.
These are relative-expression proxies, not clinical hormone-receptor or
HER2-amplification categories and not PAM50 assignments.

The committed [DepMap manifest](depmap_manifest.json) records the public
analysis contract and reported group counts. Each execution writes an expanded
run-specific manifest to
`results_v8/05_biology_validation/depmap_manifest.json`, including the
numeric thresholds, selected filenames, and SHA-256 hashes of the local 26Q1
inputs.

To execute the remaining biological validation without DepMap:

```bash
export INTERGATE_RUN_DEPMAP=0
```

This option supports analysis through Notebook 05. The full manuscript lock in
Notebook 06 requires DepMap to be complete because the reported outputs are
part of the frozen protocol.
