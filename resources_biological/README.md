# Biological resources

Notebook 05 downloads the MSigDB 2026.1.Hs Hallmark and C2:CGP GMT files into
`msigdb/` and records their release in the output manifest. The repository also
contains `PAM50_symbols.txt`, using current HGNC symbols for the original PAM50
genes, so the primary signature tests can exclude PAM50 members.

DepMap files cannot be redistributed here. To reproduce the orthogonal
essentiality analysis, download one compatible DepMap release and place these
files in `depmap/`:

```text
resources_biological/depmap/
├── CRISPRGeneEffect.csv
├── Model.csv                         # `Models.csv` is also accepted
└── OmicsExpressionProteinCodingGenesTPMLogp1.csv
```

The expression file is required only when `Model.csv` does not contain enough
molecular annotation to construct the three operational breast-cell-line
proxy groups. Alternative accepted expression filename:
`OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`.

The release and file hashes are written to `depmap_manifest.json`. These proxy
groups are not treated as PAM50 equivalents. To execute the remaining
biological validation without DepMap:

```bash
export INTERGATE_RUN_DEPMAP=0
```

This option supports analysis through notebook 05. The full manuscript lock in
notebook 06 requires DepMap to be complete, because those reported outputs are
part of the frozen protocol.
