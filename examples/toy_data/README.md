# Toy data for offline self-contained checks

This directory contains a tiny expression matrix, metadata table and replaceable prior edge list. It is not used for the manuscript results; it only verifies that the package can be imported, read a sample-by-gene dataset, parse a local prior edge list and run the preprocessing/splitting utilities without downloading Zenodo materials.

Files:

- `expr_toy.csv`: genes x samples expression matrix.
- `metadata_toy.csv`: sample, label and batch/cohort metadata.
- `prior_edges_toy.tsv`: local directed/undirected edge list with optional signs/weights.
