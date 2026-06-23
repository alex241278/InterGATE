#!/usr/bin/env bash
set -euo pipefail

python scripts/00_check_setup.py

jupyter nbconvert --to notebook --execute --inplace notebooks/0_Main_Pipeline.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/1_Results_Bootstrap.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/2_Ejes_REFINED_AXES.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/3_Ejes_Figura_residuales.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/4_Lista_genes_por_eje.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/5_Bioinformatics_Benchmarks.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/6_Benchmarks_SOTA.ipynb
