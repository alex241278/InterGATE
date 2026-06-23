#!/usr/bin/env bash
set -euo pipefail

python scripts/00_check_setup.py
jupyter nbconvert --to notebook --execute --inplace notebooks/7_Backbone_Ablation.ipynb
