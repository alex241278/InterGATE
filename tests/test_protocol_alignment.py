"""Machine-readable alignment between protocol, configuration, and manuscript facts."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
protocol = json.loads((ROOT / "configs/protocol_v6.json").read_text(encoding="utf-8"))
config = (ROOT / "intergate/config.py").read_text(encoding="utf-8")
posthoc = (ROOT / "intergate/posthoc_biology_common.py").read_text(encoding="utf-8")

assert protocol["endpoint"] == ["LumA", "LumB", "HER2-enriched", "Basal-like"]
assert protocol["development_cohorts"] == ["GSE96058", "GSE25066", "GSE162228", "GSE20711"]
assert protocol["external_cohorts"] == ["TCGA-BRCA", "GSE21653", "GSE1456"]
assert protocol["outer_seeds"] == [1234, 2025, 7, 99, 31337]
assert protocol["kappa"] == 0.01
assert protocol["biology_graph"]["training_score_terms"]["cohort_support"] == 0.25
assert protocol["biology_graph"]["definitive_locked_score_terms"]["cohort_support"] == 0.075
assert protocol["biology_graph"]["score_lock_timing"] == "development_only_after_model_fitting_before_external_opening"

patterns = {
    "KEEP_MIN": "0.01",
    "BIOLOGY_SCORE_SUPPORT_WEIGHT": "0.25",
    "DEFINITIVE_BIOLOGY_SCORE_SUPPORT_WEIGHT": "0.075",
}
for name, value in patterns.items():
    assert re.search(
        rf"^\s*{name}:.*=\s*{re.escape(value)}\s*$", config, re.MULTILINE
    ), name

assert "LOCKED_SUPPORT_WEIGHT = 0.075" in posthoc
assert "EXPECTED_EXACT_K = 529" in posthoc
assert "EXPECTED_FULL_STRICT_EDGES = 284" in posthoc
print("Protocol alignment: OK")
