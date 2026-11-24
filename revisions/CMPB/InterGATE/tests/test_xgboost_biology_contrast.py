"""Regression test for tie-safe empirical panel contrasts."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intergate.xgboost_intergate_biology import matched_null_panel_contrast


def main() -> None:
    panel_size = 288
    recovery = pd.DataFrame([
        {
            "panel": "InterGATE_strict_5of5",
            "annotation_class": "Basal-like",
            "panel_size": panel_size,
            "overlap_n": 71,
            "overlap_fraction": 71 / panel_size,
        },
        {
            "panel": "XGBoost_TreeSHAP_top288",
            "annotation_class": "Basal-like",
            "panel_size": panel_size,
            "overlap_n": 63,
            "overlap_fraction": 63 / panel_size,
        },
    ])

    # Ten null differences exceed the observed count difference and three are
    # exact ties.  All ties must contribute to the empirical upper tail.
    differences = np.zeros(1000, dtype=int)
    differences[:10] = 9
    differences[10:13] = 8
    null_rows = []
    for draw, difference in enumerate(differences):
        null_rows.extend([
            {
                "panel": "InterGATE_strict_5of5",
                "annotation_class": "Basal-like",
                "draw": draw,
                "overlap_n": 100 + int(difference),
            },
            {
                "panel": "XGBoost_TreeSHAP_top288",
                "annotation_class": "Basal-like",
                "draw": draw,
                "overlap_n": 100,
            },
        ])
    result = matched_null_panel_contrast(
        recovery, pd.DataFrame(null_rows)
    ).iloc[0]
    assert result["observed_fraction_difference_left_minus_right"] == 8 / 288
    assert np.isclose(result["empirical_p_left_greater"], 14 / 1001)
    print("XGBoost--InterGATE empirical contrast tie handling: OK")


if __name__ == "__main__":
    main()
