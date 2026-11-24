"""Summarise paired masking effects over the five fixed-split fits at each density."""
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2] / "InterGATE/results_v8/12c_kappa_retraining"
SEEDS = {1234, 2025, 7, 99, 31337}
T975_DF4 = 2.7764451051977987


def main():
    with (BASE / "kappa_branch_reliance_validation.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    complete = {(r["kappa"], r["seed"]): float(r["macro_f1"]) for r in rows if r["variant"] == "complete"}
    subsets = defaultdict(dict)
    for row in rows:
        if row["variant"] == "complete":
            continue
        key = (float(row["kappa"]), int(row["K"]), row["variant"])
        drop = complete[row["kappa"], row["seed"]] - float(row["macro_f1"])
        assert abs(drop - float(row["macro_f1_drop"])) < 1e-12
        seed = int(row["seed"])
        assert seed not in subsets[key]
        subsets[key][seed] = drop
    summary = []
    for (density, budget, variant), by_seed in sorted(subsets.items()):
        assert by_seed.keys() == SEEDS
        diffs = list(by_seed.values())
        mean = statistics.mean(diffs)
        half_width = T975_DF4 * statistics.stdev(diffs) / math.sqrt(len(diffs))
        summary.append(dict(density=density, directed_edges=budget, masked=variant, seeds=len(diffs),
                            validation_cases=743, mean_macro_F1_decrease=mean, seed_CI025=mean-half_width,
                            seed_CI975=mean+half_width, min_seed_decrease=min(diffs), max_seed_decrease=max(diffs)))
    path = HERE / "masking_reliance.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
