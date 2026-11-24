"""Direct, two-sided comparisons of seven named programmes in two fixed panels."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "InterGATE"
sys.path.insert(0, str(PROJECT))
from intergate.biology_validation import bh_adjust
from intergate.xgboost_intergate_biology import _matched_gene_draws, load_locked_annotations

SOURCE = PROJECT / "results_v8" / "08c_xgboost_intergate_biology"
LEFT = "InterGATE_strict_5of5"
RIGHT = "XGBoost_TreeSHAP_top288"
B = 4999


def main():
    covariates = pd.read_csv(SOURCE / "development_gene_matching_covariates.csv.gz")
    inventory = pd.read_csv(SOURCE / "locked_gene_panels.csv")
    named = pd.read_csv(SOURCE / "prespecified_basal_programmes.csv")
    panels = [set(inventory.loc[inventory.panel == name, "gene"]) for name in (LEFT, RIGHT)]
    left, right = panels
    genes = covariates.gene.to_numpy()
    assert len(genes) == len(set(genes)) == 7131
    assert len(left) == len(right) == 288 and len(left & right) == 25
    _, hallmark, c2, _ = load_locked_annotations(
        PROJECT / "resources_biological", PROJECT / "results_v8/05_biology_validation")
    draws = [_matched_gene_draws(panel, covariates, n_draws=B, random_seed=20261123+i)[0]
             for i, panel in enumerate(panels)]
    rows = []
    for term, sets in named.groupby("term", sort=False):
        members = (hallmark | c2)[term]
        shared = len(left & right & members)
        a, b = (len(panel & members) for panel in panels)
        archive = sets.set_index("panel").overlap_n
        assert (a, b) == (archive[LEFT], archive[RIGHT])
        # Conditional on shared genes and on the programme count in the symmetric difference.
        exclusive = len(left - right)
        table = [[a - shared, exclusive - a + shared], [b - shared, exclusive - b + shared]]
        p_exact = fisher_exact(table, alternative="two-sided").pvalue
        hits = np.isin(genes, list(members))
        ref = hits[draws[0]].sum(1) - hits[draws[1]].sum(1)
        obs = a - b
        # Symmetric two-tailed empirical test about the matched-null mean.
        null = ref.mean()
        total = int(ref.sum())
        # Multiplication by B gives exact integer distances from the null mean.
        p_matched = (1 + np.sum(np.abs(B * ref - total) >= abs(B * obs - total))) / (B+1)
        rows.append(dict(term=term, InterGATE_hits=a, XGBoost_hits=b, shared_hits=shared,
                         difference=obs, matched_null_mean=null, exact_p=p_exact,
                         matched_p=p_matched, null_q025=np.quantile(ref,.025), null_q975=np.quantile(ref,.975)))
    result = pd.DataFrame(rows)
    assert len(result) == 7
    for kind in ("exact","matched"):
        result[kind+"_BH7"] = bh_adjust(result[kind+"_p"])
    result.to_csv(HERE/"direct_programme_contrasts.csv",index=False)
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
