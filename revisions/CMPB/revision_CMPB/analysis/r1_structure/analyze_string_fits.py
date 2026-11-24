"""Aggregate five fully trained STRING models in directed and unordered units."""
from itertools import combinations
import json
import numpy as np
import pandas as pd
from structural_sensitivity import (
    CandidateStatistics, HERE, PROJECT, PRIMARY, RESULTS, K, module_aware_exact_k, weak_topology_summary,
)
from intergate.biology import combine_biology_scores

OUT = HERE/"string_training"
RUN = OUT/"artifacts/INTERGATE_STRING_PHYSICAL"

def summarize():
    contract=json.loads((OUT/"string_training_contract.json").read_text())
    prior=pd.read_csv(OUT/"candidates.csv.gz")
    metrics=CandidateStatistics(prior)
    if len(prior)!=contract["eligible_directed_edges"] or not np.array_equal(metrics.ids,np.arange(len(prior))):
        raise RuntimeError("STRING candidate table/contract mismatch")
    full_ref=pd.read_csv(RESULTS/"09_ablation_biology_w0075/strict_5of5_edges/strict_5of5_FULL.csv")
    all_sets=[]; panels=[]; pairwise=[]; fitted=[]
    for seed in PRIMARY:
        folder=RUN/f"seed_{seed}"
        completed=json.loads((folder/"revision_result.json").read_text())
        meta=json.loads((folder/"meta.json").read_text())
        if not ((folder/"transport_checkpoint_COMPLETE.txt").is_file()
                and completed["metrics"]["evaluation_split"]=="development_validation"
                and meta["model_config_sha256"]==contract["model_configuration"]
                and meta["implementation_sha256"]==contract["implementation"]
                and meta["dataset_signature"]==contract["dataset"]):
            raise RuntimeError(f"STRING seed {seed} does not match its contract")
        fitted.append({"seed":seed,"development_macro_F1":completed["metrics"]["macro_f1"]})
        with np.load(folder/"transport_support_dense.npz", allow_pickle=True) as dense:
            if not np.array_equal(dense["structural_unit_id"],np.arange(len(prior))):
                raise RuntimeError("Expected one ordered structural unit per directed candidate")
            if not np.all(dense["biology_selected"].sum(axis=1)==K): raise RuntimeError("Unequal budgets")
            c=list(dense["class_names"]).index("Basal-like")
            scores=combine_biology_scores(dense["differential_logits"].astype(float),dense["context_score"].astype(float),
                dense["selection_frequency"].astype(float),context_weight=.35,support_weight=.075)
            ids=module_aware_exact_k(metrics.edges,scores[c],k_edges=K,pool_multiplier=6,
                                     target_components=12,min_component_edges=3)
        chosen=set(map(int,ids))
        if len(chosen)!=K: raise RuntimeError(f"Unexpected edge budget for seed {seed}")
        all_sets.append(chosen)
        panel=prior.iloc[ids].copy(); panel.insert(0,"seed",seed); panels.append(panel)
    ref_arc=set(zip(full_ref.source_gene,full_ref.target_gene))
    ref_pairs=set(tuple(sorted(pair)) for pair in ref_arc)
    ref_nodes=set(full_ref.source_gene)|set(full_ref.target_gene)
    strict=set.intersection(*all_sets)
    for i,j in combinations(range(len(PRIMARY)),2):
        pairwise.append({"seed_a":PRIMARY[i],"seed_b":PRIMARY[j],"directed_jaccard":
            len(all_sets[i]&all_sets[j])/len(all_sets[i]|all_sets[j])})
    frame=prior.iloc[sorted(strict)].copy()
    arcs=set(zip(frame.source_gene,frame.target_gene))
    pairs=set(tuple(sorted(pair)) for pair in arcs)
    nodes=set(frame.source_gene)|set(frame.target_gene)
    summary={"n_seeds":len(PRIMARY),"per_seed_directed_budget":K,
             **weak_topology_summary(metrics.edges[:,sorted(strict)]),
             "mean_seed_jaccard":float(np.mean([r["directed_jaccard"] for r in pairwise])),
             "core_undirected_pairs":len(pairs),"core_reciprocal_pairs":len(arcs)-len(pairs),
             "primary_core_directed_overlap":len(arcs&ref_arc),"primary_core_undirected_overlap":len(pairs&ref_pairs),
             "primary_core_gene_overlap":len(nodes&ref_nodes),
             "directed_jaccard_vs_primary":len(arcs&ref_arc)/len(arcs|ref_arc),
             "undirected_jaccard_vs_primary":len(pairs&ref_pairs)/len(pairs|ref_pairs),
             "gene_jaccard_vs_primary":len(nodes&ref_nodes)/len(nodes|ref_nodes),
             "signature_gene_count":int(metrics.flags[np.unique(metrics.edges[:,sorted(strict)])].sum())}
    summary.update(metrics.matched(strict))
    pd.DataFrame([summary]).to_csv(OUT/"five_seed_summary.csv",index=False)
    frame.to_csv(OUT/"five_seed_core_edges.csv.gz",index=False)
    pd.concat(panels,ignore_index=True).to_csv(OUT/"seed_panels.csv.gz",index=False)
    pd.DataFrame(pairwise).to_csv(OUT/"pairwise_overlap.csv",index=False)
    pd.DataFrame(fitted).to_csv(OUT/"development_performance.csv",index=False)

if __name__=="__main__": summarize()
