"""Compare fixed primary interactions to STRING physical links and fit its expression-context baseline.

Usage: python string_prior.py /directory/containing/the/two/STRING/v12.0/files
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from structural_sensitivity import Audit, HERE, PROJECT, RESULTS, B, SEED
from intergate.biology import compute_differential_edge_evidence, DifferentialEvidenceConfig
from intergate.posthoc_biology_common import load_locked_signature_resources

def read_resource(directory, genes, save_manifest=True):
    info = directory / "9606.protein.info.v12.0.txt.gz"
    physical = directory / "9606.protein.physical.links.v12.0.txt.gz"
    meta = pd.read_csv(info,sep="\t",usecols=["#string_protein_id","preferred_name"])
    assert meta["#string_protein_id"].is_unique
    symbols = pd.Series(meta.preferred_name.str.upper().values,index=meta["#string_protein_id"])
    gene_index = {gene:i for i,gene in enumerate(genes)}
    mapping = symbols.map(gene_index).dropna().astype(int)
    edges=pd.read_csv(physical,sep=" ")
    links=edges[["protein1","protein2"]].apply(lambda col:col.map(mapping))
    keep=links.notna().all(axis=1)&links.protein1.ne(links.protein2)
    pairs=np.sort(links[keep].to_numpy(dtype=int),axis=1)
    table=pd.DataFrame(pairs,columns=["i","j"])
    table["string_score"]=edges.loc[keep,"combined_score"].to_numpy()
    table=table.groupby(["i","j"],as_index=False,sort=True).string_score.max()
    inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [info,physical]}
    resource_manifest = {
        "release":"STRING v12.0, Homo sapiens, physical links (undirected)",
        "urls":["https://stringdb-downloads.org/download/protein.info.v12.0/"+info.name,
                "https://stringdb-downloads.org/download/protein.physical.links.v12.0/"+physical.name],
        "sha256":inputs,"mapping":"exact uppercase STRING preferred name to submitted ordered 7131 genes",
        "gene_universe":len(genes),"matched_proteins":len(mapping),
        "unique_physical_pairs_within_universe":len(table),
        "context_score_baseline_minimum":700}
    if save_manifest:
        (HERE/"string_manifest.json").write_text(json.dumps(resource_manifest,indent=2)+"\n")
    return table

def primary_coverage(table, audit, genes):
    pairs=np.sort(audit.edges.T,axis=1)
    full_idx=pairs[:,0]*len(genes)+pairs[:,1]
    physical_idx=table.i.to_numpy()*len(genes)+table.j.to_numpy()
    scores=pd.Series(table.string_score.values,index=physical_idx)
    core=np.array(sorted(audit.reference))
    unique_ids,first,inv = np.unique(full_idx,return_index=True,return_inverse=True)
    core_ids=np.unique(inv[core])
    universe=scores.reindex(unique_ids).fillna(0).to_numpy(dtype=float)
    frame=pd.DataFrame({"gene_a":np.asarray(genes)[unique_ids[core_ids]//len(genes)],
                        "gene_b":np.asarray(genes)[unique_ids[core_ids]%len(genes)],
                        "STRING_score":universe[core_ids]})
    frame.to_csv(HERE/"string_core_coverage.csv",index=False)
    strata=[]
    for key,q in zip(*np.unique(audit.groups[core],return_counts=True)):
        pool=np.setdiff1d(audit.pools[key],core,assume_unique=True)
        if len(pool)<q: raise RuntimeError("Cannot perform primary matched null")
        strata.append((pool,q))
    rng=np.random.default_rng(SEED); thresholds=(150,400,700,900)
    draws=[]
    for _ in range(B):
        selected=np.concatenate([rng.choice(pool,q,replace=False) for pool,q in strata])
        observed=universe[np.unique(inv[selected])]
        draws.append([np.mean(observed>=t) for t in thresholds])
    result=[]
    for col,t in enumerate(thresholds):
        obs=np.mean(universe[core_ids]>=t)
        null=np.asarray(draws)[:,col]
        result.append(dict(STRING_minimum_score=t, n_reference_pairs=len(core_ids),
                    n_in_STRING=int((universe[core_ids]>=t).sum()),observed_fraction=obs,
                    n_prior_pairs=len(unique_ids),prior_fraction=np.mean(universe>=t),
                    null_mean=null.mean(),null_q025=np.quantile(null,.025),null_q975=np.quantile(null,.975),
                    fold=obs/null.mean(),B=B))
    pd.DataFrame(result).to_csv(HERE/"string_coverage_vs_matched_prior.csv",index=False)
    return set(unique_ids[core_ids])

def alternate_context(table,ref,genes,info):
    # The original gene-wise transform is affine. Reuse the exact registered
    # training scaler and compare q on the shared pairs to the submitted array.
    from intergate.data import load_expression_and_metadata, scale_features
    X, labels, cohorts=load_expression_and_metadata()
    y=np.array([info["label_map"][label] for label in labels],dtype=np.int64)
    train_idx=np.array(info["splits"]["train_idx"])
    assert hashlib.sha256(y.tobytes()).hexdigest()==info["dataset_signature"]["labels_sha256"]
    assert hashlib.sha256("\x1f".join(map(str,cohorts)).encode()).hexdigest()==info["dataset_signature"]["cohorts_sha256"]
    from intergate.config import CFG
    X=scale_features(X[genes],train_idx,mode=CFG.SCALE_MODE)
    edges=table.loc[table.string_score>=700].copy().reset_index(drop=True)
    ij=edges[["i","j"]].to_numpy().T
    result=compute_differential_edge_evidence(X,y,cohorts,train_idx,ij,
                    ["LumA","LumB","HER2-enriched","Basal-like"],CFG.TRAIN_COHORTS,
                    config=DifferentialEvidenceConfig())
    edges["q_basal"]=result["context_score"][3]
    edges["pair"]=edges.i*len(genes)+edges.j
    audit=Audit()
    old_ij=np.sort(audit.edges.T,axis=1)
    old_ids=old_ij[:,0]*len(genes)+old_ij[:,1]
    with np.load(PROJECT/"artifacts/intrinsic4/INTERGATE_V8/seed_1234/transport_support_dense.npz") as dense:
        original=pd.Series(dense["context_score"][3,audit.ids],index=old_ids).groupby(level=0).mean()
    common=edges[edges.pair.isin(original.index)]
    err=float(np.max(abs(common.q_basal.to_numpy()-original.loc[common.pair].to_numpy())))
    if len(common)<1 or err>3e-4: raise RuntimeError(f"Context anchor mismatch on {len(common)} shared pairs: {err}")
    locked=load_locked_signature_resources(results_root=RESULTS,resource_dir=PROJECT/"resources_biological")[0]["Basal-like"]
    sig=np.isin(genes,list(locked)); indicators=sig[ij].sum(axis=0)
    sizes=np.bincount(ij.ravel(),minlength=len(genes))
    ranks={"physical_confidence":edges.string_score.to_numpy(),"undirected_degree_product":sizes[ij[0]]*sizes[ij[1]],
           "absolute_development_q":abs(edges.q_basal.to_numpy())}
    summaries=[]; panels=[]
    # For an undirected prior the annotation is symmetric. Match the sorted
    # pair of degree-bin indices in the eligible high-confidence graph.
    quantiles=np.unique(np.quantile(sizes[ij].ravel(),np.linspace(0,1,11)))
    bins=np.sort(np.searchsorted(quantiles[1:-1],sizes[ij],side="right"),axis=0)
    groups=bins[0]*11+bins[1]
    all_ids=np.arange(len(edges))
    for name,rank in ranks.items():
        ids=np.lexsort((all_ids,-rank))[:529]
        nodes=np.unique(ij[:,ids])
        selected_pairs=set(edges.iloc[ids].pair)
        genes_in_ref=set([int(x)//len(genes) for x in ref]+[int(x)%len(genes) for x in ref])
        row=dict(selector=name,n_STRING_eligible_pairs=len(edges),n_eligible_genes=len(np.unique(ij)),
                 n_pairs=len(ids),n_genes=len(nodes),n_signature_genes=int(sig[nodes].sum()),
                 one_fraction=float((indicators[ids]>=1).mean()),both_fraction=float((indicators[ids]==2).mean()),
                 node_fraction=float(sig[nodes].mean()),pairs_shared_with_primary=len(selected_pairs&ref),
                 genes_shared_with_primary=len(set(nodes)&genes_in_ref),
                 n_shared_pairs_q_checked=len(common),max_abs_q_difference=err)
        pools=[]; short=[]
        for g,n in zip(*np.unique(groups[ids],return_counts=True)):
            pool=np.setdiff1d(np.flatnonzero(groups==g),ids,assume_unique=True)
            if len(pool)<n: short.append((int(g),int(n),len(pool)))
            pools.append((pool,n))
        row["infeasible_strata"]=str(short) if short else ""
        if not short:
            rng=np.random.default_rng(SEED); vals=[]
            for _ in range(B):
                select=np.concatenate([rng.choice(pool,n,replace=False) for pool,n in pools])
                vals.append(((indicators[select]>=1).mean(),(indicators[select]==2).mean(),sig[np.unique(ij[:,select])].mean()))
            for source,null_mean in zip(("one","both","node"),np.mean(vals,axis=0)):
                row[source+"_null_mean"]=null_mean
                row[source+"_fold"]=row[source+"_fraction"]/null_mean if null_mean else np.nan
        summaries.append(row)
        frame=edges.iloc[ids].copy()
        frame["gene_a"]=np.asarray(genes)[frame.i];frame["gene_b"]=np.asarray(genes)[frame.j]
        frame["selector"]=name
        panels.append(frame)
    pd.DataFrame(summaries).to_csv(HERE/"string_independent_rankings.csv",index=False)
    pd.concat(panels).to_csv(HERE/"string_selected_pairs.csv.gz",index=False)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("source",type=Path);args=parser.parse_args()
    p=PROJECT/"artifacts/intrinsic4/INTERGATE_V8/seed_1234"
    genes=list(torch.load(p/"graph_bundle.pt",map_location="cpu",weights_only=False)["genes_full"])
    tables=read_resource(args.source,genes)
    ref=primary_coverage(tables,Audit(),genes)
    alternate_context(tables,ref,genes,json.loads((p/"meta.json").read_text()))
