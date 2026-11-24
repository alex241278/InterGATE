"""Review-time Basal-like re-export. Uses the submitted selector and all five dense arrays.

Run from any working directory. The optional --extended argument processes the nested
3/5/10/20 design up to the specified number of completed fits.
"""
import argparse
from collections import Counter
from itertools import combinations, product
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "InterGATE"
sys.path.insert(0, str(PROJECT))
from intergate.biology import _rank_scale, module_aware_exact_k, weak_topology_summary
from intergate.posthoc_biology_common import load_locked_signature_resources

RESULTS = PROJECT / "results_v8"
PRIMARY = (1234, 2025, 7, 99, 31337)
REFERENCE = (1., .35, .075)
K = 529
B = 1000
SEED = 20261123


class CandidateStatistics:
    """Calculate endpoint recovery against the exact source/degree-stratified null."""
    def __init__(self, prior):
        self.prior=prior.sort_values("candidate_edge_id").reset_index(drop=True)
        self.ids = self.prior.candidate_edge_id.to_numpy()
        self.lookup = pd.Index(self.ids)
        self.edges = self.prior[["source_idx","target_idx"]].to_numpy(dtype=int).T
        self.types = self.prior.edge_type.to_numpy()
        resource = load_locked_signature_resources(results_root=RESULTS, resource_dir=PROJECT / "resources_biological")
        signature = resource[0]["Basal-like"]
        flags = np.zeros(self.edges.max()+1, dtype=bool)
        for side in ("source", "target"):
            idx = self.prior[side+"_idx"].to_numpy()
            flags[idx] = self.prior[side+"_gene"].isin(signature)
        self.flags = flags
        self.signature_count = flags[self.edges].sum(axis=0)
        self.groups = np.array([f"{t}/{int(s)}/{i}/{j}" for t,s,i,j in self.prior[["edge_type","sign","source_degree_bin","target_degree_bin"]].itertuples(index=False,name=None)])
        self.pools = {s: np.flatnonzero(self.groups==s) for s in np.unique(self.groups)}
        self.null_cache = {}

    def matched(self, selected):
        """Sample without replacement, exactly matching type/sign/directed degree bins."""
        ids = np.array(sorted(selected),dtype=int)
        counts = Counter(self.groups[ids]); short=[]
        strata = []
        for key, n in sorted(counts.items()):
            pool = np.setdiff1d(self.pools[key], ids, assume_unique=True)
            if len(pool)<n: short.append(f"{key}={n}/{len(pool)}")
            strata.append((pool,n))
        one,both,node = self.observed(ids)
        out=dict(one_fraction=one, both_fraction=both, signature_node_fraction=node,
                 null_infeasible_strata=" ".join(short), null_draws=0)
        if short or not len(ids): return out
        key = tuple(ids)
        if key in self.null_cache: return self.null_cache[key]
        rng=np.random.default_rng(SEED)
        draws=np.empty((B,len(ids)),dtype=int)
        start=0
        for pool,n in strata:
            # For sparse samples, fixing collisions among independent draws is
            # rejection sampling from uniform ordered samples without replacement.
            if n==1:
                draws[:,start]=rng.choice(pool,B)
            elif n*n <= len(pool):
                take=rng.integers(0,len(pool),size=(B,n))
                bad=np.any(np.diff(np.sort(take,axis=1),axis=1)==0,axis=1)
                while bad.any():
                    take[bad]=rng.integers(0,len(pool),size=(int(bad.sum()),n))
                    bad[bad]=np.any(np.diff(np.sort(take[bad],axis=1),axis=1)==0,axis=1)
                draws[:,start:start+n]=pool[take]
            else:
                draws[:,start:start+n]=np.stack([rng.choice(pool,n,replace=False) for _ in range(B)])
            start+=n
        values=self.signature_count[draws]
        nulls=[(values>=1).mean(axis=1), (values==2).mean(axis=1),
               np.array([self.flags[np.unique(self.edges[:,i])].mean() for i in draws])]
        for name,observed,samples in zip(("one","both","node"),(one,both,node),nulls):
            out[name+"_null_mean"]=float(samples.mean())
            out[name+"_fold"]=observed/samples.mean() if samples.mean() else np.nan
            out[name+"_null_mc_se"]=float(samples.std(ddof=1)/np.sqrt(B))
        out["null_draws"]=B
        self.null_cache[key]=out
        return out

    def observed(self, ids):
        if not len(ids): return (np.nan,)*3
        counts=self.signature_count[ids]
        return (np.mean(counts>=1),np.mean(counts==2),self.flags[np.unique(self.edges[:,ids])].mean())


class Audit(CandidateStatistics):
    def __init__(self, include_external=True):
        prior_file = ("08b_external_structural_concordance/external_candidate_differential_effects.csv.gz" if include_external else "05_biology_validation/candidate_prior_for_null.csv.gz")
        self.include_external = include_external
        super().__init__(pd.read_csv(RESULTS / prior_file))
        self.reference = set(self.pos(pd.read_csv(RESULTS / "09_ablation_biology_w0075/strict_5of5_edges/strict_5of5_FULL.csv").candidate_edge_id))
        if len(self.ids)!=52878 or len(self.reference)!=299:
            raise RuntimeError("Unexpected candidate or reference panel")
        self.consensus = set(self.pos(pd.read_csv(RESULTS / "12_reviewer_controls/factorial_InterGATE_module_Basal-like.csv").candidate_edge_id))
        if len(self.consensus)!=K: raise RuntimeError("Submitted consensus has the wrong size")
        self.rank_terms = {}
        self.members = []
        self.selections = []
        self.rows = []

    def pos(self, candidate_ids):
        positions = self.lookup.get_indexer(candidate_ids)
        if (positions<0).any(): raise ValueError("Unknown candidate identifier")
        return positions

    def terms(self, seed):
        if seed not in self.rank_terms:
            path = (PROJECT / "artifacts/intrinsic4/INTERGATE_V8" if seed in PRIMARY else
                    HERE / "training/artifacts/INTERGATE_REVISION_SEEDS") / f"seed_{seed}"
            with np.load(path / "transport_support_dense.npz",allow_pickle=True) as dense:
                if not np.array_equal(np.flatnonzero(dense["structural_unit_id"]>=0), self.ids):
                    raise ValueError("Candidate order disagrees with dense arrays")
                units = dense["structural_unit_id"][self.ids]
                if not np.all(np.diff(units)>0): raise ValueError("Structural units are not monotone singletons")
                idx = list(dense["class_names"]).index("Basal-like")
                delta = dense["differential_logits"][idx].astype(float)
                q = dense["context_score"][idx].astype(float)
                f = dense["selection_frequency"][idx].astype(float)
                self.rank_terms[seed] = [_rank_scale(abs(delta))[self.ids],
                                         _rank_scale(np.sign(delta)*q,preserve_zeros=True)[self.ids],
                                         _rank_scale(f)[self.ids]]
        return self.rank_terms[seed]

    def select(self, seed, weights=REFERENCE, modules=12, quota=3, pool=6, source="all"):
        delta, context, support = self.terms(seed)
        a,b,c = weights
        scores = a*delta+b*context+c*support
        eligible = np.arange(len(self.ids))
        if source=="HuRI": eligible=eligible[self.types==0]
        if source=="OmniPath": eligible=eligible[self.types!=0]
        if modules==0:
            selected=eligible[np.lexsort((eligible, -scores[eligible]))[:K]]
        else:
            selected=eligible[module_aware_exact_k(self.edges[:,eligible],scores[eligible],
                              k_edges=K,target_components=modules,min_component_edges=quota,
                              pool_multiplier=pool)]
        if len(set(selected))!=K: raise RuntimeError("Invalid edge budget")
        return set(selected)

    def record(self,label,group,sets,row=None,store_selection=True):
        reference=self.reference
        core=set.intersection(*sets.values())
        cp=np.array(sorted(core),dtype=int)
        nodes=set(self.edges[:,cp].ravel())
        ref_nodes=set(self.edges[:,sorted(reference)].ravel())
        pairwise=[len(left & right)/len(left | right) for left,right in combinations(sets.values(),2)]
        fixed_prior=group=="prior_only"
        if store_selection:
            self.selections.extend((label,seed,int(self.ids[i])) for seed,ids in sets.items() for i in sorted(ids))
        self.members.extend((label,int(self.ids[i])) for i in cp)
        source=self.prior.iloc[cp]
        dev=source.development_meta_effect.to_numpy() if self.include_external else np.array([])
        ext=source["external_effect::EXTERNAL_META"].to_numpy() if self.include_external else np.array([])
        finite=np.isfinite(dev)&np.isfinite(ext)
        output=dict(specification=label,family=group,n_networks=len(sets),
                    mean_seed_jaccard=np.mean(pairwise) if pairwise else (1.0 if fixed_prior else np.nan),
                    jaccard_by_construction=fixed_prior, **weak_topology_summary(self.edges[:,cp]),
                    overlap_with_primary_529=len(core & self.consensus),jaccard_with_primary_529=len(core & self.consensus)/len(core | self.consensus),
                    core_edges_in_primary=len(core&reference),core_edge_jaccard=len(core&reference)/len(core|reference),
                    core_genes_in_primary=len(nodes&ref_nodes),core_gene_jaccard=len(nodes&ref_nodes)/len(nodes|ref_nodes),
                    signature_genes=int(self.flags[list(nodes)].sum()) if nodes else 0,
                    n_huri=int((self.types[cp]==0).sum()),external_evaluated=self.include_external,external_finite=int(finite.sum()),
                    external_meta_spearman=spearmanr(dev[finite],ext[finite]).statistic if finite.sum()>2 else np.nan,
                    external_sign_agreement=np.mean(np.sign(dev[finite])==np.sign(ext[finite])) if finite.any() else np.nan,
                    **self.matched(core),**(row or {}))
        self.rows.append(output)
        return output

    def run(self,extended_size=0):
        sets={s:self.select(s) for s in PRIMARY}
        for s, selected in sets.items():
            expected=pd.read_csv(RESULTS / f"12_reviewer_controls/factorial_seed_graphs/InterGATE_module_seed_{s}_Basal-like.csv")
            assert selected==set(self.pos(expected.candidate_edge_id)),f"Original panel mismatch in seed {s}"
        assert set.intersection(*sets.values())==self.reference
        self.record("primary","reference",sets,dict(a=1,b=.35,c=.075,modules=12,quota=3,pool=6))
        for subset in combinations(PRIMARY,3):
            self.record("trio_"+"_".join(map(str,subset)),"seed_subsets",{s:sets[s] for s in subset},store_selection=False)
        if extended_size:
            for n,extras in [(10,5),(20,15)]:
                if n>extended_size: continue
                added={s:self.select(s) for s in range(10001,10001+extras)}
                self.record(f"nested_{n}","extended_seeds",sets|added)
            return
        self.record("learned_529_consensus","equal_budget",{"aggregate":self.consensus})
        weights=list(product((.5,1.,2.),(0.,.35,.7),(0.,.075,.15)))
        weights += [(0.,.35,.075),(0.,.35,0.),(0.,0.,.075),(1.,.175,.0375),(1.,.35,.25),(1.,1.,.075)]
        for a,b,c in weights:
            if (a,b,c)==REFERENCE: continue
            label=f"weights_{a:g}_{b:g}_{c:g}"
            self.record(label,"weights",{s:self.select(s,(a,b,c)) for s in PRIMARY},
                        dict(a=a,b=b,c=c,modules=12,quota=3,pool=6))
        designs=[(m,n,6) for m,n in product((1,6,12,24),(1,3,6))]+[(0,0,0),(12,3,3),(12,3,9)]
        for m,n,p in designs:
            if (m,n,p)==(12,3,6): continue
            label=f"selector_{m}_{n}_{p}"
            self.record(label,"selector",{s:self.select(s,modules=m,quota=n,pool=p) for s in PRIMARY},
                        dict(a=1,b=.35,c=.075,modules=m,quota=n,pool=p))
        for name in ["HuRI","OmniPath"]:
            self.record("mask_"+name,"source_mask",{s:self.select(s,source=name) for s in PRIMARY})
        confidence=self.prior.prior_weight.abs().to_numpy()
        degree=(self.prior.source_out_degree*self.prior.target_in_degree).to_numpy()
        ids=np.arange(len(self.prior))
        for name, scores in [("absolute_prior_weight",confidence),("directed_degree_product",degree)]:
            chosen=np.lexsort((ids,-scores))[:K]
            cutoff=scores[chosen[-1]]
            self.record(name,"prior_only",{"fixed":set(chosen)},
                        dict(cutoff=cutoff,tied_at_cutoff=int((scores==cutoff).sum()),
                             places_at_cutoff=int((scores[chosen]==cutoff).sum())))
        # Source-balanced control uses the same three type totals as the full
        # prior (largest-remainder apportionment at K), not those of the core.
        types,counts=np.unique(self.types,return_counts=True)
        counts=K*counts/counts.sum(); sizes=np.floor(counts).astype(int)
        sizes[np.argsort(-(counts-sizes),kind="stable")[:K-sizes.sum()]]+=1
        fixed=set()
        for t,n in zip(types,sizes):
            idx=ids[self.types==t]
            fixed.update(idx[np.lexsort((idx,-confidence[idx]))[:n]])
        self.record("source_balanced_absolute_prior_weight","prior_only",{"fixed":fixed})
        ties=[]
        for label,values in [("absolute_prior_weight",confidence),("directed_degree_product",degree)]:
            cutoff=np.sort(values)[-K]; above=np.flatnonzero(values>cutoff); block=np.flatnonzero(values==cutoff)
            rng=np.random.default_rng(SEED)
            for replicate in range(B):
                choice=set(np.r_[above,rng.choice(block,K-len(above),replace=False)])
                ties.append((label,replicate,len(choice&self.reference),
                             len(choice&self.reference)/len(choice|self.reference),
                             *self.observed(np.array(sorted(choice)))))
        pd.DataFrame(ties,columns=["selector","tie_replicate","overlap_primary_core","jaccard_primary_core","one_fraction","both_fraction","signature_node_fraction"]).to_csv(HERE/"prior_random_tie_audit.csv.gz",index=False)

    def save(self, prefix):
        pd.DataFrame(self.rows).to_csv(HERE/f"{prefix}_summary.csv",index=False)
        pd.DataFrame(self.members,columns=["specification","candidate_edge_id"]).merge(
            self.prior[["candidate_edge_id","source_gene","target_gene","edge_type"]],on="candidate_edge_id",validate="many_to_one"
        ).to_csv(HERE/f"{prefix}_core_edges.csv.gz",index=False)
        pd.DataFrame(self.selections,columns=["specification","seed","candidate_edge_id"]).to_csv(HERE/f"{prefix}_seed_panels.csv.gz",index=False)
        print(pd.DataFrame(self.rows).groupby("family").size().to_string(),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended",type=int,choices=(10,20),default=0)
    options=parser.parse_args()
    audit=Audit(include_external=not options.extended)
    audit.run(options.extended)
    audit.save(f"extended_{options.extended}" if options.extended else "structure")
