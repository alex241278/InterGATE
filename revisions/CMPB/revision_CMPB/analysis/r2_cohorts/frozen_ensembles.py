"""Evaluate frozen InterGATE and XGBoost predictions under changed target composition."""
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from common import ADJUSTED, CLASSES, DATA, OUT, REPO, metadata, structural_split
from mcombat_sensitivity import target_transform

sys.path.insert(0,str(REPO))
from intergate.ablation import predict_bundle_on_test, register_runtime
from intergate.config import CFG
from intergate.data import build_regulator_features

# Keep documented relative config defaults for architecture; use CPU inference.
seeds = [1234,2025,7,99,31337]
bench = REPO/'results_v8/08a_paired_predictive_benchmark'
archive = pd.read_csv(bench/'external/external_predictions_paired.csv.gz')
meta = metadata('test')
dev = metadata('train')
train, _ = structural_split(dev)
raw = pd.read_csv(DATA/'output_prepared/expr_prepared.csv',index_col=0)
contract = json.loads((ADJUSTED/'artefacto_mcombat/artefacto.json').read_text())
raw = raw.loc[contract['genes']]
full_dev = pd.read_csv(ADJUSTED/'expr_train_mcombat.csv',index_col=0)
full_ext = pd.read_csv(ADJUSTED/'expr_test_mcombat.csv',index_col=0)
prior = np.load(next((ADJUSTED/'backbone_cache').glob('*.npz')))['edge_index']
used = np.unique(prior)
features = list(raw.index[used])
expected = json.loads((bench/'development_benchmark_manifest.json').read_text())
assert hashlib.sha256('\n'.join(features).encode()).hexdigest() == expected['ordered_gene_sha256']
ref = raw[dev.index[dev.dataset.eq('GSE96058')]]
mu,sigma = ref.mean(axis=1).to_numpy(),ref.std(axis=1,ddof=0).to_numpy()
scaler = StandardScaler().fit(full_dev.loc[features,train].T.to_numpy(dtype=np.float32))
# Evaluate exactly the same 240 sentinel comparisons as the batch sensitivity.
# Keep all three rows per observation in their matched order for direct contrasts.
rng=np.random.default_rng(20261123)
frames, keys=[], []
for cohort, block in meta.groupby('dataset'):
    for rep in range(10):
        idx=pd.Index(np.concatenate([rng.choice(g.index,2,replace=False) for _,g in block.groupby('label')]))
        for group in ['Basal-like','LumA','full']:
            if group=='full':
                matrix=full_ext.loc[features,idx].T
            else:
                avail=block.index[block.label.eq(group)].difference(idx)
                buddies=rng.choice(avail,16,replace=False)
                joint=block.index[block.index.isin([*idx,*buddies])]
                matrix=target_transform(raw[joint],idx,mu,sigma)[features]
            frames.append(matrix.to_numpy(dtype=np.float32))
            keys.extend(dict(cohort=cohort,replicate=rep+1,sample=s,background=group) for s in idx)
features_raw=np.concatenate(frames)
keys=pd.DataFrame(keys)
assert len(keys)==720
archive_positions=meta.index.get_indexer(keys['sample'])+len(dev)
pcols=['p_'+c for c in CLASSES]
checks=[]


def publish(model, seed_prob):
    full=keys.background.eq('full').to_numpy()
    for seed,prob in zip(seeds,seed_prob):
        frozen=archive.query('model==@model and seed==@seed').set_index('sample_index')
        target=frozen.loc[archive_positions[full],pcols].to_numpy()
        check=float(np.abs(prob[full]-target).max())
        checks.append(dict(model=model,seed=seed,full_batch_max_probability_error=check,
                           n_argmax_mismatch=int(np.sum(prob[full].argmax(1)!=target.argmax(1)))))
        if check>5e-4 or checks[-1]['n_argmax_mismatch']:
            raise ValueError(f'Archive replay error: {checks[-1]}')
    avg=pd.DataFrame(np.mean(seed_prob,axis=0),columns=pcols)
    data=pd.concat([keys, avg],axis=1)
    data.to_csv(OUT/f'sentinel_probabilities_{model}.csv.gz',index=False)
    levels={k:g.set_index(['cohort','replicate','sample'])[pcols].sort_index() for k,g in data.groupby('background')}
    aa,bb,ff=[levels[k] for k in ['Basal-like','LumA','full']]
    assert aa.index.equals(bb.index) and aa.index.equals(ff.index)
    pairs=pd.DataFrame({'max_abs_probability_change':(aa-bb).abs().max(axis=1),
          'different_predictions':aa.idxmax(axis=1)!=bb.idxmax(axis=1),
          'changed_basal_vs_full':aa.idxmax(axis=1)!=ff.idxmax(axis=1),
          'changed_luma_vs_full':bb.idxmax(axis=1)!=ff.idxmax(axis=1)})
    stats=pairs.groupby(level='cohort').agg(evaluations=('different_predictions','size'),
               different_predictions=('different_predictions','sum'),
               changed_basal_vs_full=('changed_basal_vs_full','sum'),
               changed_luma_vs_full=('changed_luma_vs_full','sum'),
               median_max_probability_change=('max_abs_probability_change','median'))
    stats.to_csv(OUT/f'sentinel_summary_{model}.csv')
    pd.DataFrame(checks).to_csv(OUT/'frozen_model_replay.csv',index=False)
    print(model,stats.to_string(),flush=True)


probs=[]
for seed in seeds:
    path=bench/f'frozen_tabular_models/seed_{seed}/xgboost.joblib'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected['model_sha256'][str(seed)]['xgboost']
    xgb=joblib.load(path)
    xgb.set_params(device='cpu',n_jobs=2)
    probs.append(xgb.predict_proba(features_raw))
publish('xgboost',probs)
# The GNN and hybrid branch consume standardized expression; regulator features
# are recomputed for every changed matrix using the full, ordered candidate prior.
scaled=scaler.transform(features_raw)
new_index=np.full(len(raw),-1,dtype=np.int64)
new_index[used]=np.arange(len(used))
reg_feats, names=build_regulator_features(scaled,features,new_index[prior],
    stats=CFG.REG_STATS,min_targets=CFG.REG_MIN_GENES,max_regulators=CFG.REG_MAX_REGULATORS)
device=torch.device('cpu')
torch.set_num_threads(3)
CFG.BATCH_SIZE=8
register_runtime(Xs_gene=scaled,genes_kegg=features,X_h=reg_feats,n_classes=4,
    y=np.array([CLASSES.index(meta.loc[k,'label'])for k in keys['sample']]),
    cohort=keys['cohort'].to_numpy(),test_idx=np.arange(len(keys)))
probs=[]
for seed in seeds:
    bundle=REPO/f'artifacts/intrinsic4/INTERGATE_V8/seed_{seed}'
    ans=predict_bundle_on_test(str(bundle),device=device)
    assert ans['class_names']==CLASSES
    probs.append(ans['proba'])
    print(f'InterGATE seed {seed} complete',flush=True)
publish('InterGATE',probs)
