"""Fit existing-batch adjustments in structural training, and vary external batch composition."""
import importlib.metadata
import json
import warnings

import numpy as np
import pandas as pd
from inmoose.pycombat.pycombat_norm import fit_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ADJUSTED, CLASSES, DATA, OUT, REPO, metadata, structural_split


def new_batch_coefficients(values, mu, sigma):
    """No-covariate, parametric reference ComBat; fit_model returns posterior location/scale.

    With a fixed reference, batch estimation is separable. fit_model is applied to a
    single standardized target batch. Reference moments have already been removed.
    Inputs and outputs are in raw, prepared-expression units.
    """
    n = values.shape[1]
    x = np.asmatrix((values - mu[:, None]) / sigma[:, None])
    gamma, delta, _ = fit_model(np.ones((1,n)), 1, x, [np.arange(n)], False, True, None, None)
    slope = 1 / np.sqrt(delta[0])
    intercept = mu * (1 - slope) - sigma * gamma[0] * slope
    return slope, intercept


def transform(values, coef):
    slope, intercept = coef
    return slope[:, None] * values + intercept[:, None]


def target_transform(batch, selected, mu, sigma):
    # Reproduce the original code's deterministic jitter, including its use in the
    # transformed output for constant genes, and the original biological sample order.
    values = batch.to_numpy(dtype=float, copy=True)
    constant = np.var(values, axis=1, ddof=1) <= 1e-8
    if constant.any():
        values[constant] += np.random.default_rng(0).normal(0, 1e-6, (constant.sum(), values.shape[1]))
    result = transform(values[:,batch.columns.get_indexer(selected)],new_batch_coefficients(values, mu, sigma))
    return pd.DataFrame(result.T,index=selected,columns=batch.index)


def fit_predictor(frame, samples, labels):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver='lbfgs',
                          class_weight='balanced', max_iter=1000, tol=1e-4))
    model.fit(frame.loc[samples], labels.loc[samples])
    if np.max(model[-1].n_iter_) == model[-1].max_iter:
        raise RuntimeError('Classifier failed to converge')
    return model


def absolute_differences(values):
    values = np.abs(np.asarray(values))
    return dict(median_abs=float(np.median(values)),p95_abs=float(np.quantile(values,.95)),
                max_abs=float(np.max(values)))


if __name__ == '__main__':
    warnings.filterwarnings('ignore',message='Degrees of freedom <= 0')
    raw = pd.read_csv(DATA/'output_prepared/expr_prepared.csv',index_col=0)
    contract = json.loads((ADJUSTED/'artefacto_mcombat/artefacto.json').read_text())
    raw = raw.loc[contract['genes']]
    assert len(raw)==11145
    mdev, mext = metadata('train'), metadata('test')
    train, val = structural_split(mdev)
    ref = mdev.dataset.eq('GSE96058')
    original_dev = pd.read_csv(ADJUSTED/'expr_train_mcombat.csv',index_col=0).T
    original_ext = pd.read_csv(ADJUSTED/'expr_test_mcombat.csv',index_col=0).T
    candidate = pd.read_csv(REPO/'results_v8/08c_xgboost_intergate_biology/development_gene_matching_covariates.csv.gz').gene
    assert candidate.is_unique and len(candidate)==7131
    # Repeat the batch variance eligibility rule in the fitting split, conditional on
    # the published upstream probe, floor and gene-universe filters.
    valid = np.ones(len(raw),bool)
    for cohort, block in mdev.loc[train].groupby('dataset'):
        valid &= raw[block.index].var(axis=1).to_numpy()>1e-8
    kept = raw.index[valid]
    # The public comparison workbook uses upper-case symbols, including C*orf genes.
    raw_symbols = {gene.upper(): gene for gene in kept}
    assert len(raw_symbols) == len(kept)
    features = [raw_symbols[gene] for gene in candidate]
    assert len(features) == 7131
    print('Fit start',len(kept),len(features),flush=True)
    mu = raw.loc[kept,train.intersection(mdev.index[ref])].mean(axis=1).to_numpy()
    sigma = raw.loc[kept,train.intersection(mdev.index[ref])].std(axis=1,ddof=0).to_numpy()
    mu0 = raw.loc[:,mdev.index[ref]].mean(axis=1).to_numpy()
    sigma0 = raw.loc[:,mdev.index[ref]].std(axis=1,ddof=0).to_numpy()
    strict_blocks = []
    checks, differences = [], []
    for cohort, block in mdev.groupby('dataset'):
        training = train.intersection(block.index)
        validation = val.intersection(block.index)
        values = raw.loc[kept,block.index]
        if cohort == 'GSE96058':
            adjusted = values.T.copy()
            old = raw[block.index].T
        else:
            coef = new_batch_coefficients(raw.loc[kept,training].to_numpy(),mu,sigma)
            adjusted = pd.DataFrame(transform(values.to_numpy(),coef).T,index=block.index,columns=kept)
            old_coef = new_batch_coefficients(raw[block.index].to_numpy(),mu0,sigma0)
            old = pd.DataFrame(transform(raw[block.index].to_numpy(),old_coef).T,index=block.index,columns=raw.index)
        check = np.max(np.abs(original_dev.loc[old.index,old.columns].to_numpy()-old.to_numpy()))
        checks.append(dict(cohort=cohort,block='development',max_abs_replay_error=check))
        if check>1e-5: raise RuntimeError(f'Development replay error {cohort}: {check}')
        differences.append(dict(cohort=cohort,validation_cases=len(validation),
            **absolute_differences(adjusted.loc[validation,features]-original_dev.loc[validation,features])))
        strict_blocks.append(adjusted[features])
    strict_dev = pd.concat(strict_blocks)
    fitted, outcome = {}, {}
    for method, matrix in [('pooled_development',original_dev[features]), ('training_only',strict_dev)]:
        fitted[method] = fit_predictor(matrix,train,mdev.label)
        outcome[method] = fitted[method].predict(matrix.loc[val,features])
    rows = []
    for cohort, idx in [('All',val),*[(g,b.index) for g,b in mdev.loc[val].groupby('dataset')]]:
        pos = val.get_indexer(idx)
        truth = mdev.loc[idx].label.to_numpy()
        a,b = outcome['pooled_development'][pos],outcome['training_only'][pos]
        row = dict(cohort=cohort,n=len(idx),n_changed=int(np.sum(a!=b)),
                macro_f1_pooled=f1_score(truth,a,average='macro',labels=CLASSES,zero_division=0),
                macro_f1_training_only=f1_score(truth,b,average='macro',labels=CLASSES,zero_division=0))
        row['delta']=row['macro_f1_training_only']-row['macro_f1_pooled']
        if cohort=='All':
            rng=np.random.default_rng(20261123)
            strata=mdev.loc[idx].groupby(['dataset','label']).indices
            interval=[]
            for rep in range(1000):
                q=np.concatenate([rng.choice(v,len(v),replace=True) for v in strata.values()])
                interval.append(f1_score(truth[q],b[q],average='macro',labels=CLASSES,zero_division=0)
                                -f1_score(truth[q],a[q],average='macro',labels=CLASSES,zero_division=0))
            row['delta_ci025'],row['delta_ci975']=np.quantile(interval,[.025,.975])
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'validation_classifier.csv',index=False)
    pd.DataFrame(differences).to_csv(OUT/'validation_expression_changes.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    # Every pair of 24-patient batches has eight identical sentinels, two per subtype,
    # and 16 additional tumors of a single subtype (Basal-like vs LumA). Repeat ten
    # times with independent draws; a retested sentinel is not an independent patient.
    rng = np.random.default_rng(20261123)
    sample_rows = []
    for cohort, block in mext.groupby('dataset'):
        model = fitted['pooled_development']
        full = target_transform(raw[block.index],block.index,mu0,sigma0)
        check = np.max(np.abs(full.to_numpy()-original_ext.loc[block.index,full.columns].to_numpy()))
        checks.append(dict(cohort=cohort,block='external',max_abs_replay_error=check))
        if check>1e-5: raise RuntimeError(f'External replay error {cohort}: {check}')
        for rep in range(10):
            sentinel = pd.Index(np.concatenate([rng.choice(g.index,2,replace=False) for _,g in block.groupby('label')]))
            backgrounds = {}
            for group in ['Basal-like','LumA']:
                available = block.index[block.label.eq(group)].difference(sentinel)
                chosen = rng.choice(available,16,replace=False)
                joint = block.index[block.index.isin([*sentinel,*chosen])]
                backgrounds[group] = target_transform(raw[joint],sentinel,mu0,sigma0)[features]
            aa,bb=backgrounds['Basal-like'],backgrounds['LumA']
            base = full.loc[sentinel,features]
            def probabilities(matrix):
                return model.predict_proba(matrix)
            pa,pb,pf=probabilities(aa),probabilities(bb),probabilities(base)
            for j,sample in enumerate(sentinel):
                sample_rows.append(dict(cohort=cohort,replicate=rep+1,sample=sample,label=block.loc[sample,'label'],
                   **absolute_differences(aa.iloc[j]-bb.iloc[j]),max_abs_prob=float(np.max(np.abs(pa[j]-pb[j]))),
                   different_label=bool(pa[j].argmax()!=pb[j].argmax()),
                   basal_background_disagrees_with_full=bool(pa[j].argmax()!=pf[j].argmax()),
                   lumA_background_disagrees_with_full=bool(pb[j].argmax()!=pf[j].argmax())))
        print('Complete external',cohort,flush=True)
    table=pd.DataFrame(sample_rows)
    table.to_csv(OUT/'external_composition_sentinel_results.csv',index=False)
    stats=table.groupby('cohort').agg(sentinel_evaluations=('sample','size'),distinct_sentinels=('sample','nunique'),
          median_gene_change=('median_abs','median'),median_max_probability_change=('max_abs_prob','median'),
          q95_max_probability_change=('max_abs_prob',lambda a:a.quantile(.95)),
          different_predictions=('different_label','sum'),
          changed_basal_vs_full=('basal_background_disagrees_with_full','sum'),
          changed_luma_vs_full=('lumA_background_disagrees_with_full','sum'))
    stats.to_csv(OUT/'external_composition_summary.csv')
    pd.DataFrame(checks).to_csv(OUT/'harmonization_replay.csv',index=False)
    manifest=dict(software={name:importlib.metadata.version(name) for name in ['inmoose','numpy','pandas','scikit-learn']},
        original_reference_n=int(ref.sum()),training_reference_n=int(len(train.intersection(mdev.index[ref]))),
        original_gene_universe=len(raw),training_valid_genes=len(kept),dropped_genes=list(raw.index[~valid]),
        common_predictive_genes=len(features),development_train=len(train),development_validation=len(val),
        composition_seed=20261123,n_repeats=10,segregated_companions_per_batch=16,sentinels_per_subtype=2)
    (OUT/'harmonization_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(stats)
