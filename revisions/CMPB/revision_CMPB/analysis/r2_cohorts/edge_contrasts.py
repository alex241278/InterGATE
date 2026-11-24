"""Exploratory post-review analyses of subtype composition and tumor purity."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import ADJUSTED, CLASSES, OUT, REPO, metadata, structural_split


sources = {
    'TCGA_mastercalls.abs_tables_JSedit.fixed.txt': '4f277128-f793-4354-a13d-30cc7fe9f6b5',
    'TCGA_all_leuk_estimate.masked.20170107.tsv': '6f75c9d7-5134-4ed1-b8f3-72856c98a4e8',
}
edges = pd.read_csv(REPO / 'results_v8/08b_external_structural_concordance/basal_strict_5of5_external_edge_effects.csv')
genes = sorted(set(edges.source_gene) | set(edges.target_gene))
source = np.array([genes.index(gene) for gene in edges.source_gene])
target = np.array([genes.index(gene) for gene in edges.target_gene])
keep_pair = ~pd.Series(list(zip(np.minimum(source, target), np.maximum(source, target)))).duplicated().to_numpy()
assert (len(edges), len(genes), sum(keep_pair)) == (299, 288, 214)


def edge_z(values, covariates=None):
    centered = values - values.mean(axis=0)
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=float)
        covariates = covariates - covariates.mean(axis=0)
        centered = centered - covariates @ np.linalg.lstsq(covariates, centered, rcond=None)[0]
    norms = np.sqrt(np.sum(centered ** 2, axis=0))
    denom = norms[source] * norms[target]
    numerator = np.einsum('ij,ij->j', centered[:, source], centered[:, target])
    # A zero-variance endpoint makes the correlation undefined. Fail visibly here.
    if np.any(denom < 1e-12):
        raise ValueError('Undefined correlation in a retained group')
    return np.arctanh(np.clip(numerator / denom, -.999999, .999999))


def group_contrast(expr, meta, comparator, columns):
    group = meta.label.eq('Basal-like')
    other = ~group if comparator == 'pooled-rest' else meta.label.eq(comparator)
    a, b = meta.loc[group], meta.loc[other]
    z_a = edge_z(expr.loc[a.index].to_numpy(), a[columns] if columns else None)
    z_b = edge_z(expr.loc[b.index].to_numpy(), b[columns] if columns else None)
    return z_a-z_b, len(a), len(b)


def load_expression(phase):
    chunks = pd.read_csv(ADJUSTED / f'expr_{phase}_mcombat.csv', index_col=0, chunksize=300)
    selected = pd.concat([chunk.loc[chunk.index.isin(genes)] for chunk in chunks])
    return selected.loc[genes].T


def summarize(delta, reference, **scope):
    for unit, mask in [('directed_rows', np.ones(len(edges), bool)), ('unordered_pairs', keep_pair)]:
        a, b = reference[mask], delta[mask]
        summaries.append(dict(**scope, unit=unit, units=len(a), spearman=spearmanr(a, b).statistic,
                              concordant_signs=int((np.sign(a)==np.sign(b)).sum()),
                              mean_absolute_effect=float(np.abs(b).mean())))


def tcga_covariates(meta):
    absolute = pd.read_csv(OUT / tuple(sources)[0], sep='\t').query('`call status` == "called"')
    leuk = pd.read_csv(OUT / tuple(sources)[1], sep='\t', header=None,
                       names=['study','barcode','leukocyte_fraction']).query('study == "BRCA"')
    absolute['id'] = absolute['array'].str[:15]
    leuk['id'] = leuk.barcode.str[:15]
    ids = meta.index.str.split(':').str[-1]
    assert ids.is_unique and all(len(s)==15 for s in ids)
    # RNA assay metadata identifies the sample-type barcode, not a particular aliquot.
    # Aggregate replicate DNA aliquots at that same sample type with a fixed median.
    out = meta.assign(purity=ids.map(absolute.groupby('id').purity.median()),
                      leukocyte_fraction=ids.map(leuk.groupby('id').leukocyte_fraction.median()))
    for col in ['purity','leukocyte_fraction']:
        assert out[col].dropna().between(0,1).all()
    return out


if __name__ == '__main__':
    metadata_dev, metadata_ext = metadata('train'), metadata('test')
    train, _ = structural_split(metadata_dev)
    metadata_dev = metadata_dev.loc[train]
    expr_dev, expr_ext = load_expression('train'), load_expression('test')
    counts, summaries, effects = [], [], {}
    comparators = ['pooled-rest', *CLASSES[:3]]
    for comp in comparators:
        meta_effects = {}
        for phase, frame, matrix in [('development',metadata_dev,expr_dev), ('external',metadata_ext,expr_ext)]:
            block, weights = [], []
            for cohort, subset in frame.groupby('dataset'):
                delta, nb, no = group_contrast(matrix, subset, comp, [])
                counts.append(dict(comparator=comp, phase=phase, cohort=cohort, n_basal=nb, n_comparator=no))
                weight = 1 / (1/(nb-3) + 1/(no-3))
                block.append(delta)
                weights.append(weight)
                effects[f'{phase}::{cohort}::{comp}'] = delta
                if phase == 'external':
                    summarize(delta, meta_effects['development'], comparator=comp,
                              cohort=cohort, adjustment='all_cases_unadjusted')
            meta_effects[phase] = np.average(block, weights=weights, axis=0)
            effects[f'{phase}::meta::{comp}'] = meta_effects[phase]
        summarize(meta_effects['external'], meta_effects['development'], comparator=comp,
                  cohort='external_meta', adjustment='all_cases_unadjusted')
        if comp=='pooled-rest':
            for phase,col in [('development','development_meta_effect'),('external','external_effect::EXTERNAL_META')]:
                np.testing.assert_allclose(meta_effects[phase], edges[col], atol=1e-5)
        tcga = tcga_covariates(metadata_ext.loc[metadata_ext.dataset.eq('TCGA-BRCA')])
        complete = tcga.dropna(subset=['purity','leukocyte_fraction'])
        for model, covs in [('complete_cases_unadjusted',[]), ('purity_only',['purity']),
                            ('purity_and_leukocytes',['purity','leukocyte_fraction'])]:
            delta, nb, no = group_contrast(expr_ext, complete, comp, covs)
            effects[f'external::TCGA-BRCA::{comp}::{model}'] = delta
            counts.append(dict(comparator=comp, phase='external', cohort=f'TCGA-BRCA::{model}',
                               n_basal=nb, n_comparator=no))
            summarize(delta, meta_effects['development'], comparator=comp,
                      cohort='TCGA-BRCA', adjustment=model)

    pd.DataFrame(counts).to_csv(OUT/'edge_contrast_denominators.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'edge_concordance.csv',index=False)
    pd.concat([edges[['candidate_edge_id','source_gene','target_gene']],pd.DataFrame(effects)],axis=1).to_csv(
        OUT/'edge_effects.csv.gz',index=False)
    coverage = tcga.groupby('label').agg(total=('label','size'),purity=('purity','count'),
                                          leukocyte_fraction=('leukocyte_fraction','count'))
    coverage['both'] = complete.groupby('label').size()
    coverage.to_csv(OUT/'tcga_covariate_coverage.csv')
    provenance = {f:{'url':f'https://api.gdc.cancer.gov/data/{uid}',
                     'sha256':hashlib.sha256((OUT/f).read_bytes()).hexdigest()} for f,uid in sources.items()}
    (OUT/'covariate_sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(pd.DataFrame(summaries).query('unit=="directed_rows" and (cohort=="external_meta" or adjustment != "all_cases_unadjusted")').to_string(index=False))
    print(coverage)
