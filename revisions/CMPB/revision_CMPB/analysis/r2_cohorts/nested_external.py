"""Summarise archived external effects in completed, nested Basal-like cores."""
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[1]
STRUCTURE = HERE.parent / 'r1_structure'
MANUSCRIPT = REVISION / 'manuscript'
WORKBOOK_NAME = 'Supplementary_Data_1_Basal_like_strict_core.xlsx'
WORKBOOK = MANUSCRIPT / 'Supplementary Data' / WORKBOOK_NAME
if not WORKBOOK.is_file():
    WORKBOOK = REVISION.parent / '2027_InterGATE' / 'Supplementary Data' / WORKBOOK_NAME
SCOPES = ('TCGA-BRCA', 'GSE21653', 'GSE1456', 'EXTERNAL_META')
DEV = 'development::meta::pooled-rest'


def spearman(x, y):
    """Pearson correlation of average ranks, undefined for too few or constant values."""
    if len(x) != len(y):
        raise ValueError('Development and external effects have different lengths')
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0:
        return np.nan
    ranks_x = pd.Series(x).rank(method='average').to_numpy()
    ranks_y = pd.Series(y).rank(method='average').to_numpy()
    return float(np.corrcoef(ranks_x,ranks_y)[0,1])


def normalized(genes):
    if genes.isna().any():
        raise ValueError('Missing gene symbol')
    return genes.astype(str).str.strip().str.upper()


def add_pair(frame):
    frame = frame.copy()
    left, right = normalized(frame.source_gene), normalized(frame.target_gene)
    assert left.ne(right).all(), 'Self-loop in nested strict core'
    frame['pair_a'] = left.where(left < right, right)
    frame['pair_b'] = right.where(left < right, left)
    return frame


def read_original_sheet(workbook, name):
    rows = workbook[name].values
    for row in rows:
        if row and row[0] in ('candidate_edge_id', 'panel'):
            return pd.DataFrame(list(rows), columns=row)
    raise ValueError(f'Header missing from {name}')


def archived_analysis():
    wb = load_workbook(WORKBOOK, read_only=True, data_only=True)
    try:
        pairs = add_pair(read_original_sheet(wb, 'Core_pair_effects'))
        reference = read_original_sheet(wb, 'Core_concordance').query('unit == "unique_unordered_pairs"')
    finally:
        wb.close()
    assert normalized(pairs.pair_gene_a).equals(pairs.pair_a)
    assert normalized(pairs.pair_gene_b).equals(pairs.pair_b)
    key = ['pair_a', 'pair_b']
    pairs = pairs.set_index(key, verify_integrity=True).sort_index()
    full = add_pair(pd.read_csv(HERE/'edge_effects.csv.gz')).set_index('candidate_edge_id', verify_integrity=True)
    assert len(full) == 299 and len(pairs) == 214
    # Confirm pair membership against ALL archived candidate IDs, including cases
    # in which the representative direction is absent from a later intersection.
    all_ids = []
    for pair, row in pairs.iterrows():
        ids = [int(token) for token in str(row.candidate_ids).split(';')]
        all_ids.extend(ids)
        assert len(set(ids)) == len(ids) == row.n_directed_rows
        assert [(r.pair_a, r.pair_b) for r in full.loc[ids].itertuples()] == [pair]*len(ids)
    assert len(all_ids) == len(set(all_ids)) == len(full) and set(all_ids) == set(full.index)
    effect_cols = [DEV]+[f'external::{"meta" if scope=="EXTERNAL_META" else scope}::pooled-rest' for scope in SCOPES]
    grouped = full.groupby(key)[effect_cols]
    assert (grouped.max()-grouped.min()).to_numpy().max() < 1e-12, 'Disagreement between reciprocal rows'
    precise = grouped.first().reindex(pairs.index)
    assert np.isfinite(precise).all().all()
    return full, pairs, precise, reference.set_index('scope', verify_integrity=True)


def nested_directed_sets(full):
    previous = None
    for n in (5, 10, 20):
        name = 'structure' if n==5 else f'extended_{n}'
        spec = 'primary' if n==5 else f'nested_{n}'
        summary_path = STRUCTURE/f'{name}_summary.csv'
        edges_path = STRUCTURE/f'{name}_core_edges.csv.gz'
        if n!=5 and not (summary_path.is_file() and edges_path.is_file()):
            # Aggregator exports the predeclared sizes in order. Do not infer
            # missing results from incomplete folders or partial checkpoints.
            break
        summary = pd.read_csv(summary_path).query('specification == @spec').iloc[0]
        frame = add_pair(pd.read_csv(edges_path).query('specification == @spec')).set_index(
            'candidate_edge_id', verify_integrity=True)
        assert int(summary.n_networks)==n and len(frame)==int(summary.n_edges)
        current = set(frame.index)
        if n==5:
            assert current == set(full.index)
        else:
            assert current.issubset(previous)
        assert normalized(frame.source_gene).equals(normalized(full.loc[frame.index].source_gene))
        assert normalized(frame.target_gene).equals(normalized(full.loc[frame.index].target_gene))
        multiplicity = frame.groupby(['pair_a', 'pair_b'], sort=True).size()
        assert multiplicity.isin([1, 2]).all() and multiplicity.sum() == len(frame)
        yield n, len(frame), multiplicity
        previous = current


def calculate():
    directed, pairs, precise, original = archived_analysis()
    rows = []
    twenty_pairs = None
    for n, edge_count, multiplicity in nested_directed_sets(directed):
        selected = multiplicity.index
        archived = pairs.loc[selected]
        high = precise.loc[selected]
        for scope in SCOPES:
            ext = 'meta' if scope=='EXTERNAL_META' else scope
            x = high[DEV].to_numpy()
            y = high[f'external::{ext}::pooled-rest'].to_numpy()
            x_archive = archived.development_meta_effect.to_numpy(dtype=float)
            y_archive = archived[f'external_effect::{scope}'].to_numpy(dtype=float)
            z = archived[f'external_z::{scope}'].to_numpy(dtype=float)
            fdr = archived[f'fdr_BH_within_unique_pairs::{scope}'].to_numpy(dtype=float)
            assert np.isfinite(z).all() and np.isfinite(fdr).all() and ((fdr>=0)&(fdr<=1)).all()
            assert np.array_equal(np.sign(y),np.sign(z)) and np.array_equal(np.sign(y),np.sign(y_archive))
            assert np.array_equal(np.sign(x),np.sign(x_archive))
            discrepancy = float(max(np.max(np.abs(x-x_archive)),np.max(np.abs(y-y_archive)))) if len(x) else np.nan
            assert not len(x) or discrepancy < 1e-6, f'Unexpected discrepancy against archived effects for {scope}'
            nonzero = (x!=0) & (y!=0)
            agree = nonzero & (np.sign(x)==np.sign(y))
            bh = fdr < .05
            rho = spearman(x,y)
            archived_rho = spearman(x_archive,y_archive)
            row=dict(n_seeds=n,scope=scope,n_directed_edges=edge_count,n_unordered_pairs=len(selected),
                     n_nonzero_sign_pairs=int(nonzero.sum()),n_sign_concordant=int(agree.sum()),
                     sign_agreement=float(agree.sum()/nonzero.sum()) if nonzero.any() else np.nan,spearman=rho,
                     n_archived_abs_z_ge_1_96=int((np.abs(z)>=1.96).sum()),n_original_bh_family_pairs=214,
                     n_archived_bh_lt_0_05=int(bh.sum()),n_archived_bh_and_sign_concordant=int((agree&bh).sum()),
                     spearman_from_archived_workbook=archived_rho,max_abs_difference_vs_archived_effect=discrepancy)
            if n==5:
                orig=original.loc[scope]
                assert len(selected)==orig.n_effects and nonzero.sum()==orig.n_nonzero_sign_pairs
                np.testing.assert_allclose(rho,orig.spearman_development_external,atol=1e-12,rtol=0)
                np.testing.assert_allclose(row['sign_agreement'],orig.sign_agreement_development_external,atol=1e-12,rtol=0)
                if scope=='EXTERNAL_META':
                    assert edge_count==299 and (len(selected),int(agree.sum()),int(bh.sum()),int((agree&bh).sum()))==(214,174,31,28)
                    np.testing.assert_allclose(rho,0.63592,atol=5e-6,rtol=0)
            if n == 20 and scope == 'EXTERNAL_META':
                twenty_pairs = pd.DataFrame({
                    'n_directed_edges': multiplicity,
                    'development_contrast': x,
                    'external_meta_contrast': y,
                    'external_meta_bh_q_original_214': fdr,
                    'sign_concordant': agree,
                    'concordant_and_bh_lt_0_05': agree & bh,
                }).rename_axis(['gene_a', 'gene_b']).reset_index()
            rows.append(row)
    return pd.DataFrame(rows), twenty_pairs


def statement(r):
    if not r.n_unordered_pairs:
        return f'The strict {r.n_seeds}-seed intersection contained no retained pairs.'
    if not r.n_nonzero_sign_pairs:
        return (f'The strict {r.n_seeds}-seed intersection contained {r.n_unordered_pairs} unordered pairs. '
                'None had nonzero effects both in development and in the external meta-analysis.')
    rho = f'Spearman $\\rho={r.spearman:.3f}$' if np.isfinite(r.spearman) else 'rank correlation undefined'
    return (f'For the {r.n_seeds}-seed intersection ({r.n_directed_edges} directed edges), '
            f'{r.n_sign_concordant}/{r.n_nonzero_sign_pairs} unordered pairs had concordant '
            f'development and external meta-analytic signs ({100*r.sign_agreement:.1f}\\%, '
            f'{rho}). Of the {r.n_archived_bh_lt_0_05} retained '
            f'pairs below the archived FDR threshold, {r.n_archived_bh_and_sign_concordant} '
            'had concordant signs.')


def describe_examples(pairs):
    eligible = pairs.loc[pairs.concordant_and_bh_lt_0_05]
    ordered = (eligible.assign(magnitude=eligible.development_contrast.abs())
               .sort_values(['magnitude', 'gene_a', 'gene_b'], ascending=[False, True, True]))
    if len(ordered) < 3:
        return ''
    examples = [
        f'{r.gene_a}--{r.gene_b} (${r.development_contrast:+.3f}$, ${r.external_meta_contrast:+.3f}$)'
        for r in ordered.head(3).itertuples(index=False)
    ]
    return (' We ranked the pairs meeting both criteria by the absolute development contrast '
            'in Fisher-transformed correlation. The three highest-ranked pairs, with development '
            'and external meta-analytic contrasts given in that order, were '
            + ', '.join(examples[:-1]) + ', and ' + examples[-1]
            + '.')


def publish(path, text):
    if path.is_file() and path.read_text()==text:
        return
    with NamedTemporaryFile(mode='w',dir=path.parent,prefix='.'+path.name,suffix='.tmp',
                            encoding='utf-8',delete=False) as tmp:
        tmp.write(text)
    os.replace(tmp.name,path)


def main():
    result, twenty_pairs=calculate()
    meta=result[result.scope.eq('EXTERNAL_META')]
    paragraphs = [statement(r) for r in meta.itertuples(index=False)]
    supplement=('After fixing the development intersections, we collapsed reciprocal edges and '
                'summarised the archived external effects for the retained Basal-like pairs. '
                + ' '.join(paragraphs) + ' We used the same procedure within each external cohort. '
                'The FDR threshold was 0.05 after the original Benjamini--Hochberg correction for '
                '214 pairs within each external scope. These post hoc summaries reuse the external '
                'estimates previously reported for the five-seed core.')
    publish(HERE/'nested_external_concordance.csv',result.to_csv(index=False))
    twenty=meta[meta.n_seeds.eq(20)]
    main_text=''
    if not twenty.empty:
        r=next(twenty.itertuples(index=False))
        assert twenty_pairs is not None and len(twenty_pairs) == r.n_unordered_pairs
        publish(HERE/'nested_20_pairs.csv', twenty_pairs.to_csv(index=False))
        supplement += (' Supplementary Data~5 (sheet ``20-seed external pairs\'\') lists all '
                       'retained pairs alphabetically, with both contrasts, their archived external '
                       'meta-analytic BH-adjusted $P$ values and indicators of concordance.')
        main_text=(statement(r) + ' This post hoc comparison reuses the original 214-pair external '
                   'meta-analytic FDR threshold of 0.05.' + describe_examples(twenty_pairs) + '\n')
    publish(MANUSCRIPT/'nested_external_results.tex',supplement+'\n')
    publish(MANUSCRIPT/'nested_external_main.tex',main_text)
    print(meta[['n_seeds','n_directed_edges','n_unordered_pairs','n_sign_concordant','spearman',
                'n_archived_bh_lt_0_05','n_archived_bh_and_sign_concordant']].to_string(index=False))


if __name__=='__main__':
    main()
