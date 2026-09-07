"""Patient-paired bootstrap of frozen InterGATE versus q external concordance.

Additional post-hoc analysis; never trains, selects edges or fits preprocessing.
The caller supplies the same prepared runtime used for the original external
analysis. Patients are resampled within external cohort and case/rest stratum;
each resample is shared by both methods and both selectors. Development effects,
preprocessing, panel membership and inverse-variance cohort weights stay fixed.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .review_posthoc import bh_adjust, validate_edges, sha256


def fisher_effect(X, case, rest, source, target, batch_size=256):
    X = np.asarray(X)
    def correlations(idx):
        result = np.empty(len(source), dtype=np.float64)
        for start in range(0, len(source), batch_size):
            sl = slice(start, start + batch_size)
            left = np.array(X[np.ix_(idx, source[sl])], dtype=np.float64)
            right = np.array(X[np.ix_(idx, target[sl])], dtype=np.float64)
            left -= left.mean(0); right -= right.mean(0)
            numerator = np.einsum('ij,ij->j', left, right)
            denominator = np.sqrt(np.einsum('ij,ij->j', left, left) * np.einsum('ij,ij->j', right, right))
            result[sl] = np.clip(np.divide(numerator, denominator, out=np.zeros_like(numerator),
                                           where=denominator > 1e-12), -.999999, .999999)
        return result
    return np.arctanh(correlations(case)) - np.arctanh(correlations(rest))


def statistic(a, b, metric):
    valid = np.isfinite(a) & np.isfinite(b)
    if metric == 'spearman':
        if valid.sum() < 3 or np.unique(a[valid]).size < 2 or np.unique(b[valid]).size < 2:
            return np.nan
        return float(spearmanr(a[valid], b[valid]).statistic)
    valid &= (a != 0) & (b != 0)
    return float(np.mean(np.sign(a[valid]) == np.sign(b[valid]))) if valid.any() else np.nan


def run_patient_bootstrap(runtime: Mapping[str, Any], results_root: str | Path, *,
                          external_cohorts: Sequence[str], target_class='Basal-like',
                          selectors=('module', 'raw'), n_bootstrap=1000,
                          random_seed=20260907, batch_size=256,
                          cache_atol=1e-4) -> pd.DataFrame:
    if n_bootstrap < 1000:
        raise ValueError('Use at least 1,000 patient resamples for the reported analysis.')
    if not runtime.get('external_opened', False):
        raise ValueError('Supply the prepared post-lock external runtime, not a training-only runtime.')
    root = Path(results_root).resolve()
    cache_path = root / '08b_external_structural_concordance/external_candidate_differential_effects.csv.gz'
    counts_path = root / '08b_external_structural_concordance/external_cohort_counts.csv'
    candidates = validate_edges(pd.read_csv(cache_path)).set_index('candidate_edge_id', drop=False)
    paths = [cache_path, counts_path]
    panels = {}
    for selector in selectors:
        for method in ('InterGATE', 'q'):
            p = root / '12_reviewer_controls' / f'factorial_{method}_{selector}_Basal-like.csv'
            x = validate_edges(pd.read_csv(p))
            ref = candidates.loc[x.candidate_edge_id]
            for col in ('source_gene', 'target_gene'):
                if not np.array_equal(x[col].to_numpy(), ref[col].to_numpy()):
                    raise ValueError(f'Candidate annotation mismatch: {p.name}/{col}.')
            panels[(selector, method)] = x.candidate_edge_id.to_numpy(int)
            paths.append(p)
        if len(panels[(selector, 'InterGATE')]) != len(panels[(selector, 'q')]):
            raise ValueError(f'Unequal directed budgets for {selector}.')
    union_ids = sorted(set(np.concatenate(list(panels.values())).tolist()))
    used = candidates.loc[union_ids].copy()
    for col in ('source_idx', 'target_idx', 'development_meta_effect'):
        if col not in used:
            raise ValueError(f'Required cached column missing: {col}.')
    representatives = used.drop_duplicates(['pair_gene_a', 'pair_gene_b'])
    pair_index = {(r.pair_gene_a, r.pair_gene_b): i for i, r in enumerate(representatives.itertuples())}
    id_to_pair = {int(r.candidate_edge_id): pair_index[(r.pair_gene_a, r.pair_gene_b)] for r in used.itertuples()}
    source = representatives.source_idx.to_numpy(int)
    target = representatives.target_idx.to_numpy(int)
    genes = np.char.upper(np.char.strip(np.asarray(runtime['genes']).astype(str)))
    if not (np.array_equal(genes[source], representatives.source_gene.to_numpy(str))
            and np.array_equal(genes[target], representatives.target_gene.to_numpy(str))):
        raise ValueError('Runtime gene order differs from the frozen candidate cache.')
    development = representatives.development_meta_effect.to_numpy(float)
    for r in used.itertuples():
        if not np.isclose(r.development_meta_effect, development[id_to_pair[r.candidate_edge_id]],
                          atol=1e-6, rtol=1e-5):
            raise ValueError('Development effects are not symmetric within an unordered pair.')
    panel_indices = {}
    for key, ids in panels.items():
        mapped = np.asarray([id_to_pair[int(i)] for i in ids])
        panel_indices[(*key, 'directed_rows')] = mapped
        panel_indices[(*key, 'unique_unordered_pairs')] = np.unique(mapped)
    X = np.asarray(runtime['X_scaled'])
    y = np.asarray(runtime['y'])
    cohort = np.char.upper(np.char.strip(np.asarray(runtime['cohort']).astype(str)))
    test = np.asarray(runtime['test_idx'], dtype=int)
    if len(np.unique(test)) != len(test):
        raise ValueError('Duplicate external sample indices.')
    target_id = list(map(str, runtime['classes'])).index(target_class)
    counts = pd.read_csv(counts_path).set_index('scope')
    groups, weights = {}, {}
    for name in external_cohorts:
        selected = test[cohort[test] == str(name).upper()]
        case, rest = selected[y[selected] == target_id], selected[y[selected] != target_id]
        if min(len(case), len(rest)) < 8:
            raise ValueError(f'Insufficient class counts in {name}.')
        if name not in counts.index or (len(selected), len(case), len(rest)) != tuple(
            int(counts.loc[name, c]) for c in ('n_external', 'n_basal_like', 'n_rest')
        ):
            raise ValueError(f'{name}: sample counts differ from the frozen 08b cache. Do not refreeze; resolve the original sample manifest.')
        groups[name] = (case, rest)
        weights[name] = 1 / (1/(len(case)-3) + 1/(len(rest)-3))
    scopes = [*groups, 'EXTERNAL_META']
    out = root / '14_review_posthoc/patient_bootstrap'
    out.mkdir(parents=True, exist_ok=True)
    contract = {'analysis': 'additional_post_hoc_patient_paired_bootstrap',
                'n_bootstrap': int(n_bootstrap), 'random_seed': int(random_seed),
                'selectors': list(selectors), 'scopes': scopes, 'target_class': target_class,
                'source_sha256': {str(p): sha256(p) for p in paths},
                'resampling_unit': 'patient within external cohort and class-vs-rest stratum',
                'fixed': ['development effects', 'selected panels', 'preprocessing', 'cohort weights'],
                'not_quantified': ['development sampling uncertainty', 'selection uncertainty'],
                'p_value': 'two-sided centered nonparametric bootstrap approximation, not an exact permutation p value'}
    contract_path = out / 'bootstrap_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('A different bootstrap contract already exists; do not overwrite it silently.')
    contract_path.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    def effects(bootstrap=False, rng=None):
        values = {}
        for name, (case, rest) in groups.items():
            if bootstrap:
                case, rest = rng.choice(case, len(case), replace=True), rng.choice(rest, len(rest), replace=True)
            values[name] = fisher_effect(X, case, rest, source, target, batch_size)
        values['EXTERNAL_META'] = sum(values[n] * weights[n] for n in groups) / sum(weights.values())
        return values
    baseline = effects()
    for scope in scopes:
        cached = representatives[f'external_effect::{scope}'].to_numpy(float)
        if not np.allclose(baseline[scope], cached, rtol=1e-4, atol=cache_atol):
            raise ValueError(f'{scope}: runtime does not reproduce frozen effects. Stop and check samples/preprocessing.')
    keys = [(selector, unit, scope, metric) for selector in selectors
            for unit in ('directed_rows', 'unique_unordered_pairs') for scope in scopes
            for metric in ('spearman', 'sign_agreement')]
    def differences(values):
        result = []
        for selector, unit, scope, metric in keys:
            a = panel_indices[(selector, 'InterGATE', unit)]
            b = panel_indices[(selector, 'q', unit)]
            result.append(statistic(development[a], values[scope][a], metric)
                          - statistic(development[b], values[scope][b], metric))
        return np.asarray(result)
    observed = differences(baseline)
    distribution = np.empty((n_bootstrap, len(keys)))
    rng = np.random.default_rng(random_seed)
    for b in range(n_bootstrap):
        distribution[b] = differences(effects(True, rng))
        if (b+1) % 100 == 0:
            print(f'Bootstrap pareado de pacientes: {b+1}/{n_bootstrap}', flush=True)
    rows = []
    for j, (selector, unit, scope, metric) in enumerate(keys):
        draws = distribution[:, j]; draws = draws[np.isfinite(draws)]
        if len(draws) < .95 * n_bootstrap or not np.isfinite(observed[j]):
            low, high, p = np.nan, np.nan, np.nan
        else:
            low, high = np.quantile(draws, [.025, .975])
            p = (1 + np.sum(np.abs(draws - observed[j]) >= abs(observed[j]))) / (1 + len(draws))
        rows.append({'selector': selector, 'unit': unit, 'scope': scope, 'metric': metric,
                     'intergate_minus_q': observed[j], 'ci025_patient_bootstrap': low,
                     'ci975_patient_bootstrap': high, 'p_bootstrap_centered_approx': p,
                     'n_valid_bootstraps': len(draws),
                     'n_intergate_units': len(panel_indices[(selector, 'InterGATE', unit)]),
                     'n_q_units': len(panel_indices[(selector, 'q', unit)])})
    summary = pd.DataFrame(rows)
    summary['fdr_all_bootstrap_contrasts'] = bh_adjust(summary.p_bootstrap_centered_approx)
    summary.to_csv(out / 'intergate_vs_q_patient_paired_bootstrap.csv', index=False)
    np.savez_compressed(out / 'paired_bootstrap_distribution.npz', differences=distribution,
                        observed=observed, key_labels=np.array(['|'.join(k) for k in keys]))
    return summary
