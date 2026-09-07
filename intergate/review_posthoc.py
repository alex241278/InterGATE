"""Read-only CMPB review analyses of already exported InterGATE results.

No training, no runtime construction, no checkpoint loading, no graph selection.
Comparisons are descriptive: cached edge effects do not support patient-level
paired confidence intervals, and edges sharing genes are not independent units.
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bh_adjust(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if np.any((p[valid] < 0) | (p[valid] > 1)):
        raise ValueError('P values must be in [0, 1].')
    x = p[valid]
    if not len(x):
        return out
    order = np.argsort(x, kind='stable')
    adjusted = np.minimum.accumulate((x[order] * len(x) / np.arange(1, len(x)+1))[::-1])[::-1]
    restored = np.empty_like(x)
    restored[order] = np.clip(adjusted, 0, 1)
    out[valid] = restored
    return out


def validate_edges(frame: pd.DataFrame) -> pd.DataFrame:
    required = {'candidate_edge_id', 'source_gene', 'target_gene'}
    missing = required - set(frame)
    if missing:
        raise ValueError(f'Missing edge columns: {sorted(missing)}')
    x = frame.copy()
    ids = pd.to_numeric(x.candidate_edge_id, errors='raise').to_numpy(float)
    if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all() or (ids < 0).any():
        raise ValueError('Candidate IDs must be non-negative finite integers.')
    x['candidate_edge_id'] = ids.astype(np.int64)
    if x.candidate_edge_id.duplicated().any():
        raise ValueError('Duplicate candidate IDs; do not count them twice.')
    for col in ('source_gene', 'target_gene'):
        if x[col].isna().any() or x[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'Missing gene labels in {col}.')
        x[col] = x[col].astype(str).str.strip().str.upper()
    if x.source_gene.eq(x.target_gene).any():
        raise ValueError('Self-loops are not part of this biological analysis.')
    pairs = np.sort(x[['source_gene', 'target_gene']].to_numpy(str), axis=1)
    x['pair_gene_a'], x['pair_gene_b'] = pairs[:, 0], pairs[:, 1]
    return x


def collapse_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per unordered gene pair, not per direction or edge type.

    Symmetric expression-based effects must agree within each pair. Learned
    directional logits are deliberately NOT averaged into a symmetric effect.
    """
    x = validate_edges(frame).sort_values('candidate_edge_id')
    effect_cols = [c for c in x if c == 'development_meta_effect'
                   or c.startswith('external_effect::') or c.startswith('external_z::')]
    rows = []
    for (a, b), group in x.groupby(['pair_gene_a', 'pair_gene_b'], sort=True):
        for col in effect_cols:
            values = pd.to_numeric(group[col], errors='raise').to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite {col} for pair {a}/{b}.')
            if not np.allclose(values, values[0], rtol=1e-5, atol=1e-6):
                raise ValueError(f'Non-symmetric cached effects for pair {a}/{b}: {col}.')
        row = group.iloc[0].to_dict()
        row['n_directed_rows'] = len(group)
        row['n_orientations'] = len(group[['source_gene', 'target_gene']].drop_duplicates())
        row['candidate_ids'] = ';'.join(group.candidate_edge_id.astype(str))
        if 'edge_type' in group:
            row['represented_edge_types'] = ';'.join(sorted(set(group.edge_type.astype(str))))
        rows.append(row)
    return pd.DataFrame(rows).drop(columns=[
        c for c in ('median_learned_differential', 'median_differential_logit', 'differential_logit')
        if c in x
    ], errors='ignore')


def metrics(frame: pd.DataFrame, scope: str) -> dict[str, Any]:
    a = frame.development_meta_effect.to_numpy(float)
    b = frame[f'external_effect::{scope}'].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    sign_ok = ok & (a != 0) & (b != 0)
    rho = float(spearmanr(a[ok], b[ok]).statistic) if (
        ok.sum() >= 3 and np.unique(a[ok]).size > 1 and np.unique(b[ok]).size > 1
    ) else float('nan')
    return {'n_effects': int(ok.sum()), 'n_nonzero_sign_pairs': int(sign_ok.sum()),
            'spearman_development_external': rho,
            'sign_agreement_development_external': float(
                np.mean(np.sign(a[sign_ok]) == np.sign(b[sign_ok]))
            ) if sign_ok.any() else float('nan')}


def join_effects(selected: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    selected = validate_edges(selected)
    lookup = candidates.set_index('candidate_edge_id', drop=False)
    missing = set(selected.candidate_edge_id) - set(lookup.index)
    if missing:
        raise ValueError(f'{len(missing)} selected IDs missing from candidate cache.')
    current = lookup.loc[selected.candidate_edge_id].reset_index(drop=True)
    for col in ('source_gene', 'target_gene'):
        if not np.array_equal(selected[col].to_numpy(), current[col].to_numpy()):
            raise ValueError(f'Cached prior and selected graph disagree in {col}.')
    return current


def run_review(results_root: str | Path, output_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    root = Path(results_root).expanduser().resolve()
    out = (Path(output_dir).expanduser().resolve() if output_dir is not None
           else root / '14_review_posthoc')
    source_dir = root / '08b_external_structural_concordance'
    candidate_path = source_dir / 'external_candidate_differential_effects.csv.gz'
    core_path = source_dir / 'basal_strict_5of5_external_edge_effects.csv'
    missing = [p for p in (candidate_path, core_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing cached outputs from notebook 08b:\n' +
                                '\n'.join(str(p) for p in missing) +
                                '\nNo training or external evaluation has been started.')
    candidates = validate_edges(pd.read_csv(candidate_path))
    strict = join_effects(pd.read_csv(core_path), candidates)
    if strict.empty:
        raise ValueError('Empty strict-core export.')
    scopes = [c.split('::', 1)[1] for c in candidates if c.startswith('external_effect::')]
    if not scopes or 'development_meta_effect' not in candidates:
        raise ValueError('Candidate cache does not contain the required differential effects.')
    forbidden = {root, source_dir.resolve(), (root / '12_reviewer_controls').resolve()}
    if out in forbidden or any(p.resolve().is_relative_to(out) for p in (candidate_path, core_path)):
        raise ValueError('Choose a separate post-hoc output directory.')
    out.mkdir(parents=True, exist_ok=True)
    sources = [candidate_path, core_path]
    original_hashes = {str(p): sha256(p) for p in sources}
    unique = collapse_pairs(strict)
    strict_rows = []
    for unit, panel in [('directed_rows', strict), ('unique_unordered_pairs', unique)]:
        for scope in scopes:
            strict_rows.append({'panel': 'strict_5of5', 'unit': unit, 'scope': scope,
                                **metrics(panel, scope)})
    tables: dict[str, pd.DataFrame] = {
        'strict_core_concordance_by_unit': pd.DataFrame(strict_rows),
        'strict_core_reciprocity': pd.DataFrame([{
            'directed_rows': len(strict), 'unique_unordered_pairs': len(unique),
            'reciprocal_pairs': int(unique.n_orientations.eq(2).sum()),
            'pairs_with_multiple_edge_types': int(unique.represented_edge_types.str.contains(';').sum())
                if 'represented_edge_types' in unique else 0,
        }]),
    }
    zcols = [c for c in unique if c.startswith('external_z::')]
    for col in zcols:
        scope = col.split('::', 1)[1]
        p = 2 * norm.sf(np.abs(unique[col].to_numpy(float)))
        unique[f'p_nominal::{scope}'] = p
        unique[f'fdr_BH_within_unique_pairs::{scope}'] = bh_adjust(p)
    tables['strict_core_unique_pair_effects'] = unique
    rows, contrasts, availability = [], [], []
    factorial = root / '12_reviewer_controls'
    for selector in ('raw', 'module'):
        panels = {}
        for method in ('InterGATE', 'q'):
            path = factorial / f'factorial_{method}_{selector}_Basal-like.csv'
            availability.append({'selector': selector, 'method': method,
                                 'file': str(path), 'available': path.is_file()})
            if path.is_file():
                panels[method] = join_effects(pd.read_csv(path), candidates)
                sources.append(path)
                original_hashes[str(path)] = sha256(path)
        if len(panels) != 2:
            continue
        if len(panels['InterGATE']) != len(panels['q']):
            raise ValueError(f'Unequal directed budgets for selector {selector}; comparison stopped.')
        if not len(panels['q']):
            raise ValueError('Empty factorial panels.')
        for unit in ('directed_rows', 'unique_unordered_pairs'):
            p = {k: (v if unit == 'directed_rows' else collapse_pairs(v)) for k, v in panels.items()}
            for scope in scopes:
                result = {}
                for method, panel in p.items():
                    result[method] = metrics(panel, scope)
                    rows.append({'method': method, 'selector': selector, 'unit': unit,
                                 'scope': scope, 'directed_budget': len(panels[method]),
                                 'n_unique_pairs': len(collapse_pairs(panels[method])),
                                 **result[method]})
                for metric in ('spearman_development_external', 'sign_agreement_development_external'):
                    contrasts.append({'selector': selector, 'unit': unit, 'scope': scope,
                                      'metric': metric,
                                      'intergate': result['InterGATE'][metric], 'q_only': result['q'][metric],
                                      'intergate_minus_q': result['InterGATE'][metric] - result['q'][metric],
                                      'inference': 'descriptive; no independent-edge p value'})
    tables['q_comparator_input_status'] = pd.DataFrame(availability)
    if rows:
        tables['intergate_vs_q_external'] = pd.DataFrame(rows)
        tables['intergate_vs_q_external_differences'] = pd.DataFrame(contrasts)
    matched_path = root / '05_biology_validation' / 'matched_candidate_signature_tests.csv'
    if matched_path.is_file():
        m = pd.read_csv(matched_path)
        required = {'graph_class', 'term', 'statistic', 'observed', 'null_mean'}
        if not required <= set(m):
            raise ValueError(f'Malformed matched-signature table: {required - set(m)}')
        m = m[m.statistic.eq('unique_node_fraction')].copy()
        m['excess_over_matched_null'] = m.observed - m.null_mean
        m['ratio_over_matched_null'] = np.divide(m.observed, m.null_mean,
            out=np.full(len(m), np.nan), where=m.null_mean.to_numpy() > 0)
        m['null_standardized_excess'] = (m.excess_over_matched_null / m.null_sd.replace(0, np.nan)
                                         if 'null_sd' in m else np.nan)
        tables['consensus_signature_recovery_normalized'] = m
        sources.append(matched_path)
        original_hashes[str(matched_path)] = sha256(matched_path)
    for name, table in tables.items():
        table.to_csv(out / f'{name}.csv', index=False)
    if any(sha256(Path(path)) != digest for path, digest in original_hashes.items()):
        raise RuntimeError('A source file changed during this read-only analysis.')
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_status': 'additional_post_hoc_review_analysis',
        'training_performed': False, 'runtime_or_checkpoint_loaded': False,
        'new_graph_selection': False, 'source_hashes_sha256': original_hashes,
        'available_q_comparisons': sorted(set(pd.DataFrame(rows).get('selector', []))),
        'notes': [
            'The 5/5 strict core is not compared against a larger Exact-K q panel.',
            'InterGATE and q comparisons use the frozen factorial panels and the same selector.',
            'Directed budgets are equal; deduplication can yield different numbers of unique pairs.',
            'Unique-pair sign concordance concerns differential correlations, not causal orientation.',
            'BH adjustment is per external scope across unique strict-core pairs; it is conditional on prior core selection.',
            'Cached edge effects do not provide patient-level paired inference; no edge-independent p value is invented.',
            'The original directed matched nulls are not reused as a unique-pair matched null.',
        ],
        'outputs_sha256': {p.name: sha256(p) for p in sorted(out.glob('*.csv'))},
    }
    (out / 'review_posthoc_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return tables
