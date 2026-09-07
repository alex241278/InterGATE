"""Run: python -m unittest discover -s tests -p 'test_cmpb_review.py'"""
import json
from pathlib import Path
import tempfile
import unittest
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intergate.review_posthoc import bh_adjust, collapse_pairs, join_effects, run_review
from intergate.publication_style import apply_style, prepare_figure


class ReviewTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            'candidate_edge_id': [10, 12, 20, 31, 99, 101],
            'source_gene': ['A', 'B', 'B', 'C', 'D', 'E'],
            'target_gene': ['B', 'A', 'C', 'D', 'E', 'F'],
            'edge_type': [0, 0, 1, 2, 1, 0],
            'development_meta_effect': [.3, .3, -.2, .5, .8, -.7],
            'external_effect::EXTERNAL_META': [.25, .25, -.3, .4, .6, -.1],
            'external_z::EXTERNAL_META': [2.5, 2.5, -3, 4, 6, -1],
            'median_learned_differential': [.1, -.2, .4, .3, .1, .2],
        })

    def test_collapse_reciprocal_pairs(self):
        out = collapse_pairs(self.frame())
        self.assertEqual(len(out), 5)
        self.assertEqual(int(out.n_orientations.eq(2).sum()), 1)
        self.assertNotIn('median_learned_differential', out)

    def test_mismatched_symmetric_effect_rejected(self):
        x = self.frame(); x.loc[1, 'development_meta_effect'] = .9
        with self.assertRaises(ValueError):
            collapse_pairs(x)

    def test_duplicate_ids_rejected(self):
        x = self.frame(); x.loc[1, 'candidate_edge_id'] = 10
        with self.assertRaises(ValueError):
            collapse_pairs(x)

    def test_noncontiguous_ids_join_by_id_not_row(self):
        x = self.frame()
        out = join_effects(x.iloc[[4, 0]], x.iloc[::-1])
        self.assertEqual(out.candidate_edge_id.tolist(), [99, 10])
        self.assertEqual(out.development_meta_effect.tolist(), [.8, .3])

    def test_bh_order_and_missing(self):
        result = bh_adjust([.04, .001, np.nan, .02])
        np.testing.assert_allclose(result[[0, 1, 3]], [.04, .003, .03])
        self.assertTrue(np.isnan(result[2]))

    def test_missing_cache_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                run_review(tmp)
            self.assertFalse((Path(tmp) / '14_review_posthoc').exists())

    def test_full_read_only_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); d = root / '08b_external_structural_concordance'; d.mkdir()
            x = self.frame()
            x.to_csv(d / 'external_candidate_differential_effects.csv.gz', index=False)
            x.iloc[:4].to_csv(d / 'basal_strict_5of5_external_edge_effects.csv', index=False)
            f = root / '12_reviewer_controls'; f.mkdir()
            for selector in ('module', 'raw'):
                x.iloc[:4].to_csv(f / f'factorial_InterGATE_{selector}_Basal-like.csv', index=False)
                x.iloc[2:].to_csv(f / f'factorial_q_{selector}_Basal-like.csv', index=False)
            before = {p: p.read_bytes() for p in root.rglob('*.csv*')}
            tables = run_review(root)
            self.assertEqual(len(tables['strict_core_unique_pair_effects']), 3)
            self.assertEqual(len(tables['intergate_vs_q_external']), 8)
            self.assertTrue(all(p.read_bytes() == b for p, b in before.items()))
            manifest = json.loads((root / '14_review_posthoc/review_posthoc_manifest.json').read_text())
            self.assertFalse(manifest['training_performed'])

    def test_unequal_budget_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); d = root / '08b_external_structural_concordance'; d.mkdir()
            x = self.frame()
            x.to_csv(d / 'external_candidate_differential_effects.csv.gz', index=False)
            x.iloc[:4].to_csv(d / 'basal_strict_5of5_external_edge_effects.csv', index=False)
            f = root / '12_reviewer_controls'; f.mkdir()
            x.iloc[:4].to_csv(f / 'factorial_InterGATE_module_Basal-like.csv', index=False)
            x.iloc[3:].to_csv(f / 'factorial_q_module_Basal-like.csv', index=False)
            with self.assertRaises(ValueError):
                run_review(root)

    def test_font_application_idempotent(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        style = apply_style()
        fig, ax = plt.subplots(layout='constrained')
        ax.plot([0, 1], [0, 1]); t = ax.text(.5, .5, 'annotation', fontsize=7)
        prepare_figure(fig, style); a = t.get_fontsize()
        prepare_figure(fig, style)
        self.assertEqual(a, t.get_fontsize())
        self.assertGreaterEqual(a, 12)
        plt.close(fig)


class PatientBootstrapTests(unittest.TestCase):
    def test_fisher_effect_is_symmetric(self):
        from intergate.review_external_bootstrap import fisher_effect
        X = np.random.default_rng(8).normal(size=(40, 3))
        d = fisher_effect(X, np.arange(20), np.arange(20, 40),
                          np.array([0, 1]), np.array([1, 0]))
        np.testing.assert_allclose(d[0], d[1])

    def test_minimum_draw_guard(self):
        from intergate.review_external_bootstrap import run_patient_bootstrap
        with self.assertRaises(ValueError):
            run_patient_bootstrap({}, '.', external_cohorts=['TEST'], n_bootstrap=5)


if __name__ == '__main__':
    unittest.main()
