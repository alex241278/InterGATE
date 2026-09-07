#!/usr/bin/env python3
"""Run cached post-hoc analyses; no model fitting or graph reselection."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from intergate.review_posthoc import run_review

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'results_v8')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    tables = run_review(args.results_root, args.output_dir)
    for name in ('strict_core_reciprocity', 'strict_core_concordance_by_unit',
                 'q_comparator_input_status', 'intergate_vs_q_external_differences'):
        if name in tables:
            print('\n' + name)
            print(tables[name].to_string(index=False))
