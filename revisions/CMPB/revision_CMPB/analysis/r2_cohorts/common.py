"""Paths and immutable, class-stratified development split for R2 sensitivities."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2] / 'InterGATE'
DATA = REPO / 'data_prepare'
ADJUSTED = DATA / 'output_mcombat'
CLASSES = ['LumA', 'LumB', 'HER2-enriched', 'Basal-like']


def metadata(part):
    frame = pd.read_csv(ADJUSTED / f'meta_{part}.csv').set_index('sample')
    ordered = pd.read_csv(ADJUSTED / f'expr_{part}_mcombat.csv', nrows=0).columns[1:]
    return frame.loc[ordered]


def structural_split(meta):
    labels = meta.label.map(dict(zip(CLASSES, range(4))))
    train, val = train_test_split(range(len(meta)), stratify=labels,
                                  random_state=43, test_size=0.20)
    counts = pd.read_csv(REPO / 'results_v8/01_preflight/biology_group_counts.csv')
    observed = meta.iloc[train].groupby(['dataset', 'label']).size()
    assert all(observed[row.cohort, row.class_name] == row.n_case_train
               for row in counts.itertuples(index=False))
    return meta.index[train], meta.index[val]
