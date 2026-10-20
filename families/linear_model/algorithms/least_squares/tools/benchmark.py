"""Fit-time baseline: native LinearRegression vs sklearn (see family tools/bench.py)."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('MOXIE_THREADS', '1'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', os.environ.get('MOXIE_THREADS', '1'))
import numpy as np
from sklearn.linear_model import LinearRegression
import bench

rows = []
for n, p, positive in [(200, 10, 0), (1000, 20, 0), (2000, 50, 0), (1000, 20, 1)]:
    for row in bench.native(ROOT, n, p, 5, positive):
        X = np.random.default_rng(0).uniform(-1, 1, (n, p)).astype(row['dtype'])
        y = X @ (np.arange(p) % 5 - 2.0).astype(row['dtype'])
        row.update(positive=positive, sklearn_seconds=bench.best(lambda: LinearRegression(positive=bool(positive)).fit(X, y)))
        rows.append(row)
bench.record(ROOT, 'linear_regression_fit', rows)
