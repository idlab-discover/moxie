"""Fit-time baseline: native Ridge solvers vs sklearn (see family tools/bench.py)."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('MOXIE_THREADS', '1'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', os.environ.get('MOXIE_THREADS', '1'))
import numpy as np
from sklearn.linear_model import Ridge
import bench

rows = []
for n, p in [(200, 10), (2000, 50), (100, 400)]:
    for solver in ['cholesky', 'svd', 'lsqr', 'sparse_cg']:
        if solver == 'svd' and n * p > 20000:
            continue
        for row in bench.native(ROOT, n, p, 3, solver):
            X = np.random.default_rng(0).uniform(-1, 1, (n, p)).astype(row['dtype'])
            y = X @ (np.arange(p) % 5 - 2.0).astype(row['dtype'])
            row.update(solver=solver, sklearn_seconds=bench.best(lambda: Ridge(solver=solver).fit(X, y)))
            rows.append(row)
bench.record(ROOT, 'ridge_fit', rows)
