"""Compare the native dense adapter with the pinned sklearn splitters and DummyRegressor."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[5]
ORACLE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES = ['sklearn/model_selection/_split.py', 'sklearn/dummy.py']
assert sklearn.__version__ == '1.9.1'
assert subprocess.check_output(['git', '-C', ORACLE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
assert not subprocess.check_output(['git', '-C', ORACLE, 'status', '--porcelain', '--', *FILES], text=True)


def load(name, source):
    spec = importlib.util.spec_from_file_location(name, ORACLE / source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split = load('sklearn.model_selection._moxie_split_reference', FILES[0])
dummy = load('sklearn._moxie_dummy_reference', FILES[1])
rows = np.arange(12)
X = rows.astype(np.float32).reshape(-1, 1)
y = (rows % 5).astype(np.float32)
weights = (rows + 1).astype(np.float32)

expected = {'score': [], 'index': []}
for fold, (train, test) in enumerate(split.check_cv(None, y).split(X, y)):
    model = dummy.DummyRegressor().fit(X[train], y[train], sample_weight=weights[train])
    expected['score'].append([fold, model.score(X[test], y[test], sample_weight=weights[test])])
    expected['index'].extend([fold, row] for row in test)
model = dummy.DummyRegressor().fit(X, y, sample_weight=weights)
expected['prediction'] = [[model.predict(X)[0]]]

actual = {}
for line in subprocess.check_output([sys.argv[1]], text=True).splitlines():
    key, *values = line.split()
    actual.setdefault(key, []).append([float(value) for value in values])
assert actual.keys() == expected.keys()
for key, values in expected.items():
    # Float32 predictions and score inputs can differ by rounding of the weighted mean.
    atol = 2e-6 if key in ('score', 'prediction') else 0
    np.testing.assert_allclose(actual[key], values, rtol=0, atol=atol, err_msg=key)
print(f'PASS pinned dense adapter parity with {REVISION}')
