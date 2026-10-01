"""Dense FeatureUnion parity against the verified local sklearn class body."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES = ['sklearn/pipeline.py', 'sklearn/base.py', 'sklearn/utils/metaestimators.py',
         'sklearn/tests/test_pipeline.py', 'doc/modules/compose.rst']
assert subprocess.check_output(['git', '-C', SOURCE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
assert not subprocess.check_output(['git', '-C', SOURCE, 'status', '--porcelain', '--', *FILES], text=True)
spec = importlib.util.spec_from_file_location('sklearn._union_parity', SOURCE / 'sklearn/pipeline.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
FeatureUnion = module.FeatureUnion
for name in ('fit', 'fit_transform', 'transform', '_hstack'):
    function = getattr(FeatureUnion, name)
    while hasattr(function, '__wrapped__'):
        function = function.__wrapped__
    assert function.__code__.co_filename == str(SOURCE / 'sklearn/pipeline.py')


class Fallback(BaseEstimator):
    def __init__(self, factor=1, kind=0):
        self.factor, self.kind = factor, kind
        self.fitted = False
        self.fit_calls = self.fused_calls = 0

    def _learn(self, X, y):
        self.fitted = False
        if self.kind == 4:
            raise ValueError('requested fixture fit failure')
        self.n_features_in_ = X.shape[1]
        self.target_sum = 0 if y is None else np.sum(y)
        self.fitted = True

    def fit(self, X, y=None):
        self.fit_calls += 1
        self._learn(X, y)
        return self

    def _output(self, X, extra):
        if not self.fitted:
            raise NotFittedError('fixture')
        if self.kind == 2:
            return np.zeros(X.shape[0], dtype=X.dtype)
        if self.kind == 3:
            return np.zeros((X.shape[0] + 1, X.shape[1]), dtype=X.dtype)
        if self.kind == 5:
            return np.zeros((X.shape[0], 0), dtype=X.dtype)
        data = X[:, :1] if self.kind == 1 else X
        return (data.astype(np.float64) * self.factor + extra).astype(X.dtype)

    def transform(self, X):
        return self._output(X, 0)

    def __sklearn_is_fitted__(self):
        return self.fitted


class Fused(Fallback):
    def fit_transform(self, X, y=None):
        self.fused_calls += 1
        self._learn(X, y)
        return self._output(X, 10)


def is_fitted(estimator):
    try:
        return estimator.__sklearn_is_fitted__()
    except NotFittedError:
        return False


native = subprocess.check_output([ROOT / 'builds/tests/pipeline/union_parity_driver'], text=True)
lines = iter(native.splitlines())
actual, states, names = {}, {}, {32: [], 64: []}
for line in lines:
    fields = line.split()
    if fields[0] == 'matrix':
        _, name, bits, rows, cols = fields
        bits, rows, cols = int(bits), int(rows), int(cols)
        actual[bits, name] = np.array([float(next(lines)) for _ in range(rows * cols)]).reshape(rows, cols)
    elif fields[0] == 'state':
        states[int(fields[1]), int(fields[2])] = list(map(float, fields[3:]))
    elif fields[0] == 'name':
        names[int(fields[1])].append(fields[2])
    else:
        raise AssertionError(f'unexpected native output: {line}')

expected, expected_states = {}, {}
for bits, dtype in [(32, np.float32), (64, np.float64)]:
    for shape in range(5):
        rows = 0 if shape == 1 else (1 if shape == 2 else 3)
        cols = 0 if shape == 2 else shape + 1
        X = (np.arange(rows * cols).reshape(rows, cols) - 3).astype(dtype)
        y = np.arange(1, rows + 1, dtype=dtype)
        union = FeatureUnion([('a', Fallback(2)), ('b', Fused(3, 1)), ('raw', 'passthrough'), ('drop', 'drop')],
                             transformer_weights={'a': 0.1, 'b': -2., 'raw': 0.})
        expected[bits, f'fused-{shape}'] = union.fit_transform(X, y)
        expected[bits, f'transform-{shape}'] = union.transform(X)
        union.fit(X, y)
        expected[bits, f'fit-{shape}'] = union.transform(X)
        copied = clone(union).set_params(a__factor=-4)
        expected[bits, f'clone-{shape}'] = copied.fit_transform(X, y)
        expected_states[bits, shape] = [union['a'].fit_calls, union['b'].fit_calls,
                                       union['b'].fused_calls, union['a'].target_sum, int(is_fitted(clone(union)))]
    backing = np.arange(1, 13, dtype=dtype).reshape(3, 4)
    view = backing[2:0:-1, ::2]
    passes = FeatureUnion([('raw', 'passthrough'), ('again', 'passthrough')])
    expected[bits, 'strided'] = passes.transform(view)
    expected[bits, 'single'] = FeatureUnion([('raw', 'passthrough')]).transform(view)
    empty = FeatureUnion([('a', 'drop'), ('b', 'drop')]).fit_transform(view)
    assert empty.dtype == np.float64  # documented static Mojo dtype adaptation
    expected[bits, 'dropped'] = empty.astype(dtype)
    backing[2, 0], backing[2, 2] = np.nan, np.inf
    passes.set_params(transformer_weights={'raw': 0., 'again': -1.5})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected[bits, 'nonfinite'] = passes.transform(view)
    passes.set_params(transformer_weights={'raw': np.inf, 'again': np.nan})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected[bits, 'nonfinite-weights'] = passes.transform(view)
    passes.fit(view)
    assert names[bits] == list(passes.get_feature_names_out())

assert actual.keys() == expected.keys()
assert states == expected_states, (states, expected_states)
for key, result in actual.items():
    reference = expected[key]
    assert result.shape == reference.shape, key
    assert reference.dtype == np.dtype(f'float{key[0]}'), key
    tolerance = 2e-6 if key[0] == 32 else 1e-12
    np.testing.assert_allclose(result, reference, rtol=tolerance, atol=tolerance, equal_nan=True, err_msg=str(key))
report = {'reference': REVISION, 'support_sklearn': sklearn.__version__, 'matrix_cases': len(actual),
          'dispatch_state_cases': len(states), 'name_cases': len(names),
          'source_sha256': {f: hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in FILES},
          'tolerances': {'float32': 2e-6, 'float64': 1e-12},
          'adaptations': ['All-drop Float32 union retains Float32; NumPy defaults to Float64.']}
(ROOT / 'builds/union-parity.json').write_text(json.dumps(report, indent=2) + '\n')
print(f'PASS: {len(actual)} FeatureUnion matrix cases, {len(states)} dispatch/state cases, both dtypes and feature names')
