"""TransformedTargetRegressor parity against the pinned scikit-learn checkout.

`TransformedTargetRegressor`, `FunctionTransformer` and `LinearRegression` are
compiled from the pinned `sklearn/compose/_target.py`,
`sklearn/preprocessing/_function_transformer.py` and
`sklearn/linear_model/_base.py`; the support wheel supplies only the
utilities those modules import. The native side is `builds/tests/compose/target/parity_driver`,
which reads the same fixture file and must report every case.

Compared per case: regressor coef_/intercept_, predictions (shape included),
R^2 scores, n_features_in_, the exact ordered warning texts, and for error
cases the exception type and message.

Tolerances (absolute on value/scale, relative to max(1, |reference|)):
- Float64 1e-8: the only numerical difference is LinearRegression's SVD
  (LAPACK gelsd vs mojo-linalg one-sided Jacobi), rounding-level on these
  well-conditioned designs; target functions are applied elementwise alike.
- Float32 3e-4: sklearn solves in float32 LAPACK while the native SVD
  accumulates in Float64 (see moxie-linear_model least_squares parity).
- R^2: the native score accumulates in Float64 while sklearn's r2_score uses
  the input dtype, so Float32 scores get 3e-3.
Warnings depend on exact allclose decisions and must match exactly, except
for Float32 log/exp cases (WARNING_EXEMPT): their round-trip error straddles
FunctionTransformer's rtol=1e-7 (measured: 8% of float32 values fail with
NumPy's SIMD log/exp, 1.3% with correctly rounded functions; NumPy's log is up
to 2 ulp from correctly rounded), so the outcome measures the user function's
ulp accuracy, not the library. sqrt/square are correctly rounded on both sides
and stay compared. Numerical results are compared for every case.
"""
from pathlib import Path
import json
import subprocess
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "builds/tests/compose/target"
sys.path.insert(0, str(ROOT / 'tools'))
import compose_parity_io as io
from compose_parity_io import environment

SOURCES = ['sklearn/compose/_target.py', 'sklearn/preprocessing/_function_transformer.py',
           'sklearn/linear_model/_base.py', 'sklearn/base.py']
hashes = io.verify_reference(SOURCES)

# Bind the pinned FunctionTransformer and LinearRegression under the package
# names that the pinned _target.py imports, then load _target itself.
import sklearn.preprocessing
import sklearn.linear_model
function_module = io.load_pinned('sklearn/preprocessing/_function_transformer.py', register=True)
linear_module = io.load_pinned('sklearn/linear_model/_base.py', register=True)
sklearn.preprocessing.FunctionTransformer = function_module.FunctionTransformer
sklearn.linear_model.LinearRegression = linear_module.LinearRegression
target_module = io.load_pinned('sklearn/compose/_target.py')
Reference = target_module.TransformedTargetRegressor
io.pinned_code(Reference.fit, SOURCES[0])
io.pinned_code(Reference.predict, SOURCES[0])
io.pinned_code(Reference._fit_transformer, SOURCES[0])
assert target_module.FunctionTransformer is function_module.FunctionTransformer
assert target_module.LinearRegression is linear_module.LinearRegression
io.pinned_code(function_module.FunctionTransformer.fit, SOURCES[1])
io.pinned_code(linear_module.LinearRegression.fit, SOURCES[2])

FUNCTIONS = {
    'identity': (None, None),
    'log': (np.log, np.exp),
    'sqrt': (np.sqrt, np.square),
    'shift': (lambda y: y + 1, lambda y: y),
    'clip': (lambda y: np.minimum(y, 5), lambda y: y),
    'forward_only': (np.log, None),
}

# The X/y length check belongs to the regressor (adopted LinearRegression,
# whose message differs); only the exception type is a compose obligation.
TYPE_ONLY = ('error-length-',)
WARNING_EXEMPT = {'log-32', 'log-weighted-32', 'log-multi-32'}
rng = np.random.default_rng(20261001)
cases = []


def add(name, X, y, Z, yz, func='identity', sw=None, check_inverse=True, bits=(64, 32)):
    for b in bits:
        dtype = np.float64 if b == 64 else np.float32
        cases.append(dict(id=f'{name}-{b}', bits=b, dtype=dtype, func=func, check_inverse=check_inverse,
                          X=np.asarray(X, dtype), y=np.asarray(y, dtype), Z=np.asarray(Z, dtype),
                          yz=np.asarray(yz, dtype), sw=None if sw is None else np.asarray(sw, dtype)))


def dataset(n, p, k=None, positive=False):
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(p,) if k is None else (p, k)) * 0.3
    noise = 0.05 * rng.normal(size=(n,) if k is None else (n, k))
    Z = rng.normal(size=(9, p))
    if positive:
        return X, np.exp(X @ coef + 1 + noise), Z, np.exp(Z @ coef + 1)
    return X, X @ coef + 0.5 + noise, Z, Z @ coef + 0.5


add('identity', *dataset(30, 4))
add('identity-wide', *dataset(5, 8))
X, y, Z, yz = dataset(20, 3)
add('identity-column', X, y.reshape(-1, 1), Z, yz.reshape(-1, 1))
add('identity-multi', *dataset(25, 3, k=2))
add('log', *dataset(40, 3, positive=True), func='log')
X, y, Z, yz = dataset(30, 3, positive=True)
add('log-weighted', X, y, Z, yz, func='log', sw=rng.uniform(0, 2, 30) * (rng.uniform(size=30) > 0.2))
add('log-multi', *dataset(30, 2, k=3, positive=True), func='log')
X, y, Z, yz = dataset(25, 3, positive=True)
add('sqrt-column', X, y.reshape(-1, 1), Z, yz.reshape(-1, 1), func='sqrt')
add('shift', *dataset(20, 2), func='shift')
add('shift-unchecked', *dataset(20, 2), func='shift', check_inverse=False)
# clip: only rows with y > 5 break the round trip. With n = 300,
# FunctionTransformer checks y[::3] and TransformedTargetRegressor y[::30];
# place the violation outside both, in the first only, or in both samples.
X, y, Z, yz = dataset(300, 3)
for rows, label in (([4], 'clip-none'), ([3], 'clip-transformer-only'), ([30], 'clip-both')):
    target = np.clip(y.copy(), -4, 4)
    target[rows] = 9.0
    add(label, X, target, Z, yz, func='clip')
add('single-sample', [[1.0, 2.0]], [3.0], [[0.0, 1.0], [2.0, 1.0]], [1.0, 2.0])
# Error cases (fit must raise identically).
X, y, Z, yz = dataset(10, 2)
add('error-forward-only', X, y, Z, yz, func='forward_only')
bad = y.copy()
bad[3] = np.nan
add('error-nan', X, bad, Z, yz)
bad = y.copy()
bad[5] = np.inf
add('error-inf', X, bad, Z, yz)
add('error-empty', X, np.zeros(0), Z, yz)
add('error-length', X, y[:7], Z, yz)

folder = BUILD / 'parity'
folder.mkdir(parents=True, exist_ok=True)
lines = []
for case in cases:
    arrays = dict(X=case['X'], y=case['y'], Z=case['Z'], yz=case['yz'])
    if case['sw'] is not None:
        arrays['sw'] = case['sw']
    params = dict(func=case['func'], check_inverse=int(case['check_inverse']))
    lines.append(io.case_line(case['id'], case['bits'], params, arrays))
(folder / 'fixtures.txt').write_text('\n'.join(lines) + '\n')
output = subprocess.run([str(BUILD / 'parity_driver'), str(folder / 'fixtures.txt')], text=True,
                        capture_output=True, check=True, env=environment(ROOT)).stdout
(folder / 'native.txt').write_text(output)
native = io.parse_results(output)
native_warnings = {}
for line in output.splitlines():
    if line.startswith('warn|'):
        _, ident, text = line.split('|', 2)
        native_warnings.setdefault(ident, []).append(text)
assert set(native) == {c['id'] for c in cases}, 'native driver did not report every case'

report = []
for case in cases:
    got = native[case['id']]
    func, inverse = FUNCTIONS[case['func']]
    model = Reference(func=func, inverse_func=inverse, check_inverse=case['check_inverse'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            model.fit(case['X'], case['y'], sample_weight=case['sw']) if case['sw'] is not None \
                else model.fit(case['X'], case['y'])
        except Exception as error:  # error cases: same type and message prefix
            expected = f'{type(error).__name__}: {error}'
            assert 'error' in got, (case['id'], 'native fit succeeded', expected)
            if case['id'].startswith(TYPE_ONLY):
                assert got['error'].split(':')[0] == type(error).__name__, (case['id'], got['error'], expected)
            else:
                assert expected.startswith(got['error']) or got['error'].startswith(expected.split('\n')[0]), \
                    (case['id'], got['error'], expected)
            report.append(dict(case=case['id'], error=got['error']))
            continue
        fit_warnings = [f'{w.category.__name__}: {w.message}' for w in caught]
    assert 'error' not in got, (case['id'], got)
    if case['id'] not in WARNING_EXEMPT:
        assert native_warnings.get(case['id'], []) == fit_warnings, \
            (case['id'], native_warnings.get(case['id']), fit_warnings)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pred = model.predict(case['Z'])
        score = model.score(case['Z'], case['yz'])
        train_score = model.score(case['X'], case['y'], sample_weight=case['sw']) if case['sw'] is not None else None
    assert pred.dtype == case['dtype'], (case['id'], pred.dtype)
    f32 = case['bits'] == 32
    tol = 3e-4 if f32 else 1e-8
    scale = max(1.0, float(np.abs(case['y']).max()))
    errors = {}
    regressor = model.regressor_
    errors['coef'] = io.close(got['coef'], regressor.coef_, tol, tol * scale, case['id'] + ' coef')
    expected_intercept = np.broadcast_to(np.atleast_1d(regressor.intercept_), got['intercept'].shape)
    errors['intercept'] = io.close(got['intercept'], expected_intercept, tol, tol * scale, case['id'] + ' intercept')
    errors['pred'] = io.close(got['pred'], pred, tol, tol * scale, case['id'] + ' pred')
    stol = 3e-3 if f32 else 1e-8
    errors['score'] = io.close(got['score'], score, stol, stol, case['id'] + ' score')
    if train_score is not None:
        errors['train_score'] = io.close(got['train_score'], train_score, stol, stol, case['id'] + ' train_score')
    assert int(got['n_features_in']) == model.n_features_in_, case['id']
    report.append(dict(case=case['id'], warnings=len(fit_warnings), **errors))

metadata = dict(revision=io.REVISION, sources=hashes, cases=len(cases), seed=20261001, numpy=np.__version__,
                support_sklearn=__import__('sklearn').__version__, results=report)
(folder / 'report.json').write_text(json.dumps(metadata, indent=2) + '\n')
numeric = [r for r in report if 'error' not in r]
worst = {b: max(max(v for k, v in r.items() if k not in ('case', 'warnings')) for r in numeric if r['case'].endswith(str(b)))
         for b in (32, 64)}
print(f'TransformedTargetRegressor parity passed: {len(cases)} cases ({len(report) - len(numeric)} error cases) '
      f'vs pinned sklearn {io.REVISION[:10]}; max abs error float64={worst[64]:.2e}, float32={worst[32]:.2e}')
