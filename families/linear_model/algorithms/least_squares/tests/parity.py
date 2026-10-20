"""LinearRegression parity against the pinned scikit-learn checkout.

The estimator class is compiled from the pinned `sklearn/linear_model/_base.py`
(support wheel supplies only imported utilities, SciPy's lstsq/nnls and the
R2 metric). The native side is the compiled `builds/parity_driver`, which reads
the same fixture file; every case must produce a native result line.

Tolerances (absolute on value/scale, relative to max(1, |reference|)):
- Float64: 1e-8 for well-conditioned fits (LAPACK gelsd vs one-sided Jacobi SVD
  differ only by rounding), 1e-5 for the cond~1e7 case (error ~ cond * eps).
- Float32: sklearn solves in float32 LAPACK while the native SVD accumulates in
  Float64 and rounds, so the reference itself carries ~cond * 6e-8 error:
  3e-4 generally, 5e-2 on predictions only for the ill-conditioned case.
- rank_ must match exactly; NNLS (positive=True) is Float64 on both sides.
"""
from pathlib import Path
import json
import subprocess
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
import parity_io as io
from childcheck import environment

SOURCES = ['sklearn/linear_model/_base.py', 'sklearn/base.py']
hashes = io.verify_reference(SOURCES)
base = io.load_pinned('sklearn/linear_model/_base.py')
Reference = base.LinearRegression
io.pinned_code(Reference.fit, SOURCES[0])
io.pinned_code(Reference.predict, SOURCES[0])

rng = np.random.default_rng(20260926)
cases = []


def add(name, X, y, Z, yz, sw=None, fit_intercept=True, positive=False, tol=1e-6, loose=False):
    for bits, dtype in ((64, np.float64), (32, np.float32)):
        cases.append(dict(id=f'{name}-{bits}', bits=bits, dtype=dtype, X=np.asarray(X, dtype), y=np.asarray(y, dtype),
                          Z=np.asarray(Z, dtype), yz=np.asarray(yz, dtype),
                          sw=None if sw is None else np.asarray(sw, dtype), fit_intercept=fit_intercept,
                          positive=positive, tol=tol, loose=loose))


def dataset(n, p, k=None, noise=0.1):
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(p,) if k is None else (p, k))
    y = X @ coef + 0.5 + noise * rng.normal(size=(n,) if k is None else (n, k))
    Z = rng.normal(size=(7, p))
    yz = Z @ coef + 0.5
    return X, y, Z, yz


for fit_intercept in (True, False):
    tag = 'icpt' if fit_intercept else 'noicpt'
    add(f'tall-{tag}', *dataset(30, 5), fit_intercept=fit_intercept)
    add(f'wide-{tag}', *dataset(5, 12), fit_intercept=fit_intercept)
    add(f'multi-{tag}', *dataset(25, 4, k=3), fit_intercept=fit_intercept)
    X, y, Z, yz = dataset(30, 5)
    add(f'weighted-{tag}', X, y, Z, yz, sw=rng.uniform(0, 3, 30) * (rng.uniform(size=30) > 0.2), fit_intercept=fit_intercept)
    X, y, Z, yz = dataset(20, 6, k=2)
    add(f'positive-{tag}', X, y, Z, yz, fit_intercept=fit_intercept, positive=True)
    add(f'positive-weighted-{tag}', X, y, Z, yz, sw=rng.uniform(0.5, 2, 20), fit_intercept=fit_intercept, positive=True)
X, y, Z, yz = dataset(12, 4)
X[:, 3] = X[:, 0] + X[:, 1]  # exact rank deficiency
add('rank-deficient', X, y, Z, yz)
X, y, Z, yz = dataset(15, 4)
X[:, 3] = X[:, 0] + 1e-7 * rng.normal(size=15)  # cond ~ 1e7
add('ill-conditioned', X, y, Z, yz, loose=True)
add('ill-conditioned-cutoff', X, y, Z, yz, tol=1e-3)
add('tiny', [[1.0], [3.0]], [3.0, 7.0], [[2.0], [5.0]], [5.0, 11.0])
add('single-sample', [[1.0, 1.0]], [2.0], [[1.0, 0.0], [0.0, 2.0]], [1.0, 2.5], fit_intercept=False)
X, y, Z, yz = dataset(10, 3)
add('column-target', X, y.reshape(-1, 1), Z, yz.reshape(-1, 1))
add('constant-feature', np.c_[X, np.ones(10)], y, np.c_[Z, np.ones(7)], yz)

folder = ROOT / 'builds/parity'
folder.mkdir(parents=True, exist_ok=True)
lines = []
for case in cases:
    arrays = dict(X=case['X'], y=case['y'], Z=case['Z'], yz=case['yz'])
    if case['sw'] is not None:
        arrays['sw'] = case['sw']
    params = dict(fit_intercept=int(case['fit_intercept']), positive=int(case['positive']), tol=repr(case['tol']))
    lines.append(io.case_line(case['id'], case['bits'], params, arrays))
(folder / 'fixtures.txt').write_text('\n'.join(lines) + '\n')
output = subprocess.run([str(ROOT / 'builds/parity_driver'), str(folder / 'fixtures.txt')], text=True,
                        capture_output=True, check=True, env=environment(ROOT)).stdout
(folder / 'native.txt').write_text(output)
native = io.parse_results(output)
assert set(native) == {c['id'] for c in cases}, 'native driver did not report every case'

report = []
for case in cases:
    got = native[case['id']]
    assert 'error' not in got, (case['id'], got)
    model = Reference(fit_intercept=case['fit_intercept'], positive=case['positive'], tol=case['tol'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(case['X'], case['y'], sample_weight=case['sw'])
        pred = model.predict(case['Z'])
        score = model.score(case['Z'], case['yz'])
    # sklearn keeps NNLS's float64 coef_ when positive=True and fit_intercept=False
    # (only _set_intercept casts); the native coef_ always has the input dtype.
    if not (case['positive'] and not case['fit_intercept']):
        assert model.coef_.dtype == case['dtype'], case['id']
    f32 = case['bits'] == 32
    tol = (3e-4 if f32 else 1e-8) if not case['loose'] else (None if f32 else 1e-5)
    scale = max(1.0, float(np.abs(case['y']).max()))
    errors = {}
    if tol is not None:
        errors['coef'] = io.close(got['coef'], model.coef_, tol, tol * scale, case['id'] + ' coef')
        # sklearn stores the scalar 0.0 without an intercept even for several
        # targets; the native intercept_ is always (targets,) (documented).
        expected = np.broadcast_to(np.atleast_1d(model.intercept_), got['intercept'].shape)
        errors['intercept'] = io.close(got['intercept'], expected, tol, tol * scale, case['id'] + ' intercept')
    ptol = tol if tol is not None else 5e-2
    errors['pred'] = io.close(got['pred'], pred, ptol, ptol * scale, case['id'] + ' pred')
    errors['score'] = io.close(got['score'], score, ptol * 10, ptol * 10, case['id'] + ' score')
    if not case['positive']:
        assert int(got['rank']) == model.rank_, (case['id'], got['rank'], model.rank_)
        errors['singular'] = io.close(got['singular'], model.singular_, tol or 3e-4, (tol or 3e-4) * float(model.singular_.max()), case['id'] + ' singular')
    else:
        assert 'rank' not in got and not hasattr(model, 'rank_')
    report.append(dict(case=case['id'], **errors))

metadata = dict(revision=io.REVISION, sources=hashes, cases=len(cases), seed=20260926, numpy=np.__version__,
                scipy=__import__('scipy').__version__, support_sklearn=__import__('sklearn').__version__, errors=report)
(folder / 'report.json').write_text(json.dumps(metadata, indent=2) + '\n')
worst = {bits: max(max(r[k] for k in r if k != 'case') for r in report if r['case'].endswith(str(bits))) for bits in (32, 64)}
print(f'LinearRegression parity passed: {len(cases)} cases (Float32/Float64) vs pinned sklearn {io.REVISION[:10]}; '
      f'max abs error float64={worst[64]:.2e}, float32={worst[32]:.2e}')
