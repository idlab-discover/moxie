"""Ridge / RidgeClassifier / ridge_regression parity against pinned scikit-learn.

`sklearn/linear_model/_base.py` and `_ridge.py` are compiled from the pinned
checkout (`_base` is registered first so `_ridge` binds the pinned
preprocessing); SciPy supplies LAPACK/lsqr/cg/L-BFGS-B and the support wheel
the remaining utilities. The native side is builds/parity_driver.

Tolerances (value error <= tol * max(1, |reference|) scaled by max|y|):
- direct solvers (cholesky/svd, incl. kernel form and fallbacks): Float64 1e-8;
  Float32 3e-4 (sklearn factorizes in float32, native products accumulate in
  Float64).
- lsqr: Float64 1e-7 with exact n_iter_. For Float32 data SciPy mixes float32
  bidiagonalization vectors with a float64 iterate while native is all Float64:
  3e-4, n_iter_ exact unless tol < 1e-6 (below float32 resolution).
- sparse_cg: Float64 1e-7 (same CG sequence); Float32 SciPy runs CG in
  float32 while native runs Float64: 2e-2 at tol=1e-4, 3e-4 at tol=1e-10.
- lbfgs (positive=True): same L-BFGS-B port on Float64 objective: 1e-6.
- classifier labels/classes exact; decision like its solver.
sag/saga are reported as unfinished (NotImplementedError) and excluded.
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

SOURCES = ['sklearn/linear_model/_base.py', 'sklearn/linear_model/_ridge.py']
hashes = io.verify_reference(SOURCES)
io.load_pinned(SOURCES[0], register=True)
ridge = io.load_pinned(SOURCES[1], register=True)
io.pinned_code(ridge.Ridge.fit, SOURCES[1])
io.pinned_code(ridge._BaseRidge.fit, SOURCES[1])
io.pinned_code(ridge.ridge_regression, SOURCES[1])
io.pinned_code(ridge.RidgeClassifier.fit, SOURCES[1])
SOLVERS = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']

rng = np.random.default_rng(20260927)
cases = []


def add(kind, name, X, y, Z=None, yz=None, *, alpha=1.0, fit_intercept=True, solver='auto', tol=1e-4,
        max_iter=None, positive=False, sw=None, class_weight=None, fitted_only=False, float64_only=False):
    for bits, dtype in ((64, np.float64),) + (() if float64_only else ((32, np.float32),)):
        cases.append(dict(id=f'{kind}-{name}-{bits}', kind=kind, bits=bits, dtype=dtype, X=np.asarray(X, dtype),
                          y=np.asarray(y, dtype if kind != 'classifier' else np.int64),
                          Z=None if Z is None else np.asarray(Z, dtype),
                          yz=None if yz is None else np.asarray(yz, dtype if kind != 'classifier' else np.int64),
                          alpha=alpha, fit_intercept=fit_intercept, solver=solver, tol=tol, max_iter=max_iter,
                          positive=positive, sw=None if sw is None else np.asarray(sw, dtype), class_weight=class_weight,
                          fitted_only=fitted_only and bits == 32))


def regression(n, p, k=None):
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(p,) if k is None else (p, k))
    y = X @ coef + 1.5 + 0.2 * rng.normal(size=(n,) if k is None else (n, k))
    Z = rng.normal(size=(6, p))
    return X, y, Z, Z @ coef + 1.5


for solver in ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']:
    add('ridge', f'tall-{solver}', *regression(30, 5), solver=solver)
    add('ridge', f'wide-{solver}', *regression(8, 15), solver=solver, alpha=0.5)
    add('ridge', f'multi-{solver}', *regression(25, 4, 3), solver=solver, alpha=[0.1, 1.0, 10.0])
    add('ridge', f'shared-alpha-multi-{solver}', *regression(25, 4, 2), solver=solver, alpha=2.0)
    X, y, Z, yz = regression(30, 5)
    add('ridge', f'weighted-{solver}', X, y, Z, yz, solver=solver, sw=rng.uniform(0, 2, 30))
    add('ridge', f'noicpt-{solver}', X, y, Z, yz, solver=solver, fit_intercept=False, alpha=3.0)
    add('ridge', f'tight-{solver}', X, y, Z, yz, solver=solver, tol=1e-10, alpha=0.3)
    add('ridge', f'column-target-{solver}', X, y.reshape(-1, 1), Z, yz.reshape(-1, 1), solver=solver)
add('ridge', 'lsqr-maxiter', *regression(30, 8), solver='lsqr', max_iter=3, tol=1e-12)
add('ridge', 'cg-maxiter', *regression(30, 8), solver='sparse_cg', max_iter=2, tol=1e-12)
X, y, Z, yz = regression(12, 4)
X[:, 3] = X[:, 0] + X[:, 1]
# alpha=0 on an exactly singular X is ill-posed along the null space. In Float32
# sklearn's float32 SVD keeps a rounding-level singular value (> 1e-15) and
# returns ~1e6 coefficients whose fitted values are wrong by ~0.5, so this
# fallback case is compared in Float64 only.
add('ridge', 'singular-cholesky-fallback', X, y, Z, yz, solver='cholesky', alpha=0.0, float64_only=True)
add('ridge', 'wide-alpha0-lstsq-fallback', *regression(5, 9), solver='cholesky', alpha=0.0)
add('ridge', 'wide-multi-alpha0-svd-fallback', *regression(5, 9, 2), solver='cholesky', alpha=[0.0, 0.0])
add('ridge', 'positive', *regression(30, 6), positive=True)
add('ridge', 'positive-multi', *regression(30, 4, 2), positive=True, solver='lbfgs', alpha=[0.5, 2.0])
add('ridge', 'positive-weighted', *regression(30, 4), positive=True, sw=rng.uniform(0.5, 2, 30))
add('ridge', 'tiny', [[1.0], [3.0]], [3.0, 7.0], [[2.0]], [5.0])

for solver in ['auto', 'svd', 'lsqr']:
    X = rng.normal(size=(40, 4))
    yb = (X[:, 0] + 0.3 * rng.normal(size=40) > 0).astype(int) * 4 - 1  # labels {-1, 3}
    Z = rng.normal(size=(10, 4))
    add('classifier', f'binary-{solver}', X, yb, Z, (Z[:, 0] > 0).astype(int) * 4 - 1, solver=solver)
    ym = np.digitize(X[:, 1] + 0.2 * rng.normal(size=40), [-0.7, 0.0, 0.7]) * 2 + 1  # {1,3,5,7}
    add('classifier', f'multiclass-{solver}', X, ym, Z, np.digitize(Z[:, 1], [-0.7, 0.0, 0.7]) * 2 + 1, solver=solver, alpha=0.5)
X = rng.normal(size=(40, 4))
Z = rng.normal(size=(10, 4))
ym = np.digitize(X[:, 1], [-0.7, 0.0, 0.7])
ymz = np.digitize(Z[:, 1], [-0.7, 0.0, 0.7])
ym[:3] = 3  # imbalance
add('classifier', 'column-target', X, ym.reshape(-1, 1), Z, ymz.reshape(-1, 1))
add('classifier', 'single-class', X, np.full(40, 7), Z, np.full(10, 7))
add('classifier', 'balanced', X, ym, Z, ymz, class_weight='balanced')
add('classifier', 'dict-weight', X, ym, Z, ymz, class_weight={0: 2.0, 3: 0.5})
add('classifier', 'sample-weight', X, ym, Z, ymz, sw=rng.uniform(0.2, 2, 40))
indicator = np.c_[X[:, 0] > 0, X[:, 2] > 0.3, X[:, 0] + X[:, 2] > 0].astype(int)
indicator_z = np.c_[Z[:, 0] > 0, Z[:, 2] > 0.3, Z[:, 0] + Z[:, 2] > 0].astype(int)
add('classifier', 'multilabel', X, indicator, Z, indicator_z)
add('classifier', 'multilabel-balanced', X, indicator, Z, indicator_z, class_weight='balanced')

for solver in ['auto', 'svd', 'lsqr', 'sparse_cg']:
    add('function', f'plain-{solver}', *regression(20, 5)[:2], solver=solver)
    add('function', f'multi-{solver}', *regression(20, 5, 2)[:2], solver=solver, alpha=[0.3, 3.0])
    X, y = regression(20, 5)[:2]
    add('function', f'weighted-{solver}', X, y, solver=solver, sw=rng.uniform(0.5, 2, 20))
add('function', 'positive', *regression(20, 5)[:2], positive=True)

folder = ROOT / 'builds/parity'
folder.mkdir(parents=True, exist_ok=True)
lines = []
for case in cases:
    arrays = dict(X=case['X'], y=case['y'], alpha=np.atleast_1d(np.asarray(case['alpha'], dtype=np.float64)))
    for key in ('Z', 'yz', 'sw'):
        if case[key] is not None:
            arrays[key] = case[key]
    cw = case['class_weight']
    params = dict(kind=case['kind'], fit_intercept=int(case['fit_intercept']), solver=case['solver'], tol=repr(case['tol']),
                  max_iter=-1 if case['max_iter'] is None else case['max_iter'], positive=int(case['positive']),
                  scalar_alpha=int(np.isscalar(case['alpha'])),
                  class_weight='none' if cw is None else ('balanced' if cw == 'balanced' else ','.join(f'{k}:{v!r}' for k, v in cw.items())))
    lines.append(io.case_line(case['id'], case['bits'], params, arrays))
(folder / 'fixtures.txt').write_text('\n'.join(lines) + '\n')
output = subprocess.run([str(ROOT / 'builds/parity_driver'), str(folder / 'fixtures.txt')], text=True,
                        capture_output=True, check=True, env=environment(ROOT)).stdout
(folder / 'native.txt').write_text(output)
native = io.parse_results(output)
assert set(native) == {c['id'] for c in cases}, 'native driver did not report every case'


def tolerance(case, solver):
    f32 = case['bits'] == 32
    if solver == 'lsqr':
        return 3e-4 if f32 else 1e-7
    if solver == 'sparse_cg':
        return (2e-2 if case['tol'] > 1e-8 else 3e-4) if f32 else 1e-7
    if solver == 'lbfgs':
        return 1e-5 if f32 else 1e-6
    return 3e-4 if f32 else 1e-8


report = []
for case in cases:
    got = native[case['id']]
    assert 'error' not in got, (case['id'], got)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if case['kind'] == 'function':
            coef, n_iter = ridge.ridge_regression(case['X'], case['y'], case['alpha'], sample_weight=case['sw'],
                                                  solver=case['solver'], max_iter=case['max_iter'], tol=case['tol'],
                                                  positive=case['positive'], return_n_iter=True)
            solver = ridge.resolve_solver_for_numpy(case['positive'], False, False) if case['solver'] == 'auto' else case['solver']
            t = tolerance(case, solver)
            scale = max(1.0, float(np.abs(case['y']).max()))
            errors = dict(coef=io.close(got['coef'], coef, t, t * scale, case['id'] + ' coef'))
            if n_iter is not None:
                assert list(got['n_iter'].astype(int)) == list(n_iter), (case['id'], got['n_iter'], n_iter)
            else:
                assert got['n_iter'].size == 0, case['id']
            report.append(dict(case=case['id'], **errors))
            continue
        Model = ridge.Ridge if case['kind'] == 'ridge' else ridge.RidgeClassifier
        kwargs = dict(alpha=case['alpha'], fit_intercept=case['fit_intercept'], solver=case['solver'], tol=case['tol'],
                      max_iter=case['max_iter'], positive=case['positive'])
        if case['kind'] == 'classifier':
            kwargs['class_weight'] = case['class_weight']
        model = Model(**kwargs).fit(case['X'], case['y'], sample_weight=case['sw'])
    assert SOLVERS[int(got['solver'])] == model.solver_, (case['id'], SOLVERS[int(got['solver'])], model.solver_)
    t = tolerance(case, model.solver_)
    scale = max(1.0, float(np.abs(case['y']).max()))
    if case['fitted_only']:
        report.append(dict(case=case['id'], solver=model.solver_,
                           pred=io.close(got['pred'], model.predict(case['Z']), 1e-3, 1e-3 * scale, case['id'] + ' fitted')))
        continue
    errors = dict(coef=io.close(got['coef'], model.coef_, t, t * scale, case['id'] + ' coef'),
                  intercept=io.close(got['intercept'], np.broadcast_to(np.atleast_1d(model.intercept_), got['intercept'].shape),
                                     t, t * scale, case['id'] + ' intercept'))
    if case['kind'] == 'ridge':
        if model.n_iter_ is None:
            assert got['n_iter'].size == 0, case['id']
        elif case['bits'] == 64 or case['tol'] >= 1e-6:
            # SciPy's LSQR keeps float32 bidiagonalization vectors for float32
            # data, so near float32 resolution its stopping step can differ.
            assert list(got['n_iter'].astype(int)) == list(model.n_iter_), (case['id'], got['n_iter'], model.n_iter_)
        errors['pred'] = io.close(got['pred'], model.predict(case['Z']), t, t * scale, case['id'] + ' pred')
        errors['score'] = io.close(got['score'], model.score(case['Z'], case['yz']), 10 * t, 10 * t, case['id'] + ' score')
    else:
        assert np.array_equal(got['classes'], model.classes_), (case['id'], got['classes'], model.classes_)
        errors['decision'] = io.close(got['decision'], model.decision_function(case['Z']), t, t * scale, case['id'] + ' decision')
        assert np.array_equal(got['pred'], model.predict(case['Z'])), (case['id'], got['pred'], model.predict(case['Z']))
        assert abs(got['score'] - model.score(case['Z'], case['yz'])) < 1e-12, case['id']
    report.append(dict(case=case['id'], solver=model.solver_, **errors))

# sag/saga: the SAG engine is not ported; confirm the explicit refusal.
X, y = regression(10, 3)[:2]
probe = [io.case_line(f'sag-{s}', 64, dict(kind='ridge', fit_intercept=1, solver=s, tol='0.0001', max_iter=-1, positive=0,
                                            scalar_alpha=1, class_weight='none'),
                      dict(X=X, y=y, alpha=np.array([1.0]), Z=X, yz=y)) for s in ('sag', 'saga')]
(folder / 'unfinished.txt').write_text('\n'.join(probe) + '\n')
refusals = io.parse_results(subprocess.run([str(ROOT / 'builds/parity_driver'), str(folder / 'unfinished.txt')], text=True,
                                           capture_output=True, check=True).stdout)
assert all('NotImplementedError' in refusals[k]['error'] for k in ('sag-sag', 'sag-saga')), refusals

(folder / 'report.json').write_text(json.dumps(dict(revision=io.REVISION, sources=hashes, cases=len(cases), seed=20260927,
                                                    numpy=np.__version__, scipy=__import__('scipy').__version__,
                                                    support_sklearn=__import__('sklearn').__version__, errors=report), indent=2) + '\n')
worst = {bits: max(max(v for k, v in r.items() if k not in ('case', 'solver')) for r in report if r['case'].endswith(str(bits)))
         for bits in (32, 64)}
print(f'Ridge parity passed: {len(cases)} cases (Ridge, RidgeClassifier, ridge_regression; Float32/Float64; solvers '
      f'auto/svd/cholesky/lsqr/sparse_cg/lbfgs) vs pinned sklearn {io.REVISION[:10]}; max abs error '
      f'float64={worst[64]:.2e}, float32={worst[32]:.2e}; sag/saga refused as unported')
