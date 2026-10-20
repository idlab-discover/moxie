"""Differential BayesianRidge checks against the local pinned sklearn source."""
from pathlib import Path
import json
import os
import platform
import subprocess
import sys
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
import parity_io as io
from childcheck import environment

SOURCES = ['sklearn/linear_model/_base.py', 'sklearn/linear_model/_bayes.py', 'sklearn/base.py']
hashes = io.verify_reference(SOURCES)
io.load_pinned('sklearn/linear_model/_base.py', register=True)
Reference = io.load_pinned('sklearn/linear_model/_bayes.py').BayesianRidge
io.pinned_code(Reference.fit, SOURCES[1])
io.pinned_code(Reference.predict, SOURCES[1])
rng = np.random.default_rng(20261016)
cases = []


def add(name, X, y, Z, sw=None, **overrides):
    params = dict(fit_intercept=True, compute_score=True, max_iter=300, tol=1e-3,
                  alpha_init=None, lambda_init=None, alpha_1=1e-6, alpha_2=1e-6,
                  lambda_1=1e-6, lambda_2=1e-6)
    params.update(overrides)
    for bits, dtype in ((64, np.float64), (32, np.float32)):
        arrays = {k: np.asarray(v, dtype) for k, v in dict(X=X, y=y, Z=Z).items()}
        if sw is not None:
            arrays['sw'] = np.asarray(sw, dtype)
        cases.append(dict(id=f'{name}-{bits}', bits=bits, params=params.copy(), arrays=arrays))


for n, p in [(30, 5), (5, 12), (7, 7), (12, 1), (1, 3)]:
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + rng.normal(size=n) * 0.2 + 3
    Z = rng.normal(size=(6, p))
    for intercept in (True, False):
        tag = f'{n}x{p}-{intercept}'
        add(tag, X, y, Z, fit_intercept=intercept)
        add(tag + '-weighted', X, y, Z, sw=rng.uniform(0.5, 2, n), fit_intercept=intercept)
        add(tag + '-one-step', X, y, Z, fit_intercept=intercept, max_iter=1,
            alpha_init=0.5, lambda_init=0.3, alpha_1=0.1, alpha_2=0.2, lambda_1=0.1, lambda_2=0.2)
X = rng.normal(size=(20, 4))
y = X @ np.array([1, 2, -0.5, 3]) + rng.normal(size=20)
Z = rng.normal(size=(6, 4))
add('column-target', X, y[:, None], Z)
add('zero-weights', X, y, Z, sw=np.arange(20) % 3)
add('constant-y', X, np.full(20, 2.0), Z)
add('zero-y', X, np.zeros(20), Z)
add('constant-X', np.ones((20, 4)), y, Z)
add('zero-X', np.zeros((20, 4)), y, Z)
add('shifted', X + 100, y, Z + 100)
add('zero-alpha-init', X, y, Z, alpha_init=0, compute_score=False)
add('zero-lambda-init', X, y, Z, lambda_init=0, compute_score=False)
add('zero-hyperpriors', X, y, Z, alpha_1=0, alpha_2=0, lambda_1=0, lambda_2=0)
X[:, 3] = X[:, 0] + X[:, 1]
add('rank-deficient', X, y, Z)
X[:, 3] += 1e-7 * rng.normal(size=20)
add('ill-conditioned', X, y, Z)
add('custom-init', X, y, Z, alpha_init=0.1, lambda_init=1e-3)

folder = ROOT / 'builds/parity'
folder.mkdir(parents=True, exist_ok=True)
lines = []
for case in cases:
    params = {k: int(v) if isinstance(v, bool) else (-1 if v is None else v)
              for k, v in case['params'].items()}
    lines.append(io.case_line(case['id'], case['bits'], params, case['arrays']))
(folder / 'fixtures.txt').write_text('\n'.join(lines) + '\n')
output = subprocess.run([str(ROOT / 'builds/parity_driver'), str(folder / 'fixtures.txt')],
                        text=True, capture_output=True, check=True, env=environment(ROOT)).stdout
(folder / 'native.txt').write_text(output)
native = io.parse_results(output)
assert set(native) == {c['id'] for c in cases}
report = []
for case in cases:
    got = native[case['id']]
    arrays = case['arrays']
    model = Reference(**case['params'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            model.fit(arrays['X'], arrays['y'], sample_weight=arrays.get('sw'))
            pred, std = model.predict(arrays['Z'], return_std=True)
        except (ValueError, FloatingPointError):
            assert 'error' in got, (case['id'], 'reference failed but native fit succeeded')
            report.append(dict(case=case['id'], rejected=True))
            continue
    assert 'error' not in got, (case['id'], got)
    expected = dict(coef=model.coef_, intercept=np.atleast_1d(model.intercept_), sigma=model.sigma_,
                    scores=model.scores_, alpha=model.alpha_, **{'lambda': model.lambda_},
                    offset=model.X_offset_, scale=model.X_scale_, pred=pred, std=std)
    # Float64 Jacobi versus LAPACK rounding; Float32 reference reductions and
    # factorizations accumulate in Float32, unlike the native Float64 path.
    tol = 3e-4 if case['bits'] == 32 else 1e-7
    errors = {}
    assert got['n_iter'] == model.n_iter_, (case['id'], got['n_iter'], model.n_iter_)
    for key, value in expected.items():
        # Strong shrinkage makes lambda sensitive to the small coefficient
        # norm. The wide Float32 fixture differs by 0.19% from LAPACK.
        # Keep stricter tolerances for every observable array and prediction.
        precision_tol = 3e-3 if case['bits'] == 32 and key in ('alpha', 'lambda') else tol
        errors[key] = io.close(got[key], value, precision_tol, tol, case['id'] + ' ' + key)
    # Independent posterior identity covers null-space directions in wide X.
    centered = arrays['X'].astype(np.float64) - got['offset']
    if 'sw' in arrays:
        centered *= np.sqrt(arrays['sw'])[:, None]
    precision = got['lambda'] * np.eye(centered.shape[1]) + got['alpha'] * centered.T @ centered
    # Identity residual is conditioning-sensitive, so use the measured condition.
    residual = np.linalg.norm(precision @ got['sigma'] - np.eye(centered.shape[1]), ord=np.inf)
    bound = (2e-6 if case['bits'] == 32 else 2e-12) * max(1, np.linalg.cond(precision))
    assert residual < bound, (case['id'], residual, bound)
    report.append(dict(case=case['id'], errors=errors))
(folder / 'report.json').write_text(json.dumps(dict(revision=io.REVISION, sources=hashes,
    cases=len(cases), seed=20261016, numpy=np.__version__, scipy=__import__('scipy').__version__,
    support_sklearn=__import__('sklearn').__version__, python=sys.version,
    mojo=subprocess.check_output(['mojo', '--version'], text=True).strip(),
    flags=os.environ.get('MOXIE_MOJO_FLAGS', '-O3 --fp-mode=contract=off'),
    threads=environment(ROOT)['OMP_NUM_THREADS'], machine=platform.machine(),
    processor=platform.processor(), platform=platform.platform(), command=sys.argv,
    errors=report), indent=2) + '\n')
print(f'BayesianRidge parity passed: {len(cases)} Float32/Float64 cases vs pinned sklearn {io.REVISION[:10]}')
