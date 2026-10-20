"""Differential tests executing the pinned local sklearn method bodies."""
from pathlib import Path
import ast
import subprocess
import sys
import warnings
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# The default runtime build supplies this driver; the opt-in Numax build overrides it.
DRIVER = Path(os.environ.get('MOXIE_IPCA_DRIVER', ROOT / 'builds/incremental_pca_driver'))
# Reuse the PCA oracle's source verifier/loader without running its test matrix.
path = ROOT / 'tests/pca/parity.py'
plumbing = {'__file__': str(path)}
exec(compile(path.read_text().split('\ncount = 0\n')[0], str(path), 'exec'), plumbing)
ns, load = plumbing['namespace'], plumbing['load']

def validate_data(estimator, x, reset=True, copy=False, **kwargs):
    a = np.array(x, copy=True) if copy else np.asarray(x)
    if reset: estimator.n_features_in_ = a.shape[1]
    else: assert a.shape[1] == estimator.n_features_in_
    return a

ns['validate_data'] = validate_data
ns.update(get_namespace_and_device=lambda *a, **k: (np, False, None),
          _max_precision_float_dtype=lambda *a, **k: np.float64,
          _safe_accumulator_op=lambda op, x, *a, **k: op(x.astype(np.float64), *a, **k))
load('sklearn/utils/extmath.py', ('_incremental_mean_and_var',))
load('sklearn/utils/_chunking.py', ('gen_batches',))
source = plumbing['SOURCE'] / 'sklearn/decomposition/_incremental_pca.py'
assert not subprocess.check_output(['git', '-C', str(plumbing['SOURCE']), 'status', '--porcelain', '--', str(source)], text=True)
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
node.body = [n for n in node.body if isinstance(n, ast.FunctionDef)]
for method in node.body: method.decorator_list = []
module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
exec(compile(module, str(source), 'exec'), ns)
IPCA = ns['IncrementalPCA']
assert IPCA.partial_fit.__code__.co_filename == str(source)


def compare(x, k, batch, whiten=False, copy=True, mode='fit', degenerate=False):
    reference = IPCA(k, batch_size=batch, whiten=whiten, copy=copy)
    work = x.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if mode == 'fit':
            reference.fit(work)
            fitted = reference.transform(work)
        else:
            for start in range(0, len(work), batch): reference.partial_fit(work[start:start + batch])
        expected = dict(mean=reference.mean_, var=reference.var_,
                        variance=reference.explained_variance_, ratio=reference.explained_variance_ratio_,
                        singular=reference.singular_values_, noise=reference.noise_variance_,
                        covariance=reference.get_covariance(), input=work)
        projected = reference.transform(x)
        expected['inverse'] = reference.inverse_transform(projected)
        if not degenerate:
            expected.update(components=reference.components_, transform=projected)
            if mode == 'fit': expected['fit_transform'] = fitted
    args = [DRIVER, 'f32' if x.dtype == np.float32 else 'f64',
            *x.shape, k or 0, batch or 0, int(whiten), int(copy), mode, *x.ravel()]
    result = subprocess.run(list(map(str, args)), text=True, capture_output=True, check=True)
    got = {}
    for line in result.stdout.splitlines():
        name, value = line.split()
        got.setdefault(name, []).append(float(value))
    assert got['seen'] == [len(x)] and got['k'] == [reference.n_components_]
    tol = 8e-4 if x.dtype == np.float32 else 3e-9
    for key, value in expected.items():
        np.testing.assert_allclose(got[key], np.asarray(value).ravel(), rtol=tol, atol=tol,
                                   equal_nan=True, err_msg=f'{x.dtype} {x.shape} {k=} {batch=} {mode=} {whiten=} {copy=} {key}')


rng = np.random.RandomState(818)
count = 0
for dtype in (np.float64, np.float32):
    for n, p in [(17, 3), (10, 7), (11, 1), (3, 7)]:
        x = rng.normal(size=(n, p)).astype(dtype)
        x[n // 2:] += 4  # Explicit mean shift exposes missing correction rows.
        for k in (None, 1, min(3, p)):
            for batch in (4, 6, None):
                for whiten in (False, True):
                    for copy in (False, True):
                        compare(x, k, batch, whiten, copy, degenerate=n <= p)
                        count += 1
        if n > 3:
            for k in (1, min(3, p)):
                for whiten in (False, True):
                    compare(x, k, 4, whiten, mode='partial')
                    count += 1
    for x in [np.zeros((9, 3), dtype=dtype), np.ones((9, 3), dtype=dtype),
              np.arange(9, dtype=dtype)[:, None] * np.ones((1, 3), dtype=dtype)]:
        compare(x, 2, 4, degenerate=True)
        count += 1
print(f'PASS: {count} dense IncrementalPCA cases against local sklearn {plumbing["REVISION"]}')
