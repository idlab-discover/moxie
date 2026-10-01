"""Covariance solves/inverses against LAPACK-backed NumPy/SciPy and local sklearn."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import warnings

import numpy as np
import scipy
from scipy.linalg import solve
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
SKLEARN = ROOT.parents[2] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
SEED = 842901


def metric_oracle():
    path = SKLEARN / 'sklearn/metrics/pairwise.py'
    assert subprocess.check_output(['git', '-C', SKLEARN, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(['git', '-C', SKLEARN, 'diff', 'HEAD', '--', path], text=True)
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_precompute_metric_params')
    ns = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    assert ns[node.name].__code__.co_filename == str(path)
    return ns[node.name], hashlib.sha256(path.read_bytes()).hexdigest()


def run(a, b, layout, label):
    bits = a.dtype.itemsize * 8
    n = a.shape[0]
    cols = -1 if b.ndim == 1 else b.shape[1]
    path = ROOT/'builds/parity-fixtures'/f'{label}.txt'
    path.write_text(' '.join(map(str, [bits, n, cols, layout, *a.ravel(), *b.ravel()]))+'\n')
    lines = subprocess.check_output([ROOT/'builds/driver', path], text=True).splitlines()
    if lines[0] == 'ERROR':
        return None, lines[1]
    assert lines[0] == 'OK' and int(lines[1]) == b.ndim
    values = np.array(lines[2:], dtype=np.float64)
    assert values.size == b.size + 2*n*n
    return (values[:b.size].reshape(b.shape),
            values[b.size:b.size+n*n].reshape(n,n),
            values[b.size+n*n:].reshape(n,n)), None


def backward(a, x, b):
    if not b.size:
        return 0.0
    residual = np.abs(a.astype(float) @ x - b)
    scale = np.abs(a.astype(float)) @ np.abs(x) + np.abs(b.astype(float))
    relative = np.divide(residual, scale, out=np.zeros_like(residual), where=scale != 0)
    assert not np.any((scale == 0) & (residual != 0))
    return float(relative.max())



def main():
    oracle, source_hash = metric_oracle()
    fixture_dir = ROOT/'builds/parity-fixtures'
    fixture_dir.mkdir(exist_ok=True)
    rng = np.random.RandomState(SEED)
    cases = []
    # Modest condition numbers: tolerances cover input-dtype Cholesky rounding.
    for dtype in (np.float32, np.float64):
        for n in (1, 2, 5, 17, 48):
            raw = rng.normal(size=(n,n))
            a = (raw @ raw.T + np.eye(n) * n).astype(dtype)
            for cols in (-1, 1, 3):
                b = rng.normal(size=(n,) if cols == -1 else (n,cols)).astype(dtype)
                for layout in range(4):
                    cases.append((a, b, layout))
        for exponent in (-20, 20) if dtype == np.float32 else (-150, 150):
            a = (np.array([[4, 1], [1, 3]]) * 10.0**exponent).astype(dtype)
            b = (np.array([2, 5]) * 10.0**exponent).astype(dtype)
            cases.append((a, b, 2))
        for n, cols in ((0,-1), (0,0), (0,3), (2,0)):
            cases.append((np.eye(n,dtype=dtype), np.empty((n,) if cols == -1 else (n,cols),dtype=dtype), 0))
        # Small but representable diagonal and nearly singular correlated cases.
        tiny = 1e-12 if dtype == np.float32 else 1e-24
        cases.append((np.diag([1,tiny]).astype(dtype), np.array([2,tiny],dtype=dtype), 0))
        delta = 1e-3 if dtype == np.float32 else 1e-6
        cases.append((np.array([[1,1],[1,1+delta]],dtype=dtype), np.array([1,2],dtype=dtype), 0))
    worst = {32: 0., 64: 0.}
    for i,(a,b,layout) in enumerate(cases):
        bits = a.dtype.itemsize*8
        (x,precision,lower), error = run(a,b,layout,f'valid-{i}')
        assert error is None
        tolerance = 5e-5 if bits == 32 else 2e-10
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected = solve(a,b,assume_a='pos') if a.size and b.size else np.empty_like(b)
            expected_inverse = np.linalg.inv(a)
        np.testing.assert_allclose(x,expected,rtol=tolerance,atol=tolerance,err_msg=f'solve case {i}')
        np.testing.assert_allclose(precision,expected_inverse,rtol=tolerance,atol=tolerance * float(np.max(np.abs(expected_inverse), initial=0)),err_msg=f'inverse case {i}')
        if a.size:
            norm = np.linalg.norm(a.astype(float),ord=np.inf)
            reconstruction = np.linalg.norm(lower @ lower.T-a,ord=np.inf)/norm
            assert reconstruction < (5e-6 if bits == 32 else 2e-13), (i,reconstruction)
        err = max(backward(a,x,b), backward(a,precision,np.eye(a.shape[0])))
        assert err < (5e-6 if bits == 32 else 2e-13), (i,err)
        worst[bits] = max(worst[bits],err)
        assert np.isfinite(x).all() and np.isfinite(precision).all()
    failures = 0
    for dtype in (np.float32,np.float64):
        for a in [np.zeros((2,2)), [[1,1],[1,1]], [[1,2],[2,1]], [[1,0],[0,-1]],
                  [[1,np.nan],[0,1]], [[1,0],[np.inf,1]], [[1,0.01],[0,1]]]:
            result,error = run(np.array(a,dtype=dtype),np.ones(2,dtype=dtype),2,f'bad-{failures}')
            assert result is None and error.startswith('ValueError:'), error
            failures += 1
    # Exact family reproducer, including the reference sample-covariance policy.
    points = np.array([[0.,0.],[1.,0.],[0.,1.]])
    want = oracle(points,points,metric='mahalanobis')['VI']
    covariance = np.cov(points.T)
    (_,precision,_),error = run(covariance,np.eye(2),3,'mahalanobis')
    assert error is None
    np.testing.assert_allclose(precision,want,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(precision,[[4,2],[2,4]],rtol=1e-12,atol=1e-12)
    distances = cdist(points,points,metric='mahalanobis',VI=precision)
    np.testing.assert_allclose(distances,[[0,2,2],[2,0,2],[2,2,0]],rtol=1e-12,atol=1e-12)
    report = dict(valid_cases=len(cases),rejected_cases=failures,seed=SEED,
                  maximum_backward_error=worst,numpy=np.__version__,scipy=scipy.__version__,
                  sklearn_revision=REVISION,pairwise_source_sha256=source_hash,
                  compiler=subprocess.check_output(['mojo','--version'],text=True).strip(),flags='-O3',
                  caveat='Tolerances apply to tested conditioning and scaling only; no general ill-conditioning guarantee.')
    (ROOT/'builds/parity-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS {len(cases)} covariance solve/inverse cases, {failures} rejected cases, and pinned sklearn Mahalanobis fixture')


if __name__ == '__main__':
    main()
