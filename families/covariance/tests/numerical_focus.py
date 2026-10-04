"""Reproduce inverse/cost and singular CV differences without Float32 CD errors.

Observation command; --strict fails on any public numerical mismatch. The
oracle is never widened. Widened inverse comparisons are separately labelled
diagnostics, not parity checks. Inputs and intermediate matrices stay in builds/.
"""
from pathlib import Path
import json
import os
import sys
import warnings

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import scipy
from scipy.linalg import inv
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import parity_io as io
import pinned_sklearn
from singular_boundaries import exact_determinant_sign

MODULES = pinned_sklearn.load()
GL = MODULES['_graph_lasso']
EMP = MODULES['_empirical_covariance']
OUT = ROOT / 'builds/numerical-focus'


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def probe(matrix, name, test=None):
    bits = 32 if matrix.dtype == np.float32 else 64
    text = f'{bits} {io.matrix(matrix)}'
    if test is not None:
        text += ' ' + io.matrix(test)
    return io.run(ROOT / 'builds/singular_driver', OUT / f'{name}.txt', text)


def logdet(matrix):
    sign, value = np.linalg.slogdet(matrix)
    return float(value) if sign > 0 else -np.inf


def residual(matrix, precision):
    a, p = matrix.astype(np.float64), precision.astype(np.float64)
    return np.linalg.norm(a @ p - np.eye(len(a)), np.inf) / max(
        1., np.linalg.norm(a, np.inf) * np.linalg.norm(p, np.inf))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    differences, inverse_records, fold_records, lars_records = [], [], [], []
    comparisons = 0

    def compare(case, method, got, expected, dtype):
        nonlocal comparisons
        comparisons += 1
        got, expected = np.asarray(got).ravel(), np.asarray(expected).ravel()
        tol = 5e-5 if dtype == np.float32 else 2e-8
        if got.shape != expected.shape or not np.allclose(
                got, expected, rtol=tol, atol=tol, equal_nan=True):
            entry = dict(case=case, method=method, native=got, reference=expected)
            if got.shape == expected.shape:
                entry['indices'] = np.flatnonzero(~np.isclose(
                    got, expected, rtol=tol, atol=tol, equal_nan=True))
            differences.append(entry)

    base = np.random.default_rng(310).normal(size=(30, 3))
    base[:, 1] += .8 * base[:, 0]
    for dtype in (np.float64, np.float32):
        X = base.astype(dtype)
        X[:, 1] = X[:, 0] + 1e-7 * X[:, 1]
        empirical = np.cov(X.T.astype(np.float64), bias=True).astype(dtype)
        for mode in ('cd', 'lars'):
            case = f'graphical-{mode}-{dtype.__name__}-ill_conditioned'
            native = io.run(ROOT / 'builds/audit_graphical_driver', OUT / f'{case}.txt',
                            f'{32 if dtype == np.float32 else 64} {io.matrix(X)} '
                            f'{io.matrix(X[:8])} {io.matrix(empirical)} {io.real(.01)} 100 {mode}')
            for alpha in ((0., .01) if mode == 'lars' else (0.,)):
                prefix = 'fn_alpha1' if alpha == 0 else 'fn_alpha0'
                expected = GL.graphical_lasso(empirical, alpha, mode=mode,
                                             return_costs=True, return_n_iter=True)
                for key, value in zip(('covariance', 'precision', 'costs', 'n_iter'), expected):
                    compare(case, prefix + '.' + key, native[prefix + '_' + key], value, dtype)
                if alpha > 0:
                    widened = GL.graphical_lasso(empirical.astype(np.float64), alpha,
                                                mode=mode, return_costs=True, return_n_iter=True)
                    lars_records.append(dict(dtype=np.dtype(dtype).name,
                                             native_costs=native[prefix + '_costs'].reshape(-1, 2),
                                             reference_costs=expected[2],
                                             widened_reference_costs=widened[2]))
        # Identify SciPy's structure-dependent inverse separately from dtype widening.
        native = probe(empirical, f'inverse-{dtype.__name__}')['inverse'].reshape(3, 3)
        variants = {}
        for workspace in (dtype, np.float64):
            for structure in (None, 'sym', 'gen', 'pos'):
                name = f'{np.dtype(workspace).name}-{structure}'
                try:
                    value = inv(empirical.astype(workspace), assume_a=structure)
                    variants[name] = dict(precision=value, residual=residual(empirical, value),
                                          relative_to_native=np.linalg.norm(value-native) / np.linalg.norm(value))
                except np.linalg.LinAlgError as error:
                    variants[name] = dict(error=str(error))
        inverse_records.append(dict(dtype=np.dtype(dtype).name, covariance=empirical,
                                    condition=np.linalg.cond(empirical.astype(np.float64)),
                                    native=native, native_residual=residual(empirical, native), variants=variants))

    for dtype in (np.float64, np.float32):
        X = np.random.default_rng(1510).normal(size=(53, 4))
        X[:, 1] += .6 * X[:, 0]
        X = (X + np.linspace(-.7, 1.3, 4))[:6].astype(dtype)
        for fold, (train, test) in enumerate(KFold(3).split(X)):
            name = f'fold-{dtype.__name__}-{fold}'
            native = probe(X[train], name, X[test])
            reference = EMP.EmpiricalCovariance().fit(X[train])
            p = native['precision'].reshape(4, 4)
            q = reference.precision_
            reference_on_native = probe(q, name + '-reference-precision')['logdet'][0]
            fold_records.append(dict(case=name, native_score=native['score'][0],
                                     reference_score=reference.score(X[test]),
                                     native_covariance=native['covariance'].reshape(4, 4),
                                     reference_covariance=reference.covariance_,
                                     native_precision=p, reference_precision=q,
                                     native_precision_logdet_native=probe(p, name + '-native-precision')['logdet'][0],
                                     native_precision_logdet_numpy=logdet(p),
                                     reference_precision_logdet_native=reference_on_native,
                                     reference_precision_logdet_numpy=logdet(q),
                                     native_exact_sign=exact_determinant_sign(p),
                                     reference_exact_sign=exact_determinant_sign(q)))
        for mode in ('cd', 'lars'):
            case = f'cv-{mode}-{dtype.__name__}-singular_baseline'
            native = io.run(ROOT / 'builds/audit_cv_driver', OUT / f'{case}.txt',
                            f'{32 if dtype == np.float32 else 64} {io.matrix(X)} '
                            f'{io.matrix(X[:5])} 3 0 2 3 100 0 {mode}')
            # Float32 CD assume_centered=True is explicitly out of this task.
            for centered in ((False,) if dtype == np.float32 and mode == 'cd' else (False, True)):
                model = GL.GraphicalLassoCV(alphas=3, n_refinements=2, cv=3,
                                           mode=mode, assume_centered=centered).fit(X)
                expected = np.column_stack([model.cv_results_[f'split{i}_test_score'] for i in range(3)])
                prefix = f'cv{int(centered)}'
                compare(case, prefix + '.splits', native[prefix + '_splits'], expected, dtype)
                compare(case, prefix + '.alpha', native[prefix + '_alpha'], model.alpha_, dtype)

    report = dict(reference=pinned_sklearn.REVISION, numpy=np.__version__, scipy=scipy.__version__,
                  compiler_flags='-O3 --fp-mode=contract=off', threads=1, seeds=[310, 1510],
                  comparisons=comparisons, differences=differences, inverses=inverse_records,
                  folds=fold_records, lars=lars_records)
    (OUT / 'results.json').write_text(json.dumps(serial(report), indent=2, allow_nan=False) + '\n')
    print(f'NUMERICAL AUDIT: {comparisons} comparisons, {len(differences)} differences; {OUT / "results.json"}')
    if '--strict' in sys.argv and differences:
        raise SystemExit(1)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
