"""Differential parity: native GraphicalLasso / graphical_lasso ('cd'/'lars') vs pinned sklearn.

The reference runs the pinned `_graph_lasso.py`; its coordinate-descent kernel
`_cd_fast.enet_coordinate_descent_gram` comes from the scikit-learn support
wheel (compiled Cython, same algorithm as the pinned .pyx), which the native
`_enet_gram.mojo` ports. LARS executes the pinned `_least_angle.py`.
Both follow identical iterate sequences, so results
agree to accumulated rounding: tolerance 1e-10 relative to each array's
magnitude times kappa(covariance)/1e2 floor 1 (Float64), 1e-6 for Float32
(iterates in Float64 on Float32-rounded inputs). n_iter must match exactly;
costs are compared elementwise with the same bound.
alpha=0 on a numerically singular emp_cov (cond > 1e10) inverts rounding
noise in both implementations and is only counted. Raised errors must coincide (e.g. FloatingPointError on ill-conditioning).
"""
from pathlib import Path
import json, math, os, sys, warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
import parity_io as io
import pinned_sklearn

modules = pinned_sklearn.load(['_empirical_covariance', '_graph_lasso'])
GL = modules['_graph_lasso']
for function in (GL.GraphicalLasso.fit, GL.graphical_lasso, GL._graphical_lasso, GL._dual_gap, GL._objective):
    assert pinned_sklearn.executed_from_pinned(function), function
DRIVER = ROOT / 'builds/parity_driver'
SEEDS = int(os.environ.get('MOXIE_PARITY_SEEDS', '3'))
OUT = ROOT / 'builds/parity'
OUT.mkdir(parents=True, exist_ok=True)
KINDS = {'small': (50, 4, 0.1, 100), 'sparse_graph': (200, 8, 0.05, 100), 'strong': (40, 5, 0.5, 100),
         'wide': (10, 12, 0.3, 100), 'few_iter': (80, 6, 0.01, 3), 'weak_penalty': (100, 6, 1e-3, 100), 'two_features': (50, 2, 0.1, 100)}


def dataset(kind, rng):
    n, p = KINDS[kind][:2]
    precision = np.eye(p)
    for j in range(p - 1):
        precision[j, j + 1] = precision[j + 1, j] = 0.4
    cov = np.linalg.inv(precision)
    return rng.multivariate_normal(np.zeros(p), cov, size=n) + rng.uniform(-1, 1, p)


def reference(X, X_test, emp, alpha, max_iter, mode):
    ref = {}
    for centered in (0, 1):
        key = f'gl_assume{centered}'
        try:
            model = GL.GraphicalLasso(alpha, assume_centered=bool(centered), max_iter=max_iter, mode=mode).fit(X)
            ref.update({key + '_location': model.location_, key + '_covariance': model.covariance_,
                        key + '_precision': model.get_precision(), key + '_costs': np.ravel(model.costs_),
                        key + '_n_iter': model.n_iter_, key + '_score': model.score(X_test),
                        key + '_mahalanobis': model.mahalanobis(X_test)})
        except FloatingPointError as e:
            ref[key] = e
        except ValueError as e:
            assert 'Buffer dtype mismatch' in str(e), e
            ref[key + '_reference_unsupported'] = None
    for zero in (0, 1):
        key = f'fn_alpha{zero}'
        try:
            c, p, costs, n_iter = GL.graphical_lasso(emp, 0.0 if zero else alpha, max_iter=max_iter, return_costs=True, return_n_iter=True, mode=mode)
            ref.update({key + '_covariance': c, key + '_precision': p, key + '_costs': np.ravel(costs), key + '_n_iter': n_iter})
        except (FloatingPointError, np.linalg.LinAlgError) as e:
            ref[key] = e
        except ValueError as e:
            # sklearn's Float32 precomputed path passes mixed float32/float64
            # buffers to the Cython kernel and fails ("Buffer dtype mismatch").
            assert 'Buffer dtype mismatch' in str(e), e
            ref[key + '_reference_unsupported'] = None
    return ref


def main(mode="cd"):
    warnings.simplefilter('ignore')
    summary = dict(cases=0, checks=0, worst={})
    for dtype in (np.float64, np.float32):
        for kind, (n, p, alpha, max_iter) in KINDS.items():
            for seed in range(SEEDS):
                rng = np.random.default_rng(700 + 10 * list(KINDS).index(kind) + seed)
                X = dataset(kind, rng).astype(dtype)
                X_test = dataset(kind, rng)[:8].astype(dtype)
                emp = np.cov(X.T.astype(np.float64), bias=1).astype(dtype)
                text = ' '.join([str(32 if dtype == np.float32 else 64), io.matrix(X), io.matrix(X_test), io.matrix(emp),
                                 io.real(alpha), str(max_iter), mode])
                native = io.run(DRIVER, OUT / f'{mode}-{kind}-{dtype.__name__}-{seed}.txt', text)
                ref = reference(X, X_test, emp, alpha, max_iter, mode)
                summary['cases'] += 1
                for key, expected in ref.items():
                    if key.endswith('_reference_unsupported'):
                        summary['reference_unsupported'] = summary.get('reference_unsupported', 0) + 1
                        continue
                    if isinstance(expected, Exception):
                        assert isinstance(native.get(key), Exception), (kind, key, 'reference raised', expected)
                        summary['checks'] += 1
                        continue
                    got = native.get(key)
                    if got is None:
                        prefix = key.split('_')[0] + '_' + key.split('_')[1]
                        assert not isinstance(native.get(prefix), Exception), (kind, dtype.__name__, seed, key, native.get(prefix))
                    assert got is not None, (kind, key)
                    expected = np.atleast_1d(np.asarray(expected, dtype=np.float64)).ravel()
                    got = np.asarray(got).ravel()
                    if key.startswith('fn_alpha1') and np.linalg.cond(np.asarray(emp, dtype=np.float64)) > 1e10:
                        # alpha=0 inverts a numerically singular emp_cov: both results are rounding noise.
                        summary['degenerate'] = summary.get('degenerate', 0) + 1
                        continue
                    if key.endswith('_n_iter'):
                        assert got[0] == expected[0], (kind, dtype.__name__, seed, key, got, expected)
                        summary['checks'] += 1
                        continue
                    assert got.shape == expected.shape, (mode, kind, key, got.shape, expected.shape)
                    cov_key = key.rsplit('_', 1)[0] + '_covariance'
                    kappa = float(np.linalg.cond(np.asarray(ref.get(cov_key, emp), dtype=np.float64)))
                    base = 1e-10 * max(1.0, kappa / 1e2) if dtype == np.float64 else 1e-6 * max(1.0, kappa / 1e2)
                    magnitude = float(np.max(np.abs(expected))) or 1.0
                    tolerance = base * magnitude
                    error = float(np.max(np.abs(got - expected)))
                    label = f"{dtype.__name__}:{key.rsplit('_', 1)[1]}"
                    summary['worst'][label] = float(max(summary['worst'].get(label, 0.0), error / tolerance))
                    summary['checks'] += 1
                    assert error <= tolerance, (kind, dtype.__name__, seed, key, got[:6], expected[:6], error, tolerance)
    (OUT / f'{mode}-results.json').write_text(json.dumps(summary, indent=1))
    worst = ', '.join(f'{k}={v:.2g}' for k, v in sorted(summary['worst'].items()))
    print(f"PASS {mode}: {summary['checks']} parity checks over {summary['cases']} cases (Float64/Float32); worst error/tolerance: {worst}")


if __name__ == '__main__':
    from lars_parity import main as lars_main
    lars_main()
    from cv_parity import main as cv_main
    for mode in ("cd", "lars"):
        main(mode)
        cv_main(mode)
