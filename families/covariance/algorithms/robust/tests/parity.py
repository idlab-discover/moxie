"""Differential parity: native MinCovDet / EllipticEnvelope / fast_mcd vs pinned sklearn.

Reference values run the pinned `_robust_covariance.py` and
`_elliptic_envelope.py` (tools/pinned_sklearn.py) with the same integer seeds;
the native code draws from the legacy RandomState port, so subsets match.

Tolerances:
- support masks and predicted labels: exact equality (combinatorial output).
- estimates, distances, scores: Float64 50 * p * eps * kappa relative to
  the array magnitude (kappa: condition number of the reference covariance),
  plus a location term for Float32 (NumPy float32 means carry n * eps error;
  the port accumulates in Float64), as derived in the empirical child.
- EllipticEnvelope score (accuracy): exact.
Tie normalization: sklearn breaks exact ties through NumPy's unstable
argsort/argpartition (platform dependent; e.g. the two endpoints of the
univariate shortest half are always equidistant from its midpoint). The
reference robust module therefore runs with a NumPy proxy whose argsort is
stable and whose argpartition keeps the lowest indices among ties, the
documented native rule. No other reference code is changed.
Environment: MOXIE_PARITY_SEEDS (default 3).
"""
from pathlib import Path
import ast
import json
import math
import os
import sys
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
import parity_io as io
import pinned_sklearn

modules = pinned_sklearn.load(['_empirical_covariance', '_robust_covariance', '_elliptic_envelope'])
ROB = modules['_robust_covariance']
ELL = modules['_elliptic_envelope']

# The inherited method also comes from the pinned checkout, not the wheel.
import sklearn.base
base_path = pinned_sklearn.REFERENCE / 'sklearn/base.py'
base_tree = ast.parse(base_path.read_text(), filename=str(base_path))
outlier = next(node for node in base_tree.body if isinstance(node, ast.ClassDef) and node.name == 'OutlierMixin')
fit_predict = next(node for node in outlier.body if isinstance(node, ast.FunctionDef) and node.name == 'fit_predict')
base_namespace = vars(sklearn.base).copy()
exec(compile(ast.Module(body=[fit_predict], type_ignores=[]), str(base_path), 'exec'), base_namespace)
ELL.EllipticEnvelope.fit_predict = base_namespace['fit_predict']
assert pinned_sklearn.executed_from_pinned(ELL.EllipticEnvelope.fit_predict)


# Scoring validation changed after the support wheel: execute the pinned
# helpers in copies of the support namespaces, including their call chain.
import sklearn.metrics._classification as classification
import sklearn.utils.validation as validation
import sklearn.utils._array_api as array_api


def pinned_definitions(relative, names, namespace):
    path = pinned_sklearn.REFERENCE / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    for name in names:
        assert pinned_sklearn.executed_from_pinned(namespace[name])
    return namespace


weight_namespace = pinned_definitions('sklearn/utils/validation.py', ['_check_sample_weight'], vars(validation).copy())
average_namespace = pinned_definitions('sklearn/utils/_array_api.py', ['_average'], vars(array_api).copy())
score_namespace = vars(classification).copy()
score_namespace.update(_check_sample_weight=weight_namespace['_check_sample_weight'], _average=average_namespace['_average'])
pinned_definitions('sklearn/metrics/_classification.py', ['_check_targets', 'accuracy_score'], score_namespace)
ELL.accuracy_score = score_namespace['accuracy_score']


class _StableNumPy:
    """NumPy proxy: deterministic lowest-index tie breaking for sorting helpers."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def argsort(a, axis=-1, kind=None, order=None):
        return np.argsort(a, axis=axis, kind='stable')

    @staticmethod
    def argpartition(a, kth, axis=-1):
        return np.argsort(a, axis=axis, kind='stable')


ROB.np = _StableNumPy()
for function in (ROB.MinCovDet.fit, ROB.fast_mcd, ROB._c_step, ROB.select_candidates, ROB._consistency_factor,
                 ROB.MinCovDet.reweight_covariance, ELL.EllipticEnvelope.error_norm, ELL.EllipticEnvelope.fit, ELL.EllipticEnvelope.predict):
    assert pinned_sklearn.executed_from_pinned(function), function

DRIVER = ROOT / 'builds/parity_driver'
SEEDS = int(os.environ.get('MOXIE_PARITY_SEEDS', '3'))
OUT = ROOT / 'builds/parity'
OUT.mkdir(parents=True, exist_ok=True)
# kind: (n, p, support_fraction or None, assume_centered, contamination)
KINDS = {
    'small': (40, 2, None, False, 0.1),
    'moderate': (120, 5, None, False, 0.2),
    'fraction': (80, 3, 0.7, False, 0.1),
    'full_support': (30, 2, 1.0, False, 0.05),
    'centered': (60, 3, None, True, 0.1),
    'univariate': (35, 1, None, False, 0.1),
    'univariate_full': (20, 1, 1.0, False, 0.3),
    'subsets': (700, 3, None, False, 0.1),
    'merged_full': (1600, 2, None, False, 0.1),
}


def dataset(kind, rng):
    n, p = KINDS[kind][:2]
    x = rng.standard_normal((n, p)) @ rng.uniform(0.5, 2.0, (p, p))
    outliers = max(1, n // 10)
    x[:outliers] += rng.uniform(5, 10, (outliers, p)) * rng.choice([-1, 1], (outliers, p))
    return x


def postprocess(ref, model, X, X_test, key):
    for step in range(3):
        post = f'{key}_post{step}'
        if step != 1:
            ref[post + '_corrected'] = model.correct_covariance(X).copy()
            ref[post + '_corrected_dist'] = model.dist_.copy()
        result = model.reweight_covariance(X)
        for name, value in zip(('location', 'covariance', 'support'), result):
            ref[post + '_returned_' + name] = value.copy()
        for name in ('location', 'covariance', 'dist', 'raw_covariance', 'raw_location', 'support'):
            ref[post + '_' + name] = getattr(model, name + '_').copy()
        ref[post + '_precision'] = model.get_precision().copy()
        if key == 'ee':
            ref[post + '_offset'] = model.offset_
            ref[post + '_predict'] = model.predict(X_test)


def reference(X, X_test, y, fraction, seed, assume, contamination):
    ref = {}
    location, covariance, support, dist = ROB.fast_mcd(X, support_fraction=fraction, random_state=seed)
    ref.update(fast_mcd_location=location, fast_mcd_covariance=covariance, fast_mcd_support=support, fast_mcd_dist=dist)
    for store in (0, 1):
        key = f'mcd_store{store}'
        model = ROB.MinCovDet(store_precision=bool(store), assume_centered=assume, support_fraction=fraction, random_state=seed).fit(X)
        for name in ('raw_location', 'raw_covariance', 'raw_support', 'location', 'covariance', 'support', 'dist'):
            ref[f'{key}_{name}'] = getattr(model, name + '_').copy()
        ref[key + '_precision'] = model.get_precision()
        ref[key + '_score'] = model.score(X_test)
        ref[key + '_mahalanobis'] = model.mahalanobis(X_test)
        postprocess(ref, model, X, X_test, key)
    detector = ELL.EllipticEnvelope(assume_centered=assume, support_fraction=fraction, contamination=contamination, random_state=seed)
    ref['ee_fit_predict'] = detector.fit_predict(X, y)
    ref.update(ee_offset=detector.offset_, ee_predict=detector.predict(X_test), ee_decision=detector.decision_function(X_test),
               ee_score_samples=detector.score_samples(X_test), ee_score=detector.score(X_test, y), ee_covariance=detector.covariance_)
    p = X.shape[1]
    comparison = np.fromfunction(lambda i, j: 2 * i - j + 1, (p, p)).astype(X.dtype)
    backing = np.zeros((2 * p + 1, 2 * p + 1), dtype=X.dtype)
    strided = backing[-2::-2, -2::-2]
    strided[:] = comparison
    ref['ee_error_default'] = detector.error_norm(comparison)
    for norm in ('frobenius', 'spectral'):
        for scaling in (0, 1):
            for squared in (0, 1):
                key = f'ee_error_{norm}_{scaling}_{squared}'
                ref[key] = detector.error_norm(comparison, norm, bool(scaling), bool(squared))
                ref[key + '_strided'] = detector.error_norm(strided, norm, bool(scaling), bool(squared))
    for key, comp, norm in [('ee_error_invalid_norm', comparison, 'invalid'),
                            ('ee_error_invalid_shape', np.zeros((0, 0)), 'spectral')]:
        try:
            ref[key] = detector.error_norm(comp, norm)
        except (ValueError, NotImplementedError) as error:
            ref[key] = error
    for kind, value in enumerate((np.nan, np.inf, -np.inf)):
        nonfinite = comparison.copy()
        nonfinite[0, 0] = value
        ref[f'ee_error_nonfinite_{kind}'] = detector.error_norm(nonfinite)
    n = len(y)
    signed = np.arange(1, n + 1, dtype=float)
    signed[0] = -2
    cancelling = np.zeros(n)
    cancelling[:2] = [1, -1]
    weights = [np.arange(1, n + 1, dtype=float), signed,
               np.ones(n - 1), np.ones(n + 1), np.ones((n, 1)), np.ones(0),
               np.r_[np.nan, np.ones(n - 1)], np.r_[np.inf, np.ones(n - 1)],
               np.r_[-np.inf, np.ones(n - 1)], np.zeros(n), cancelling]
    for kind, weight in enumerate(weights):
        try:
            ref[f'ee_weighted_{kind}'] = detector.score(X_test, y, sample_weight=weight)
        except (ValueError, ZeroDivisionError) as error:
            ref[f'ee_weighted_{kind}'] = error
    postprocess(ref, detector, X, X_test, "ee")
    return ref


def main():
    warnings.simplefilter('ignore')
    summary = dict(cases=0, checks=0, worst={})
    for dtype in (np.float64, np.float32):
        eps = np.finfo(dtype).eps
        for kind, (n, p, fraction, assume, contamination) in KINDS.items():
            seeds = SEEDS if n <= 700 else 1
            for seed in range(seeds):
                rng = np.random.default_rng(9000 + 10 * list(KINDS).index(kind) + seed)
                X = dataset(kind, rng).astype(dtype)
                X_test = dataset(kind, rng)[:15].astype(dtype)
                y = np.where(np.arange(15) < 2, -1, 1)
                text = ' '.join([str(32 if dtype == np.float32 else 64), io.matrix(X), io.matrix(X_test), io.matrix(y),
                                 io.real(fraction if fraction else -1.0), str(seed), '1' if assume else '0', io.real(contamination)])
                native = io.run(DRIVER, OUT / f'{kind}-{dtype.__name__}-{seed}.txt', text)
                ref = reference(X, X_test, y, fraction, seed, assume, contamination)
                summary['cases'] += 1
                kappa = float(np.linalg.cond(np.atleast_2d(np.asarray(ref['mcd_store1_covariance'], dtype=np.float64))))
                raw_kappa = float(np.linalg.cond(np.atleast_2d(np.asarray(ref['mcd_store1_raw_covariance'], dtype=np.float64))))
                kappa = max(kappa, raw_kappa)
                lam = 1.0 / float(np.min(np.linalg.eigvalsh(np.atleast_2d(np.asarray(ref['mcd_store1_raw_covariance'], dtype=np.float64)))))
                dm = n * eps * float(np.max(np.abs(X))) if dtype == np.float32 else 0.0
                for key, expected in ref.items():
                    got = native.get(key)
                    if isinstance(expected, Exception):
                        assert isinstance(got, Exception), (kind, key, got, expected)
                        assert str(got).startswith(type(expected).__name__ + ':'), (kind, key, got, expected)
                        summary['checks'] += 1
                        continue
                    assert got is not None and not isinstance(got, Exception), (kind, dtype.__name__, seed, key, got)
                    expected = np.atleast_1d(np.asarray(expected, dtype=np.float64)).ravel()
                    got = np.asarray(got).ravel()
                    assert got.shape == expected.shape, (kind, key, got.shape, expected.shape)
                    if key.startswith('ee_error_nonfinite_'):
                        np.testing.assert_equal(got, expected)
                        summary['checks'] += 1
                        continue
                    if key.startswith('ee_weighted_') or key.endswith(('support', 'predict')) or key in ('ee_predict', 'ee_fit_predict', 'ee_score'):
                        category, tolerance = 'exact', 0.0
                        error = float(np.max(np.abs(got - expected)))
                    else:
                        magnitude = float(np.max(np.abs(expected))) or 1.0
                        location = 2 * math.sqrt(magnitude * lam) * dm * math.sqrt(p) + lam * dm * dm * p
                        tolerance = 50 * p * eps * kappa * magnitude + location * (1 if 'location' not in key and 'covariance' not in key else 0) + 4 * n * eps * magnitude
                        error = float(np.max(np.abs(got - expected)))
                        category = key.split('_')[-1]
                    summary['checks'] += 1
                    label = f'{dtype.__name__}:{category}'
                    ratio = error / tolerance if tolerance else (0.0 if error == 0 else math.inf)
                    summary['worst'][label] = float(max(summary['worst'].get(label, 0.0), ratio))
                    assert error <= tolerance, (kind, dtype.__name__, seed, key, got[:8], expected[:8], error, tolerance)
    (OUT / 'results.json').write_text(json.dumps(summary, indent=1))
    worst = ', '.join(f'{k}={v:.2g}' for k, v in sorted(summary['worst'].items()))
    print(f"PASS {summary['checks']} parity checks over {summary['cases']} cases (Float64/Float32, exact supports/labels); worst error/tolerance: {worst}")


if __name__ == '__main__':
    main()
