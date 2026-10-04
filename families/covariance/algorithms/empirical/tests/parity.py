"""Differential parity: native empirical/fixed-shrinkage covariance vs pinned sklearn.

Every reference value is produced by the pinned scikit-learn covariance
sources (tools/pinned_sklearn.py). The native values come from the compiled
builds/parity_driver, which imports only the assembled Mojo package.

Tolerance model (stated per quantity, eps = unit roundoff of the tested dtype):
- location/covariance: first-order rounding of means and Gram entries,
  4 * n * eps relative to max|X - mean|^2 scale (NumPy float32 means
  accumulate sequentially in float32; covariance is exact up to rounding).
- precision: inverse perturbation bound 50 * p * eps * kappa; Mahalanobis
  distances and scores add a location rounding term: NumPy means carry up
  to dm = n * eps * max|X| error, which moves each distance by at most
  2 sqrt(d lambda_max(P)) dm sqrt(p) + lambda_max(P) dm^2 p, where kappa is the ratio of the largest to the
  smallest eigenvalue retained by the pseudoinverse cutoff (effective
  condition number of the reference covariance).
- Scores whose precision is rank deficient are determinants of rounding
  noise in both implementations (sign and magnitude undefined); they are
  counted as degenerate and only required to be -inf or finite, not equal.
- error norms: 100 * eps relative to the squared (or unsquared) input scale.

Environment: MOXIE_PARITY_SEEDS (default 3) seeds per dataset kind.
"""
from pathlib import Path
import json
import math
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
import parity_io as io
import pinned_sklearn

modules = pinned_sklearn.load(['_empirical_covariance', '_shrunk_covariance'])
EMP = modules['_empirical_covariance']
SHR = modules['_shrunk_covariance']
for function in (EMP.EmpiricalCovariance.fit, EMP.EmpiricalCovariance.score, EMP.empirical_covariance,
                 EMP.log_likelihood, EMP.fast_logdet, SHR.ShrunkCovariance.fit, SHR.shrunk_covariance):
    assert pinned_sklearn.executed_from_pinned(function), function

DRIVER = ROOT / 'builds/parity_driver'
SEEDS = int(os.environ.get('MOXIE_PARITY_SEEDS', '3'))
OUT = ROOT / 'builds/parity'
OUT.mkdir(parents=True, exist_ok=True)


def spd(rng, p, scale=1.0):
    a = rng.standard_normal((p, p))
    return scale * (a @ a.T / p + 0.5 * np.eye(p))


def dataset(kind, rng):
    if kind == 'tall':
        return rng.standard_normal((60, 4)) @ rng.standard_normal((4, 4)) + rng.standard_normal(4)
    if kind == 'square':
        return rng.standard_normal((12, 10))
    if kind == 'wide':
        return rng.standard_normal((5, 8))
    if kind == 'one_by_one':
        return rng.standard_normal((1, 1))
    if kind == 'two_by_one':
        return rng.standard_normal((2, 1))
    if kind == 'single_sample':
        return rng.standard_normal((1, 3))
    if kind == 'constant_column':
        x = rng.standard_normal((20, 3))
        x[:, 1] = 2.5
        return x
    if kind == 'ill_conditioned':
        base = rng.standard_normal((40, 1))
        return np.hstack([base, base + 1e-4 * rng.standard_normal((40, 1)), rng.standard_normal((40, 2))])
    if kind == 'offset':
        return 1e4 + rng.standard_normal((30, 3))
    raise ValueError(kind)


KINDS = ['tall', 'square', 'wide', 'one_by_one', 'two_by_one', 'single_sample', 'constant_column', 'ill_conditioned', 'offset']


def reference(X, X_test, comparison, shrinkage, batch):
    ref = {}
    for centered in (0, 1):
        ref[f'empirical_assume{centered}'] = EMP.empirical_covariance(X, assume_centered=bool(centered))
        for prefix, factory in (('ec', lambda s, a: EMP.EmpiricalCovariance(store_precision=s, assume_centered=a)),
                                ('sc', lambda s, a: SHR.ShrunkCovariance(store_precision=s, assume_centered=a, shrinkage=shrinkage))):
            for store in (0, 1):
                key = f'{prefix}_assume{centered}_store{store}'
                model = factory(bool(store), bool(centered)).fit(X)
                ref[key + '_location'] = model.location_
                ref[key + '_covariance'] = model.covariance_
                ref[key + '_precision'] = model.get_precision()
                ref[key + '_score'] = model.score(X_test)
                ref[key + '_mahalanobis'] = model.mahalanobis(X_test)
    model = EMP.EmpiricalCovariance().fit(X)
    ref['fast_logdet'] = EMP.fast_logdet(model.covariance_)
    ref['log_likelihood'] = EMP.log_likelihood(EMP.empirical_covariance(X_test), model.get_precision())
    for norm in ('frobenius', 'spectral'):
        for scaling in (0, 1):
            for squared in (0, 1):
                ref[f'error_{norm}_{scaling}{squared}'] = model.error_norm(comparison, norm=norm, scaling=bool(scaling), squared=bool(squared))
    ref['shrunk_batch'] = SHR.shrunk_covariance(batch, shrinkage)
    ref['shrunk_single'] = SHR.shrunk_covariance(comparison, shrinkage)
    return ref


def effective_condition(covariance, dtype):
    w = np.abs(np.linalg.eigvalsh(np.asarray(covariance, dtype=np.float64)))
    if w.max() == 0:
        return 1.0, 0
    kept = w[w > len(w) * np.finfo(dtype).eps * w.max()]
    return kept.max() / kept.min(), len(kept)


def location_term(key, ref, X, magnitude):
    prefix = key.rsplit('_', 1)[0]
    precision = ref.get(prefix + '_precision', ref['ec_assume0_store1_precision'])
    n, p = X.shape
    dm = n * np.finfo(X.dtype).eps * float(np.max(np.abs(X)))
    lam = float(np.max(np.abs(np.linalg.eigvalsh(np.asarray(precision, dtype=np.float64)))))
    return 2 * math.sqrt(magnitude * lam) * dm * math.sqrt(p) + lam * dm * dm * p


def main():
    import warnings
    warnings.simplefilter('ignore')
    summary = {'cases': 0, 'checks': 0, 'degenerate_scores': 0, 'worst': {}}
    records = []
    for dtype in (np.float64, np.float32):
        eps = np.finfo(dtype).eps
        for kind in KINDS:
            for seed in range(SEEDS):
                rng = np.random.default_rng(1000 * KINDS.index(kind) + seed)
                X = dataset(kind, rng).astype(dtype)
                n, p = X.shape
                X_test = (dataset(kind, rng)[:7] if kind not in ('one_by_one', 'two_by_one', 'single_sample') else rng.standard_normal((3, p))).astype(dtype)
                if X_test.shape[1] != p:
                    X_test = rng.standard_normal((7, p)).astype(dtype)
                comparison = spd(rng, p).astype(dtype)
                shrinkage = [0.0, 0.37, 1.0][seed % 3]
                batch = np.stack([spd(rng, p) for _ in range(3)]).astype(dtype)
                text = ' '.join([str(32 if dtype == np.float32 else 64), io.matrix(X), io.matrix(X_test), io.matrix(comparison),
                                 io.real(shrinkage), '3', *(io.matrix(b) for b in batch)])
                native = io.run(DRIVER, OUT / f'{kind}-{dtype.__name__}-{seed}.txt', text)
                ref = reference(X, X_test, comparison, shrinkage, batch)
                summary['cases'] += 1
                scale = float(np.max(np.abs(X - X.mean(0))) ** 2) or 1.0
                for key, expected in ref.items():
                    got = native.get(key)
                    if got is None:
                        prefix = key.rsplit('_', 1)[0]
                        got = native.get(prefix.rsplit('_', 0)[0])
                    assert got is not None and not isinstance(got, Exception), (kind, dtype, seed, key, got)
                    expected = np.atleast_1d(np.asarray(expected, dtype=np.float64)).ravel()
                    got = np.asarray(got).ravel()
                    assert got.shape == expected.shape, (kind, key, got.shape, expected.shape)
                    covariance_key = key.split('_store')[0] if '_store' in key else 'ec_assume0'
                    reference_covariance = ref.get(f'{covariance_key}_store1_covariance', ref['ec_assume0_store1_covariance'])
                    kappa, rank = effective_condition(reference_covariance, dtype)
                    if key.endswith('_score') or key in ('fast_logdet', 'log_likelihood'):
                        if rank < p or not np.isfinite(expected[0]):
                            summary['degenerate_scores'] += 1
                            assert np.isneginf(got[0]) or np.isfinite(got[0]), (kind, key, got)
                            continue
                        tolerance = 50 * p * eps * kappa * max(1.0, abs(expected[0])) + location_term(key, ref, X, 1.0)
                        error = abs(got[0] - expected[0])
                        category = 'score'
                    elif key.endswith('_location'):
                        tolerance = 4 * n * eps * max(1.0, float(np.max(np.abs(X))))
                        error = float(np.max(np.abs(got - expected)))
                        category = 'location'
                    elif key.endswith('_precision') or key.endswith('_mahalanobis'):
                        magnitude = float(np.max(np.abs(expected))) or 1.0
                        tolerance = 50 * p * eps * kappa * magnitude + (0.0 if key.endswith('_precision') else location_term(key, ref, X, magnitude))
                        error = float(np.max(np.abs(got - expected)))
                        category = 'precision' if key.endswith('_precision') else 'mahalanobis'
                    elif key.startswith('error_'):
                        squared = key.endswith('1')
                        magnitude = max(abs(expected[0]), (scale if squared else math.sqrt(scale)), float(np.max(np.abs(comparison))) ** (2 if squared else 1))
                        tolerance = 100 * p * eps * magnitude
                        error = abs(got[0] - expected[0])
                        category = 'error_norm'
                    else:
                        magnitude = max(float(np.max(np.abs(expected))), scale if 'shrunk' not in key else 0.0) or 1.0
                        tolerance = 4 * n * eps * magnitude + 4 * eps * magnitude
                        error = float(np.max(np.abs(got - expected)))
                        category = 'covariance'
                    summary['checks'] += 1
                    worst = summary['worst'].setdefault(f'{dtype.__name__}:{category}', 0.0)
                    summary['worst'][f'{dtype.__name__}:{category}'] = float(max(worst, error / tolerance if tolerance else 0.0))
                    records.append(dict(kind=kind, dtype=dtype.__name__, seed=seed, key=key, error=float(error), tolerance=float(tolerance)))
                    assert error <= tolerance, (kind, dtype.__name__, seed, key, got, expected, error, tolerance)
    (OUT / 'results.json').write_text(json.dumps(dict(summary=summary, records=records), indent=1))
    worst = ', '.join(f'{k}={v:.2g}' for k, v in sorted(summary['worst'].items()))
    print(f"PASS {summary['checks']} parity checks over {summary['cases']} cases (Float64/Float32; "
          f"{summary['degenerate_scores']} rank-deficient scores checked for definedness); worst error/tolerance: {worst}")


if __name__ == '__main__':
    main()
