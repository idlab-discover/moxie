"""Differential parity: native LedoitWolf/OAS/ledoit_wolf*/oas vs pinned sklearn.

Reference values run the pinned `_shrunk_covariance.py` (tools/pinned_sklearn.py).
Tolerances (eps = unit roundoff of the tested dtype):
- shrinkage coefficients: sklearn evaluates the Ledoit-Wolf/OAS sums in the
  input dtype; public Float32 native paths preserve that dtype, while native
  products can differ from BLAS. The historical conservative bound is
  200 * p * eps * c + 4 * r,
  where c = (|beta_/n| + |delta_|) / (n p |delta|) + 1 is the cancellation
  factor of the formula and r = |sklearn - exact Float64 formula| is the
  measured rounding of the reference itself (zero for Float64 input).
- covariance: 4 * n * eps * scale plus ds * max|S - mu I| (coefficient error).
- precision: 50 * p * eps * kappa * |P| + |P|^2 * covariance tolerance;
  distances and scores scale that relative error (kappa as in empirical).
The stricter family tests/shrinkage_boundaries.py gate overflow and cancellation.
Rank-deficient scores (determinant noise) are only checked for definedness.
"""
from pathlib import Path
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

modules = pinned_sklearn.load(['_empirical_covariance', '_shrunk_covariance'])
SHR = modules['_shrunk_covariance']
for function in (SHR.LedoitWolf.fit, SHR.OAS.fit, SHR.ledoit_wolf, SHR.ledoit_wolf_shrinkage, SHR.oas, SHR._ledoit_wolf, SHR._oas):
    assert pinned_sklearn.executed_from_pinned(function), function

DRIVER = ROOT / 'builds/parity_driver'
SEEDS = int(os.environ.get('MOXIE_PARITY_SEEDS', '3'))
OUT = ROOT / 'builds/parity'
OUT.mkdir(parents=True, exist_ok=True)
KINDS = {'tall': (60, 4), 'square': (12, 10), 'wide': (5, 8), 'many_features': (30, 60), 'single_feature': (20, 1),
         'two_samples': (2, 3), 'single_sample': (1, 3), 'constant_column': (25, 3), 'offset': (40, 3)}


def dataset(kind, rng):
    n, p = KINDS[kind]
    x = rng.standard_normal((n, p)) @ np.diag(rng.uniform(0.5, 3.0, p))
    if kind == 'constant_column':
        x[:, 1] = 1.5
    if kind == 'offset':
        x += 1e3
    return x


def reference(X, X_test):
    ref = {}
    for centered in (0, 1):
        tag = f'_assume{centered}'
        a = bool(centered)
        for block_size in (1000, 1, 3):
            ref[f'lw_shrinkage{tag}_bs{block_size}'] = SHR.ledoit_wolf_shrinkage(X, assume_centered=a, block_size=block_size)
        ref[f'ledoit_wolf{tag}_covariance'], ref[f'ledoit_wolf{tag}_shrinkage'] = SHR.ledoit_wolf(X, assume_centered=a, block_size=2)
        ref[f'oas{tag}_covariance'], ref[f'oas{tag}_shrinkage'] = SHR.oas(X, assume_centered=a)
        for store in (0, 1):
            for name, cls in (('LW', SHR.LedoitWolf), ('OAS', SHR.OAS)):
                key = f'{name}{tag}_store{store}'
                model = cls(store_precision=bool(store), assume_centered=a).fit(X)
                ref[key + '_location'] = model.location_
                ref[key + '_covariance'] = model.covariance_
                ref[key + '_shrinkage'] = model.shrinkage_
                ref[key + '_precision'] = model.get_precision()
                ref[key + '_score'] = model.score(X_test)
                ref[key + '_mahalanobis'] = model.mahalanobis(X_test)
    return ref


def exact_statistics(X, assume_centered):
    """Float64 evaluation of the sklearn formulas on the same (rounded) data."""
    Z = np.asarray(X, dtype=np.float64)
    if not assume_centered:
        Z = Z - np.asarray(X.mean(0), dtype=X.dtype).astype(np.float64)
    n, p = Z.shape
    S = Z.T @ Z / n
    trace = np.trace(S)
    mu = trace / p
    beta_ = np.sum((Z ** 2).sum(1) ** 2)
    delta_ = np.sum(S ** 2)
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2 * mu * trace + p * mu * mu) / p
    lw = 0.0 if p == 1 else (0.0 if min(beta, delta) == 0 else min(beta, delta) / delta)
    alpha = delta_ / p ** 2
    den = (n + 1) * (alpha - mu * mu / p)
    oas = 0.0 if p == 1 else (1.0 if den == 0 else min((alpha + mu * mu) / den, 1.0))
    cancellation = (abs(beta_ / n) + abs(delta_)) / (n * p * abs(delta)) + 1 if delta else 1.0
    return dict(lw=lw, oas=oas, cancellation=cancellation, S=S, mu=mu)


def main():
    warnings.simplefilter('ignore')
    summary = dict(cases=0, checks=0, degenerate_scores=0, worst={})
    for dtype in (np.float64, np.float32):
        eps = np.finfo(dtype).eps
        for kind in KINDS:
            for seed in range(SEEDS):
                rng = np.random.default_rng(5000 + 100 * list(KINDS).index(kind) + seed)
                X = dataset(kind, rng).astype(dtype)
                X_test = dataset(kind, rng)[:6].astype(dtype) if KINDS[kind][0] >= 6 else rng.standard_normal((4, X.shape[1])).astype(dtype)
                n, p = X.shape
                native = io.run(DRIVER, OUT / f'{kind}-{dtype.__name__}-{seed}.txt',
                                f"{32 if dtype == np.float32 else 64} {io.matrix(X)} {io.matrix(X_test)}")
                ref = reference(X, X_test)
                summary['cases'] += 1
                exact = {c: exact_statistics(X, bool(c)) for c in (0, 1)}
                for key, expected in ref.items():
                    got = native.get(key)
                    assert got is not None and not isinstance(got, Exception), (kind, key, got, [k for k in native if 'error' in str(native[k])])
                    expected = np.atleast_1d(np.asarray(expected, dtype=np.float64)).ravel()
                    got = np.asarray(got).ravel()
                    assert got.shape == expected.shape, (kind, key)
                    centered = int(key.split('assume')[1][0])
                    stats = exact[centered]
                    formula = 'oas' if key.lower().startswith('oas') else 'lw'
                    reference_shrinkage = float(ref[f"{'OAS' if formula == 'oas' else 'LW'}_assume{centered}_store1_shrinkage"])
                    shrink_tol = 200 * p * eps * stats['cancellation'] + 4 * abs(reference_shrinkage - stats[formula])
                    scale = max(float(np.max(np.abs(stats['S']))), 1e-300)
                    spread = float(np.max(np.abs(stats['S'] - stats['mu'] * np.eye(p))))
                    cov_tol = 4 * n * eps * scale + shrink_tol * spread
                    if key.endswith('shrinkage') or '_bs' in key:
                        tolerance, category = shrink_tol, 'shrinkage'
                        error = abs(got[0] - expected[0])
                    elif key.endswith('_location'):
                        tolerance, category = 4 * n * eps * max(1.0, float(np.max(np.abs(X)))), 'location'
                        error = float(np.max(np.abs(got - expected)))
                    elif key.endswith('_covariance'):
                        tolerance, category = cov_tol, 'covariance'
                        error = float(np.max(np.abs(got - expected)))
                    else:
                        covariance = np.asarray(ref[key.rsplit('_', 1)[0] + '_covariance'], dtype=np.float64)
                        w = np.abs(np.linalg.eigvalsh(covariance))
                        kept = w[w > p * eps * w.max()] if w.max() > 0 else np.array([1.0])
                        kappa = kept.max() / kept.min()
                        norm_p = 1.0 / kept.min()
                        relative = 50 * p * eps * kappa + norm_p * cov_tol
                        dm = n * eps * float(np.max(np.abs(X)))
                        location = 2 * math.sqrt(norm_p * max(1.0, float(np.max(np.abs(expected))))) * dm * math.sqrt(p) + norm_p * dm * dm * p
                        if key.endswith('_score'):
                            if len(kept) < p or not np.isfinite(expected[0]):
                                summary['degenerate_scores'] += 1
                                assert np.isneginf(got[0]) or np.isfinite(got[0]), (kind, key, got)
                                continue
                            tolerance, category = p * relative * max(1.0, abs(expected[0])) + location, 'score'
                            error = abs(got[0] - expected[0])
                        else:
                            magnitude = float(np.max(np.abs(expected))) or 1.0
                            tolerance = relative * magnitude + (location if key.endswith('_mahalanobis') else 0.0)
                            category = key.rsplit('_', 1)[1]
                            error = float(np.max(np.abs(got - expected)))
                    summary['checks'] += 1
                    label = f'{dtype.__name__}:{category}'
                    summary['worst'][label] = float(max(summary['worst'].get(label, 0.0), error / tolerance if tolerance else (0.0 if error == 0 else math.inf)))
                    assert error <= tolerance, (kind, dtype.__name__, seed, key, got, expected, error, tolerance)
    (OUT / 'results.json').write_text(json.dumps(summary, indent=1))
    worst = ', '.join(f'{k}={v:.2g}' for k, v in sorted(summary['worst'].items()))
    print(f"PASS {summary['checks']} parity checks over {summary['cases']} cases (Float64/Float32; "
          f"{summary['degenerate_scores']} rank-deficient scores checked for definedness); worst error/tolerance: {worst}")


if __name__ == '__main__':
    main()
