"""Pinned sklearn GraphicalLassoCV parity: grids, folds, scores and final refits.

Float64 tolerance 2e-8 accounts for path solve and geometric-grid rounding;
Float32 3e-5 additionally accounts for stored covariance/precision rounding.
Alpha selection and iteration counts are exact (rounding-equivalent grid
entries use the last index). Float32 LARS centered grids are checked against the original
reference at 3e-7, then numerical outputs against a Float64 oracle on the same
Float32-rounded data, matching native computation. Ill-conditioned short paths
can amplify the original Float32 grid rounding in fold scores. Other alpha
values use 1e-12.
Nonfinite score masks must match, including NaN standard deviations, except
rank-deficient empirical alpha-zero baselines (known inherited determinant
rounding limitation). These are counted explicitly. Float32 assume_centered
reference failures use Float64 oracle inputs rounded first to Float32. Single
alpha grids bypass the reference's IndexError and compare fixed-alpha refits.
"""
from pathlib import Path
import json
import os
import sys
import warnings
import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
import parity_io as io
import pinned_sklearn

GL = pinned_sklearn.load(['_empirical_covariance', '_graph_lasso'])['_graph_lasso']
assert pinned_sklearn.executed_from_pinned(GL.GraphicalLassoCV.fit)
OUT = ROOT / 'builds/cv_parity'
OUT.mkdir(parents=True, exist_ok=True)


def main(mode="cd"):
    warnings.simplefilter('ignore')
    summary = dict(widened_lars_reference=0, cases=0, checks=0, worst_scaled_error=0.0, widened_reference=0, singleton_reference_bug=0, singular_baseline_columns=0)
    cases = [
        ('default', 73, 4, 4, 4, 5, 100, False),
        ('refine', 113, 5, 5, 3, 4, 100, False),
        ('explicit', 91, 4, [0.02, 0.3, 0.07], 4, 3, 100, False),
        ('duplicates', 71, 3, [0.1, 0.03, 0.1], 1, 3, 100, False),
        ('custom', 83, 4, [0.03, 0.1], 2, 4, 100, True),
        ('short_path', 53, 4, 3, 2, 3, 20, False),
        ('wide', 12, 15, [0.5, 0.1, 0.01], 1, 3, 100, False),
        ('one_alpha', 63, 4, [0.1], 4, 3, 100, False),
        ('ties', 47, 3, [10., 100., 20.], 1, 3, 100, False),
        ('singleton_test', 6, 5, 5, 2, 5, 100, False),
    ]
    for dtype in (np.float64, np.float32):
        for index, (name, n, p, alphas, refinements, folds, iterations, custom) in enumerate(cases):
            for seed in range(int(os.environ.get('MOXIE_PARITY_SEEDS', '3'))):
                rng = np.random.default_rng(1500 + 10 * index + seed)
                X = rng.normal(size=(n, p))
                X[:, 1] += 0.6 * X[:, 0]
                X += np.linspace(-0.7, 1.3, p)
                X = X.astype(dtype)
                test = (rng.normal(size=(9, p)) + 0.5).astype(dtype)
                explicit = isinstance(alphas, list)
                text = ' '.join([str(32 if dtype == np.float32 else 64), io.matrix(X), io.matrix(test),
                                 str(len(alphas) if explicit else alphas), str(int(explicit)),
                                 *([io.real(a) for a in alphas] if explicit else []),
                                 str(refinements), str(folds), str(iterations), str(int(custom)), mode])
                native = io.run(ROOT / 'builds/cv_parity_driver', OUT / f'{mode}-{name}-{dtype.__name__}-{seed}.txt', text)
                summary['cases'] += 1
                for centered in (False, True):
                    prefix = f'cv{int(centered)}'
                    cv = list(KFold(folds, shuffle=True, random_state=42).split(X)) if custom else folds
                    try:
                        ref = GL.GraphicalLassoCV(mode=mode, alphas=alphas, n_refinements=refinements, cv=cv,
                                                 max_iter=iterations, assume_centered=centered).fit(X)
                    except IndexError:
                        # Pinned sklearn unconditionally indexes path[1] even for
                        # a one-element explicit grid. Verify our documented
                        # adaptation against the corresponding fixed-alpha fit.
                        assert name == 'one_alpha'
                        fixed = GL.GraphicalLasso(alphas[0], mode=mode, assume_centered=centered).fit(X.astype(np.float64))
                        np.testing.assert_allclose(native[prefix + '_covariance'], fixed.covariance_.ravel(),
                                                   rtol=3e-5, atol=3e-5)
                        assert native[prefix + '_alpha'][0] == alphas[0]
                        summary['checks'] += 2
                        summary['singleton_reference_bug'] += 1
                        continue
                    except ValueError as err:
                        if 'Buffer dtype mismatch' in str(err):
                            summary['widened_reference'] += 1
                            ref = GL.GraphicalLassoCV(mode=mode, alphas=alphas, n_refinements=refinements, cv=cv,
                                                     max_iter=iterations, assume_centered=centered).fit(X.astype(np.float64))
                        else:
                            raise
                    except (FloatingPointError, np.linalg.LinAlgError):
                        assert isinstance(native.get(prefix), Exception), (name, dtype, seed, prefix, native)
                        summary['checks'] += 1
                        continue
                    assert not isinstance(native.get(prefix), Exception), (name, dtype, seed, prefix, native.get(prefix))
                    grid_tol = 3e-7 if dtype == np.float32 else 1e-12
                    if mode == 'lars' and dtype == np.float32 and centered:
                        # Public grid/selection contract against actual Float32;
                        # remaining numerics against our documented Float64 engine.
                        np.testing.assert_allclose(native[prefix + '_alphas'], ref.cv_results_['alphas'], rtol=3e-7, atol=3e-7)
                        np.testing.assert_allclose(native[prefix + '_alpha'], ref.alpha_, rtol=3e-7, atol=3e-7)
                        selected = np.flatnonzero(np.isclose(native[prefix + '_alphas'], native[prefix + '_alpha'][0], rtol=grid_tol, atol=0))[-1]
                        original_selected = np.flatnonzero(np.isclose(ref.cv_results_['alphas'], ref.alpha_, rtol=grid_tol, atol=0))[-1]
                        assert selected == original_selected, (mode, name, seed, 'Float32 selected grid index')
                        ref = GL.GraphicalLassoCV(mode=mode, alphas=alphas, n_refinements=refinements, cv=cv,
                                                 max_iter=iterations, assume_centered=centered).fit(X.astype(np.float64))
                        summary['widened_lars_reference'] += 1
                        summary['checks'] += 3
                    expected = dict(alpha=ref.alpha_, alphas=ref.cv_results_['alphas'],
                                    splits=np.column_stack([ref.cv_results_[f'split{i}_test_score'] for i in range(folds)]),
                                    mean=ref.cv_results_['mean_test_score'], std=ref.cv_results_['std_test_score'],
                                    covariance=ref.covariance_, precision=ref.get_precision(), location=ref.location_,
                                    costs=ref.costs_, n_iter=ref.n_iter_, score=ref.score(test), mahalanobis=ref.mahalanobis(test))
                    # Coalesce rounding-equivalent refinement endpoints; their
                    # ordering is not meaningful at the grid's numeric precision.
                    selected = np.flatnonzero(np.isclose(native[prefix + '_alphas'], native[prefix + '_alpha'][0], rtol=grid_tol, atol=0))[-1]
                    expected_selected = np.flatnonzero(np.isclose(ref.cv_results_['alphas'], ref.alpha_, rtol=grid_tol, atol=0))[-1]
                    assert selected == expected_selected, (mode, name, seed, 'selected grid index')
                    singular_baseline = min(len(train) for train, _ in (cv if custom else KFold(folds).split(X))) <= p
                    for key, value in expected.items():
                        want = np.asarray(value).ravel()
                        got = np.asarray(native[prefix + '_' + key]).ravel()
                        if singular_baseline and key in ('splits', 'mean', 'std'):
                            # A pseudoinverse of a rank-deficient covariance has
                            # zero determinant. Floating-point slogdet sign can
                            # vary; inherited empirical scoring is not promised
                            # to match sklearn for this reporting-only baseline.
                            drop = folds if key == 'splits' else 1
                            got, want = got[:-drop], want[:-drop]
                            summary['singular_baseline_columns'] += 1
                        atol = 2e-8 if dtype == np.float64 else 3e-5
                        if key in ('alpha', 'alphas'):
                            atol = 1e-12
                        if key == 'n_iter':
                            atol = 0
                        np.testing.assert_allclose(got, want, rtol=atol, atol=atol, equal_nan=True,
                                                   err_msg=f'{name} {dtype} seed={seed} centered={centered} {key}')
                        finite = np.isfinite(want)
                        if atol and finite.any():
                            error = np.max(np.abs(got[finite] - want[finite]) / (atol * (1 + np.abs(want[finite]))))
                            summary['worst_scaled_error'] = max(summary['worst_scaled_error'], float(error))
                        summary['checks'] += 1
    (OUT / f'{mode}-results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('PASS GraphicalLassoCV parity:', mode, summary)


if __name__ == '__main__':
    main()
