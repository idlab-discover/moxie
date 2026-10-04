"""Validate native Float32 CD support separately from reference buffer failures.

The unmodified oracle is first observed and its dtype failure recorded. Only the
extension's numerical validation uses Float64 versions of the same rounded
inputs, explicitly; this is not reported as raw sklearn Float32 parity.
"""
from pathlib import Path
import json
import sys
import warnings
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import parity_io as io
import pinned_sklearn


def main():
    gl = pinned_sklearn.load()['_graph_lasso']
    out = ROOT / 'builds/graphical-extensions'
    out.mkdir(exist_ok=True)
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for seed in (310, 1510):
            X = np.random.default_rng(seed).normal(size=(53, 4)).astype(np.float32)
            X[:, 1] += .6 * X[:, 0]
            empirical = (X.T @ X / len(X)).astype(np.float32)
            got = io.run(ROOT / 'builds/audit_graphical_driver', out / f'{seed}.txt',
                         f'32 {io.matrix(X)} {io.matrix(X[:5])} {io.matrix(empirical)} {io.real(.1)} 100 cd')
            try:
                gl.graphical_lasso(empirical, .1)
            except ValueError as error:
                assert 'Buffer dtype mismatch' in str(error), error
                records.append(dict(seed=seed, reference_error=str(error)))
            else:
                raise AssertionError('Reference no longer fails; revisit extension classification')
            C, P = gl.graphical_lasso(empirical.astype(np.float64), .1)
            for key, expected in (('covariance', C), ('precision', P)):
                np.testing.assert_allclose(got['fn_alpha0_' + key], expected.ravel(), rtol=2e-5, atol=2e-6)
            for centered in (False, True):
                model = gl.GraphicalLasso(alpha=.1, assume_centered=centered).fit(X.astype(np.float64))
                key = f'gl_assume{int(centered)}_'
                np.testing.assert_allclose(got[key + 'covariance'], model.covariance_.ravel(), rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(got[key + 'precision'], model.precision_.ravel(), rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(got[key + 'score'], model.score(X[:5].astype(np.float64)), rtol=2e-5, atol=2e-6)
            grid = [.3, .1, .03]
            got = io.run(ROOT / 'builds/audit_cv_driver', out / f'cv-{seed}.txt',
                         f'32 {io.matrix(X)} {io.matrix(X[:5])} 3 1 ' + ' '.join(io.real(a) for a in grid) + ' 2 3 100 0 cd')
            for centered in (False, True):
                model = gl.GraphicalLassoCV(alphas=grid, cv=3, assume_centered=centered).fit(X.astype(np.float64))
                key = f'cv{int(centered)}_'
                np.testing.assert_array_equal(got[key + 'alpha'], [model.alpha_])
                np.testing.assert_allclose(got[key + 'covariance'], model.covariance_.ravel(), rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(got[key + 'precision'], model.precision_.ravel(), rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(got[key + 'mean'], model.cv_results_['mean_test_score'], rtol=2e-5, atol=2e-6)
    (out / 'results.json').write_text(json.dumps(dict(reference=pinned_sklearn.REVISION, extension_reference='Float64 of rounded Float32 inputs', observations=records), indent=2))
    print('PASS: Float32 CD and CV native extensions against explicitly widened reference')


if __name__ == '__main__':
    main()
