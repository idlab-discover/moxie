"""Pinned sklearn Float32 intermediate arithmetic and partial-state regression gate.

Finite estimates use 3e-5 relative tolerance for native versus BLAS products;
zero/NaN/Inf classes and failure outcomes are exact. Binary cancellation fixtures
require exact coefficients. Failed-refit covariance/precision are compared to
that implementation's own pre-fit snapshot by the audit refit gate.
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
    shr = pinned_sklearn.load(['_empirical_covariance', '_shrunk_covariance'])['_shrunk_covariance']
    out = ROOT / 'builds/shrinkage-boundaries'
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(424)
    base = rng.normal(size=(17, 3)).astype(np.float32)
    fixtures = {'ordinary': base, 'fortran': np.asfortranarray(base), 'strided-reverse': base, 'offset': base + np.float32(1e4),
                'zero': np.zeros((17, 3), np.float32),
                'constant': np.full((17, 3), 1e30, np.float32),
                'single_feature': np.full((17, 1), 1e19, np.float32),
                'single_sample': np.array([[1., 2., 3.]], np.float32),
                'pairwise_tail': rng.normal(size=(13, 13)).astype(np.float32)}
    # Distinguish overflow before/after covariance normalization and squaring.
    for exponent in (8, 9, 10, 18, 19, 20, 30):
        fixtures[f'scale-{exponent}'] = base * np.float32(10.**exponent)
    for steps in (0, 1, 4):
        X = np.vstack([np.eye(3), -np.eye(3)]).astype(np.float32)
        for _ in range(steps):
            X[0, 0] = np.nextafter(X[0, 0], np.float32(2))
        fixtures[f'isotropic-{steps}'] = X
    # Same local dataset and normalization as sklearn's test_covariance.py.
    diabetes = np.loadtxt(ROOT.parents[2] / 'scikit-learn/sklearn/datasets/data/diabetes_data_raw.csv.gz')
    diabetes -= diabetes.mean(axis=0)
    diabetes /= diabetes.std(axis=0) * np.sqrt(len(diabetes))
    fixtures['diabetes'] = diabetes.astype(np.float32)
    fixtures['diabetes-small'] = diabetes[:, :4].astype(np.float32)
    fixtures['wide-blocks'] = np.random.RandomState(0).normal(size=(10, 20)).astype(np.float32)
    cases = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, X in fixtures.items():
            for kind, cls in enumerate((shr.LedoitWolf, shr.OAS)):
                for centered in (False, True):
                    for store in (False, True):
                        for refit in (False, True):
                            for block in ((1, 2, 1000) if kind == 0 else (1000,)):
                                case = f'{name}-{kind}-{int(centered)}-{int(store)}-{int(refit)}-{block}'
                                old = rng.normal(size=(19, X.shape[1])).astype(np.float32)
                                options = dict(assume_centered=centered, store_precision=store)
                                if kind == 0:
                                    options['block_size'] = block
                                model = cls(**options)
                                if refit:
                                    model.fit(old)
                                error = None
                                try:
                                    model.fit(X)
                                except Exception as exc:
                                    error = exc
                                layout = 1 if name == 'fortran' else 2 if name == 'strided-reverse' else 0
                                wire = X.T if layout == 1 else X[::-1] if layout == 2 else X
                                got = io.run(ROOT / 'builds/shrinkage_boundary_driver', out / (case + '.txt'),
                                             f'32 {kind} {io.matrix(old)} {io.matrix(wire)} {int(centered)} {int(store)} {block} {int(refit)} {layout}')
                                assert ('fit' in got) == (error is not None), (case, got.get('fit'), error)
                                if error is not None:
                                    assert type(error).__name__ in str(got['fit']), (case, got['fit'], error)
                                for attr in ('location', 'covariance', 'precision', 'shrinkage', 'n_features_in'):
                                    if attr == 'precision' and hasattr(model, 'covariance_'):
                                        expected = model.get_precision()
                                    elif hasattr(model, attr + '_'):
                                        expected = getattr(model, attr + '_')
                                    else:
                                        assert isinstance(got[attr], Exception), (case, attr, got[attr])
                                        continue
                                    actual = got[attr]
                                    assert not isinstance(actual, Exception), (case, attr, actual)
                                    exact = attr in ('shrinkage', 'location') and (name.startswith('isotropic') or name in ('zero', 'constant', 'single_feature', 'single_sample'))
                                    values = np.asarray(expected).ravel()
                                    finite = np.abs(values[np.isfinite(values)])
                                    scale = float(finite.max()) if finite.size else 0.
                                    absolute = 3e-6 if attr == 'shrinkage' else 3e-6 * scale
                                    np.testing.assert_allclose(actual, values,
                                                               rtol=0 if exact else 3e-5, atol=0 if exact else absolute,
                                                               equal_nan=True, err_msg=f'{case} {attr}')
                                if refit and error is not None:
                                    for attr in ('covariance', 'precision'):
                                        np.testing.assert_array_equal(got[attr], got['before_' + attr], err_msg=case)
                                if kind == 0:
                                    expected = shr.ledoit_wolf_shrinkage(X, assume_centered=centered, block_size=block)
                                    np.testing.assert_allclose(got['coefficient'], expected, rtol=3e-5, atol=3e-6, equal_nan=True, err_msg=case)
                                cases.append(dict(case=case, failed=error is not None,
                                                  shrinkage=str(getattr(model, 'shrinkage_', 'absent'))))
    (out / 'results.json').write_text(json.dumps(dict(reference=pinned_sklearn.REVISION, numpy=np.__version__, cases=cases), indent=2))
    print(f'PASS: {len(cases)} Float32 shrinkage boundary/state cases')


if __name__ == '__main__':
    main()
