"""Compare the private LAR endpoint with the pinned sklearn Gram path.

Float64 2e-9 relative/absolute allows triangular reduction and 15-digit
correlation rounding differences. Random cases cover interior knots, complete
paths and sign crossings; constructed cases cover ties and rejected pivots.
Inputs must remain bitwise unchanged. No LassoLars API is claimed.
"""
from pathlib import Path
import json
import sys
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
import parity_io as io
import pinned_sklearn


def main():
    gl = pinned_sklearn.load(['_empirical_covariance', '_graph_lasso'])['_graph_lasso']
    assert pinned_sklearn.executed_from_pinned(gl.lars_path_gram)
    out = ROOT / 'builds/lars_parity'
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for seed in range(24):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(20, 9))
        X[:, 1] += 2 * X[:, 0]
        y = rng.normal(size=20)
        gram, xy = X.T @ X, X.T @ y
        for fraction in (0., 0.01, 0.2, 0.8, 1., 1.2):
            cases.append((f'random-{seed}-{fraction}', gram, xy, fraction * abs(xy).max(), np.finfo(float).eps))
    for name, gram, xy in [
        ('one', np.array([[2.]]), np.array([3.])),
        ('zero', np.eye(4), np.zeros(4)),
        ('tied', np.eye(4), np.array([2., -2., 2., -2.])),
        ('duplicate', np.ones((3, 3)), np.array([1., 1., 1.])),
        ('rejected', np.diag([1., 1e-16, 2.]), np.array([2., 1., 0.5])),
        ('nearly_duplicate', np.array([[1., 1.-1e-12], [1.-1e-12, 1.]]), np.array([1., 1.])),
    ]:
        for fraction in (0., 0.1, 0.9, 1.1):
            for eps in (np.finfo(float).eps, 1e-5):
                cases.append((f'{name}-{fraction}-{eps}', gram, xy, fraction * abs(xy).max(), eps))
    summary = {'cases': 0, 'worst_absolute_error': 0., 'worst_scaled_error': 0.}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, gram, xy, penalty, eps in cases:
            text = ' '.join([io.matrix(gram), io.matrix(xy[:, None]), io.real(penalty), io.real(eps)])
            got = io.run(ROOT / 'builds/lars_parity_driver', out / f'{name}.txt', text)
            _, _, expected = gl.lars_path_gram(Xy=xy, Gram=gram, n_samples=len(xy),
                                              alpha_min=penalty / len(xy), eps=eps,
                                              method='lar', return_path=False)
            assert not isinstance(got['coef'], Exception), (name, got['coef'])
            np.testing.assert_allclose(got['coef'], expected, rtol=2e-9, atol=2e-9, err_msg=name)
            np.testing.assert_array_equal(got['gram_after'], gram.ravel())
            np.testing.assert_array_equal(got['xy_after'], xy)
            scaled = np.max(abs(got['coef'] - expected) / (2e-9 * (1 + abs(expected))))
            summary['worst_scaled_error'] = max(summary['worst_scaled_error'], float(scaled))
            summary['cases'] += 1
            summary['worst_absolute_error'] = max(summary['worst_absolute_error'], float(np.max(abs(got['coef']-expected))))
    (out / 'results.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('PASS private LARS Gram parity:', summary)


if __name__ == '__main__':
    main()
