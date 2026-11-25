"""Compare nested GPR search with local pinned scikit-learn."""
import subprocess

import numpy as np

from parity import GPR, K, ROOT, REVISION, load_pinned


def main():
    search = load_pinned(
        'sklearn/model_selection/_search.py',
        'sklearn.model_selection._moxie_search_reference',
        'sklearn.model_selection',
    )
    splitting = load_pinned(
        'sklearn/model_selection/_split.py',
        'sklearn.model_selection._moxie_split_reference',
        'sklearn.model_selection',
    )
    rows = np.arange(15)
    x = rows / 5
    X = x.astype(np.float32).reshape(-1, 1)
    y = np.column_stack((np.sin(x) + (rows % 3) / 20, x * x / 4)).astype(np.float32)
    weights = (rows + 1).astype(float)
    length_scale = 'kernel__k1__k2__length_scale'
    grid = {'alpha': [0.001, 0.05], length_scale: [0.2, 0.7, 1.4]}
    candidates = list(search.ParameterGrid(grid))
    score = []
    means = []
    for i, params in enumerate(candidates):
        fold_scores = []
        for fold, (train, test) in enumerate(splitting.KFold().split(X, y)):
            model = GPR(K.ConstantKernel(1.0) * K.RBF(1.0) + K.WhiteKernel(0.01), optimizer=None)
            model.set_params(**params).fit(X[train], y[train])
            value = model.score(X[test], y[test], sample_weight=weights[test])
            score.append([i, fold, value])
            fold_scores.append(value)
        means.append(np.mean(fold_scores))
    best = candidates[np.argmax(means)]
    refit = GPR(K.ConstantKernel(1.0) * K.RBF(1.0) + K.WhiteKernel(0.01), optimizer=None)
    refit.set_params(**best).fit(X, y)
    expected = {
        'score': score,
        'best': [[best['alpha'], best[length_scale]]],
        'prediction': refit.predict(X),
    }

    binary = ROOT / 'builds/tests/gaussian_process/regression/search_driver'
    lines = subprocess.check_output([binary], text=True).splitlines()
    assert lines.pop() == 'contracts passed'
    actual = {}
    for line in lines:
        name, *values = line.split()
        actual.setdefault(name, []).append([float(value) for value in values])
    assert actual.keys() == expected.keys()
    np.testing.assert_array_equal(np.asarray(actual['score'])[:, :2], np.asarray(score)[:, :2])
    np.testing.assert_array_equal(actual['best'], expected['best'])
    # Targets are stored in Float32 by the oracle and widened by Moxie.
    # Weighted R² can magnify roundoff on small, low-variance validation folds.
    np.testing.assert_allclose(actual['score'], score, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual['prediction'], expected['prediction'], rtol=0, atol=1e-7)
    print(f'PASS 30 fold scores, best parameters and 30 refit values against {REVISION}')


if __name__ == '__main__':
    main()
