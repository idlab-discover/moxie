"""Public native callable metrics vs the compiled, pinned sklearn checkout.

Run after pixi run build, using a Python environment with that checkout built.
Float32 tolerance covers distance storage and prediction rounding; Float64
uses 2e-12 for these bounded, two-feature weighted-L1 sums.
"""
import itertools
from pathlib import Path
import subprocess
import warnings

import numpy as np
import sklearn
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.neighbors import (LocalOutlierFactor, NearestNeighbors, KNeighborsClassifier,
                               KNeighborsRegressor, RadiusNeighborsClassifier,
                               RadiusNeighborsRegressor)
from sklearn.exceptions import NotFittedError

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parents[2] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
assert sklearn.__version__ == '1.10.dev0'
installed = Path(sklearn.__file__).resolve().parent.parent
for source in ['sklearn/neighbors/_base.py', 'sklearn/neighbors/_classification.py',
               'sklearn/neighbors/_regression.py', 'sklearn/metrics/pairwise.py', 'sklearn/base.py']:
    assert not subprocess.check_output(['git', '-C', str(REFERENCE), 'diff', 'HEAD', '--', source])
    assert (installed / source).read_bytes() == (REFERENCE / source).read_bytes()


class WeightedL1:
    def __init__(self):
        self.weights = np.array([1.75, 0.625])

    def __call__(self, x, y):
        return np.sum(self.weights * np.abs(x.astype(float) - y.astype(float)))


def smooth_weights(distances):
    if distances.dtype == object:
        result = np.empty(len(distances), dtype=object)
        for i, row in enumerate(distances):
            result[i] = 1 / (1 + row)
        return result
    return 1 / (1 + distances)


count = 0
for dtype, radius_mode, weights, multi, self_query, algorithm in itertools.product(
        [np.float32, np.float64], [False, True], ['uniform', 'distance', 'callable'],
        [False, True], [False, True], ['brute','auto','ball_tree']):
    x = np.zeros((7, 2), dtype=dtype)
    for i in range(2, 7):
        x[i] = [i, i*i % 5]
    q = np.array([[0, 0], [2.5, 1.5], [30, 0]], dtype=dtype)
    y = np.array([[i*i/3, i/7] for i in range(7)], dtype=dtype)
    labels = np.array([[2+i*2 % 3*3, -2+i % 2*5] for i in range(7)], dtype=dtype)
    y[1], labels[1] = y[0], labels[0]
    if not multi:
        y, labels = y[:, 0], labels[:, 0]
    query = None if self_query else q
    metric = WeightedL1()
    nn = NearestNeighbors(n_neighbors=3, radius=3.5, algorithm=algorithm, metric=metric).fit(x)
    kd, ki = nn.kneighbors(query)
    rd, ri = nn.radius_neighbors(query, sort_results=True)
    args = ['f32' if dtype == np.float32 else 'f64', 'radius' if radius_mode else 'knn',
            weights, 'multi' if multi else 'single', 'self' if self_query else 'query', algorithm]
    rows = [line.split() for line in subprocess.check_output(
        [str(ROOT/'builds/evaluate_callable_metrics'), *args], text=True).splitlines()]

    def array(tag, column, shape):
        return np.array([float(row[column]) for row in rows if row[0] == tag]).reshape(shape)

    tol = 3e-6 if dtype == np.float32 else 2e-12
    lof = LocalOutlierFactor(n_neighbors=3,algorithm=algorithm,metric=metric,novelty=True).fit(x)
    assert_allclose(array('L',3,(len(x),)),lof.negative_outlier_factor_,rtol=tol,atol=tol)
    assert_allclose(array('N',3,(len(q),)),lof.score_samples(q),rtol=tol,atol=tol)
    assert_allclose(array('K', 3, kd.shape), kd, rtol=tol, atol=tol)
    # Canonicalize duplicate coordinates: sklearn may choose either copy at
    # the k boundary. Their targets agree so downstream predictions are comparable.
    got_indices = array('K', 2, ki.shape).astype(int)
    assert_array_equal(np.sort(np.where(got_indices == 1, 0, got_indices), axis=1),
                       np.sort(np.where(ki == 1, 0, ki), axis=1))
    for i, indices in enumerate(got_indices):
        assert len(set(indices)) == len(indices)
        source = x[i] if self_query else q[i]
        assert_allclose([metric(source, x[j]) for j in indices], kd[i], rtol=tol, atol=tol)
    if self_query:
        assert not np.any(got_indices == np.arange(len(x))[:, None])
    assert_array_equal(array('O', 2, (len(ri),)), [len(row) for row in ri])
    for i in range(len(ri)):
        got = [(int(row[2]), float(row[3])) for row in rows if row[0] == 'D' and int(row[1]) == i]
        expected = sorted(zip(ri[i], rd[i]))
        assert_allclose(sorted(got), expected, rtol=tol, atol=tol)
    common = dict(algorithm=algorithm, metric=metric,
                  weights=smooth_weights if weights == 'callable' else weights)
    if radius_mode:
        classifier = RadiusNeighborsClassifier(radius=3.5, outlier_label='most_frequent', **common)
        regressor = RadiusNeighborsRegressor(radius=3.5, **common)
    else:
        classifier = KNeighborsClassifier(n_neighbors=3, **common)
        regressor = KNeighborsRegressor(n_neighbors=3, **common)
    # Clone a fitted estimator and refit, matching the native evaluation path.
    classifier = clone(classifier.fit(x, labels)).fit(x, labels)
    regressor = clone(regressor.fit(x, y)).fit(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cp = classifier.predict(query)
        rp = regressor.predict(query)
        pp = classifier.predict_proba(query)
    assert_array_equal(array('C', 3, cp.shape), cp)
    assert_allclose(array('R', 3, rp.shape), rp, rtol=tol, atol=tol, equal_nan=True)
    for j, probs in enumerate(pp if multi else [pp]):
        assert_allclose(array('P'+str(j), 3, probs.shape), probs, rtol=tol, atol=tol)
    count += 1

# Reference errors corresponding to native contract probes.
x = np.zeros((4, 2))
model = NearestNeighbors(n_neighbors=2, metric=WeightedL1(), algorithm='brute')
def raises(kind, fn):
    try:
        fn()
    except kind:
        return
    raise AssertionError(f'expected {kind.__name__}')
raises(NotFittedError, lambda: model.kneighbors())
model.fit(x)
raises(ValueError, lambda: model.kneighbors(n_neighbors=4))
raises(ValueError, lambda: model.kneighbors(np.zeros((2, 3))))
raises(ValueError, lambda: model.radius_neighbors(return_distance=False, sort_results=True))
raises(ValueError, lambda: NearestNeighbors(metric=WeightedL1(), algorithm='kd_tree').fit(x))
def failing_metric(x, y):
    raise ValueError('custom metric failure')
raises(ValueError, lambda: NearestNeighbors(n_neighbors=1, metric=failing_metric, algorithm='brute').fit(x).kneighbors())
print(f'PASS {count} callable-metric sklearn configurations and 6 reference error contracts; {REVISION}')
