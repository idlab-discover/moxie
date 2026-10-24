"""End-to-end parity of public neighbors estimators with the compiled pinned sklearn.

Unlike parity_estimators.py, which feeds pinned method bodies with SciPy
neighbor arrays, this calls the real oracle estimators (fit, queries, predict,
scores, LOF and KDE) on the same seeded continuous data. Python only writes
inputs and compares outputs; the native binary performs all learning and inference.
"""
import itertools
import os
import subprocess
import tempfile
import warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.neighbors import (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
    RadiusNeighborsClassifier, RadiusNeighborsRegressor, LocalOutlierFactor, KernelDensity, NearestCentroid, VALID_METRICS)

# Reference warnings for semi-metrics and empty neighborhoods are not under test.
warnings.simplefilter('ignore')

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT.parents[2]/'scikit-learn'
REV = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git', '-C', str(SK), 'rev-parse', 'HEAD'], text=True).strip() == REV
# Check the loaded estimator sources, not just a development version string.
assert sklearn.__version__ == '1.10.dev0', sklearn.__version__
ORACLE = Path(sklearn.__file__).resolve().parent
for name in ('_base.py', '_classification.py', '_regression.py', '_lof.py',
             '_kde.py', '_nearest_centroid.py'):
    expected = subprocess.check_output(['git', '-C', str(SK), 'show',
                                        f'{REV}:sklearn/neighbors/{name}'])
    assert (ORACLE / 'neighbors' / name).read_bytes() == expected, (ORACLE, name)
BIN = Path(os.environ.get('NEIGHBORS_E2E_BIN', ROOT/'builds/evaluate_end_to_end'))

rng = np.random.RandomState(0)
X = rng.uniform(-2, 2, size=(40, 3))
Q = rng.uniform(-2.2, 2.2, size=(9, 3))
labels = (X[:, 0] + X[:, 1] > 0).astype(float) + (X[:, 2] > 0.8)
q_labels = (Q[:, 0] + Q[:, 1] > 0).astype(float) + (Q[:, 2] > 0.8)
target = np.sin(X).sum(axis=1)
q_target = np.sin(Q).sum(axis=1)


def write(path, *arrays):
    with open(path, 'w') as f:
        for a in arrays:
            a = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
            f.write(f'{a.shape[0]} {a.shape[1]} ' + ' '.join(repr(float(v)) for v in a.ravel()) + '\n')


def native(dtype, path, *options):
    out = subprocess.run([str(BIN), dtype, path, *map(str, options)], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f'{options}: {(out.stderr or out.stdout).strip()}')
    records = {}
    for line in out.stdout.splitlines():
        key, index, value = line.split(maxsplit=2)
        entries = records.setdefault(key, [])
        assert int(index) == len(entries), (key, index, len(entries))
        entries.append(value)
    return records


def values(records, key):
    return np.array([float(v) for v in records[key]])


def ints(records, key):
    return np.array([int(v) for v in records[key]])


def close(name, expected, actual, dtype, scale=1.0):
    # Float64: exact distance formulas in a different summation order (2e-12).
    # Float32: sklearn computes some reductions in float64 then casts; native
    # accumulation in the feature dtype gives a few ulps on |values| <= ~10 (3e-6).
    tol = (3e-6 if dtype == 'f32' else 2e-12) * scale
    expected = np.asarray(expected, dtype=np.float64).ravel()
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol, err_msg=name)


count = 0
with tempfile.TemporaryDirectory(prefix='neighbors-e2e-') as workspace:
    data = Path(workspace)/'data.txt'
    metrics = [('minkowski', 2), ('minkowski', 1), ('minkowski', 3), ('chebyshev', 2), ('manhattan', 2),
               ('cosine', 2), ('canberra', 2), ('braycurtis', 2), ('sqeuclidean', 2), ('correlation', 2), ('minkowski', 0.5)]
    for dtype, (metric, p), algorithm in itertools.product(['f32', 'f64'], metrics, ['auto', 'brute', 'kd_tree', 'ball_tree']):
        typ = np.float32 if dtype == 'f32' else np.float64
        Xt, Qt = X.astype(typ), Q.astype(typ)
        if metric not in VALID_METRICS['brute' if algorithm == 'auto' else algorithm]:
            continue
        if metric == 'minkowski' and p < 1 and algorithm in ('kd_tree', 'ball_tree'):
            continue  # only brute/auto support semi-metrics
        write(data, Xt, labels, Qt, q_labels)
        # Query parity: indices exact (tie-free data), distances to tolerance.
        nn = NearestNeighbors(n_neighbors=4, radius=1.3 if metric not in ('cosine', 'correlation') else 0.3,
                              algorithm=algorithm, metric=metric, p=p, leaf_size=3).fit(Xt)
        rec = native(dtype, data, 'nn', algorithm, metric, p, 'uniform', 4, nn.radius)
        d, i = nn.kneighbors(Qt)
        np.testing.assert_array_equal(ints(rec, 'KI'), i.ravel(), err_msg=f'{metric} {p} {algorithm} kneighbors')
        close('kneighbors distances', d, values(rec, 'KD'), dtype)
        d, i = nn.kneighbors()
        np.testing.assert_array_equal(ints(rec, 'SI'), i.ravel(), err_msg='self kneighbors')
        close('self kneighbors distances', d, values(rec, 'SD'), dtype)
        d, i = nn.radius_neighbors(Qt, sort_results=True)
        np.testing.assert_array_equal(ints(rec, 'RO'), np.r_[0, np.cumsum([len(r) for r in i])])
        # Bray-Curtis can tie at 1.0. Check sorted output, then match each distance
        # to its neighbor index without imposing the reference's unspecified tie order.
        native_i, native_d, offsets = ints(rec, 'RI'), values(rec, 'RD'), ints(rec, 'RO')
        for row, (rd, ri) in enumerate(zip(d, i)):
            a, b = offsets[row], offsets[row + 1]
            assert sorted(native_i[a:b]) == sorted(ri), (metric, p, algorithm, row)
            assert np.all(np.diff(native_d[a:b]) >= 0), (metric, p, algorithm, row)
            close('radius distances', rd[np.argsort(ri)],
                  native_d[a:b][np.argsort(native_i[a:b])], dtype)
        assert rec['M'][0] == nn.effective_metric_, (rec['M'], nn.effective_metric_)
        count += 1
        for weights in ['uniform', 'distance']:
            clf = KNeighborsClassifier(5, weights=weights, algorithm=algorithm, metric=metric, p=p, leaf_size=3).fit(Xt, labels)
            rec = native(dtype, data, 'knnc', algorithm, metric, p, weights, 5, 1.0)
            np.testing.assert_array_equal(values(rec, 'C'), clf.predict(Qt), err_msg=f'knn predict {metric} {weights}')
            np.testing.assert_array_equal(values(rec, 'C0'), clf.classes_)
            close('knn proba', clf.predict_proba(Qt), values(rec, 'P'), dtype)
            close('knn score', [clf.score(Qt, q_labels)], values(rec, 'A'), dtype)
            reg = KNeighborsRegressor(5, weights=weights, algorithm=algorithm, metric=metric, p=p, leaf_size=3).fit(Xt, target.astype(typ))
            write(data, Xt, target.astype(typ), Qt, q_target.astype(typ))
            rec = native(dtype, data, 'knnr', algorithm, metric, p, weights, 5, 1.0)
            close('knn regression', reg.predict(Qt), values(rec, 'V'), dtype)
            close('knn regression score', [reg.score(Qt, q_target.astype(typ))], values(rec, 'A'), dtype, 10)
            r = 1.6 if metric not in ('cosine', 'correlation') else 0.4
            if metric == 'minkowski' and p == 0.5:
                r = 6.0
            rreg = RadiusNeighborsRegressor(r, weights=weights, algorithm=algorithm, metric=metric, p=p, leaf_size=3).fit(Xt, target.astype(typ))
            rec = native(dtype, data, 'rnr', algorithm, metric, p, weights, 5, r)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                close('radius regression', rreg.predict(Qt), values(rec, 'V'), dtype)
            write(data, Xt, labels, Qt, q_labels)
            # Native outlier_most_frequent=True is the reference's 'most_frequent'. The pinned brute
            # uniform fast path (_classification.py:805) forwards the raw string instead of the
            # resolved outlier_label_, so outliers get all-zero probabilities there but not on the
            # tree/generic path. Pass the resolved most frequent class (an encoded code equal to its
            # label for these 0/1/2 labels) so every backend follows the generic semantics.
            most_frequent = np.unique(labels)[np.bincount(np.searchsorted(np.unique(labels), labels)).argmax()]
            rclf = RadiusNeighborsClassifier(r, weights=weights, algorithm=algorithm, metric=metric, p=p, leaf_size=3,
                                             outlier_label=int(most_frequent)).fit(Xt, labels)
            rec = native(dtype, data, 'rnc', algorithm, metric, p, weights, 5, r)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                np.testing.assert_array_equal(values(rec, 'C'), rclf.predict(Qt), err_msg=f'radius predict {metric}')
                close(f'radius proba {dtype} {metric} {p} {algorithm} {weights}', rclf.predict_proba(Qt), values(rec, 'P'), dtype)
                close('radius score', [rclf.score(Qt, q_labels)], values(rec, 'A'), dtype)
            count += 4
        for contamination in ['auto', 0.1]:
            lof = LocalOutlierFactor(7, algorithm=algorithm, metric=metric, p=p, leaf_size=3, contamination=contamination)
            rec = native(dtype, data, 'lof', algorithm, metric, p, contamination, 7, 1.0)
            np.testing.assert_array_equal(values(rec, 'L'), lof.fit_predict(Xt), err_msg=f'lof fit_predict {metric}')
            # LOF divides by (mean reach distance + 1e-10): relative error grows by ~1/lrd.
            close('lof factors', lof.negative_outlier_factor_, values(rec, 'F'), dtype, 10)
            close('lof offset', [lof.offset_], values(rec, 'O'), dtype, 10)
            assert int(rec['N'][0]) == lof.n_neighbors_
            nov = LocalOutlierFactor(7, algorithm=algorithm, metric=metric, p=p, leaf_size=3, contamination=contamination,
                                     novelty=True).fit(Xt)
            close('lof score_samples', nov.score_samples(Qt), values(rec, 'SS'), dtype, 10)
            close('lof decision_function', nov.decision_function(Qt), values(rec, 'DF'), dtype, 10)
            np.testing.assert_array_equal(values(rec, 'PR'), nov.predict(Qt), err_msg='lof novelty predict')
            count += 1
    for dtype, kernel, algorithm, metric in itertools.product(['f32', 'f64'],
            ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'],
            ['auto', 'kd_tree', 'ball_tree'], ['euclidean', 'manhattan', 'chebyshev']):
        typ = np.float32 if dtype == 'f32' else np.float64
        Xt, Qt = X.astype(typ), Q.astype(typ)
        write(data, Xt, labels, Qt, q_labels)
        kde = KernelDensity(bandwidth=0.7, algorithm=algorithm, kernel=kernel, metric=metric).fit(Xt)
        rec = native(dtype, data, 'kde', algorithm, metric, 2, kernel, 1, 0.7)
        expected = kde.score_samples(Qt)
        # Compare non-finite values too: out-of-support log densities must be -inf.
        # Both input dtypes use Float64 KDE arithmetic with Numax libm.
        np.testing.assert_allclose(values(rec, 'SS'), expected, rtol=2e-12, atol=2e-12,
                                   err_msg=f'kde {dtype} {kernel} {metric}')
        np.testing.assert_allclose(values(rec, 'T'), [kde.score(Qt)], rtol=2e-12, atol=2e-12,
                                   err_msg=f'kde total {dtype} {kernel} {metric}')
        count += 1
    for dtype, metric, shrink, priors in itertools.product(['f32', 'f64'], ['euclidean', 'manhattan'], [0.0, 0.4],
                                                           ['uniform', 'empirical']):
        typ = np.float32 if dtype == 'f32' else np.float64
        Xt, Qt = X.astype(typ), Q.astype(typ)
        write(data, Xt, labels, Qt, q_labels)
        nc = NearestCentroid(metric=metric, shrink_threshold=shrink or None, priors=priors).fit(Xt, labels)
        rec = native(dtype, data, 'nc', 'auto', metric, 2, priors, 1, shrink)
        np.testing.assert_array_equal(values(rec, 'C'), nc.predict(Qt), err_msg=f'centroid {metric} {shrink} {priors}')
        close('centroids', nc.centroids_, values(rec, 'CE'), dtype)
        if metric == 'euclidean':
            # The reference squares pairwise_distances' expanded-form Euclidean (x.x - 2x.c + c.c),
            # which cancels to ~1e-10 relative on these responses; same 2e-9 as parity_centroid.py.
            scale = 10 if dtype == 'f32' else 1000
            close('centroid proba', nc.predict_proba(Qt), values(rec, 'P'), dtype, scale)
            close('centroid decision', nc.decision_function(Qt), values(rec, 'DF'), dtype, scale)
        close('centroid score', [nc.score(Qt, q_labels)], values(rec, 'A'), dtype)
        count += 1
    assert count == 872, f'Unexpected coverage: {count} configurations'
    print(f'PASS {count} end-to-end configurations against compiled pinned sklearn {REV}: NearestNeighbors, '
          'KNN/radius classifiers and regressors, LOF (outlier and novelty), KDE and NearestCentroid')
