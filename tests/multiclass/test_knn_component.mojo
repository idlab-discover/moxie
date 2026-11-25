"""Native KNN component ownership and response validation."""
import numojo as nm
from moxie._multiclass_binary import positive_probability
from moxie.base import clone
from moxie.multiclass import OneVsOneClassifier, OneVsRestClassifier, pairwise_input
from moxie.neighbors import KNeighborsClassifier
from moxie.neighbors._distance import Metric
from std.testing import assert_equal, assert_false, assert_raises, assert_true


def check_copy[dtype: DType](algorithm: String) raises:
    comptime KNN = KNeighborsClassifier[dtype, Y=nm.NDArray[DType.int64]]
    var X = nm.zeros[dtype](nm.Shape(4, 1))
    var y = nm.zeros[DType.int64](nm.Shape(4))
    for i in range(4):
        X[nm.Item(i, 0)] = Float64(i).cast[dtype]()
        y[nm.Item(i)] = Int64(i // 2)

    var model = OneVsRestClassifier[KNN, dtype](
        KNN(1, algorithm=algorithm)
    )
    with assert_raises(contains="NotFittedError"):
        _ = model.predict_proba(X)
    assert_equal(model.fit(X, y).predict_proba(X)[nm.Item(0, 0)], 1.0)
    var fitted_copy = model.estimators_[0].estimator().copy()
    fitted_copy.classes_[0][nm.Item(0)] = -1
    assert_equal(model.estimators_[0].estimator().classes_[0][nm.Item(0)], 0)
    fitted_copy.classes_[0][nm.Item(0)] = 0
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var params = fresh.get_params()
    assert_equal(params.estimator.neighbors.algorithm, algorithm)
    params.estimator.neighbors.n_neighbors = 3
    _ = fresh.set_params(params).fit(X, y)
    assert_equal(fresh.get_params().estimator.neighbors.n_neighbors, 3)
    assert_equal(model.get_params().estimator.neighbors.n_neighbors, 1)

    for i in range(4):
        y[nm.Item(i)] = 1 - y[nm.Item(i)]
    _ = model.fit(X, y)
    assert_equal(model.predict_proba(X)[nm.Item(0, 1)], 1.0)
    assert_equal(fitted_copy.predict_proba(X)[0][nm.Item(0, 0)], 1.0)
    with assert_raises(contains="ValueError"):
        _ = model.fit(X, nm.zeros[DType.int64](nm.Shape(3)))
    assert_equal(model.predict_proba(X)[nm.Item(0, 1)], 1.0)


def check_probability_shape() raises:
    var X = nm.zeros[DType.float64](nm.Shape(2, 1))
    X[nm.Item(1, 0)] = 1.0
    var y = nm.zeros[DType.int64](nm.Shape(2, 2))
    y[nm.Item(1, 0)] = 1
    y[nm.Item(1, 1)] = 1
    var component = KNeighborsClassifier[Y=nm.NDArray[DType.int64]](1)
    _ = component.fit(X, y)
    assert_equal(len(component.predict_proba(X)), 2)
    with assert_raises(contains="expected one binary predict_proba matrix"):
        _ = positive_probability(component, X, 2)

    _ = component.fit(X, nm.zeros[DType.int64](nm.Shape(2)))
    with assert_raises(contains="expected one binary predict_proba matrix"):
        _ = positive_probability(component, X, 2)

    var model = OneVsRestClassifier(KNeighborsClassifier[Y=nm.NDArray[DType.int64]](1))
    var binary = nm.zeros[DType.int64](nm.Shape(2))
    binary[nm.Item(1)] = 1
    assert_equal(model.fit(X, binary).predict(X)[nm.Item(1)], 1)


def check_pairwise_fit() raises:
    comptime KNN = KNeighborsClassifier[Y=nm.NDArray[DType.int64]]
    var distances = nm.zeros[DType.float64](nm.Shape(6, 6))
    var features = nm.zeros[DType.float64](nm.Shape(6, 1))
    var y = nm.zeros[DType.int64](nm.Shape(6))
    for i in range(6):
        features[nm.Item(i, 0)] = Float64(i)
        y[nm.Item(i)] = Int64(i % 3)
        for j in range(6):
            distances[nm.Item(i, j)] = Float64(abs(i - j))

    var model = OneVsOneClassifier(KNN(1), n_jobs=2)
    assert_false(pairwise_input(model))
    _ = model.set_params(estimator=KNN(1, metric_params=Metric[DType.float64]("precomputed")))
    assert_true(pairwise_input(model))
    _ = model.fit(distances, y)
    assert_equal(model.n_features_in_, 6)
    assert_equal(model.estimators_[0].n_features_in_, 4)
    assert_equal(model.pairwise_indices_.value()[0][nm.Item(2)], 3)
    assert_equal(model.score(distances, y), 1.0)
    var fresh = clone(model)
    assert_true(pairwise_input(fresh))
    assert_false(Bool(fresh.pairwise_indices_))

    with assert_raises(contains="square"):
        _ = model.fit(features, y)
    with assert_raises(contains="ValueError"):
        _ = model.predict(features)
    _ = model.set_params(estimator=KNN(1, metric="precomputed", metric_params=Metric[DType.float64]("euclidean")))
    assert_false(pairwise_input(model))
    assert_equal(model.score(distances, y), 1.0)
    _ = model.fit(features, y)
    assert_false(Bool(model.pairwise_indices_))
    assert_equal(model.score(features, y), 1.0)


def main() raises:
    for algorithm in ["brute", "kd_tree", "ball_tree"]:
        check_copy[DType.float32](algorithm)
        check_copy[DType.float64](algorithm)
    check_probability_shape()
    check_pairwise_fit()
    print("PASS: KNN component ownership, probability shape and pairwise fit")
