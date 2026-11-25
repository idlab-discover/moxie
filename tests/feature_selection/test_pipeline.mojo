import numojo as nm
from std.math import nan
from std.testing import assert_equal, assert_false, assert_raises
from moxie.base import clone
from moxie.pipeline import Pipeline
from moxie.dummy import DummyRegressor
from moxie.feature_selection import (
    VarianceThreshold, SelectKBest, BuiltinScorer, MutualInfoScorer, DiscreteFeatures,
)
from moxie.svm import LinearSVC

def check[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(6, 3))
    var y = nm.zeros[dtype](nm.Shape(6))
    for i in range(6):
        X.unsafe_set(3 * i, 7)
        X.unsafe_set(3 * i + 1, SIMD[dtype, 1](i))
        X.unsafe_set(3 * i + 2, SIMD[dtype, 1](i % 2))
        y.unsafe_set(i, SIMD[dtype, 1](i))
    var pipe = Pipeline(VarianceThreshold[dtype](), Pipeline(SelectKBest(BuiltinScorer[dtype]("r_regression"), k=1), DummyRegressor[dtype]()))
    _ = pipe.fit(X, y)
    assert_equal(pipe.transformer.get_support(True)[List[Int]], [1, 2])
    assert_equal(pipe.estimator.transformer.get_support(True)[List[Int]], [0])
    assert_equal(pipe.predict(X).shape[0], 6)
    assert_equal(Float64(pipe.predict(X).unsafe_get(0)), 2.5)
    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(Float64(X.unsafe_get(0)), 7.0)

def check_list_pipeline() raises:
    var X = nm.array[DType.float64](
        [0, 0, 1, 0, 0, 1, 1, 1, 0, 2, 1, 2, 0, 3, 1, 3], [8, 2],
    )
    var labels: List[String] = ["z", "a", "z", "a", "z", "a", "z", "a"]
    comptime Scorer = MutualInfoScorer[target_type=List[String]]
    var pipe = Pipeline(
        SelectKBest(Scorer(discrete_features=DiscreteFeatures(True), random_state=7), k=1),
        LinearSVC[String](),
    )
    with assert_raises():
        _ = pipe.predict(X)
    _ = pipe.fit(X, labels)
    assert_equal(pipe.transformer.get_support(True)[List[Int]], [0])
    assert_equal(pipe.predict(X), labels)
    assert_equal(pipe.estimator.classes_, ["a", "z"])
    assert_equal(X.unsafe_get(1), 0)

    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    var changed: List[String] = ["first", "second", "first", "second",
                                 "first", "second", "first", "second"]
    _ = fresh.fit(X, changed)
    _ = pipe.fit(X, changed)
    assert_equal(pipe.predict(X), changed)
    assert_equal(pipe.predict(X), fresh.predict(X))

    var string_regression = Scorer(classification=False)
    with assert_raises():
        _ = string_regression.score(X, labels.copy())
    var int_regression = MutualInfoScorer[target_type=List[Int]](classification=False)
    var ints: List[Int] = [0, 1, 0, 1, 0, 1, 0, 1]
    with assert_raises():
        _ = int_regression.score(X, ints^)
    var numeric = MutualInfoScorer[target_type=List[Float64]](random_state=7)
    with assert_raises():
        _ = numeric.score(X, None)
    with assert_raises():
        _ = numeric.score(X, List[Float64]())
    var short: List[Float64] = [0, 1]
    with assert_raises():
        _ = numeric.score(X, short^)
    var invalid: List[Float64] = [0, 1, 0, 1, 0, 1, 0, 0.5]
    with assert_raises():
        _ = numeric.score(X, invalid.copy())
    invalid[7] = nan[DType.float64]()
    with assert_raises():
        _ = numeric.score(X, invalid^)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    check_list_pipeline()
