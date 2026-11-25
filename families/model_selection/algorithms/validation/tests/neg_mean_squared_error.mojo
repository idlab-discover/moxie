from moxie.dummy import DummyRegressor
from moxie.model_selection import NegMeanSquaredError
from numojo import NDArray, Shape, Strides, Item, zeros
from std.math import nan, inf
from std.testing import assert_raises


def check[dtype: DType]() raises:
    comptime Dense = NDArray[dtype]
    comptime Scorer = NegMeanSquaredError[DummyRegressor[dtype], Dense, dtype]
    var scorer = Scorer()
    var X = zeros[dtype](Shape(3, 1))
    var y = Dense(Shape(3, 2))
    for i in range(y.size):
        y.unsafe_set(i, Scalar[dtype](i * (1 + i % 2)) / 3.0)
    var model = DummyRegressor[dtype]()
    with assert_raises(contains="NotFittedError"):
        _ = scorer.score(model, X, y, None)
    _ = model.fit(X, y)

    var weights = Dense(Shape(3))
    weights[Item(0)] = 0.25
    weights[Item(1)] = -0.125
    weights[Item(2)] = 1.0
    var reversed_y = y.view_with_layout(Shape(3, 2), Strides(-2, 1), 4)
    print(String(dtype), "multioutput", scorer.score(model, X, reversed_y, None))
    var reversed_weights = weights.view_with_layout(Shape(3), Strides(-1), 2)
    print(String(dtype), "weighted", scorer.score(model, X, reversed_y, reversed_weights^))

    var vector = Dense(Shape(3))
    for i in range(3):
        vector[Item(i)] = Scalar[dtype](i)
    _ = model.fit(X, vector)
    var column = vector.view_with_layout(Shape(3, 1), Strides(1, 1), 0)
    print(String(dtype), "singleton_column", scorer.score(model, X, column, None))

    with assert_raises(contains="matching nonempty"):
        _ = scorer.score(model, X, y, None)
    with assert_raises(contains="matching nonempty"):
        _ = scorer.score(model, X, zeros[dtype](Shape(0)), None)
    with assert_raises(contains="vector or matrix"):
        _ = scorer.score(model, X, zeros[dtype](Shape(3, 1, 1)), None)
    with assert_raises(contains="one sample weight per row"):
        _ = scorer.score(model, X, vector, zeros[dtype](Shape(3, 1)))
    with assert_raises(contains="weights sum to zero"):
        _ = scorer.score(model, X, vector, zeros[dtype](Shape(3)))
    with assert_raises(contains="sample weights must be finite"):
        var invalid = zeros[dtype](Shape(3))
        invalid[Item(1)] = inf[dtype]()
        _ = scorer.score(model, X, vector, invalid^)
    with assert_raises(contains="targets and predictions must be finite"):
        vector[Item(0)] = nan[dtype]()
        _ = scorer.score(model, X, vector, None)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
