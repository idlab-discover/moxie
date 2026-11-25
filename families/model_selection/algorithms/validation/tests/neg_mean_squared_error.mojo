from moxie.dummy import DummyRegressor
from moxie.model_selection import NegMeanAbsoluteError, NegMeanSquaredError, NegRootMeanSquaredError, TypedScorer
from numojo import NDArray, Shape, Strides, Item, array, zeros
from std.math import nan, inf
from std.testing import assert_raises


def check[Scorer: TypedScorer, //, dtype: DType](scorer: Scorer, loss: StaticString) raises:
    comptime Dense = NDArray[dtype]
    comptime Model = DummyRegressor[dtype]
    comptime assert Scorer.Estimator == Model
    comptime assert Scorer.Input == Dense and Scorer.Target == Dense
    comptime assert Scorer.Metadata == Optional[Dense]

    var X = zeros[dtype](Shape(3, 1))

    def score(scorer: Scorer, model: Model, features: Dense, target: Dense, weights: Optional[Dense] = None) raises -> Float64:
        return scorer.score(
            rebind[Scorer.Estimator](model), rebind[Scorer.Input](features),
            rebind[Scorer.Target](target), rebind[Scorer.Metadata](weights),
        )

    var y = Dense(Shape(3, 2))
    for i in range(y.size):
        y.unsafe_set(i, Scalar[dtype](i * (1 + i % 2)) / 3.0)
    var model = Model()
    with assert_raises(contains="NotFittedError"):
        _ = score(scorer, model, X, y, None)
    _ = model.fit(X, y)

    var weights = Dense(Shape(3))
    weights[Item(0)] = 0.25
    weights[Item(1)] = -0.125
    weights[Item(2)] = 1.0
    var reversed_y = y.view_with_layout(Shape(3, 2), Strides(-2, 1), 4)
    print(loss, String(dtype), "multioutput", score(scorer, model, X, reversed_y, None))
    var reversed_weights = weights.view_with_layout(Shape(3), Strides(-1), 2)
    print(loss, String(dtype), "weighted", score(scorer, model, X, reversed_y, reversed_weights^))

    var vector = Dense(Shape(3))
    for i in range(3):
        vector[Item(i)] = Scalar[dtype](i)
    _ = model.fit(X, vector)
    var column = vector.view_with_layout(Shape(3, 1), Strides(1, 1), 0)
    print(loss, String(dtype), "singleton_column", score(scorer, model, X, column, None))

    var signed_weights = array[dtype]([-1.0, 3.0, 0.0], [3])
    print(loss, String(dtype), "signed_weights", score(scorer, model, X, vector, signed_weights^))

    with assert_raises(contains="matching nonempty"):
        _ = score(scorer, model, X, y, None)
    with assert_raises(contains="matching nonempty"):
        _ = score(scorer, model, X, zeros[dtype](Shape(0)), None)
    with assert_raises(contains="vector or matrix"):
        _ = score(scorer, model, X, zeros[dtype](Shape(3, 1, 1)), None)
    with assert_raises(contains="one sample weight per row"):
        _ = score(scorer, model, X, vector, zeros[dtype](Shape(3, 1)))
    with assert_raises(contains="weights sum to zero"):
        _ = score(scorer, model, X, vector, zeros[dtype](Shape(3)))
    with assert_raises(contains="sample weights must be finite"):
        var invalid = zeros[dtype](Shape(3))
        invalid[Item(1)] = inf[dtype]()
        _ = score(scorer, model, X, vector, invalid^)
    with assert_raises(contains="targets and predictions must be finite"):
        vector[Item(0)] = nan[dtype]()
        _ = score(scorer, model, X, vector, None)


def predict[dtype: DType](estimator: DummyRegressor[dtype], X: NDArray[dtype]) raises -> NDArray[dtype]:
    return estimator.predict(X)


def main() raises:
    comptime for i in range(2):
        comptime dtype = DType.float32 if i == 0 else DType.float64
        comptime Dense = NDArray[dtype]
        comptime Model = DummyRegressor[dtype]
        check[dtype](NegMeanSquaredError[Model, Dense, dtype](), "neg_mean_squared_error")
        check[dtype](NegMeanAbsoluteError[Model, Dense, dtype, predict=predict[dtype]](), "neg_mean_absolute_error")
        check[dtype](NegRootMeanSquaredError[Model, Dense, dtype](), "neg_root_mean_squared_error")
