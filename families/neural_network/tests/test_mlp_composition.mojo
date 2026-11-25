"""Dense MLP composition through the public pipeline and CV adapters."""

from numojo import NDArray, Shape, arange, ones, zeros
from std.math import isfinite
from std.testing import assert_equal, assert_false, assert_true
from moxie.base import clone
from moxie.model_selection import DenseSupervisedAdapter, cross_val_score
from moxie.neural_network import MLPClassifier, MLPRegressor
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler


def pipelines() raises:
    var X = arange[DType.float32](24).reshape(Shape(12, 2))
    var y = zeros[DType.int64](Shape(12))
    for row in range(y.size):
        y.unsafe_set(row, Int64(row % 3))
    var model = MLPClassifier[DType.float32]([4], solver="lbfgs", random_state=1, max_iter=4)
    var pipe = make_pipeline(StandardScaler[DType.float32](), clone(model))
    assert_equal(pipe.transformer_name, "standardscaler")
    assert_equal(pipe.estimator_name, "mlpclassifier")
    _ = pipe.fit(X, y)
    var X_scaled = pipe.transformer.transform(X)
    _ = model.fit(X_scaled, y)
    var actual = pipe.predict_proba(X)
    var expected = model.predict_proba(X_scaled)
    assert_equal(actual.shape, expected.shape)
    for i in range(actual.size):
        assert_equal(actual.unsafe_get(i), expected.unsafe_get(i))
    assert_equal(pipe.score(X, y), model.score(X_scaled, y))
    assert_equal(pipe.n_features_in_(), 2)
    var fresh = clone(pipe)
    assert_false(fresh.estimator.__sklearn_is_fitted__())
    assert_false(fresh.transformer.__sklearn_is_fitted__())


def cross_validation() raises:
    comptime Array = NDArray[DType.float64]
    comptime Estimator = MLPRegressor[DType.float64]
    comptime Adapter = DenseSupervisedAdapter[Estimator, Array, DType.float64]
    var X = arange[DType.float64](24).reshape(Shape(12, 2))
    var y = arange[DType.float64](12)
    var model = Estimator(
        hidden_layer_sizes=[3], random_state=0, max_iter=2,
        early_stopping=True, validation_fraction=0.25,
    )
    var weights = Optional(ones[DType.float64](Shape(12)))
    var scores = cross_val_score[Adapter](
        model, X, y, Adapter(), fit_metadata=weights, score_metadata=weights,
    )
    assert_equal(scores.size, 5)
    for i in range(scores.size):
        assert_true(isfinite(scores.unsafe_get(i)))
    assert_false(model.__sklearn_is_fitted__())


def main() raises:
    pipelines()
    cross_validation()
    print("PASS StandardScaler MLP pipeline, clone and weighted MLP cross-validation")
