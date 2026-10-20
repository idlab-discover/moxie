"""Exercise fluent mutations on real public estimators and shared traits."""
from std.testing import assert_equal, assert_true
from numojo import NDArray, Shape, zeros
from moxie.base import Predictor, SupervisedScorer, WeightedSupervisedEstimator
from moxie.linear_model import LinearRegression, Ridge, RidgeClassifier, ElasticNet, Lasso


def check_model[dtype: DType](
    mut model: LinearRegression[dtype], X: NDArray[dtype], y: NDArray[dtype], weights: NDArray[dtype]
) raises:
    var params = model.get_params()
    _ = model.set_params(params).fit(X, y).predict(X)
    var expected = model.score(X, y)
    _ = model.fit(X, y, sample_weight=weights).predict(X)
    var actual = model.score(X, y)
    assert_true(model.__sklearn_is_fitted__())
    assert_true(abs(actual - expected) < 1e-5)
    assert_true(actual > 0.9)
    var cloned = model.__sklearn_clone__()
    assert_true(not cloned.__sklearn_is_fitted__())


def check_model[dtype: DType](
    mut model: Ridge[dtype], X: NDArray[dtype], y: NDArray[dtype], weights: NDArray[dtype]
) raises:
    var params = model.get_params()
    _ = model.set_params(params).fit(X, y).predict(X)
    var expected = model.score(X, y)
    _ = model.fit(X, y, sample_weight=weights).predict(X)
    var actual = model.score(X, y)
    assert_true(model.__sklearn_is_fitted__())
    assert_true(abs(actual - expected) < 1e-5)
    assert_true(actual > 0.9)
    var cloned = model.__sklearn_clone__()
    assert_true(not cloned.__sklearn_is_fitted__())


def check_model[dtype: DType](
    mut model: RidgeClassifier[dtype], X: NDArray[dtype], y: NDArray[DType.int64], weights: NDArray[dtype]
) raises:
    var params = model.get_params()
    _ = model.set_params(params).fit(X, y).predict(X)
    var expected = model.score(X, y)
    _ = model.fit(X, y, sample_weight=weights).predict(X)
    var actual = model.score(X, y)
    assert_true(model.__sklearn_is_fitted__())
    assert_true(abs(actual - expected) < 1e-5)
    assert_true(actual > 0.9)
    var cloned = model.__sklearn_clone__()
    assert_true(not cloned.__sklearn_is_fitted__())


def check_model[dtype: DType](
    mut model: ElasticNet[dtype], X: NDArray[dtype], y: NDArray[dtype], weights: NDArray[dtype]
) raises:
    var params = model.get_params()
    _ = model.set_params(params).fit(X, y).predict(X)
    var expected = model.score(X, y)
    _ = model.fit(X, y, sample_weight=weights).predict(X)
    var actual = model.score(X, y)
    assert_true(model.__sklearn_is_fitted__())
    assert_true(abs(actual - expected) < 1e-5)
    assert_true(actual > 0.9)
    var cloned = model.__sklearn_clone__()
    assert_true(not cloned.__sklearn_is_fitted__())


def check_model[dtype: DType](
    mut model: Lasso[dtype], X: NDArray[dtype], y: NDArray[dtype], weights: NDArray[dtype]
) raises:
    var params = model.get_params()
    _ = model.set_params(params).fit(X, y).predict(X)
    var expected = model.score(X, y)
    _ = model.fit(X, y, sample_weight=weights).predict(X)
    var actual = model.score(X, y)
    assert_true(model.__sklearn_is_fitted__())
    assert_true(abs(actual - expected) < 1e-5)
    assert_true(actual > 0.9)
    var cloned = model.__sklearn_clone__()
    assert_true(not cloned.__sklearn_is_fitted__())


def check_dtype[dtype: DType]() raises:
    var X = zeros[dtype](Shape(4, 1))
    var y = zeros[dtype](Shape(4))
    var labels = zeros[DType.int64](Shape(4))
    var weights = zeros[dtype](Shape(4))
    for i in range(4):
        X.unsafe_set(i, Scalar[dtype](i))
        y.unsafe_set(i, Scalar[dtype](2 * i + 1))
        labels.unsafe_set(i, Int64(i >= 2))
        weights.unsafe_set(i, 1)
    var ols = LinearRegression[dtype]()
    check_model(ols, X, y, weights)
    var ridge = Ridge[dtype](alpha=0.1)
    check_model(ridge, X, y, weights)
    var classifier = RidgeClassifier[dtype]()
    check_model(classifier, X, labels, weights)
    var elastic = ElasticNet[dtype](alpha=0.1)
    check_model(elastic, X, y, weights)
    var lasso = Lasso[dtype](alpha=0.1)
    check_model(lasso, X, y, weights)
    assert_equal(ridge.set_params(tol=1e-5).fit(X, y, sample_weight=2.0).predict(X).shape[0], 4)
    assert_equal(elastic.set_params(tol=1e-5).fit(X, y, sample_weight=2.0).predict(X).shape[0], 4)
    assert_equal(lasso.set_params(tol=1e-5).fit(X, y, sample_weight=2.0).predict(X).shape[0], 4)


def main() raises:
    check_dtype[DType.float32]()
    check_dtype[DType.float64]()
    print("PASS: real estimators through fluent shared traits, both dtypes")
