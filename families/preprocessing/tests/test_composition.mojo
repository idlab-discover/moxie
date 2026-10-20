"""Preprocessing estimators inside Moxie compositions.

Pipelines, nested pipelines, ColumnTransformer and clone must give exactly
what the same steps give when called by hand, as in sklearn. cross_val_score
over a Pipeline is not covered: with Mojo 1.0 the call crashes the compiler
during parameter inference (a plain Ridge estimator compiles).
"""
from numojo import NDArray, Shape, Item, zeros
from std.testing import assert_equal, assert_true, assert_false
from moxie.base import clone
from moxie.compose import ColumnTransformer, ColumnEntry
from moxie.linear_model import Ridge
from moxie.pipeline import Pipeline
from moxie.preprocessing import (
    StandardScaler, PolynomialFeatures, FunctionTransformer,
    QuantileTransformer, PowerTransformer, SplineTransformer, KBinsDiscretizer,
)

comptime Dense = NDArray[DType.float64]


def data(rows: Int, cols: Int, shift: Float64 = 0) raises -> Dense:
    var X = zeros[DType.float64](Shape(rows, cols))
    for i in range(rows):
        for j in range(cols):
            var k = Float64(i * cols + j)
            X.unsafe_set(i * cols + j, shift + Float64((i * 7 + j * 3) % 11) + 0.1 * k + Float64(j * j))
    return X^


def target(X: Dense) raises -> Dense:
    var y = zeros[DType.float64](Shape(X.shape[0]))
    for i in range(X.shape[0]):
        y.unsafe_set(i, 2.0 * X.item(i, 0) - X.item(i, 1) + 0.5 * Float64(i % 3))
    return y^


def assert_same(a: Dense, b: Dense, what: String) raises:
    assert_equal(a.size, b.size, what + ": size")
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k), what + ": element " + String(k))


def test_pipeline() raises:
    var X = data(30, 3)
    var y = target(X)
    var X_new = data(7, 3, 4.0)
    var pipe = Pipeline(StandardScaler(), Ridge(alpha=0.5), transformer_name="scale", estimator_name="ridge")
    assert_false(pipe.__sklearn_is_fitted__())
    var predicted = pipe.fit(X, y).predict(X_new)
    var scaler = StandardScaler()
    var model = Ridge(alpha=0.5)
    _ = model.fit(scaler.fit_transform(X), y)
    assert_same(predicted, model.predict(scaler.transform(X_new)), "Pipeline(StandardScaler, Ridge)")
    assert_same(pipe.transformer.mean_.value(), scaler.mean_.value(), "fitted scaler inside the pipeline")
    # Clone is unfitted with the same configuration; refitting the clone reproduces the result.
    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_false(fresh.transformer.__sklearn_is_fitted__())
    assert_same(fresh.fit(X, y).predict(X_new), predicted, "refit clone")
    # set_params on the nested scaler record, then refit: with_mean=False changes the scaling.
    var params = pipe.get_params()
    params.transformer.with_mean = False
    _ = pipe.set_params(params).fit(X, y)
    assert_false(pipe.transformer.with_mean)
    var plain = StandardScaler(with_mean=False)
    var refit = Ridge(alpha=0.5)
    _ = refit.fit(plain.fit_transform(X), y)
    assert_same(pipe.predict(X_new), refit.predict(plain.transform(X_new)), "set_params then refit")


def test_nested_pipelines() raises:
    var X = data(25, 2)
    var y = target(X)
    var tail = Pipeline(StandardScaler(), Ridge(alpha=1.0))
    var pipe = Pipeline(PolynomialFeatures(2), tail^)
    var predicted = pipe.fit(X, y).predict(X)
    var poly = PolynomialFeatures(2)
    var scaler = StandardScaler()
    var model = Ridge(alpha=1.0)
    _ = model.fit(scaler.fit_transform(poly.fit_transform(X)), y)
    assert_same(predicted, model.predict(scaler.transform(poly.transform(X))), "Pipeline(PolynomialFeatures, Pipeline(StandardScaler, Ridge))")
    # Every Float64 NDArray transformer slots into the same position.
    var spline = Pipeline(SplineTransformer(n_knots=4, degree=2), Ridge(alpha=1e-3))
    var direct = SplineTransformer(n_knots=4, degree=2)
    var ridge = Ridge(alpha=1e-3)
    _ = ridge.fit(direct.fit_transform(X), y)
    assert_same(spline.fit(X, y).predict(X), ridge.predict(direct.transform(X)), "Pipeline(SplineTransformer, Ridge)")
    var power = Pipeline(PowerTransformer(), Ridge(alpha=1.0))
    var pt = PowerTransformer()
    var ridge2 = Ridge(alpha=1.0)
    _ = ridge2.fit(pt.fit_transform(X), y)
    assert_same(power.fit(X, y).predict(X), ridge2.predict(pt.transform(X)), "Pipeline(PowerTransformer, Ridge)")
    var quantile = Pipeline(QuantileTransformer(n_quantiles=10), Ridge(alpha=1.0))
    var qt = QuantileTransformer(n_quantiles=10)
    var ridge3 = Ridge(alpha=1.0)
    _ = ridge3.fit(qt.fit_transform(X), y)
    assert_same(quantile.fit(X, y).predict(X), ridge3.predict(qt.transform(X)), "Pipeline(QuantileTransformer, Ridge)")
    var bins = Pipeline(KBinsDiscretizer(3, encode="onehot-dense", strategy="uniform"), Ridge(alpha=1.0))
    var kb = KBinsDiscretizer(3, encode="onehot-dense", strategy="uniform")
    var ridge4 = Ridge(alpha=1.0)
    _ = ridge4.fit(kb.fit_transform(X), y)
    assert_same(bins.fit(X, y).predict(X), ridge4.predict(kb.transform(X)), "Pipeline(KBinsDiscretizer, Ridge)")


def test_column_transformer() raises:
    # ColumnTransformer copies its components, so they must be Copyable. Estimators
    # with a `copy` constructor field (the scalers, Normalizer, Binarizer and the
    # quantile/power transformers) cannot conform yet: Copyable reserves copy().
    var X = data(12, 4)
    var X_new = data(5, 4, 3.0)
    var ct = ColumnTransformer(
        ColumnEntry("poly", PolynomialFeatures(2, include_bias=False), [0, 1]),
        ColumnEntry("bins", KBinsDiscretizer(3, encode="ordinal", strategy="uniform"), [3]),
        remainder="passthrough",
    )
    var Z = ct.fit_transform(X)
    assert_equal(Z.shape[0], 12)
    assert_equal(Z.shape[1], 7)
    var poly = PolynomialFeatures(2, include_bias=False)
    var bins = KBinsDiscretizer(3, encode="ordinal", strategy="uniform")
    var left = poly.fit_transform(_columns(X, [0, 1]))
    var right = bins.fit_transform(_columns(X, [3]))
    for i in range(12):
        for j in range(5):
            assert_equal(Z.item(i, j), left.item(i, j))
        assert_equal(Z.item(i, 5), right.item(i, 0))
        assert_equal(Z.item(i, 6), X.item(i, 2))
    var names = ct.get_feature_names_out()
    assert_equal(names[0], "poly__x0")
    assert_equal(names[4], "poly__x1^2")
    assert_equal(names[5], "bins__x3")
    assert_equal(names[6], "remainder__x2")
    var T = ct.transform(X_new)
    var expected_left = poly.transform(_columns(X_new, [0, 1]))
    for i in range(5):
        assert_equal(T.item(i, 3), expected_left.item(i, 3))
        assert_equal(T.item(i, 5), bins.transform(_columns(X_new, [3])).item(i, 0))
    var spline = ColumnTransformer(
        ColumnEntry("spline", SplineTransformer(n_knots=4, degree=2), [1, 2]),
        ColumnEntry("same", FunctionTransformer(), [0]),
    )
    var S = spline.fit(X).transform(X_new)
    var direct = SplineTransformer(n_knots=4, degree=2)
    _ = direct.fit(_columns(X, [1, 2]))
    var D = direct.transform(_columns(X_new, [1, 2]))
    assert_equal(S.shape[1], D.shape[1] + 1)
    for i in range(5):
        for j in range(D.shape[1]):
            assert_equal(S.item(i, j), D.item(i, j))
        assert_equal(S.item(i, D.shape[1]), X_new.item(i, 0))
    assert_false(clone(spline).__sklearn_is_fitted__())


def _columns(X: Dense, columns: List[Int]) raises -> Dense:
    var out = zeros[DType.float64](Shape(X.shape[0], len(columns)))
    for i in range(X.shape[0]):
        for j in range(len(columns)):
            out.unsafe_set(i * len(columns) + j, X.item(i, columns[j]))
    return out^


def main() raises:
    test_pipeline()
    test_nested_pipelines()
    test_column_transformer()
    print("PASS preprocessing composition: pipelines, nested pipelines, ColumnTransformer, clone")
