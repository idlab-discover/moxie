"""Lifecycle contracts for QuantileTransformer, PowerTransformer and their functions."""
from numojo import NDArray, Shape, zeros
from std.math import isnan, abs
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie._random import RandomState, seed_global_random_state
from moxie.preprocessing import (
    InvertiblePreprocessor, QuantileTransformer, PowerTransformer, quantile_transform, power_transform,
)


def generic_transform[P: InvertiblePreprocessor](model: P, X: P.Input) raises -> P.Transformed:
    # Compile-time trait conformance plus generic dispatch.
    return model.transform(X)


def data(rows: Int, cols: Int, shift: Float64 = 0) raises -> NDArray[DType.float64]:
    var x = zeros[DType.float64](Shape(rows, cols))
    for i in range(rows):
        for j in range(cols):
            var k = Float64((i * 7 + j * 13) % rows)
            x.unsafe_set(i * cols + j, shift + 0.5 + k * k / Float64(rows) + Float64(j))
    return x^


def assert_same(a: NDArray[DType.float64], b: NDArray[DType.float64]) raises:
    assert_equal(a.size, b.size)
    for k in range(a.size):
        var x = a.unsafe_get(k)
        var y = b.unsafe_get(k)
        assert_true((isnan(x) and isnan(y)) or x == y)


def test_quantile_lifecycle() raises:
    var model = QuantileTransformer(n_quantiles=10, output_distribution="normal", random_state=3)
    var params = model.get_params()
    assert_equal(params.n_quantiles, 10)
    assert_equal(params.output_distribution, "normal")
    assert_equal(params.subsample.value(), 10000)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(data(4, 2))
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()
    var X = data(30, 2)
    _ = model.fit(X)
    assert_equal(model.n_quantiles_, 10)
    assert_equal(model.quantiles_.shape[0], 10)
    assert_equal(model.references_.item(9), 1.0)
    var back = model.inverse_transform(generic_transform(model, X))
    for k in range(X.size):
        assert_true(abs(back.unsafe_get(k) - X.unsafe_get(k)) < 1e-9)
    # copy=False path: in place, identical to the copying transform.
    var inplace = X.copy()
    model.transform_inplace(inplace)
    assert_same(inplace, model.transform(X))
    model.inverse_transform_inplace(inplace)
    assert_same(inplace, model.inverse_transform(model.transform(X)))
    with assert_raises(contains="X has 3 features, but QuantileTransformer is expecting 2"):
        _ = model.transform(data(4, 3))
    # Repeat fit replaces learned state (feature count and quantile count).
    _ = model.set_params(n_quantiles=1000)
    _ = model.fit(data(7, 3))
    assert_equal(model.n_features_in_, 3)
    assert_equal(model.n_quantiles_, 7)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.n_quantiles, 1000)
    assert_equal(fresh.random_state.value(), 3)
    # Inner None clears an optional parameter; outer None leaves it.
    _ = model.set_params(subsample=Optional[Optional[Int]](Optional[Int](None)))
    assert_false(Bool(model.subsample))
    _ = model.set_params(n_quantiles=5)
    assert_false(Bool(model.subsample))
    var names: List[String] = ["a", "b", "c"]
    _ = model.fit_named(data(7, 3), names)
    assert_equal(model.get_feature_names_out()[2], "c")
    _ = model.fit(data(7, 3))
    assert_equal(model.get_feature_names_out()[0], "x0")
    with assert_raises(contains="only 'default'"):
        model.set_output(transform=String("pandas"))


def test_quantile_params_and_errors() raises:
    var X = data(12, 2)
    var bad = QuantileTransformer(n_quantiles=0)
    with assert_raises(contains="'n_quantiles' parameter"):
        _ = bad.fit(X)
    bad = QuantileTransformer(output_distribution="gamma")
    with assert_raises(contains="'output_distribution' parameter"):
        _ = bad.fit(X)
    bad = QuantileTransformer(subsample=10)
    with assert_raises(contains="cannot be greater than the number of samples used"):
        _ = bad.fit(X)
    bad = QuantileTransformer(random_state=-1)
    with assert_raises(contains="'random_state' parameter"):
        _ = bad.fit(X)
    # ignore_implicit_zeros has no effect on dense input (sklearn only warns).
    var a = QuantileTransformer(n_quantiles=5, ignore_implicit_zeros=True)
    var b = QuantileTransformer(n_quantiles=5)
    assert_same(a.fit_transform(X), b.fit_transform(X))
    with assert_raises(contains="'axis' parameter"):
        _ = quantile_transform(X, axis=2)
    # A failed fit leaves the previous state intact.
    var kept = QuantileTransformer(n_quantiles=4)
    _ = kept.fit(X)
    _ = kept.set_params(n_quantiles=0)
    with assert_raises(contains="n_quantiles"):
        _ = kept.fit(data(5, 3))
    assert_equal(kept.n_features_in_, 2)


def test_quantile_random_state() raises:
    var X = data(200, 2)
    # Integer seeds reseed on every fit.
    var seeded = QuantileTransformer(n_quantiles=20, subsample=50, random_state=11)
    _ = seeded.fit(X)
    var first = seeded.quantiles_.copy()
    _ = seeded.fit(X)
    assert_same(first, seeded.quantiles_)
    # A borrowed RandomState draws the same rows as the equal integer seed, then advances.
    var rng = RandomState(11)
    var borrowed = QuantileTransformer(n_quantiles=20, subsample=50)
    _ = borrowed.fit(X, rng)
    assert_same(first, borrowed.quantiles_)
    _ = borrowed.fit(X, rng)
    var differs = False
    for k in range(first.size):
        if first.unsafe_get(k) != borrowed.quantiles_.unsafe_get(k):
            differs = True
    assert_true(differs)
    # None draws from the process default stream.
    seed_global_random_state(11)
    var default = QuantileTransformer(n_quantiles=20, subsample=50)
    _ = default.fit(X)
    assert_same(first, default.quantiles_)


def test_quantile_dtypes_and_edges() raises:
    var X32 = zeros[DType.float32](Shape(6, 1))
    for i in range(6):
        X32.unsafe_set(i, Float32(i * i))
    X32.unsafe_set(2, Float32(Float64("nan")))
    var model = QuantileTransformer[DType.float32](n_quantiles=4)
    var out = model.fit_transform(X32)
    assert_true(isnan(out.item(2, 0)))
    assert_equal(out.item(0, 0), 0)
    assert_equal(out.item(5, 0), 1)
    var single = QuantileTransformer(n_quantiles=3)
    _ = single.fit(data(1, 2))
    assert_equal(single.n_quantiles_, 1)
    var t = quantile_transform(data(5, 4), axis=1, n_quantiles=4)
    assert_equal(t.shape[0], 5)
    assert_equal(t.shape[1], 4)


def test_power_lifecycle() raises:
    var model = PowerTransformer()
    assert_equal(model.get_params().method, "yeo-johnson")
    assert_true(model.get_params().standardize)
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(data(4, 2))
    var X = data(25, 2, -3.0)
    var direct = model.fit_transform(X)
    assert_equal(model.lambdas_.shape[0], 2)
    var fitted = PowerTransformer()
    _ = fitted.fit(X)
    for j in range(2):
        assert_equal(fitted.lambdas_.item(j), model.lambdas_.item(j))
    var t = fitted.transform(X)
    for k in range(t.size):
        assert_true(abs(t.unsafe_get(k) - direct.unsafe_get(k)) < 1e-12)
    var back = fitted.inverse_transform(generic_transform(fitted, X))
    for k in range(X.size):
        assert_true(abs(back.unsafe_get(k) - X.unsafe_get(k)) < 1e-9)
    var inplace = X.copy()
    fitted.transform_inplace(inplace)
    assert_same(inplace, fitted.transform(X))
    fitted.inverse_transform_inplace(inplace)
    assert_same(inplace, fitted.inverse_transform(fitted.transform(X)))
    with assert_raises(contains="X has 3 features, but PowerTransformer is expecting 2"):
        _ = fitted.transform(data(4, 3))
    _ = fitted.set_params(method="box-cox", standardize=False)
    _ = fitted.fit(data(9, 3))
    assert_equal(fitted.n_features_in_, 3)
    with assert_raises(contains="strictly positive"):
        _ = fitted.transform(data(4, 3, -10.0))
    var copy = clone(fitted)
    assert_equal(copy.method, "box-cox")
    assert_false(copy.__sklearn_is_fitted__())
    _ = fitted.set_params(method="log")
    with assert_raises(contains="'method' parameter"):
        _ = fitted.fit(data(4, 3))


def test_power_edges() raises:
    # A constant Yeo-Johnson feature gets lambda 1 and is left as is by fit_transform.
    var X = zeros[DType.float64](Shape(4, 2))
    for i in range(4):
        X.unsafe_set(i * 2, -2.0)
        X.unsafe_set(i * 2 + 1, Float64(i) + 0.5)
    var model = PowerTransformer(standardize=False)
    var out = model.fit_transform(X)
    assert_equal(model.lambdas_.item(0), 1.0)
    assert_equal(out.item(3, 0), -2.0)
    var nan_col = X.copy()
    for i in range(4):
        nan_col.unsafe_set(i * 2, Float64("nan"))
    with assert_raises(contains="not enough values to unpack"):
        _ = model.fit(nan_col)
    var box = PowerTransformer("box-cox")
    with assert_raises(contains="Column must not be all nan."):
        _ = box.fit(nan_col)
    var X32 = zeros[DType.float32](Shape(5, 1))
    for i in range(5):
        X32.unsafe_set(i, Float32(i + 1) * Float32(i + 1))
    var f = power_transform(X32, "box-cox")
    assert_equal(f.shape[0], 5)
    var p32 = PowerTransformer[DType.float32]("box-cox")
    _ = p32.fit(X32)
    var back = p32.inverse_transform(p32.transform(X32))
    for i in range(5):
        assert_true(abs(Float64(back.item(i, 0)) - Float64(X32.item(i, 0))) < 1e-3)


def main() raises:
    test_quantile_lifecycle()
    test_quantile_params_and_errors()
    test_quantile_random_state()
    test_quantile_dtypes_and_edges()
    test_power_lifecycle()
    test_power_edges()
    print("PASS quantile and power transformer lifecycle contracts")
