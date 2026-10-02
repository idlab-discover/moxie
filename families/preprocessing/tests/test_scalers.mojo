"""Lifecycle contracts for StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler."""
from numojo import NDArray, Shape, zeros
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.preprocessing import (
    InvertiblePreprocessor, OnlinePreprocessor,
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
    scale, minmax_scale, maxabs_scale, robust_scale,
)


def generic_inverse[P: InvertiblePreprocessor](model: P, Z: P.Transformed) raises -> P.Input:
    # Compile-time trait conformance plus generic dispatch.
    return model.inverse_transform(Z)


def online[P: OnlinePreprocessor]():
    # Conformance only: generic code cannot drop the trait's Optional[Target]
    # default, whose element type is not required to be destructible.
    pass


def matrix(rows: Int, cols: Int, start: Float64 = 0) raises -> NDArray[DType.float64]:
    var x = zeros[DType.float64](Shape(rows, cols))
    for k in range(rows * cols):
        x.unsafe_set(k, start + Float64(k * k) - 2.5)
    return x^


def assert_close(a: Float64, b: Float64) raises:
    assert_true(abs(a - b) <= 1e-12 * max(1.0, abs(b)), String(a) + " != " + String(b))


def test_standard_scaler() raises:
    var model = StandardScaler(with_mean=False)
    assert_false(model.get_params().with_mean)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(matrix(2, 2))
    var X = matrix(4, 2)
    model.fit(X)
    assert_false(Bool(model.mean_.value().size == 0))
    var back = generic_inverse(model, model.transform(X))
    for k in range(8):
        assert_close(back.unsafe_get(k), X.unsafe_get(k))
    # A normal fit replaces learned state; partial_fit accumulates.
    model.set_params(with_mean=True)
    model.fit(matrix(3, 2))
    assert_equal(model.n_samples_seen_.value(), 3)
    model.partial_fit(X)
    assert_equal(model.n_samples_seen_.value(), 7)
    assert_true(model.n_samples_seen_.is_scalar)
    assert_false(model.n_samples_seen_.is_integer)
    # A NaN makes counts differ by feature: sklearn's vector n_samples_seen_.
    var holes = matrix(3, 2)
    holes.unsafe_set(0, Float64("nan"))
    model.fit(holes)
    assert_false(model.n_samples_seen_.is_scalar)
    assert_equal(model.n_samples_seen_.values.unsafe_get(0), 2)
    with assert_raises(contains="per-feature"):
        _ = model.n_samples_seen_.value()
    # Neither statistic: sklearn's integer count and no moments.
    var counter = StandardScaler(with_mean=False, with_std=False)
    counter.fit(X)
    assert_true(counter.n_samples_seen_.is_integer)
    assert_false(Bool(counter.mean_))
    with assert_raises(contains="partial_fit"):
        counter.set_params(with_std=True)
        counter.partial_fit(X)
    # Scalar weights scale counts without expanding a vector.
    var weighted = StandardScaler()
    weighted.fit(X, sample_weight=2.5)
    assert_equal(weighted.n_samples_seen_.value(), 10)
    with assert_raises(contains="features"):
        _ = weighted.transform(matrix(2, 3))
    var copy = clone(weighted)
    assert_false(copy.__sklearn_is_fitted__())
    assert_true(copy.with_mean)
    var inplace = X.copy()
    weighted.transform_inplace(inplace)
    var expected = weighted.transform(X)
    for k in range(8):
        assert_equal(inplace.unsafe_get(k), expected.unsafe_get(k))
    weighted.inverse_transform_inplace(inplace)
    for k in range(8):
        assert_close(inplace.unsafe_get(k), X.unsafe_get(k))
    # The borrowed input is unchanged by copying transforms.
    assert_equal(X.item(0, 0), -2.5)
    var names: List[String] = ["a", "b"]
    weighted.fit_named(X, names)
    assert_equal(weighted.get_feature_names_out()[1], "b")
    weighted.fit(X)
    assert_equal(weighted.get_feature_names_out()[0], "x0")


def test_minmax_and_maxabs() raises:
    var X = matrix(4, 3)
    var mm = MinMaxScaler((-1.0, 1.0), clip=True)
    var out = mm.fit_transform(X)
    assert_equal(out.item(0, 0), -1)
    assert_equal(out.item(3, 0), 1)
    assert_equal(mm.n_samples_seen_, 4)
    mm.partial_fit(matrix(2, 3, 100))
    assert_equal(mm.n_samples_seen_, 6)
    # clip bounds transforms of out-of-range data.
    var far = mm.transform(matrix(1, 3, 1e6))
    assert_equal(far.item(0, 0), 1)
    mm.set_params(feature_range=(2.0, 1.0))
    with assert_raises(contains="feature range"):
        mm.fit(X)
    var cloned = clone(mm)
    assert_equal(cloned.feature_range[0], 2)
    assert_false(cloned.__sklearn_is_fitted__())
    var ma = MaxAbsScaler[DType.float32]()
    var X32 = zeros[DType.float32](Shape(2, 2))
    X32.unsafe_set(0, -4)
    X32.unsafe_set(3, 2)
    ma.fit(X32)
    assert_equal(ma.max_abs_.unsafe_get(0), 4)
    assert_equal(ma.scale_.unsafe_get(1), 2)
    var t = ma.transform(X32)
    assert_equal(t.item(0, 0), -1)
    var row = zeros[DType.float32](Shape(1, 2))
    ma.fit(row)
    assert_equal(ma.n_samples_seen_, 1)
    # All-zero features are scaled by 1, never divided by zero.
    assert_equal(ma.scale_.unsafe_get(0), 1)


def test_robust_scaler() raises:
    var X = matrix(5, 2)
    var model = RobustScaler(with_scaling=False)
    model.fit(X)
    assert_false(Bool(model.scale_))
    assert_equal(model.center_.value().unsafe_get(0), X.item(2, 0))
    var back = generic_inverse(model, model.transform(X))
    assert_close(back.item(4, 1), X.item(4, 1))
    model.set_params(quantile_range=(80.0, 20.0))
    with assert_raises(contains="quantile range"):
        model.fit(X)
    # The failed fit kept earlier learned state.
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(clone(model).quantile_range[0], 80)


def test_functions() raises:
    var X = matrix(3, 2)
    var vector = zeros[DType.float64](Shape(3))
    with assert_raises(contains="axis 1"):
        _ = scale(vector, axis=1)
    with assert_raises(contains="'axis' parameter"):
        _ = maxabs_scale(X, axis=2)
    assert_equal(minmax_scale(vector, axis=1).ndim, 1)
    var rows = robust_scale(X, axis=1)
    assert_equal(rows.shape[0], 3)
    assert_equal(rows.shape[1], 2)
    var columns = scale(X)
    var total = Float64(0)
    for i in range(3):
        total += columns.item(i, 0)
    assert_true(abs(total) < 1e-12)


def main() raises:
    online[StandardScaler[DType.float64]]()
    online[MinMaxScaler[DType.float32]]()
    online[MaxAbsScaler[DType.float64]]()
    test_standard_scaler()
    test_minmax_and_maxabs()
    test_robust_scaler()
    test_functions()
    print("PASS scaler lifecycle contracts")
