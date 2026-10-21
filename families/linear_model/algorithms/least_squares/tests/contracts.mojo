"""Native LinearRegression contracts: lifecycle, shapes, weights, rank, NNLS, errors."""
import numojo as nm
from numojo import NDArray, Shape
from std.math import sqrt
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from moxie.base import clone
from moxie.linear_model import LinearRegression, LinearRegressionParams


def matrix[dtype: DType](rows: Int, cols: Int, values: List[Float64]) raises -> NDArray[dtype]:
    var a = nm.zeros[dtype](Shape(rows, cols))
    for i in range(rows * cols):
        a.unsafe_set(i, values[i].cast[dtype]())
    return a^


def vector[dtype: DType](values: List[Float64]) raises -> NDArray[dtype]:
    var a = nm.zeros[dtype](Shape(len(values)))
    for i in range(len(values)):
        a.unsafe_set(i, values[i].cast[dtype]())
    return a^


def tiny[dtype: DType]() raises:
    # Handoff fixture: X=[[1,1]], y=[2], no intercept -> min-norm coef [1,1].
    var model = LinearRegression[dtype](fit_intercept=False)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(matrix[dtype](1, 2, [1.0, 1.0]))
    model.fit(matrix[dtype](1, 2, [1.0, 1.0]), vector[dtype]([2.0]))
    assert_almost_equal(Float64(model.coef_.unsafe_get(0)), 1.0, atol=1e-6)
    assert_almost_equal(Float64(model.coef_.unsafe_get(1)), 1.0, atol=1e-6)
    assert_equal(model.rank_.value(), 1)
    assert_almost_equal(Float64(model.singular_.value().unsafe_get(0)), sqrt(2.0), atol=1e-6)
    assert_equal(Float64(model.intercept_.unsafe_get(0)), 0.0)
    # positive=True with y=-2 -> coef 0; rank_/singular_ unset.
    var positive = LinearRegression[dtype](fit_intercept=False, positive=True)
    positive.fit(matrix[dtype](1, 1, [1.0]), vector[dtype]([-2.0]))
    assert_equal(Float64(positive.coef_.unsafe_get(0)), 0.0)
    assert_false(Bool(positive.rank_))
    assert_false(Bool(positive.singular_))


def intercept_and_shapes[dtype: DType]() raises:
    # y = 2 x0 - x1 + 3 exactly.
    var X = matrix[dtype](4, 2, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    var y = vector[dtype]([3.0, 5.0, 2.0, 4.0])
    var model = LinearRegression[dtype]()
    model.fit(X, y)
    assert_equal(model.coef_.ndim, 1)
    assert_equal(model.intercept_.shape[0], 1)
    assert_almost_equal(Float64(model.coef_.unsafe_get(0)), 2.0, atol=1e-4)
    assert_almost_equal(Float64(model.coef_.unsafe_get(1)), -1.0, atol=1e-4)
    assert_almost_equal(Float64(model.intercept_.unsafe_get(0)), 3.0, atol=1e-4)
    assert_equal(model.n_features_in_, 2)
    assert_equal(model.rank_.value(), 2)
    var prediction = model.predict(X)
    assert_equal(prediction.ndim, 1)
    assert_almost_equal(Float64(prediction.unsafe_get(3)), 4.0, atol=1e-4)
    assert_almost_equal(model.score(X, y), 1.0, atol=1e-6)
    # Matrix target with one column keeps its second axis.
    var Y = matrix[dtype](4, 1, [3.0, 5.0, 2.0, 4.0])
    model.fit(X, Y)
    assert_equal(model.coef_.ndim, 2)
    assert_equal(model.coef_.shape[0], 1)
    assert_equal(model.predict(X).ndim, 2)
    # Two targets: second is y2 = x0 + 1.
    var Y2 = matrix[dtype](4, 2, [3.0, 1.0, 5.0, 2.0, 2.0, 1.0, 4.0, 3.0])
    model.fit(X, Y2)
    assert_equal(model.coef_.shape[0], 2)
    assert_almost_equal(Float64(model.coef_.unsafe_get(2)), 1.0, atol=1e-4)
    assert_almost_equal(Float64(model.intercept_.unsafe_get(1)), 1.0, atol=1e-4)
    assert_equal(model.predict(X).shape[1], 2)
    # Feature-count and finiteness checks at inference.
    with assert_raises(contains="features"):
        _ = model.predict(matrix[dtype](1, 3, [1.0, 2.0, 3.0]))
    with assert_raises(contains="NaN"):
        _ = model.predict(matrix[dtype](1, 2, [1.0, Float64("nan")]))


def weights[dtype: DType]() raises:
    var scalar = LinearRegression[dtype]()
    var xs = matrix[dtype](3, 1, [0.0, 1.0, 2.0])
    var ys = vector[dtype]([1.0, 4.0, 4.0])
    scalar.fit(xs, ys, sample_weight=2.0)
    var expanded = LinearRegression[dtype]()
    expanded.fit(xs, ys, sample_weight=vector[dtype]([2.0, 2.0, 2.0]))
    assert_almost_equal(Float64(scalar.coef_.unsafe_get(0)), Float64(expanded.coef_.unsafe_get(0)), atol=1e-6)
    with assert_raises():
        scalar.fit(xs, ys, sample_weight=-1.0)
    # A zero weight equals dropping the sample; integer weights equal repetition.
    var X = matrix[dtype](4, 1, [0.0, 1.0, 2.0, 10.0])
    var y = vector[dtype]([1.0, 3.0, 5.0, 0.0])
    var weighted = LinearRegression[dtype]()
    weighted.fit(X, y, sample_weight=vector[dtype]([1.0, 1.0, 1.0, 0.0]))
    assert_almost_equal(Float64(weighted.coef_.unsafe_get(0)), 2.0, atol=1e-4)
    assert_almost_equal(Float64(weighted.intercept_.unsafe_get(0)), 1.0, atol=1e-4)
    var repeated = LinearRegression[dtype]()
    repeated.fit(matrix[dtype](4, 1, [0.0, 1.0, 1.0, 2.0]), vector[dtype]([1.0, 2.0, 2.0, 7.0]))
    var doubled = LinearRegression[dtype]()
    doubled.fit(matrix[dtype](3, 1, [0.0, 1.0, 2.0]), vector[dtype]([1.0, 2.0, 7.0]), sample_weight=vector[dtype]([1.0, 2.0, 1.0]))
    assert_almost_equal(Float64(repeated.coef_.unsafe_get(0)), Float64(doubled.coef_.unsafe_get(0)), atol=1e-4)
    assert_almost_equal(Float64(repeated.intercept_.unsafe_get(0)), Float64(doubled.intercept_.unsafe_get(0)), atol=1e-4)
    with assert_raises():
        weighted.fit(X, y, sample_weight=vector[dtype]([1.0, -1.0, 1.0, 1.0]))
    with assert_raises():
        weighted.fit(X, y, sample_weight=vector[dtype]([1.0, 1.0]))
    # Failed fit keeps the previous learned state.
    assert_almost_equal(Float64(weighted.coef_.unsafe_get(0)), 2.0, atol=1e-4)
    # Weighted score.
    assert_almost_equal(weighted.score(X, y, vector[dtype]([1.0, 1.0, 1.0, 0.0])), 1.0, atol=1e-5)


def rank_deficient[dtype: DType]() raises:
    # Duplicate columns: minimum-norm solution splits the coefficient.
    var X = matrix[dtype](3, 2, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    var model = LinearRegression[dtype](fit_intercept=False)
    model.fit(X, vector[dtype]([2.0, 4.0, 6.0]))
    assert_equal(model.rank_.value(), 1)
    assert_almost_equal(Float64(model.coef_.unsafe_get(0)), 1.0, atol=1e-4)
    assert_almost_equal(Float64(model.coef_.unsafe_get(1)), 1.0, atol=1e-4)
    assert_equal(model.singular_.value().shape[0], 2)
    # tol=0 keeps every nonzero singular value; an exactly zero one is still dropped.
    model.set_params(tol=Optional(0.0))
    model.fit(X, vector[dtype]([2.0, 4.0, 6.0]))
    assert_true(model.rank_.value() >= 1)


def positive_multi[dtype: DType]() raises:
    # y1 = x0 - x1 (x1 clipped), y2 = x0 + x1.
    var X = matrix[dtype](4, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0])
    var Y = matrix[dtype](4, 2, [1.0, 1.0, -1.0, 1.0, 0.0, 2.0, 1.0, 3.0])
    var model = LinearRegression[dtype](fit_intercept=False, positive=True)
    model.fit(X, Y)
    assert_equal(model.coef_.shape[0], 2)
    for i in range(4):
        assert_true(Float64(model.coef_.unsafe_get(i)) >= 0.0)
    assert_almost_equal(Float64(model.coef_.unsafe_get(2)), 1.0, atol=1e-4)
    assert_almost_equal(Float64(model.coef_.unsafe_get(3)), 1.0, atol=1e-4)
    assert_equal(Float64(model.coef_.unsafe_get(1)), 0.0)


def lifecycle() raises:
    var model = LinearRegression[DType.float64](fit_intercept=False, tol=1e-3, n_jobs=Optional(2), positive=True)
    var params = model.get_params()
    assert_false(params.fit_intercept)
    assert_equal(params.tol, 1e-3)
    assert_equal(params.n_jobs.value(), 2)
    model.fit(matrix[DType.float64](2, 1, [1.0, 2.0]), vector[DType.float64]([2.0, 4.0]))
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    assert_true(copy.positive)
    assert_equal(copy.get_params().tol, 1e-3)
    copy.set_params(LinearRegressionParams(True, True, 1e-6, None, False))
    assert_true(copy.fit_intercept)
    assert_true(model.positive)  # clone parameters are independent
    # Repeat fit replaces learned state, including feature counts.
    model.fit(matrix[DType.float64](2, 2, [1.0, 0.0, 0.0, 1.0]), vector[DType.float64]([1.0, 2.0]))
    assert_equal(model.n_features_in_, 2)
    # Invalid parameters and inputs.
    var bad = LinearRegression[DType.float64](tol=-1.0)
    with assert_raises(contains="tol"):
        bad.fit(matrix[DType.float64](2, 1, [1.0, 2.0]), vector[DType.float64]([2.0, 4.0]))
    # sklearn ignores n_jobs=0 except for positive=True with a matrix target,
    # where joblib.Parallel(n_jobs=0) raises.
    var zero_jobs = LinearRegression[DType.float64](n_jobs=Optional(0))
    zero_jobs.fit(matrix[DType.float64](2, 1, [1.0, 2.0]), vector[DType.float64]([2.0, 4.0]))
    assert_almost_equal(Float64(zero_jobs.coef_.unsafe_get(0)), 2.0, atol=1e-12)
    zero_jobs.positive = True
    zero_jobs.fit(matrix[DType.float64](2, 1, [1.0, 2.0]), vector[DType.float64]([2.0, 4.0]))
    assert_almost_equal(Float64(zero_jobs.coef_.unsafe_get(0)), 2.0, atol=1e-12)
    with assert_raises(contains="n_jobs"):
        zero_jobs.fit(matrix[DType.float64](2, 1, [1.0, 2.0]), matrix[DType.float64](2, 1, [2.0, 4.0]))
    assert_equal(zero_jobs.coef_.ndim, 1)  # the failed fit kept the previous state
    var fresh = LinearRegression[DType.float64]()
    with assert_raises():
        fresh.fit(matrix[DType.float64](2, 1, [1.0, Float64("inf")]), vector[DType.float64]([2.0, 4.0]))
    with assert_raises():
        fresh.fit(matrix[DType.float64](2, 1, [1.0, 2.0]), vector[DType.float64]([2.0, 4.0, 5.0]))
    with assert_raises():
        fresh.fit(nm.zeros[DType.float64](Shape(0, 2)), nm.zeros[DType.float64](Shape(0)))
    assert_false(fresh.__sklearn_is_fitted__())
    # Input arrays are never mutated (copy_X semantics are always preserving).
    var X = matrix[DType.float64](3, 1, [1.0, 2.0, 3.0])
    var y = vector[DType.float64]([1.0, 2.0, 4.0])
    var keep = LinearRegression[DType.float64](copy_X=False)
    keep.fit(X, y, sample_weight=vector[DType.float64]([1.0, 4.0, 9.0]))
    assert_equal(X.unsafe_get(0), 1.0)
    assert_equal(y.unsafe_get(2), 4.0)
    # A strided (reversed) view fits like its materialized copy.
    var view = X.view_with_layout(Shape(3, 1), nm.NDArrayStrides(-1, 1), 2)
    var viewed = LinearRegression[DType.float64]()
    viewed.fit(view, vector[DType.float64]([4.0, 2.0, 1.0]))
    var direct = LinearRegression[DType.float64]()
    direct.fit(matrix[DType.float64](3, 1, [3.0, 2.0, 1.0]), vector[DType.float64]([4.0, 2.0, 1.0]))
    assert_almost_equal(viewed.coef_.unsafe_get(0), direct.coef_.unsafe_get(0), atol=1e-12)
    # R2 needs two samples.
    assert_true(direct.score(matrix[DType.float64](1, 1, [1.0]), vector[DType.float64]([1.0])) != direct.score(matrix[DType.float64](1, 1, [1.0]), vector[DType.float64]([1.0])))


def main() raises:
    tiny[DType.float32]()
    tiny[DType.float64]()
    intercept_and_shapes[DType.float32]()
    intercept_and_shapes[DType.float64]()
    weights[DType.float32]()
    weights[DType.float64]()
    rank_deficient[DType.float32]()
    rank_deficient[DType.float64]()
    positive_multi[DType.float32]()
    positive_multi[DType.float64]()
    lifecycle()
    print("LinearRegression contracts passed: lifecycle, shapes, weights, rank, NNLS, errors (Float32/Float64)")
