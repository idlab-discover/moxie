"""Native contracts for EmpiricalCovariance, ShrunkCovariance and their helpers."""
import numojo as nm
from numojo import NDArray, Shape
from std.math import abs, isinf, sqrt
from std.testing import assert_equal, assert_true, assert_false
from moxie.base import clone
from moxie.covariance import (
    CovarianceEstimator, EmpiricalCovariance, ShrunkCovariance, empirical_covariance, log_likelihood, shrunk_covariance,
)
from moxie.covariance._empirical_covariance import EmpiricalCovarianceEngine, fast_logdet, pinvh
from moxie.covariance._engines import CovarianceEngine


def close(a: Float64, b: Float64, tol: Float64 = 1e-5) -> Bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def matrix[dtype: DType](rows: Int, cols: Int, values: List[Float64]) raises -> NDArray[dtype]:
    var result = nm.zeros[dtype](Shape(rows, cols))
    for k in range(rows * cols):
        result.unsafe_set(k, values[k].cast[dtype]())
    return result^


def raises_with(message: String, error: Error) -> Bool:
    return String(error).find(message) >= 0


def sample[dtype: DType]() raises -> NDArray[dtype]:
    return matrix[dtype](4, 2, [0.0, 1.0, 2.0, 1.5, 1.0, 0.5, 3.0, 2.5])


def generic_engine[E: CovarianceEngine](x: NDArray[E.dtype]) raises -> Float64:
    var covariance = E.empirical(x, False)
    return E.likelihood(covariance, E.precision(covariance))


def check_lifecycle[dtype: DType]() raises:
    var X = sample[dtype]()
    var model = EmpiricalCovariance[dtype]()
    assert_false(model.__sklearn_is_fitted__())
    try:
        _ = model.covariance_()
        assert_true(False)
    except e:
        assert_true(raises_with("NotFittedError: This EmpiricalCovariance instance is not fitted", e))
    try:
        _ = model.score(X)
        assert_true(False)
    except e:
        assert_true(raises_with("NotFittedError", e))
    model.fit(X)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.n_features_in_(), 2)
    var covariance = model.covariance_()
    assert_equal(covariance.shape[0], 2)
    assert_equal(covariance.shape[1], 2)
    assert_true(close(Float64(covariance.unsafe_get(0)), 1.25))
    assert_true(close(Float64(covariance.unsafe_get(1)), 0.6875))
    assert_true(close(Float64(model.location_().unsafe_get(1)), 1.375))
    assert_true(Bool(model.precision_()))
    assert_equal(model.mahalanobis(X).shape[0], 4)
    assert_true(close(Float64(model.mahalanobis(X).unsafe_get(0)), 3.0, 1e-4))
    # Feature-count mismatch and nonfinite inference input.
    try:
        _ = model.mahalanobis(nm.ones[dtype](Shape(2, 3)))
        assert_true(False)
    except e:
        assert_true(raises_with("X has 3 features, but EmpiricalCovariance is expecting 2 features", e))
    var bad = sample[dtype]()
    bad.unsafe_set(3, Float64("nan").cast[dtype]())
    try:
        _ = model.score(bad)
        assert_true(False)
    except e:
        assert_true(raises_with("NaN or infinity", e))
    # Failed refit preserves the previous state.
    try:
        model.fit(bad)
        assert_true(False)
    except e:
        assert_true(raises_with("EmpiricalCovariance.fit", e))
    assert_true(close(Float64(model.covariance_().unsafe_get(0)), 1.25))
    # Clone is unfitted and independent; repeat fit replaces state.
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    copy.set_params(assume_centered=True)
    assert_false(model.assume_centered)
    copy.fit(X)
    assert_true(close(Float64(copy.location_().unsafe_get(0)), 0.0))
    assert_true(close(Float64(copy.covariance_().unsafe_get(0)), (0.0 + 4.0 + 1.0 + 9.0) / 4.0))
    model.fit(matrix[dtype](2, 2, [1.0, 0.0, 3.0, 0.0]))
    assert_true(close(Float64(model.covariance_().unsafe_get(0)), 1.0))
    # store_precision=False: no stored precision, recomputed on demand.
    var lazy = EmpiricalCovariance[dtype](store_precision=False)
    lazy.fit(X)
    assert_false(Bool(lazy.precision_()))
    var eager = EmpiricalCovariance[dtype]()
    eager.fit(X)
    assert_true(close(Float64(lazy.get_precision().unsafe_get(0)), Float64(eager.precision_().value().unsafe_get(0))))
    var params = lazy.get_params()
    assert_false(params.store_precision)
    lazy.set_params(params)
    # Invalid shapes.
    for rows in range(2):
        try:
            var empty = EmpiricalCovariance[dtype]()
            empty.fit(nm.zeros[dtype](Shape(rows * 3, 1 - rows)))
            assert_true(False)
        except e:
            assert_true(raises_with("minimum of 1 is required", e))
    try:
        var flat = EmpiricalCovariance[dtype]()
        flat.fit(nm.zeros[dtype](Shape(3)))
        assert_true(False)
    except e:
        assert_true(raises_with("expected a 2-D array", e))
    try:
        _ = model.error_norm(covariance, norm="nuclear")
        assert_true(False)
    except e:
        assert_true(raises_with("NotImplementedError: Only spectral and frobenius norms are implemented", e))
    print("PASS EmpiricalCovariance lifecycle", dtype)


def check_numerics[dtype: DType]() raises:
    # Singleton sample: zero covariance, zero pseudoinverse (not infinity).
    var single = EmpiricalCovariance[dtype]()
    single.fit(matrix[dtype](1, 1, [1.0]))
    assert_equal(Float64(single.covariance_().unsafe_get(0)), 0.0)
    assert_equal(Float64(single.get_precision().unsafe_get(0)), 0.0)
    assert_true(isinf(single.score(matrix[dtype](1, 1, [1.0]))))
    # Rank-one ones((2,2)) has pseudoinverse 0.25 * ones.
    var ones = nm.ones[dtype](Shape(2, 2))
    var inverse = pinvh(ones)
    for k in range(4):
        assert_true(close(Float64(inverse.unsafe_get(k)), 0.25))
    # Stable logdet without forming the determinant; nonpositive -> -inf.
    var scale: Float64 = 1e200 if dtype == DType.float64 else 1e18
    var big = matrix[dtype](2, 2, [scale, 0.0, 0.0, scale])
    assert_true(close(fast_logdet(big), 2.0 * 460.51701859880916 if dtype == DType.float64 else 2.0 * 41.44653167389282, 1e-12 if dtype == DType.float64 else 1e-6))
    var indefinite = matrix[dtype](2, 2, [1.0, 2.0, 2.0, 1.0])
    assert_true(isinf(fast_logdet(indefinite)))
    # Spectral error norm: zeros vs [[1,2],[0,0]] -> largest eig of EᵀE is 5.
    var zero = EmpiricalCovariance[dtype](assume_centered=True)
    zero.fit(nm.zeros[dtype](Shape(3, 2)))
    var comparison = matrix[dtype](2, 2, [1.0, 2.0, 0.0, 0.0])
    assert_true(close(zero.error_norm(comparison, norm="spectral"), 2.5, 1e-12))
    assert_true(close(zero.error_norm(comparison, norm="spectral", scaling=False, squared=False), sqrt(5.0), 1e-12))
    assert_true(close(zero.error_norm(comparison), 2.5, 1e-12))
    # 1-D input is a single sample; strided views are read in logical order.
    var row = nm.zeros[dtype](Shape(3))
    var one_d = empirical_covariance(row)
    assert_equal(one_d.shape[0], 3)
    var X = sample[dtype]()
    var reversed = X.view_with_layout(Shape(4, 2), nm.NDArrayStrides(-2, 1), 6)
    var forward = empirical_covariance(X)
    var backward = empirical_covariance(reversed)
    for k in range(4):
        assert_true(close(Float64(forward.unsafe_get(k)), Float64(backward.unsafe_get(k)), 1e-6))
    # Broadcasting must also preserve logical indexing on reversed views.
    for rows in [1, 2]:
        assert_true(close(log_likelihood(nm.ones[dtype](Shape(rows, 1)), forward),
                          log_likelihood(nm.ones[dtype](Shape(2, 2)), forward), 1e-6))
    var column = matrix[dtype](2, 1, [2.0, 3.0])
    var reverse_column = column.view_with_layout(Shape(2, 1), nm.NDArrayStrides(-1, 1), 1)
    var expanded = matrix[dtype](2, 2, [3.0, 3.0, 2.0, 2.0])
    assert_true(close(log_likelihood(reverse_column, forward), log_likelihood(expanded, forward), 1e-6))
    # log_likelihood shape validation and generic trait dispatch.
    try:
        _ = log_likelihood(forward, nm.ones[dtype](Shape(3, 3)))
        assert_true(False)
    except e:
        assert_true(raises_with("log_likelihood", e))
    var estimator = EmpiricalCovariance[dtype]()
    estimator.fit(X)
    assert_true(close(estimator.score(X), generic_engine[EmpiricalCovarianceEngine[dtype]](X), 1e-5))
    print("PASS empirical numerics", dtype)


def check_shrinkage[dtype: DType]() raises:
    var X = sample[dtype]()
    var S = empirical_covariance(X)
    for index in range(3):
        var s = [0.0, 0.3, 1.0][index]
        var model = ShrunkCovariance[dtype](shrinkage=s)
        model.fit(X)
        var c = model.covariance_()
        var mu = (Float64(S.unsafe_get(0)) + Float64(S.unsafe_get(3))) / 2.0
        assert_true(close(Float64(c.unsafe_get(1)), (1.0 - s) * Float64(S.unsafe_get(1))))
        assert_true(close(Float64(c.unsafe_get(0)), (1.0 - s) * Float64(S.unsafe_get(0)) + s * mu))
    var invalid = ShrunkCovariance[dtype](shrinkage=1.5)
    try:
        invalid.fit(X)
        assert_true(False)
    except e:
        assert_true(raises_with("InvalidParameterError: The 'shrinkage' parameter of ShrunkCovariance", e))
    assert_false(invalid.__sklearn_is_fitted__())
    var copy = clone(ShrunkCovariance[dtype](shrinkage=0.7))
    assert_true(close(copy.shrinkage, 0.7))
    # Batched shrinkage keeps (batch, p, p).
    var batch = nm.ones[dtype](Shape(3, 2, 2))
    var shrunk = shrunk_covariance(batch, 0.5)
    assert_equal(shrunk.ndim, 3)
    assert_true(close(Float64(shrunk.unsafe_get(4)), 1.0))
    assert_true(close(Float64(shrunk.unsafe_get(5)), 0.5))
    try:
        _ = shrunk_covariance(nm.ones[dtype](Shape(2, 3)))
        assert_true(False)
    except e:
        assert_true(raises_with("square", e))
    try:
        _ = shrunk_covariance(S, -0.1)
        assert_true(False)
    except e:
        assert_true(raises_with("InvalidParameterError", e))
    print("PASS ShrunkCovariance", dtype)


def check_indefinite_mahalanobis[dtype: DType]() raises:
    from moxie.covariance._empirical_covariance import squared_mahalanobis
    from std.math import isnan
    var X = nm.zeros[dtype](Shape(3, 2))
    X.unsafe_set(0, 1)
    X.unsafe_set(3, 1)
    var precision = nm.zeros[dtype](Shape(2, 2))
    precision.unsafe_set(0, -1)
    precision.unsafe_set(3, 2)
    var values = squared_mahalanobis(X, nm.zeros[dtype](Shape(2)), precision)
    assert_true(isnan(values.unsafe_get(0)))
    assert_equal(values.unsafe_get(1), 2)
    assert_equal(values.unsafe_get(2), 0)



def check_nonsymmetric_likelihood[dtype: DType]() raises:
    from moxie.covariance._empirical_covariance import libm_log
    # Pivoted, positive determinant; the lower triangle alone has det=-9.
    var precision = matrix[dtype](2, 2, [0.0, 2.0, -3.0, 1.0])
    var empirical = matrix[dtype](2, 2, [1.0, 0.0, 0.0, 1.0])
    assert_true(close(fast_logdet(precision), libm_log(6.0), 1e-12))
    assert_true(close(log_likelihood(empirical, precision),
                      (-1.0 + libm_log(6.0) - 2.0 * 1.8378770664093453) / 2.0, 1e-12))
    var reversed = precision.view_with_layout(Shape(2, 2), nm.NDArrayStrides(-2, -1), 3)
    assert_true(close(fast_logdet(reversed), libm_log(6.0), 1e-12))
    assert_equal(precision.unsafe_get(1), 2)
    assert_equal(fast_logdet(matrix[dtype](2, 2, [0.0, 2.0, 3.0, 1.0])), -Float64.MAX * 2.0)
    assert_equal(fast_logdet(matrix[dtype](2, 2, [0.0, 2.0, 0.0, 1.0])), -Float64.MAX * 2.0)
    assert_true(close(fast_logdet(matrix[dtype](3, 3, [0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 1.0])), libm_log(24.0), 1e-12))


def check_failed_numerical_refit[dtype: DType]() raises:
    var model = EmpiricalCovariance[dtype]()
    model.fit(sample[dtype]())
    var old_covariance = model.covariance_()
    model.set_params(assume_centered=True)
    var large: Float64 = 1e200 if dtype == DType.float64 else 1e30
    var invalid_estimate = matrix[dtype](2, 2, [large, large, large, large])
    var failed = False
    try:
        model.fit(invalid_estimate)
    except:
        failed = True
    assert_true(failed)
    assert_equal(model.location_().unsafe_get(0), 0)
    assert_equal(model.covariance_().unsafe_get(0), old_covariance.unsafe_get(0))
    var different_features = nm.zeros[dtype](Shape(2, 3))
    for k in range(6):
        different_features.unsafe_set(k, large.cast[dtype]())
    try:
        model.fit(different_features)
    except:
        pass
    assert_equal(model.n_features_in_(), 3)
    failed = False
    try:
        _ = model.mahalanobis(nm.zeros[dtype](Shape(2, 3)))
    except e:
        failed = raises_with("different feature count", e)
    assert_true(failed)
    # A failed first fit exposes location but cannot perform inference.
    var first = EmpiricalCovariance[dtype](assume_centered=True)
    try:
        first.fit(invalid_estimate)
    except:
        pass
    assert_false(first.__sklearn_is_fitted__())
    assert_equal(first.location_().unsafe_get(0), 0)
    failed = False
    try:
        _ = first.score(sample[dtype]())
    except:
        failed = True
    assert_true(failed)


def main() raises:
    check_nonsymmetric_likelihood[DType.float32]()
    check_nonsymmetric_likelihood[DType.float64]()
    check_failed_numerical_refit[DType.float32]()
    check_failed_numerical_refit[DType.float64]()
    check_indefinite_mahalanobis[DType.float64]()
    check_indefinite_mahalanobis[DType.float32]()
    check_lifecycle[DType.float64]()
    check_lifecycle[DType.float32]()
    check_numerics[DType.float64]()
    check_numerics[DType.float32]()
    check_shrinkage[DType.float64]()
    check_shrinkage[DType.float32]()
