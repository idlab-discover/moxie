"""Native regressions for the shared mixture kernels (both dtypes)."""
from numojo import NDArray, Shape, zeros
from std.math import abs, isnan, isinf, exp, sqrt
from std.utils.numerics import inf
from std.testing import assert_true, assert_equal, assert_raises
from moxie._random import RandomState
from moxie.mixture._responsibilities import logsumexp_rows, log_responsibilities, argmax_rows, StableResponsibilities
from moxie.mixture._precision import (
    compute_precision_cholesky, precision_cholesky_from_precisions, log_det_cholesky,
    precisions_from_cholesky, check_precisions, CholeskyPrecision,
)
from moxie.mixture._gaussian_kernels import estimate_gaussian_parameters, estimate_log_gaussian_prob
from moxie.mixture._kmeans_init import kmeans_labels, kmeans_plusplus_indices
from moxie.mixture._initialization import initial_responsibilities, MixtureInitializer
from moxie.mixture._scalar_math import log64, exp64, log1p64


def close(a: Float64, b: Float64, tol: Float64) -> Bool:
    return abs(a - b) <= tol * (1.0 + abs(b))


def matrix[dtype: DType](rows: Int, cols: Int, values: List[Float64]) raises -> NDArray[dtype]:
    var out = zeros[dtype](Shape(rows, cols))
    for i in range(rows * cols):
        out.unsafe_set(i, values[i].cast[dtype]())
    return out^


def scalar_math() raises:
    # Correctly rounded helpers (stdlib log/exp are ~1e-10 accurate).
    assert_equal(log64(10.1), 2.312535423847214)
    assert_equal(exp64(10.1), 24343.00942440838)
    assert_equal(log1p64(1e-20), 1e-20)
    assert_true(close(log1p64(0.3), 0.26236426446749106, 1e-16))


def logsumexp[dtype: DType](tol: Float64) raises:
    var x = matrix[dtype](3, 3, [1000.0, 1000.0, 0.0, -1e300, 0.0, 1.0, 5.0, 5.0, 5.0])
    comptime if dtype == DType.float32:
        x.unsafe_set(3, -1e30)
    var lse = logsumexp_rows(x)
    # Large logits stay finite: log(2 e^1000 + 1) = 1000 + log 2.
    assert_true(close(Float64(lse.unsafe_get(0)), 1000.0 + log64(2.0), tol))
    assert_true(close(Float64(lse.unsafe_get(1)), log64(1.0 + exp64(1.0)), tol))
    assert_true(close(Float64(lse.unsafe_get(2)), 5.0 + log64(3.0), tol))
    var normalized = log_responsibilities(x)
    for i in range(3):
        var total = 0.0
        for j in range(3):
            total += exp64(Float64(normalized[1].unsafe_get(i * 3 + j)))
        assert_true(close(total, 1.0, 100 * tol), "rows normalize up to ulp(1000)")
    # Impossible rows follow the reference: -inf norm and NaN responsibilities.
    var impossible = zeros[dtype](Shape(1, 2))
    var neg_inf = -inf[dtype]()
    impossible.unsafe_set(0, neg_inf)
    impossible.unsafe_set(1, neg_inf)
    var bad = log_responsibilities(impossible)
    assert_true(isinf(bad[0].unsafe_get(0)) and bad[0].unsafe_get(0) < 0)
    assert_true(isnan(bad[1].unsafe_get(0)))
    # argmax: first maximum wins.
    var ties = matrix[dtype](1, 3, [2.0, 2.0, 1.0])
    assert_equal(argmax_rows(ties).unsafe_get(0), 0)
    var engine = StableResponsibilities[dtype]()
    assert_equal(engine.normalize(x)[1].shape[1], 3)
    with assert_raises():
        _ = logsumexp_rows(zeros[dtype](Shape(3)))


def precision[dtype: DType](tol: Float64) raises:
    # Full: cov = [[4, 2], [2, 3]] -> U upper with U U^T = inv(cov).
    var cov = zeros[dtype](Shape(1, 2, 2))
    var values: List[Float64] = [4.0, 2.0, 2.0, 3.0]
    for i in range(4):
        cov.unsafe_set(i, values[i].cast[dtype]())
    var chol = compute_precision_cholesky(cov, "full")
    assert_true(abs(Float64(chol.unsafe_get(2))) == 0.0, "precision factor is upper triangular")
    var precision = precisions_from_cholesky(chol, "full")
    # inv([[4,2],[2,3]]) = [[3,-2],[-2,4]] / 8
    assert_true(close(Float64(precision.unsafe_get(0)), 0.375, tol))
    assert_true(close(Float64(precision.unsafe_get(1)), -0.25, tol))
    assert_true(close(Float64(precision.unsafe_get(3)), 0.5, tol))
    # From precisions must reproduce the same factor.
    var again = precision_cholesky_from_precisions(precision, "full")
    for i in range(4):
        assert_true(close(Float64(again.unsafe_get(i)), Float64(chol.unsafe_get(i)), 10 * tol))
    # log det of U = -0.5 log det(cov) = -0.5 log 8.
    assert_true(close(log_det_cholesky(chol, "full", 2)[0], -0.5 * log64(8.0), tol))
    check_precisions(precision, "full", 1, 2)
    # Non-SPD covariance raises the reference message.
    var singular = zeros[dtype](Shape(1, 2, 2))
    singular.unsafe_set(0, 1)
    singular.unsafe_set(1, 1)
    singular.unsafe_set(2, 1)
    singular.unsafe_set(3, 1)
    var raised = False
    try:
        _ = compute_precision_cholesky(singular, "full")
    except error:
        raised = String(error).startswith("ValueError: Fitting the mixture model failed")
    assert_true(raised)
    # Diag/spherical: 1/sqrt, zero variance rejected, non-positive precision rejected.
    var diag = matrix[dtype](1, 2, [4.0, 0.25])
    var diag_chol = compute_precision_cholesky(diag, "diag")
    assert_true(close(Float64(diag_chol.unsafe_get(0)), 0.5, tol) and close(Float64(diag_chol.unsafe_get(1)), 2.0, tol))
    assert_true(close(log_det_cholesky(diag_chol, "diag", 2)[0], 0.0, tol))
    var zero = matrix[dtype](1, 2, [0.0, 1.0])
    with assert_raises(contains="ill-defined empirical covariance"):
        _ = compute_precision_cholesky(zero, "diag")
    with assert_raises(contains="spherical precision' should be positive"):
        check_precisions(matrix[dtype](1, 2, [1.0, -1.0]).reshape(Shape(2)), "spherical", 2, 3)
    var asymmetric = zeros[dtype](Shape(2, 2))
    asymmetric.unsafe_set(0, 1)
    asymmetric.unsafe_set(1, 0.5)
    asymmetric.unsafe_set(3, 1)
    with assert_raises(contains="symmetric, positive-definite"):
        check_precisions(asymmetric, "tied", 3, 2)
    with assert_raises(contains="should have the shape"):
        check_precisions(asymmetric, "full", 1, 2)
    var engine = CholeskyPrecision[dtype]()
    assert_equal(engine.from_covariances(diag, "diag").size, 2)


def gaussian_statistics[dtype: DType](tol: Float64) raises:
    # Two samples, hard responsibilities: one per component.
    var X = matrix[dtype](3, 2, [0.0, 0.0, 2.0, 0.0, 10.0, 10.0])
    var resp = matrix[dtype](3, 2, [1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    var params = estimate_gaussian_parameters(X, resp, 0.0, "diag")
    assert_true(close(Float64(params.nk.unsafe_get(0)), 2.0, tol))
    assert_true(close(Float64(params.means.unsafe_get(0)), 1.0, tol))
    assert_true(close(Float64(params.covariances.unsafe_get(0)), 1.0, 10 * tol))
    var full = estimate_gaussian_parameters(X, resp, 0.5, "full")
    assert_true(close(Float64(full.covariances.unsafe_get(0)), 1.5, 10 * tol))
    assert_true(close(Float64(full.covariances.unsafe_get(3)), 0.5, 10 * tol))
    var spherical = estimate_gaussian_parameters(X, resp, 0.0, "spherical")
    assert_true(close(Float64(spherical.covariances.unsafe_get(0)), 0.5, 10 * tol))
    var tied = estimate_gaussian_parameters(X, resp, 0.0, "tied")
    assert_equal(tied.covariances.ndim, 2)
    with assert_raises():
        _ = estimate_gaussian_parameters(X, resp, 0.0, "banana")
    # Standard normal log density at the mean: -log(2 pi) in 2-D.
    var means = matrix[dtype](1, 2, [0.0, 0.0])
    var unit = matrix[dtype](1, 2, [1.0, 1.0])
    var log_prob = estimate_log_gaussian_prob(X, means, unit, "diag", log_det_cholesky(unit, "diag", 2))
    assert_true(close(Float64(log_prob.unsafe_get(0)), -1.8378770664093453, tol))
    assert_true(close(Float64(log_prob.unsafe_get(1)), -1.8378770664093453 - 2.0, tol))
    # The same density through every covariance representation.
    var sph = matrix[dtype](1, 1, [1.0]).reshape(Shape(1))
    var eye = zeros[dtype](Shape(2, 2))
    eye.unsafe_set(0, 1)
    eye.unsafe_set(3, 1)
    var eye3 = eye.reshape(Shape(1, 2, 2))
    var a = estimate_log_gaussian_prob(X, means, sph, "spherical", log_det_cholesky(sph, "spherical", 2))
    var b = estimate_log_gaussian_prob(X, means, eye, "tied", log_det_cholesky(eye, "tied", 2))
    var c = estimate_log_gaussian_prob(X, means, eye3, "full", log_det_cholesky(eye3, "full", 2))
    for i in range(3):
        assert_true(close(Float64(a.unsafe_get(i)), Float64(log_prob.unsafe_get(i)), tol))
        assert_true(close(Float64(b.unsafe_get(i)), Float64(log_prob.unsafe_get(i)), tol))
        assert_true(close(Float64(c.unsafe_get(i)), Float64(log_prob.unsafe_get(i)), tol))


def initialization[dtype: DType]() raises:
    var values = List[Float64]()
    for i in range(12):
        var offset = 0.0 if i < 6 else 20.0
        values.append(offset + Float64(i % 3) * 0.1)
        values.append(offset - Float64(i % 2) * 0.1)
    var X = matrix[dtype](12, 2, values)
    var rng = RandomState(0)
    var labels = kmeans_labels(X, 2, rng)
    for i in range(12):
        assert_equal(labels[i] == labels[0], i < 6)
    var seeds = kmeans_plusplus_indices(X, 2, rng)
    assert_true((seeds[0] < 6) != (seeds[1] < 6), "k-means++ picks one seed per blob")
    with assert_raises(contains="should be >= n_clusters"):
        _ = kmeans_plusplus_indices(X, 13, rng)
    var engine = MixtureInitializer[dtype]()
    for method in [String("kmeans"), String("k-means++"), String("random"), String("random_from_data")]:
        var resp = engine.responsibilities(X, 3, method, rng)
        assert_equal(resp.shape[0], 12)
        assert_equal(resp.shape[1], 3)
        var column_total = 0.0
        for idx in range(resp.size):
            column_total += Float64(resp.unsafe_get(idx))
        if method == "kmeans" or method == "random":
            assert_true(close(column_total, 12.0, 1e-5), method)
        else:
            assert_true(close(column_total, 3.0, 1e-12), method + " seeds exactly one row per component")
    with assert_raises(contains="init_params"):
        _ = initial_responsibilities(X, 2, "kmeans2", rng)
    # Same seed, same stream.
    var r1 = RandomState(5)
    var r2 = RandomState(5)
    var first = initial_responsibilities(X, 2, "random", r1)
    var second = initial_responsibilities(X, 2, "random", r2)
    for idx in range(first.size):
        assert_equal(first.unsafe_get(idx), second.unsafe_get(idx))


def main() raises:
    scalar_math()
    logsumexp[DType.float64](1e-14)
    logsumexp[DType.float32](1e-6)
    precision[DType.float64](1e-14)
    precision[DType.float32](1e-6)
    gaussian_statistics[DType.float64](1e-14)
    gaussian_statistics[DType.float32](1e-6)
    initialization[DType.float64]()
    initialization[DType.float32]()
    print("PASS mixture kernels: logsumexp, precision factors, sufficient statistics, initialization (Float32/Float64)")
