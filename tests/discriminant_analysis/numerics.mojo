"""Numerical regressions for the shared covariance helpers, generalized eigh and solvers.

Reference constants were produced by the pinned sklearn/scipy (see docs/private).
"""
from std.testing import assert_true, assert_equal, assert_raises, TestSuite
from std.math import abs
from numojo import NDArray, Shape, zeros
from numojo.core.layout.ndstrides import NDArrayStrides
from moxie.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from moxie._discriminant_analysis._work import Work, matmul, transpose, log, exp
from moxie._discriminant_analysis._covariance import (
    ledoit_wolf,
    standardized_ledoit_wolf,
    shrunk_covariance,
    empirical_covariance,
)
from moxie._discriminant_analysis._generalized_eigh import generalized_symmetric_eigh
from support.fixtures import matrix, assert_close


def sample() raises -> Work:
    return matrix[DType.float64](5, 3, [1.0, 2, 0.5, 2, 1, 1.5, 0, 3, 2.5, 4, 1, 0, 3, 3, 1])


def test_ledoit_wolf_matches_sklearn() raises:
    var expected = matrix[DType.float64](3, 3, [1.50847363, -0.24034656, -0.36051983, -0.24034656, 1.02778051, 0.16023104, -0.36051983, 0.16023104, 1.00374586])
    assert_close(ledoit_wolf(sample()), expected, 1e-8, "ledoit_wolf")


def test_standardized_ledoit_wolf_matches_cov_auto() raises:
    var expected = matrix[DType.float64](3, 3, [2.0, -0.14574626, -0.2186194, -0.14574626, 0.8, 0.09716418, -0.2186194, 0.09716418, 0.74])
    assert_close(standardized_ledoit_wolf(sample()), expected, 1e-8, "_cov auto")


def test_fixed_shrinkage_matches_cov_float() raises:
    var expected = matrix[DType.float64](3, 3, [1.754, -0.42, -0.63, -0.42, 0.914, 0.28, -0.63, 0.28, 0.872])
    assert_close(shrunk_covariance(empirical_covariance(sample()), 0.3), expected, 1e-12, "_cov 0.3")


def test_single_feature_ledoit_wolf_is_variance() raises:
    var x = matrix[DType.float64](4, 1, [1.0, 2, 3, 6])
    var c = ledoit_wolf(x)
    assert_true(abs(c.unsafe_get(0) - 3.5) < 1e-12)


def test_generalized_eigh_contract() raises:
    var a = matrix[DType.float64](3, 3, [2.0, 1, 0, 1, 3, 1, 0, 1, 4])
    var b = matrix[DType.float64](3, 3, [4.0, 1, 0, 1, 2, 0.5, 0, 0.5, 3])
    var eig = generalized_symmetric_eigh(a, b, "test")
    assert_close(eig.values, matrix[DType.float64](1, 3, [0.4607514, 1.24382158, 1.57042702]).reshape(Shape(3)), 1e-7, "values")
    # V^T B V = I and A V = B V diag(lambda)
    var v = eig.vectors.copy()
    var vtbv = matmul(transpose(v), matmul(b, v))
    var identity = matrix[DType.float64](3, 3, [1.0, 0, 0, 0, 1, 0, 0, 0, 1])
    assert_close(vtbv, identity, 1e-12, "B-orthonormal")
    var av = matmul(a, v)
    var bv = matmul(b, v)
    for i in range(3):
        for j in range(3):
            assert_true(abs(av.unsafe_get(i * 3 + j) - bv.unsafe_get(i * 3 + j) * eig.values.unsafe_get(j)) < 1e-12)


def test_generalized_eigh_rejects_indefinite_b() raises:
    var a = matrix[DType.float64](2, 2, [1.0, 0, 0, 1])
    var b = matrix[DType.float64](2, 2, [1.0, 0, 0, 0])
    with assert_raises(contains="LinAlgError"):
        _ = generalized_symmetric_eigh(a, b, "test")


def blobs[dtype: DType]() raises -> NDArray[dtype]:
    return matrix[dtype](10, 2, [0.0, 0.1, 0.3, -0.2, -0.1, 0.4, 0.2, 0.2, 0.5, 0.0, 3.0, 3.1, 3.3, 2.8, 2.9, 3.4, 3.2, 3.0, 2.7, 2.9])


def test_solvers_agree_on_coefficients() raises:
    var X = blobs[DType.float64]()
    var y: List[Int] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    var reference = LinearDiscriminantAnalysis(solver="svd")
    reference.fit(X, y)
    for solver in ["lsqr", "eigen"]:
        var other = LinearDiscriminantAnalysis(solver=solver)
        other.fit(X, y)
        var expected = zeros[DType.float64](Shape(1, 2))
        for j in range(2):
            expected.unsafe_set(j, reference.coef_.unsafe_get(j))
        assert_close(other.coef_, expected, 1e-9, solver)


def test_float32_matches_float64() raises:
    var y: List[Int] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    for solver in ["svd", "lsqr", "eigen"]:
        var m64 = LinearDiscriminantAnalysis(solver=solver)
        m64.fit(blobs[DType.float64](), y)
        var m32 = LinearDiscriminantAnalysis[DType.float32](solver=solver)
        m32.fit(blobs[DType.float32](), y)
        assert_close(m32.coef_, m64.coef_, 1e-5, solver + " float32")
    var q64 = QuadraticDiscriminantAnalysis()
    q64.fit(blobs[DType.float64](), y)
    var q32 = QuadraticDiscriminantAnalysis[DType.float32]()
    q32.fit(blobs[DType.float32](), y)
    assert_close(q32.means_, q64.means_, 1e-6, "qda float32 means")


def test_svd_rank_deficient_duplicate_feature() raises:
    # Third column duplicates the first: svd solver drops the null direction.
    var X = matrix[DType.float64](8, 3, [0.0, 1, 0, 1, 0, 1, 0.5, 0.2, 0.5, 0.1, 0.9, 0.1, 3, 3, 3, 3.5, 2.8, 3.5, 2.9, 3.3, 2.9, 3.2, 3.1, 3.2])
    var y: List[Int] = [0, 0, 0, 0, 1, 1, 1, 1]
    var model = LinearDiscriminantAnalysis(solver="svd")
    model.fit(X, y)
    var pred = model.predict(X)
    for i in range(8):
        assert_equal(pred[i], y[i])
    # lsqr on the singular within-class covariance returns the minimum-norm solution.
    var lsqr = LinearDiscriminantAnalysis(solver="lsqr")
    lsqr.fit(X, y)
    assert_true(abs(Float64(lsqr.coef_.unsafe_get(0)) - Float64(lsqr.coef_.unsafe_get(2))) < 1e-6)


def test_strided_view_input() raises:
    # A transposed-layout view (column strides) gives the same model as a compact copy.
    var X = blobs[DType.float64]()
    var y: List[Int] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    var backing = zeros[DType.float64](Shape(2, 10))
    for i in range(10):
        for j in range(2):
            backing.unsafe_set(j * 10 + i, X.unsafe_get(i * 2 + j))
    var view = backing.view_with_layout(Shape(10, 2), NDArrayStrides(1, 10), 0)
    var a = LinearDiscriminantAnalysis()
    a.fit(X, y)
    var b = LinearDiscriminantAnalysis()
    b.fit(view, y)
    assert_close(b.coef_, a.coef_, 0.0, "strided")


def test_float32_log_probability_underflow() raises:
    var X = matrix[DType.float32](4, 1, [-2.0, -1, 1, 2])
    var y: List[Int] = [0, 0, 1, 1]
    var T = matrix[DType.float32](3, 1, [-30.0, 0, 30])
    for solver in ["svd", "lsqr", "eigen"]:
        var model = LinearDiscriminantAnalysis[DType.float32](solver=solver)
        model.fit(X, y)
        var probabilities = model.predict_proba(T)
        assert_equal(probabilities.unsafe_get(1), Float32(0))
        assert_equal(probabilities.unsafe_get(4), Float32(0))
        # Pinned sklearn: log(float32.smallest_normal) for exact zero probabilities.
        var expected = matrix[DType.float64](3, 2, [0.0, -87.33654475, -0.69314718, -0.69314718, -87.33654475, 0])
        assert_close(model.predict_log_proba(T), expected, 1e-6, solver + " float32 log underflow")


def test_log_exp_have_libm_accuracy() raises:
    # Python math.log/math.exp (libm) values. std.math.log was off by ~1e-9 here, which
    # moved QDA predict_proba by ~8e-10 against sklearn (notebooks/discriminant_analysis).
    var xs: List[Float64] = [0.707088106940819, 0.5, 10.1, 3.744536002348496]
    var logs: List[Float64] = [-0.34660000000028257, -0.6931471805599453, 2.312535423847214, 1.32029771138893]
    for i in range(len(xs)):
        assert_true(abs(log(xs[i]) - logs[i]) <= 2.3e-16 * abs(logs[i]), String("log ", xs[i]))
    var ys: List[Float64] = [-0.8817413, -2.0, 0.5]
    var exps: List[Float64] = [0.4140612786695923, 0.1353352832366127, 1.6487212707001282]
    for i in range(len(ys)):
        assert_true(abs(exp(ys[i]) - exps[i]) <= 2.3e-16 * exps[i], String("exp ", ys[i]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
