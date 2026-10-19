"""Native contracts for PLSSVD."""
import numojo as nm
from std.testing import assert_almost_equal, assert_equal, assert_false, assert_raises, assert_true, TestSuite
from moxie.base import clone
from moxie.cross_decomposition import PLSSVD, PLSSVDParams, PLSRegression, CrossDecompositionTransformer

comptime F64 = DType.float64


def block[dtype: DType = F64](rows: Int, cols: Int, values: List[Float64]) raises -> nm.NDArray[dtype]:
    var a = nm.zeros[dtype](nm.Shape(rows, cols))
    for i in range(rows * cols):
        a.unsafe_set(i, Scalar[dtype](values[i]))
    return a^


def sample_x[dtype: DType = F64]() raises -> nm.NDArray[dtype]:
    return block[dtype](6, 3, [5.0, 162, 60, 2, 110, 60, 12, 101, 101, 12, 105, 37, 13, 155, 58, 4, 101, 42])


def sample_y[dtype: DType = F64]() raises -> nm.NDArray[dtype]:
    return block[dtype](6, 3, [191.0, 36, 50, 189, 37, 52, 193, 38, 58, 162, 35, 62, 189, 35, 46, 182, 36, 56])


def through_trait[E: CrossDecompositionTransformer](estimator: E, X: E.Input, y: E.Target) raises -> E.PairedTransform:
    return estimator.transform(X, y)


def test_defaults_params_clone() raises:
    var est = PLSSVD()
    assert_equal(est.n_components, 2)
    assert_true(est.scale)
    assert_true(est.copy_input)
    est.set_params(n_components=1, copy=False)
    assert_equal(est.get_params().n_components, 1)
    assert_false(est.copy_input)
    est.fit(sample_x(), sample_y())
    var c = clone(est)
    assert_false(c.__sklearn_is_fitted__())
    assert_equal(c.n_components, 1)


def test_errors() raises:
    var est = PLSSVD()
    with assert_raises(contains="NotFittedError: This PLSSVD instance is not fitted yet"):
        _ = est.transform(sample_x())
    with assert_raises(contains="`n_components` upper bound is 3. Got 4 instead."):
        var e = PLSSVD(4)
        e.fit(sample_x(), sample_y())
    with assert_raises(contains="The 'n_components' parameter of PLSSVD must be an int in the range [1, inf). Got 0 instead."):
        var e = PLSSVD(0)
        e.fit(sample_x(), sample_y())
    with assert_raises(contains="inconsistent numbers of samples"):
        est.fit(sample_x(), nm.ones[F64](nm.Shape(5, 3)))
    est.fit(sample_x(), sample_y())
    with assert_raises(contains="X has 2 features, but PLSSVD is expecting 3 features as input."):
        _ = est.transform(nm.ones[F64](nm.Shape(3, 2)))


def test_shapes_orthonormal_and_signs() raises:
    var est = PLSSVD(2)
    var scores = est.fit_transform(sample_x(), sample_y())
    assert_equal(scores.x_scores.shape[0], 6)
    assert_equal(scores.y_scores.shape[1], 2)
    assert_equal(est.x_weights_.shape[0], 3)
    assert_equal(est.y_weights_.shape[0], 3)
    for c in range(2):
        var norm: Float64 = 0
        var best: Float64 = 0
        var signed: Float64 = 0
        for i in range(3):
            var w = est.x_weights_.item(i, c)
            norm += w * w
            if abs(w) > best:
                best = abs(w)
                signed = w
        assert_almost_equal(norm, 1.0, atol=1e-12)
        assert_true(signed > 0)  # svd_flip: largest |u| entry positive
    var names = est.get_feature_names_out()
    assert_equal(names[1], "plssvd1")
    var paired = through_trait(est, sample_x(), sample_y())
    assert_equal(paired.x_scores.shape[1], 2)


def test_one_component_equivalence_with_pls() raises:
    """Mirror sklearn test_one_component_equivalence (single target)."""
    var y1 = nm.zeros[F64](nm.Shape(6))
    var y = sample_y()
    for i in range(6):
        y1.unsafe_set(i, y.item(i, 0))
    var svd = PLSSVD(1)
    svd.fit(sample_x(), y1)
    var reg = PLSRegression(1)
    reg.fit(sample_x(), y1)
    var a = svd.transform(sample_x())
    var b = reg.transform(sample_x())
    for i in range(6):
        assert_almost_equal(a.item(i, 0), b.item(i, 0), atol=1e-8)


def test_float32_and_copy_semantics() raises:
    var a = PLSSVD[DType.float32](2)
    a.fit(sample_x[DType.float32](), sample_y[DType.float32]())
    var b = PLSSVD(2)
    b.fit(sample_x(), sample_y())
    for i in range(6):
        assert_almost_equal(a.x_weights_.unsafe_get(i), b.x_weights_.unsafe_get(i), atol=1e-12)
    var X = sample_x()
    var y = sample_y()
    var keep = PLSSVD(copy=False)
    keep.fit(X, y)
    assert_equal(X.item(0, 1), 162.0)
    keep.fit_inplace(X, y)
    assert_true(abs(X.item(0, 1) - 162.0) > 1.0)
    var X32 = sample_x[DType.float32]()
    var y32 = sample_y[DType.float32]()
    var f32 = PLSSVD[DType.float32](copy=False)
    f32.fit_inplace(X32, y32)
    assert_equal(X32.item(0, 1), Float32(162))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
