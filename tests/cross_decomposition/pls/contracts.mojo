"""Native estimator contracts for PLSRegression, PLSCanonical and CCA."""
import numojo as nm
from std.math import isnan
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)
from moxie.base import clone
from moxie.cross_decomposition import (
    CCA,
    CrossDecompositionTransformer,
    PLSCanonical,
    PLSParams,
    PLSPredictor,
    PLSRegression,
)

comptime F64 = DType.float64
comptime F32 = DType.float32

def linnerud_x() -> List[Float64]:
    return [5, 162, 60, 2, 110, 60, 12, 101, 101, 12, 105, 37, 13, 155, 58, 4, 101, 42, 8, 101, 38, 6, 125, 40, 15, 200, 40, 17, 251, 250, 17, 120, 38, 13, 210, 115, 14, 215, 105, 1, 50, 50, 6, 70, 31, 12, 210, 120, 4, 60, 25, 11, 230, 80, 15, 225, 73, 2, 110, 43]


def linnerud_y() -> List[Float64]:
    return [191, 36, 50, 189, 37, 52, 193, 38, 58, 162, 35, 62, 189, 35, 46, 182, 36, 56, 211, 38, 56, 167, 34, 60, 176, 31, 74, 154, 33, 56, 169, 34, 50, 166, 33, 52, 154, 34, 64, 247, 46, 50, 193, 36, 46, 202, 37, 62, 176, 37, 54, 157, 32, 52, 156, 33, 54, 138, 33, 68]


def block[dtype: DType](rows: Int, cols: Int, values: List[Float64]) raises -> nm.NDArray[dtype]:
    var a = nm.zeros[dtype](nm.Shape(rows, cols))
    for i in range(rows * cols):
        a.unsafe_set(i, Scalar[dtype](values[i]))
    return a^


def linnerud[dtype: DType = F64]() raises -> Tuple[nm.NDArray[dtype], nm.NDArray[dtype]]:
    return (block[dtype](20, 3, linnerud_x()), block[dtype](20, 3, linnerud_y()))


def pseudo_random(rows: Int, cols: Int, seed: Int) raises -> nm.NDArray[F64]:
    """Deterministic well-spread values (LCG), independent of any RNG API."""
    var a = nm.zeros[F64](nm.Shape(rows, cols))
    var state = UInt64(seed * 2654435761 + 12345)
    for i in range(rows * cols):
        state = state * 6364136223846793005 + 1442695040888963407
        a.unsafe_set(i, Float64(state >> 11) / Float64(UInt64(1) << 53) - 0.5)
    return a^


def column(a: nm.NDArray[F64], j: Int) raises -> nm.NDArray[F64]:
    var out = nm.zeros[F64](nm.Shape(a.shape[0]))
    for i in range(a.shape[0]):
        out.unsafe_set(i, a.item(i, j))
    return out^


def assert_matrix_orthogonal(m: nm.NDArray[F64]) raises:
    for a in range(m.shape[1]):
        for b in range(m.shape[1]):
            if a == b:
                continue
            var dot: Float64 = 0
            for i in range(m.shape[0]):
                dot += m.item(i, a) * m.item(i, b)
            assert_almost_equal(dot, 0.0, atol=1e-6)


def assert_close(actual: nm.NDArray[F64], expected: nm.NDArray[F64], atol: Float64 = 1e-10) raises:
    assert_equal(actual.ndim, expected.ndim)
    assert_equal(actual.size, expected.size)
    for i in range(actual.size):
        assert_almost_equal(actual.unsafe_get(i), expected.unsafe_get(i), atol=atol)


def uses_contracts[E: CrossDecompositionTransformer & PLSPredictor](mut estimator: E, X: E.Input, y: E.Target) raises -> E.Prediction:
    """Compile-time proof that the estimators satisfy the family traits."""
    estimator.fit(X, y)
    return estimator.predict(X)


def paired_through_trait[E: CrossDecompositionTransformer](estimator: E, X: E.Input, y: E.Target) raises -> E.PairedTransform:
    return estimator.transform(X, y)


def test_defaults_and_params() raises:
    var reg = PLSRegression()
    assert_equal(reg.n_components, 2)
    assert_true(reg.scale)
    assert_equal(reg.max_iter, 500)
    assert_equal(reg.tol, 1e-6)
    assert_true(reg.copy_input)
    assert_equal(reg.algorithm, "nipals")
    var can = PLSCanonical(3, algorithm="svd", copy=False)
    var params = can.get_params()
    assert_equal(params.n_components, 3)
    assert_equal(params.algorithm, "svd")
    assert_false(params.copy_input)
    var cca = CCA(1, scale=False, max_iter=10, tol=0.5)
    assert_false(cca.scale)
    assert_equal(cca.max_iter, 10)
    assert_equal(cca.tol, 0.5)
    assert_equal(PLSRegression.name(), "PLSRegression")
    assert_equal(PLSCanonical.name(), "PLSCanonical")
    assert_equal(CCA.name(), "CCA")


def test_set_params_and_clone() raises:
    var reg = PLSRegression()
    reg.set_params(n_components=1, tol=1e-3)
    assert_equal(reg.n_components, 1)
    assert_equal(reg.tol, 1e-3)
    assert_equal(reg.max_iter, 500)
    with assert_raises(contains="Invalid parameter 'algorithm'"):
        reg.set_params(algorithm="svd")
    with assert_raises(contains="Invalid parameter 'algorithm'"):
        var c = CCA()
        c.set_params(PLSParams(2, True, "svd", 500, 1e-6, True))
    var data = linnerud()
    reg.fit(data[0], data[1])
    var copy = clone(reg)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(copy.n_components, 1)
    copy.set_params(n_components=2)
    assert_equal(reg.n_components, 1)
    # A fitted copy owns independent learned buffers.
    var fitted_copy = reg.copy()
    fitted_copy.coef_.unsafe_set(0, 1234.0)
    assert_true(reg.coef_.unsafe_get(0) != 1234.0)


def test_unfitted_errors() raises:
    var reg = PLSRegression()
    var data = linnerud()
    with assert_raises(contains="NotFittedError: This PLSRegression instance is not fitted yet"):
        _ = reg.transform(data[0])
    with assert_raises(contains="NotFittedError"):
        _ = reg.predict(data[0])
    with assert_raises(contains="NotFittedError"):
        _ = reg.inverse_transform(data[0])
    with assert_raises(contains="NotFittedError"):
        _ = reg.score(data[0], data[1])
    with assert_raises(contains="NotFittedError"):
        _ = reg.get_feature_names_out()


def test_invalid_parameters() raises:
    var data = linnerud()
    with assert_raises(contains="The 'n_components' parameter of PLSRegression must be an int in the range [1, inf). Got 0 instead."):
        var e = PLSRegression(0)
        e.fit(data[0], data[1])
    with assert_raises(contains="`n_components` upper bound is 3. Got 4 instead. Reduce `n_components`."):
        var e = PLSRegression(4)
        e.fit(data[0], data[1])
    with assert_raises(contains="The 'max_iter' parameter of CCA"):
        var e = CCA(max_iter=0)
        e.fit(data[0], data[1])
    var zero = Float64(0)
    with assert_raises(contains="The 'tol' parameter of PLSCanonical"):
        var e = PLSCanonical(tol=zero / zero)
        e.fit(data[0], data[1])
    with assert_raises(contains="The 'tol' parameter"):
        var e = PLSCanonical(tol=-1.0)
        e.fit(data[0], data[1])
    with assert_raises(contains="The 'algorithm' parameter of PLSCanonical must be a str among"):
        var e = PLSCanonical(algorithm="lanczos")
        e.fit(data[0], data[1])
    # Canonical bound uses min(n, p, q).
    var y1 = block[F64](20, 1, linnerud_y())
    with assert_raises(contains="upper bound is 1"):
        var e = PLSCanonical(2)
        e.fit(data[0], y1)
    var ok = PLSRegression(2)
    ok.fit(data[0], y1)  # regression bound is min(n, p)


def test_invalid_inputs() raises:
    var data = linnerud()
    var reg = PLSRegression()
    var bad = data[0].copy()
    var zero = Float64(0)
    bad.unsafe_set(4, zero / zero)
    with assert_raises(contains="Input X contains NaN."):
        reg.fit(bad, data[1])
    bad.unsafe_set(4, 1.0 / zero)
    with assert_raises(contains="Input X contains infinity or a value too large for dtype('float64')."):
        reg.fit(bad, data[1])
    var bad_y = data[1].copy()
    bad_y.unsafe_set(0, zero / zero)
    with assert_raises(contains="Input y contains NaN."):
        reg.fit(data[0], bad_y)
    with assert_raises(contains="inconsistent numbers of samples: [20, 3]"):
        reg.fit(data[0], nm.ones[F64](nm.Shape(3, 3)))
    with assert_raises(contains="Found array with 1 sample(s) (shape=(1, 3)) while a minimum of 2 is required by PLSRegression."):
        reg.fit(nm.ones[F64](nm.Shape(1, 3)), nm.ones[F64](nm.Shape(1, 2)))
    with assert_raises(contains="Expected 2D array, got 1D array instead"):
        reg.fit(nm.ones[F64](nm.Shape(20)), data[1])
    with assert_raises(contains="Found array with 0 feature(s)"):
        reg.fit(nm.ones[F64](nm.Shape(20, 0)), data[1])
    with assert_raises(contains="Found array with dim 3"):
        reg.fit(data[0], nm.ones[F64](nm.Shape(20, 2, 2)))
    # A failed fit leaves the estimator unfitted.
    assert_false(reg.__sklearn_is_fitted__())
    reg.fit(data[0], data[1])
    with assert_raises(contains="X has 2 features, but PLSRegression is expecting 3 features as input."):
        _ = reg.predict(nm.ones[F64](nm.Shape(4, 2)))
    with assert_raises(contains="X has 4 columns, but PLSRegression has 2 components."):
        _ = reg.inverse_transform(nm.ones[F64](nm.Shape(3, 4)))
    with assert_raises(contains="y has 2 targets"):
        _ = reg.transform(data[0], nm.ones[F64](nm.Shape(20, 2)))
    # Failed validation of a refit keeps the previous learned state.
    with assert_raises():
        reg.fit(bad, data[1])
    assert_true(reg.__sklearn_is_fitted__())
    assert_equal(reg.n_features_in_, 3)


def test_shapes_and_sanity_values() raises:
    """Mirror sklearn test_sanity_check_pls_regression, test_attributes_shapes, coef shape."""
    var data = linnerud()
    var pls = PLSRegression(3)
    var paired = pls.fit_transform(data[0], data[1])
    assert_close(paired.x_scores, pls.x_scores_, 1e-10)
    var expected_x_weights: List[Float64] = [-0.61330704, -0.00443647, 0.78983213, -0.74697144, -0.32172099, -0.58183269, -0.25668686, 0.94682413, -0.19399983]
    var expected_x_loadings: List[Float64] = [-0.61470416, -0.24574278, 0.78983213, -0.65625755, -0.14396183, -0.58183269, -0.51733059, 1.00609417, -0.19399983]
    var expected_y_weights: List[Float64] = [0.32456184, 0.29892183, 0.20316322, 0.42439636, 0.61970543, 0.19320542, -0.13143144, -0.26348971, -0.17092916]
    for i in range(9):
        assert_almost_equal(abs(pls.x_weights_.unsafe_get(i)), abs(expected_x_weights[i]), atol=1e-6)
        assert_almost_equal(abs(pls.x_loadings_.unsafe_get(i)), abs(expected_x_loadings[i]), atol=1e-6)
        assert_almost_equal(abs(pls.y_weights_.unsafe_get(i)), abs(expected_y_weights[i]), atol=1e-6)
        assert_almost_equal(abs(pls.y_loadings_.unsafe_get(i)), abs(expected_y_weights[i]), atol=1e-6)
        # Signs are consistent between weights and loadings.
        var sx = pls.x_weights_.unsafe_get(i) / expected_x_weights[i]
        var sl = pls.x_loadings_.unsafe_get(i) / expected_x_loadings[i]
        assert_true((sx > 0) == (sl > 0))
    assert_equal(pls.coef_.shape[0], 3)
    assert_equal(pls.coef_.shape[1], 3)
    assert_equal(pls.intercept_.shape[0], 3)
    assert_equal(len(pls.n_iter_), 3)
    var two = PLSCanonical(2)
    two.fit(data[0], data[1])
    assert_equal(two.x_weights_.shape[1], 2)
    assert_equal(two.y_rotations_.shape[0], 3)
    var names = two.get_feature_names_out()
    assert_equal(len(names), 2)
    assert_equal(names[1], "plscanonical1")
    var one = CCA(1)
    one.fit(data[0], data[1])
    assert_equal(one.get_feature_names_out()[0], "cca0")
    assert_equal(one.transform(data[0]).shape[1], 1)


def test_canonical_basics() raises:
    """Mirror sklearn test_pls_canonical_basics: orthogonality, X = T P', transform, inverse."""
    var data = linnerud()
    var pls = PLSCanonical(3)
    pls.fit(data[0], data[1])
    assert_matrix_orthogonal(pls.x_weights_)
    assert_matrix_orthogonal(pls.y_weights_)
    assert_matrix_orthogonal(pls.x_scores_)
    assert_matrix_orthogonal(pls.y_scores_)
    var xt = pls.transform(data[0])
    assert_close(xt, pls.x_scores_, 1e-8)
    var paired = pls.transform(data[0], data[1])
    assert_close(paired.y_scores, pls.y_scores_, 1e-8)
    var x_back = pls.inverse_transform(xt)
    assert_close(x_back, data[0], 1e-8)
    var back = pls.inverse_transform(paired.x_scores, paired.y_scores)
    assert_close(back.y_scores, data[1], 1e-8)


def test_univariate_equivalence_and_1d_predict() raises:
    var data = linnerud()
    var y1 = nm.zeros[F64](nm.Shape(20))
    var y2 = nm.zeros[F64](nm.Shape(20, 1))
    for i in range(20):
        y1.unsafe_set(i, data[1].item(i, 0))
        y2.unsafe_set(i, data[1].item(i, 0))
    var a = PLSRegression(1)
    a.fit(data[0], y1)
    var b = PLSRegression(1)
    b.fit(data[0], y2)
    assert_close(a.coef_, b.coef_, 1e-12)
    assert_equal(a.predict(data[0]).ndim, 1)
    assert_equal(b.predict(data[0]).ndim, 2)
    var cca1 = CCA(1)
    cca1.fit(data[0], y1)
    var cca2 = CCA(1)
    cca2.fit(data[0], y2)
    assert_close(cca1.coef_, cca2.coef_, 1e-10)
    # test_pls_regression_fit_1d_y: exact quadratic features.
    var X = block[F64](6, 2, [1.0, 1, 2, 4, 3, 9, 4, 16, 5, 25, 6, 36])
    var y = nm.zeros[F64](nm.Shape(6))
    var targets: List[Float64] = [2.0, 6, 12, 20, 30, 42]
    for i in range(6):
        y.unsafe_set(i, targets[i])
    var plsr = PLSRegression()
    plsr.fit(X, y)
    var pred = plsr.predict(X)
    assert_equal(pred.ndim, 1)
    assert_equal(pred.shape[0], 6)


def test_prediction_formula_and_score() raises:
    """Mirror sklearn test_pls_prediction: intercept is y mean; predict = Xc @ coef.T + b."""
    var data = linnerud()
    for scale in range(2):
        var est = CCA(scale=scale == 1)
        est.fit(data[0], data[1])
        var pred = est.predict(data[0])
        for t in range(3):
            var mean: Float64 = 0
            for i in range(20):
                mean += data[1].item(i, t)
            assert_almost_equal(est.intercept_.item(t), mean / 20, atol=1e-10)
        for i in range(20):
            for t in range(3):
                var value = est.intercept_.item(t)
                for j in range(3):
                    var xm: Float64 = 0
                    for r in range(20):
                        xm += data[0].item(r, j)
                    value += (data[0].item(i, j) - xm / 20) * est.coef_.item(t, j)
                assert_almost_equal(pred.item(i, t), value, atol=1e-8)
    var reg = PLSRegression(3)
    reg.fit(data[0], data[1])
    var r2 = reg.score(data[0], data[1])
    assert_true(r2 > 0 and r2 < 1)
    var weights = nm.ones[F64](nm.Shape(20))
    assert_almost_equal(reg.score(data[0], data[1], weights^), r2, atol=1e-12)
    # Fewer than two samples: R^2 is undefined (NaN), as in r2_score.
    assert_true(isnan(reg.score(block[F64](1, 3, [5.0, 162, 60]), block[F64](1, 3, [191.0, 36, 50]))))


def test_scaling_coef_recovers_linear_map() raises:
    """Mirror sklearn test_pls_regression_scaling_coef: coef_ uses both X and y stds."""
    var X = pseudo_random(30, 5, 1)
    for i in range(X.size):
        X.unsafe_set(i, X.unsafe_get(i) * 20)
    var coef = pseudo_random(3, 5, 2)
    var y = X @ coef.T()
    var pls = PLSRegression(5, scale=True)
    pls.fit(X, y)
    assert_close(pls.coef_, coef, 1e-8)
    assert_close(pls.predict(X), y, 1e-8)


def test_constant_targets_and_convergence_diagnostics() raises:
    """Mirror sklearn test_pls_constant_y and test_convergence_fail as native diagnostics."""
    var X = pseudo_random(100, 3, 42)
    var y = nm.zeros[F64](nm.Shape(100))
    var pls = PLSRegression()
    pls.fit(X, y)
    assert_equal(len(pls.diagnostics_), 1)
    assert_equal(pls.diagnostics_[0], "UserWarning: y residual is constant at iteration 0")
    assert_equal(len(pls.n_iter_), 0)
    for i in range(pls.x_rotations_.size):
        assert_equal(pls.x_rotations_.unsafe_get(i), 0.0)
    var data = linnerud()
    var capped = PLSCanonical(3, max_iter=2)
    capped.fit(data[0], data[1])
    assert_true(len(capped.diagnostics_) >= 1)
    assert_equal(capped.diagnostics_[0], "ConvergenceWarning: Maximum number of iterations reached")
    for n in capped.n_iter_:
        assert_true(n <= 2)
    # A later successful fit replaces diagnostics.
    capped.set_params(max_iter=500)
    capped.fit(data[0], data[1])
    assert_equal(len(capped.diagnostics_), 0)
    var svd = PLSCanonical(2, algorithm="svd")
    svd.fit(data[0], data[1])
    assert_equal(len(svd.n_iter_), 0)


def test_float32_inputs() raises:
    var data64 = linnerud()
    var data32 = linnerud[F32]()
    var a = PLSRegression[F32](2)
    a.fit(data32[0], data32[1])
    var b = PLSRegression(2)
    b.fit(data64[0], data64[1])
    # Linnerud values are exact in Float32, so fits are identical.
    assert_close(a.coef_, b.coef_, 1e-12)
    var pred = a.predict(data32[0])
    assert_close(pred, b.predict(data64[0]), 1e-4)
    var scores = a.transform(data32[0])
    assert_equal(scores.shape[1], 2)


def test_copy_semantics() raises:
    """Mirror sklearn test_copy: fit never mutates; fit_inplace with copy=False does."""
    var data = linnerud()
    var X = data[0].copy()
    var y = data[1].copy()
    var reg = PLSRegression(copy=False)
    reg.fit(X, y)
    assert_close(X, data[0], 0.0)
    var keep = PLSRegression(copy=True)
    keep.fit_inplace(X, y)
    assert_close(X, data[0], 0.0)
    reg.fit_inplace(X, y)
    assert_true(abs(X.item(0, 0) - data[0].item(0, 0)) > 1e-3)
    # Float32 inputs are converted, so even copy=False leaves them intact.
    var d32 = linnerud[F32]()
    var X32 = d32[0].copy()
    var y32 = d32[1].copy()
    var reg32 = PLSRegression[F32](copy=False)
    reg32.fit_inplace(X32, y32)
    assert_equal(X32.item(0, 1), Float32(162))
    # copy=False inference agrees with copying inference and normalizes X.
    var fitted = PLSRegression()
    fitted.fit(data[0], data[1])
    var Xc = data[0].copy()
    var inplace = fitted.transform_inplace(Xc)
    assert_close(inplace, fitted.transform(data[0]), 1e-12)
    assert_true(abs(Xc.item(0, 1) - 162.0) > 1.0)
    var Xp = data[0].copy()
    assert_close(fitted.predict_inplace(Xp), fitted.predict(data[0]), 1e-12)


def test_strided_views_and_refit() raises:
    var data = linnerud()
    var backing = nm.zeros[F64](nm.Shape(20, 6))
    for i in range(20):
        for j in range(3):
            backing.unsafe_set(i * 6 + 2 * j, data[0].item(i, j))
    var view = backing.view_with_layout(nm.Shape(20, 3), nm.NDArrayStrides(6, 2), 0)
    var a = PLSRegression()
    a.fit(view, data[1])
    var b = PLSRegression()
    b.fit(data[0], data[1])
    assert_close(a.coef_, b.coef_, 1e-12)
    assert_close(a.predict(view), b.predict(data[0]), 1e-10)
    # Refit on different shapes replaces every learned attribute.
    var y1 = nm.zeros[F64](nm.Shape(20))
    for i in range(20):
        y1.unsafe_set(i, data[1].item(i, 2))
    a.set_params(n_components=1)
    a.fit(data[0], y1)
    assert_equal(a.coef_.shape[0], 1)
    assert_equal(a.x_weights_.shape[1], 1)
    assert_equal(a.predict(data[0]).ndim, 1)


def test_trait_conformance_and_tiny_inputs() raises:
    var data = linnerud()
    var reg = PLSRegression()
    _ = uses_contracts(reg, data[0], data[1])
    var through = paired_through_trait(reg, data[0], data[1])
    assert_close(through.y_scores, reg.transform(data[0], data[1]).y_scores, 0.0)
    var can = PLSCanonical()
    _ = uses_contracts(can, data[0], data[1])
    var cca = CCA()
    _ = uses_contracts(cca, data[0], data[1])
    # Two samples, a constant X column and rank-deficient X all fit.
    var tiny = PLSRegression(1)
    tiny.fit(block[F64](2, 2, [1.0, 2, 3, 5]), block[F64](2, 1, [1.0, 2]))
    assert_true(tiny.__sklearn_is_fitted__())
    var constant = data[0].copy()
    for i in range(20):
        constant.unsafe_set(i * 3 + 2, 1.0)
    var c = PLSCanonical(2)
    c.fit(constant, data[1])
    assert_equal(c._x_std.item(2), 1.0)
    var rank1 = nm.zeros[F64](nm.Shape(10, 3))
    for i in range(10):
        for j in range(3):
            rank1.unsafe_set(i * 3 + j, Float64(i) * Float64(j + 1))
    var r = PLSRegression(2)
    r.fit(rank1, pseudo_random(10, 2, 3))
    assert_equal(r.x_rotations_.shape[1], 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
