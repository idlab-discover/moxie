"""Lifecycle contracts for PolynomialFeatures and SplineTransformer."""
from std.math import isnan, abs
from numojo import NDArray, Shape, zeros
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from moxie.base import clone
from moxie.preprocessing import Preprocessor, PolynomialFeatures, SplineTransformer


def generic_transform[P: Preprocessor](model: P, X: P.Input) raises -> P.Transformed:
    return model.transform(X)


def arange[d: DType](rows: Int, cols: Int, start: Float64 = 0) raises -> NDArray[d]:
    var x = zeros[d](Shape(rows, cols))
    for k in range(rows * cols):
        x.unsafe_set(k, Scalar[d](start + Float64(k)))
    return x^


def test_polynomial_values() raises:
    var X = arange[DType.float64](3, 2)
    var model = PolynomialFeatures(2)
    var out = model.fit_transform(X)
    var expected: List[Float64] = [1, 0, 1, 0, 0, 1, 1, 2, 3, 4, 6, 9, 1, 4, 5, 16, 20, 25]
    assert_equal(out.shape[1], 6)
    for k in range(18):
        assert_equal(out.unsafe_get(k), expected[k])
    var names = model.get_feature_names_out()
    var want: List[String] = ["1", "x0", "x1", "x0^2", "x0 x1", "x1^2"]
    assert_true(names == want)
    var powers = model.powers_()
    assert_equal(powers.shape[0], 6)
    assert_equal(powers.unsafe_get(3 * 2), 2)       # x0^2 row
    assert_equal(powers.unsafe_get(4 * 2 + 1), 1)   # x0 x1 row
    var interaction = PolynomialFeatures(interaction_only=True)
    interaction.fit(X)
    var inter = generic_transform(interaction, X)
    var want_inter: List[Float64] = [1, 0, 1, 0, 1, 2, 3, 6, 1, 4, 5, 20]
    for k in range(12):
        assert_equal(inter.unsafe_get(k), want_inter[k])
    # Tuple degree: only degrees 2..3 plus the bias column.
    var ranged = PolynomialFeatures((2, 3))
    var r = ranged.fit_transform(X)
    assert_equal(ranged.n_output_features_, 8)
    assert_equal(r.unsafe_get(r.shape[1] * 2 + 0), 1)
    assert_equal(ranged.get_feature_names_out()[1], "x0^2")
    assert_equal(ranged.get_feature_names_out()[7], "x1^3")
    # Float32 and Fortran-ordered output.
    var f32 = PolynomialFeatures[DType.float32](3, order="F")
    var o32 = f32.fit_transform(arange[DType.float32](4, 2, 1))
    assert_true(o32.flags.F_CONTIGUOUS)
    assert_equal(Int(o32.strides[0]), 1)
    assert_equal(o32.item(3, 9), Float32(8 * 8 * 8))
    # Degree 0 keeps only the bias column.
    var bias = PolynomialFeatures(0)
    var ones = bias.fit_transform(X)
    assert_equal(ones.shape[1], 1)
    assert_equal(ones.item(2, 0), 1)
    # interaction_only with too few features: an empty (n, 0) expansion.
    var empty = PolynomialFeatures((2, 2), interaction_only=True, include_bias=False)
    var e = empty.fit_transform(arange[DType.float64](3, 1))
    assert_equal(e.shape[0], 3)
    assert_equal(e.shape[1], 0)
    assert_equal(len(empty.get_feature_names_out()), 0)


def test_polynomial_lifecycle() raises:
    var model = PolynomialFeatures((1, 3), interaction_only=True, include_bias=False, order="F")
    var params = model.get_params()
    assert_equal(params.degree, 3)
    assert_true(Bool(params.degree_range))
    assert_equal(params.degree_range.value()[0], 1)
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(copy.degree_range.value()[1], 3)
    assert_true(copy.interaction_only)
    assert_equal(copy.order, "F")
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(arange[DType.float64](2, 2))
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()
    with assert_raises(contains="NotFittedError"):
        _ = model.powers_()
    model.fit(arange[DType.float64](3, 3))
    assert_equal(model.n_features_in_, 3)
    assert_equal(model.n_output_features_, 7)
    with assert_raises(contains="X has 2 features, but PolynomialFeatures is expecting 3"):
        _ = model.transform(arange[DType.float64](2, 2))
    # Repeat fit replaces the learned state; set_params(degree=int) clears the tuple.
    model.set_params(degree=2, include_bias=True)
    assert_false(Bool(model.degree_range))
    model.fit(arange[DType.float64](3, 2))
    assert_equal(model.n_features_in_, 2)
    assert_equal(model.n_output_features_, 4)
    model.set_params(degree=(0, 1))
    assert_equal(model.degree_range.value()[1], 1)
    # Invalid parameters raise at fit and leave the fitted state intact.
    model.set_params(degree=-1)
    with assert_raises(contains="'degree' parameter of PolynomialFeatures"):
        model.fit(arange[DType.float64](3, 2))
    assert_equal(model.n_output_features_, 4)
    model.set_params(degree=(3, 2))
    with assert_raises(contains="min_degree <= max_degree, got (3, 2)"):
        model.fit(arange[DType.float64](3, 2))
    model.set_params(degree=0, include_bias=False)
    with assert_raises(contains="empty output array"):
        model.fit(arange[DType.float64](3, 2))
    model.set_params(degree=2, include_bias=True, order="K")
    with assert_raises(contains="'order' parameter"):
        model.fit(arange[DType.float64](3, 2))
    model.set_params(order="C")
    var nan = arange[DType.float64](2, 2)
    nan.unsafe_set(1, Float64("nan"))
    with assert_raises(contains="Input X contains NaN"):
        model.fit(nan)
    # The overflow check reports the exact Python-integer count.
    var wide = PolynomialFeatures((0, 100))
    with assert_raises(contains="would have 90548514656103281165404177077484163874504589675413336841320 features"):
        wide.fit(arange[DType.float64](1, 100))
    # Feature names: fit_named, explicit names, cleared by an unnamed fit.
    var named = PolynomialFeatures(2)
    var names: List[String] = ["a", "b"]
    named.fit_named(arange[DType.float64](2, 2), names)
    assert_equal(named.get_feature_names_out()[4], "a b")
    with assert_raises(contains="not equal"):
        var other: List[String] = ["a", "c"]
        _ = named.get_feature_names_out(other^)
    named.fit(arange[DType.float64](2, 2))
    assert_false(Bool(named.feature_names_in_))
    with assert_raises(contains="only 'default'"):
        named.set_output(transform=String("pandas"))


def test_spline_values() raises:
    # sklearn docstring: SplineTransformer(degree=2, n_knots=3) on 0..5.
    var X = arange[DType.float64](6, 1)
    var model = SplineTransformer(n_knots=3, degree=2)
    var out = model.fit_transform(X)
    var expected: List[Float64] = [0.5, 0.5, 0, 0, 0.18, 0.74, 0.08, 0, 0.02, 0.66, 0.32, 0, 0, 0.32, 0.66, 0.02, 0, 0.08, 0.74, 0.18, 0, 0, 0.5, 0.5]
    for k in range(24):
        assert_almost_equal(out.unsafe_get(k), expected[k], atol=1e-12)
    assert_equal(model.n_features_out_, 4)
    assert_equal(model.bsplines_[0].n_splines(), 4)
    assert_equal(model.bsplines_[0].t.shape[0], 7)
    assert_equal(model.bsplines_[0].t.unsafe_get(0), -5.0)
    assert_equal(model.get_feature_names_out()[3], "x0_sp_3")
    var row = model.bsplines_[0].evaluate(1.0)
    assert_almost_equal(row.unsafe_get(1), 0.74, atol=1e-12)
    # include_bias=False drops the last spline; Float32 and F order.
    var nobias = SplineTransformer[DType.float32](n_knots=3, degree=2, include_bias=False, order="F")
    var o32 = nobias.fit_transform(arange[DType.float32](6, 2))
    assert_equal(o32.shape[1], 6)
    assert_true(o32.flags.F_CONTIGUOUS)
    assert_equal(nobias.get_feature_names_out()[3], "x1_sp_0")
    # Periodic splines wrap: x and x + period encode identically.
    var periodic = SplineTransformer(n_knots=4, degree=3, extrapolation="periodic")
    periodic.fit(arange[DType.float64](6, 1))
    var c = periodic.bsplines_[0].coefficients()
    assert_equal(c.shape[0], 6)
    assert_equal(c.shape[1], 3)
    assert_equal(c.item(3, 0), 1)
    var a = periodic.transform(arange[DType.float64](1, 1, 1.25))
    var b = periodic.transform(arange[DType.float64](1, 1, 6.25))
    for k in range(3):
        assert_almost_equal(a.unsafe_get(k), b.unsafe_get(k), atol=1e-12)
    # Extrapolation modes outside the fitted range.
    var outside = arange[DType.float64](1, 1, 9.0)
    var error = SplineTransformer(extrapolation="error")
    error.fit(X)
    with assert_raises(contains="beyond the limits of the knots"):
        _ = error.transform(outside)
    var linear = SplineTransformer(extrapolation="linear")
    linear.fit(X)
    var lin = linear.transform(outside)
    var edge = linear.transform(arange[DType.float64](1, 1, 5.0))
    assert_true(lin.unsafe_get(6) > edge.unsafe_get(6))  # continued linearly past the last knot


def test_spline_lifecycle() raises:
    var model = SplineTransformer(4, 2, knots="quantile", extrapolation="continue", include_bias=False)
    var params = model.get_params()
    assert_equal(params.knots, "quantile")
    assert_false(Bool(params.knot_positions))
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(copy.n_knots, 4)
    assert_equal(copy.extrapolation, "continue")
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(arange[DType.float64](2, 1))
    var X = arange[DType.float64](8, 2)
    model.fit(X)
    assert_equal(model.n_features_in_, 2)
    assert_equal(model.n_features_out_, 8)
    with assert_raises(contains="X has 3 features, but SplineTransformer is expecting 2"):
        _ = model.transform(arange[DType.float64](2, 3))
    # Explicit knots via the overload; a string `knots` clears them again.
    var K = zeros[DType.float64](Shape(3, 2))
    var kv: List[Float64] = [0, 1, 5, 6, 14, 15]
    for k in range(6):
        K.unsafe_set(k, kv[k])
    var explicit = SplineTransformer(degree=1, knots=K)
    explicit.fit(X)
    assert_equal(explicit.bsplines_[1].t.unsafe_get(1), 1)
    assert_equal(explicit.n_features_out_, 6)
    var explicit_copy = clone(explicit)
    assert_true(Bool(explicit_copy.knot_positions))
    explicit.set_params(knots=String("uniform"))
    assert_false(Bool(explicit.knot_positions))
    explicit.set_params(knots=K)
    assert_true(Bool(explicit.knot_positions))
    var bad = zeros[DType.float64](Shape(2, 2))
    bad.unsafe_set(0, 2.0)
    explicit.set_params(knots=bad)
    with assert_raises(contains="sorted without duplicates"):
        explicit.fit(X)
    assert_equal(explicit.n_features_out_, 6)  # failed fit keeps state
    # Parameter and input errors.
    var sparse = SplineTransformer(sparse_output=True)
    with assert_raises(contains="sparse output is deferred"):
        sparse.fit(X)
    var periodic = SplineTransformer(n_knots=3, degree=3, extrapolation="periodic")
    with assert_raises(contains="Periodic splines require degree < n_knots"):
        periodic.fit(X)
    var one = SplineTransformer()
    with assert_raises(contains="minimum of 2 is required by SplineTransformer"):
        one.fit(arange[DType.float64](1, 2))
    var nan = X.copy()
    nan.unsafe_set(3, Float64("nan"))
    with assert_raises(contains="handle_missing='error'"):
        one.fit(nan)
    # handle_missing="zeros": missing cells encode as all-zero splines.
    var zeros_model = SplineTransformer(n_knots=3, degree=2, handle_missing="zeros")
    var encoded = zeros_model.fit_transform(nan)
    for c in range(4, 8):
        assert_equal(encoded.item(1, c), 0)
    # Sample weights: validation and zero-weight rows ignored by uniform knots.
    var w = zeros[DType.float64](Shape(8))
    for i in range(8):
        w.unsafe_set(i, 1.0)
    w.unsafe_set(7, 0.0)
    var weighted = SplineTransformer(n_knots=3, degree=1)
    weighted.fit(X, sample_weight=w)
    assert_equal(weighted.bsplines_[0].t.unsafe_get(3), 12)  # max over weighted rows
    var zero_w = zeros[DType.float64](Shape(8))
    with assert_raises(contains="at least one non-zero"):
        weighted.fit(X, sample_weight=zero_w)
    with assert_raises(contains="expected (8,)"):
        weighted.fit(X, sample_weight=zeros[DType.float64](Shape(3)))
    weighted.fit(X, sample_weight=2.0)
    assert_equal(weighted.bsplines_[0].t.unsafe_get(3), 14)
    _ = weighted.fit_transform(X, sample_weight=w)
    # Names.
    var names: List[String] = ["a", "b"]
    weighted.fit_named(X, names)
    assert_equal(weighted.get_feature_names_out()[3], "b_sp_0")
    weighted.fit(X)
    assert_false(Bool(weighted.feature_names_in_))


def main() raises:
    test_polynomial_values()
    test_polynomial_lifecycle()
    test_spline_values()
    test_spline_lifecycle()
    print("PASS polynomial and spline lifecycle contracts")
