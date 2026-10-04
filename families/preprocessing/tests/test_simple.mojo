"""Lifecycle contracts for Normalizer, Binarizer and KernelCenterer."""
from numojo import NDArray, Shape, full, zeros
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.preprocessing import (
    Preprocessor, Normalizer, Binarizer, KernelCenterer, binarize, normalize,
)
from moxie.preprocessing._simple import normalize_inplace, binarize_inplace


def generic_transform[P: Preprocessor](model: P, X: P.Input) raises -> P.Transformed:
    # Compile-time trait conformance plus generic dispatch.
    return model.transform(X)


def matrix(rows: Int, cols: Int, start: Float64 = 0) raises -> NDArray[DType.float64]:
    var x = zeros[DType.float64](Shape(rows, cols))
    for k in range(rows * cols):
        x.unsafe_set(k, start + Float64(k) - 2.5)
    return x^


def test_normalizer() raises:
    var model = Normalizer("l1", copy=False)
    var params = model.get_params()
    assert_equal(params.norm, "l1")
    assert_false(params.copy_input)
    var X = matrix(3, 2)
    # Stateless: transform works before fit and is "fitted" for check_is_fitted.
    assert_true(model.__sklearn_is_fitted__())
    var unfitted = model.transform(X)
    _ = model.fit(X)
    var out = generic_transform(model, X)
    assert_equal(out.item(0, 0), unfitted.item(0, 0))
    assert_equal(model.n_features_in_, 2)
    with assert_raises(contains="features"):
        _ = model.transform(matrix(2, 3))
    var copy = clone(model)
    assert_equal(copy.norm, "l1")
    assert_equal(copy.n_features_in_, 0)
    _ = model.set_params(norm="l3")
    with assert_raises(contains="'norm' parameter"):
        _ = model.fit(X)
    _ = model.set_params(NormalizerParamsLike())
    assert_equal(model.norm, "max")
    with assert_raises(contains="only 'default'"):
        model.set_output(transform=String("pandas"))
    model.set_output(transform=String("default"))
    var names: List[String] = ["a", "b"]
    _ = model.fit_named(X, names)
    assert_equal(model.get_feature_names_out()[1], "b")
    with assert_raises(contains="not equal"):
        var other: List[String] = ["a", "c"]
        _ = model.get_feature_names_out(other^)
    _ = model.fit(X)
    assert_false(Bool(model.feature_names_in_))
    assert_equal(model.get_feature_names_out()[0], "x0")
    var inplace = X.copy()
    model.transform_inplace(inplace)
    var expected = model.transform(X)
    for k in range(6):
        assert_equal(inplace.unsafe_get(k), expected.unsafe_get(k))
    # The borrowed input is unchanged by copying transforms.
    assert_equal(X.item(0, 0), -2.5)


def NormalizerParamsLike() -> Normalizer[DType.float64].Params:
    return Normalizer[DType.float64].Params("max", True)


def test_binarizer() raises:
    var model = Binarizer[DType.float32](threshold=0.5)
    var X = full[DType.float32](Shape(2, 2), 0.5)
    X.unsafe_set(3, 0.75)
    var out = model.fit_transform(X)
    assert_equal(out.item(0, 0), 0)
    assert_equal(out.item(1, 1), 1)
    var nan = X.copy()
    nan.unsafe_set(0, Float32(Float64("nan")))
    with assert_raises(contains="NaN"):
        _ = model.transform(nan)
    binarize_inplace(X, threshold=0.6)
    assert_equal(X.item(1, 1), 1)
    assert_equal(X.item(0, 0), 0)
    var cloned = clone(model)
    assert_equal(cloned.threshold, 0.5)
    _ = model.set_params(copy=False)
    assert_false(model.get_params().copy_input)


def test_kernel_centerer() raises:
    var K = matrix(3, 3)
    var model = KernelCenterer()
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(K)
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()
    _ = model.fit(K)
    var centered = generic_transform(model, K)
    assert_equal(model.n_features_in_, 3)
    var total = Float64(0)
    for k in range(9):
        total += centered.unsafe_get(k)
    assert_true(abs(total) < 1e-12)
    assert_equal(model.get_feature_names_out()[2], "kernelcenterer2")
    with assert_raises(contains="features"):
        _ = model.transform(matrix(3, 2))
    with assert_raises(contains="square"):
        _ = model.fit(matrix(2, 3))
    # A failed fit leaves earlier learned state intact.
    assert_equal(model.n_features_in_, 3)
    var refit = clone(model)
    assert_false(refit.__sklearn_is_fitted__())


def test_functions() raises:
    var X = matrix(2, 3)
    var vector = zeros[DType.float64](Shape(3))
    with assert_raises(contains="2D"):
        _ = normalize(vector)
    with assert_raises(contains="return_norm"):
        _ = normalize(X, return_norm=True)
    var copy = X.copy()
    normalize_inplace(copy, "max", axis=0)
    var expected = normalize(X, "max", axis=0)
    for k in range(6):
        assert_equal(copy.unsafe_get(k), expected.unsafe_get(k))
    var empty = zeros[DType.float64](Shape(0, 2))
    assert_equal(binarize(empty).shape[0], 0)


def main() raises:
    test_normalizer()
    test_binarizer()
    test_kernel_centerer()
    test_functions()
    print("PASS simple estimator lifecycle contracts")
