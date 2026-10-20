"""TruncatedSVD lifecycle, ownership, invalid settings and dtype contracts."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import abs
from numojo import NDArray, Shape, zeros, NDArrayStrides
from moxie.base import clone
from moxie.decomposition import TruncatedSVD


def matrix[dtype: DType](n: Int, p: Int) raises -> NDArray[dtype]:
    var x = zeros[dtype](Shape(n, p))
    for i in range(n):
        for j in range(p): x.unsafe_set(i * p + j, Scalar[dtype](Float64((i * 7 + j * 3) % 11) - 4 + Float64(i) / 5))
    return x^


def check[dtype: DType]() raises:
    var x = matrix[dtype](8, 4)
    var model = TruncatedSVD[dtype](2, random_state=0)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises(): _ = model.transform(x)
    with assert_raises(): _ = model.get_feature_names_out()
    var scores = model.fit_transform(x)
    assert_equal(scores.shape[0], 8)
    assert_equal(scores.shape[1], 2)
    assert_equal(model.components_.value().shape[1], 4)
    assert_equal(model.n_features_in_, 4)
    var names = model.get_feature_names_out()
    assert_equal(names[1], "truncatedsvd1")
    # Seeded fits repeat exactly; fitted state does not alias the input.
    var first = model.singular_values_.value().unsafe_get(0)
    var before = model.components_.value().unsafe_get(0)
    x.unsafe_set(0, 1000)
    assert_equal(model.components_.value().unsafe_get(0), before)
    x.unsafe_set(0, matrix[dtype](8, 4).unsafe_get(0))
    model.fit(x)
    assert_equal(model.singular_values_.value().unsafe_get(0), first)
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(copy.random_state.value(), 0)
    var params = model.get_params()
    params.algorithm = "arpack"
    copy.set_params(params)
    copy.fit(x)
    assert_true(abs(Float64(copy.singular_values_.value().unsafe_get(0)) - Float64(first)) < 1e-3 * Float64(first))
    # Negative-stride view of the same rows reversed.
    var reversed = x.view_with_layout(Shape(8, 4), NDArrayStrides(-4, 1), 28)
    copy.fit(reversed)
    assert_true(copy.__sklearn_is_fitted__())
    with assert_raises(): _ = copy.transform(zeros[dtype](Shape(2, 3)))
    with assert_raises(): _ = copy.inverse_transform(zeros[dtype](Shape(2, 3)))
    var restored = copy.inverse_transform(copy.transform(x))
    assert_equal(restored.shape[1], 4)
    copy.set_params(n_components=4)
    with assert_raises(): copy.fit(x)  # arpack requires k < min(n, p)
    copy.set_params(algorithm="randomized", n_components=5)
    with assert_raises(): copy.fit(x)  # randomized requires k <= n_features
    copy.set_params(n_components=0)
    with assert_raises(): copy.fit(x)
    copy.set_params(n_components=2, algorithm="lobpcg")
    with assert_raises(): copy.fit(x)
    copy.set_params(algorithm="randomized", power_iteration_normalizer="qr")
    with assert_raises(): copy.fit(x)
    copy.set_params(power_iteration_normalizer="QR", tol=-1.0)
    with assert_raises(): copy.fit(x)
    copy.set_params(tol=0.0, random_state=Optional[Int](-1))
    with assert_raises(): copy.fit(x)
    with assert_raises(): model.fit(matrix[dtype](8, 1))  # at least two features
    with assert_raises(): model.fit(zeros[dtype](Shape(0, 4)))
    with assert_raises(): model.set_output(transform="pandas")
    model.set_output(transform="default")
    # Rank-deficient and zero data are handled (zero variance ratio is NaN like sklearn).
    var zero = TruncatedSVD[dtype](1, algorithm="arpack")
    zero.fit(zeros[dtype](Shape(5, 3)))
    assert_equal(zero.singular_values_.value().unsafe_get(0), 0)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: TruncatedSVD contracts in Float32 and Float64")
