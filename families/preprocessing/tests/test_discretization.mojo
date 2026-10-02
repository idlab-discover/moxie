"""Lifecycle contracts for KBinsDiscretizer."""
from numojo import NDArray, Shape, zeros
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie._random import RandomState
from moxie.preprocessing import Preprocessor, KBinsDiscretizer
from moxie.preprocessing._discretization import KBinsDiscretizerParams


def generic_transform[P: Preprocessor](model: P, X: P.Input) raises -> P.Transformed:
    # Compile-time trait conformance plus generic dispatch.
    return model.transform(X)


def data(rows: Int, cols: Int) raises -> NDArray[DType.float64]:
    var x = zeros[DType.float64](Shape(rows, cols))
    for i in range(rows):
        for j in range(cols):
            x.unsafe_set(i * cols + j, Float64((i * 7 + j * 3) % 11) - 2.5 * Float64(j))
    return x^


def test_params_and_clone() raises:
    var model = KBinsDiscretizer(3, encode="ordinal", strategy="uniform", subsample=None, random_state=4)
    var params = model.get_params()
    assert_equal(params.n_bins, 3)
    assert_equal(params.encode, "ordinal")
    assert_false(Bool(params.subsample))
    assert_equal(params.random_state.value(), 4)
    model.set_params(n_bins=6, quantile_method=String("linear"), subsample=Optional[Optional[Int]](Optional[Int](50)))
    assert_equal(model.n_bins, 6)
    assert_equal(model.subsample.value(), 50)
    model.set_params(subsample=Optional[Optional[Int]](Optional[Int](None)))
    assert_false(Bool(model.subsample))
    var per_feature: List[Int] = [2, 4]
    model.set_params(per_feature)
    assert_equal(model.get_params().n_bins_per_feature.value()[1], 4)
    model.fit(data(12, 2))
    assert_true(model.__sklearn_is_fitted__())
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.n_bins_per_feature.value()[0], 2)
    assert_equal(fresh.strategy, "uniform")
    assert_equal(fresh.n_bins_.size, 0)
    fresh.set_params(KBinsDiscretizerParams(5, None, "onehot-dense", "kmeans", "inverted_cdf", Optional[Int](10), None))
    assert_equal(fresh.encode, "onehot-dense")
    assert_false(Bool(fresh.n_bins_per_feature))
    with assert_raises(contains="only 'default'"):
        model.set_output(transform=String("pandas"))


def test_defaults_and_sparse() raises:
    var model = KBinsDiscretizer()
    assert_equal(model.n_bins, 5)
    assert_equal(model.encode, "onehot")
    assert_equal(model.strategy, "quantile")
    assert_equal(model.quantile_method, "averaged_inverted_cdf")
    assert_equal(model.subsample.value(), 200000)
    # sklearn's default encode='onehot' returns a sparse matrix: deferred.
    with assert_raises(contains="sparse"):
        model.fit(data(10, 2))
    assert_false(model.__sklearn_is_fitted__())


def test_unfitted_and_validation() raises:
    var model = KBinsDiscretizer(3, encode="ordinal")
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(data(4, 2))
    with assert_raises(contains="NotFittedError"):
        _ = model.inverse_transform(data(4, 2))
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()
    model.set_params(strategy=String("median"))
    with assert_raises(contains="'strategy' parameter"):
        model.fit(data(6, 2))
    model.set_params(strategy=String("quantile"), n_bins=1)
    with assert_raises(contains="'n_bins' parameter"):
        model.fit(data(6, 2))
    var bad: List[Int] = [3, 1]
    model.set_params(bad)
    with assert_raises(contains="indices 1."):
        model.fit(data(6, 2))
    model.set_params(n_bins=3, random_state=Optional[Optional[Int]](Optional[Int](-2)))
    with assert_raises(contains="'random_state' parameter"):
        model.fit(data(6, 2))


def test_refit_and_failed_fit_keep_state() raises:
    var model = KBinsDiscretizer(3, encode="onehot-dense", strategy="uniform")
    var names: List[String] = ["a", "b", "c"]
    model.fit_named(data(9, 3), names)
    assert_equal(model.get_feature_names_out()[0], "a_0.0")
    assert_equal(len(model.get_feature_names_out()), 9)
    # A failing fit (kmeans with too few samples) leaves the fitted state intact.
    model.set_params(strategy=String("kmeans"), n_bins=5)
    with assert_raises(contains="n_samples=3 should be >= n_clusters=5"):
        model.fit(data(3, 2))
    assert_equal(model.n_features_in_, 3)
    assert_equal(model.feature_names_in_.value()[2], "c")
    # A successful refit replaces everything, including the names.
    model.set_params(strategy=String("uniform"), n_bins=4)
    model.fit(data(8, 2))
    assert_equal(model.n_features_in_, 2)
    assert_false(Bool(model.feature_names_in_))
    assert_equal(len(model.bin_edges_), 2)
    assert_equal(model.get_feature_names_out()[4], "x1_0.0")
    with assert_raises(contains="X has 3 features, but KBinsDiscretizer is expecting 2 features as input."):
        _ = model.transform(data(4, 3))
    var Z = generic_transform(model, data(8, 2))
    assert_equal(Z.shape[1], 8)


def test_dtypes_and_constant_feature() raises:
    var X = zeros[DType.float32](Shape(6, 2))
    for i in range(6):
        X.unsafe_set(i * 2, 1.5)
        X.unsafe_set(i * 2 + 1, Float32(i))
    var model = KBinsDiscretizer[DType.float32, DType.float64](3, encode="ordinal", strategy="quantile")
    var Z = model.fit_transform(X)
    assert_equal(Z.shape[1], 2)
    # Constant feature: one bin with edges [-inf, inf]; its inverse is NaN.
    assert_equal(Int(model.n_bins_.unsafe_get(0)), 1)
    assert_true(model.bin_edges_[0].unsafe_get(0) < -1e308)
    var back = model.inverse_transform(Z)
    assert_true(isnan(back.unsafe_get(0)))
    var back32 = model.inverse_transform(zeros[DType.float32](Shape(2, 2)))  # any float dtype
    assert_true(isnan(back32.unsafe_get(0)))
    with assert_raises(contains="IndexError"):
        var codes = zeros[DType.float64](Shape(1, 2))
        codes.unsafe_set(1, 7)
        _ = model.inverse_transform(codes)


def test_random_state() raises:
    var X = data(40, 2)
    # Integer seeds reseed every fit; a borrowed RandomState advances.
    var seeded = KBinsDiscretizer(4, encode="ordinal", subsample=15, random_state=3)
    seeded.fit(X)
    var rng = RandomState(3)
    var borrowed = KBinsDiscretizer(4, encode="ordinal", subsample=15, random_state=99)
    borrowed.fit(X, rng)
    for j in range(2):
        for k in range(seeded.bin_edges_[j].size):
            assert_equal(seeded.bin_edges_[j].unsafe_get(k), borrowed.bin_edges_[j].unsafe_get(k))
    # random_state=None draws from the process default stream.
    var unseeded = KBinsDiscretizer(4, encode="ordinal", subsample=15)
    unseeded.fit(X)
    assert_equal(unseeded.n_bins_.size, 2)


def test_sample_weight() raises:
    var X = data(10, 2)
    var w = zeros[DType.float64](Shape(10))
    var model = KBinsDiscretizer(3, encode="ordinal", strategy="quantile")
    with assert_raises(contains="at least one non-zero"):
        model.fit(X, sample_weight=w)
    with assert_raises(contains="sample_weight.shape == (4,), expected (10,)!"):
        model.fit(X, sample_weight=zeros[DType.float64](Shape(4)))
    for i in range(10):
        w.unsafe_set(i, Float64(i % 3))
    model.fit(X, sample_weight=w)
    assert_equal(model.n_bins_.size, 2)
    model.fit(X, sample_weight=2.0)
    model.set_params(quantile_method=String("hazen"))
    with assert_raises(contains="quantile_method should either be set to"):
        model.fit(X, sample_weight=w)


def main() raises:
    test_params_and_clone()
    test_defaults_and_sparse()
    test_unfitted_and_validation()
    test_refit_and_failed_fit_keep_state()
    test_dtypes_and_constant_feature()
    test_random_state()
    test_sample_weight()
    print("PASS discretization contracts: params, clone, sparse default, validation, refit, dtypes, random_state, sample_weight")
