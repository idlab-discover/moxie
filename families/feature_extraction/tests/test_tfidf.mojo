from std.math import inf, nan
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from numojo import NDArray, Shape, NDArrayStrides, ones, zeros
from moxie.base import clone
from moxie.feature_extraction.text import TfidfTransformer
from moxie.pipeline import TransformerPipeline
from moxie.preprocessing import Normalizer


def test_estimator_state() raises:
    var X = ones[DType.float64](Shape(3, 2))
    var model = TfidfTransformer(norm=None, smooth_idf=False)
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(X)
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()
    with assert_raises(contains="AttributeError"):
        _ = model.idf_()
    _ = model.fit(X, y=X.copy())
    assert_equal(model.n_features_in_(), 2)
    assert_equal(model.get_feature_names_out(), ["x0", "x1"])
    var names: List[String] = ["b", "a"]
    assert_equal(model.get_feature_names_out(names.copy()), names)
    with assert_raises():
        var wrong_names: List[String] = ["a"]
        _ = model.get_feature_names_out(wrong_names^)
    var weights = model.idf_()
    weights.unsafe_set(0, 9)
    assert_equal(model.idf_().unsafe_get(0), 1)
    var output = model.transform(X, copy=False)
    output.unsafe_set(0, 8)
    assert_equal(X.unsafe_get(0), 1)

    var cloned = clone(model)
    assert_false(cloned.__sklearn_is_fitted__())
    assert_false(Bool(cloned.get_params().norm))
    assert_false(cloned.get_params().smooth_idf)
    _ = model.set_params(norm="l1", use_idf=False)
    var changed = model.transform(X)
    assert_equal(changed.unsafe_get(0), 0.5)
    _ = model.set_params(norm=None)
    assert_false(Bool(model.get_params().norm))
    var smaller = ones[DType.float64](Shape(2, 1))
    with assert_raises(contains="feature count"):
        _ = model.transform(smaller)
    _ = model.fit(smaller)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.get_n_features_in(), 1)
    with assert_raises(contains="AttributeError"):
        _ = model.idf_()


def test_invalid_inputs() raises:
    var model = TfidfTransformer()
    var X = ones[DType.float64](Shape(2, 2))
    _ = model.fit(X)
    with assert_raises(contains="rank-two"):
        _ = model.fit(zeros[DType.float64](Shape(2)))
    with assert_raises(contains="rank-two"):
        _ = model.fit(zeros[DType.float64](Shape(2, 0)))
    with assert_raises(contains="rank-two"):
        _ = model.transform(zeros[DType.float64](Shape(0, 2)))
    X.unsafe_set(0, nan[DType.float64]())
    with assert_raises(contains="nonfinite count"):
        _ = model.fit(X)
    X.unsafe_set(0, inf[DType.float64]())
    with assert_raises(contains="nonfinite count"):
        _ = model.transform(X)
    assert_equal(model.n_features_in_(), 2)
    assert_equal(model.idf_().unsafe_get(0), 1)
    X.unsafe_set(0, -1)
    _ = model.set_params(sublinear_tf=True)
    with assert_raises(contains="weighted counts must be finite"):
        _ = model.transform(X)
    _ = model.set_params(norm="max")
    with assert_raises(contains="norm must be"):
        _ = model.fit(X)


def test_strided_counts() raises:
    var base = zeros[DType.int64](Shape(12))
    base.unsafe_set(7, 2)
    base.unsafe_set(9, 3)
    var X = base.view_with_layout(Shape(2, 3), NDArrayStrides(4, -1), 7)
    var model = TfidfTransformer[DType.int64](norm=None, use_idf=False)
    var got: NDArray[DType.float64] = model.fit_transform(X)
    var expected: List[Float64] = [2, 0, 0, 0, 0, 3]
    for i in range(6):
        assert_equal(got.unsafe_get(i), expected[i])


def test_pipeline() raises:
    var X = ones[DType.int64](Shape(2, 3))
    var tfidf = TfidfTransformer[DType.int64](norm=None)
    var pipeline = TransformerPipeline(tfidf^, Normalizer(norm="l1"))
    var got: NDArray[DType.float64] = pipeline.fit_transform(X)
    assert_equal(got.unsafe_get(0), 1.0 / 3.0)


def main() raises:
    test_estimator_state()
    test_invalid_inputs()
    test_strided_counts()
    test_pipeline()
    print("PASS: TF-IDF estimator state, validation, strided counts and Pipeline composition")
