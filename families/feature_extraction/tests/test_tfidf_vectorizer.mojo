from std.testing import assert_equal, assert_false, assert_raises, assert_true

from moxie.base import clone
from moxie.feature_extraction.text import TfidfVectorizer
from moxie.pipeline import TransformerPipeline
from moxie.preprocessing import Normalizer
from numojo import NDArray, NDArrayStrides, Shape, zeros


def test_fit_and_refit() raises:
    var docs: List[String] = ["Tea tea", "Coffee and tea", "coffee"]
    var model = TfidfVectorizer(stop_words=String("english"), norm=None)
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(docs)
    with assert_raises(contains="NotFittedError"):
        _ = model.idf_()
    var weighted: NDArray[DType.float64] = model.fit(docs).transform(docs)
    assert_equal(weighted.shape, Shape(3, 2))
    assert_equal(model.get_feature_names_out(), ["coffee", "tea"])
    var terms = model.inverse_transform(weighted)
    assert_equal(terms[0], ["tea"])
    var counts = model.get_params()
    counts.weighting.use_idf = False
    counts.count.binary = True
    var copied = model.copy()
    var cloned = clone(model.set_params(counts))
    assert_false(cloned.__sklearn_is_fitted__())
    assert_true(cloned.get_params().count.binary)
    assert_false(cloned.get_params().weighting.use_idf)
    assert_equal(model.idf_().unsafe_get(0), copied.idf_().unsafe_get(0))

    var empty: List[String] = []
    with assert_raises():
        _ = model.fit_transform(empty)
    assert_equal(model.get_feature_names_out(), ["coffee", "tea"])
    assert_equal(model.idf_().unsafe_get(0), copied.idf_().unsafe_get(0))
    var new_docs: List[String] = ["Cocoa cocoa", "Tea"]
    weighted = model.fit_transform(new_docs)
    assert_equal(model.get_feature_names_out(), ["cocoa", "tea"])
    assert_equal(weighted.unsafe_get(0), 1.0)
    assert_equal(copied.get_feature_names_out(), ["coffee", "tea"])
    with assert_raises(contains="AttributeError"):
        _ = model.idf_()

    counts.weighting.norm = String("max")
    _ = model.set_params(counts)
    with assert_raises(contains="norm must be"):
        _ = model.fit(docs)
    assert_equal(model.get_feature_names_out(), ["cocoa", "tea"])
    assert_equal(model.transform(new_docs).unsafe_get(0), 1.0)
    with assert_raises(contains="vocabulary width"):
        _ = model.inverse_transform(zeros[DType.float64](Shape(2, 3)))
    assert_equal(model.build_analyzer()("Coffee and tea"), ["coffee", "tea"])
    assert_equal(model.decode("Tea"), "Tea")


def test_fixed_vocabulary_and_idf() raises:
    var words: List[String] = ["tea", "coffee"]
    var model = TfidfVectorizer[DType.int64](vocabulary=words.copy(), norm=None)
    var docs: List[String] = ["tea coffee tea"]
    assert_equal(model.get_feature_names_out(), words)
    assert_true(model.fixed_vocabulary_())
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(docs)
    with assert_raises(contains="rank one"):
        _ = model.set_idf_(zeros[DType.float64](Shape(1, 2)))
    with assert_raises(contains="one value per"):
        _ = model.set_idf_(zeros[DType.float64](Shape(1)))
    assert_false(model.__sklearn_is_fitted__())
    var backing = zeros[DType.float64](Shape(4))
    backing.unsafe_set(3, 7)
    backing.unsafe_set(1, 5)
    var idf = backing.view_with_layout(Shape(2), NDArrayStrides(-2), 3)
    var weighted = model.set_idf_(idf).transform(docs)
    backing.unsafe_set(3, 11)
    assert_equal(model.idf_().unsafe_get(0), 7)
    assert_equal(weighted.unsafe_get(0), 14)
    assert_equal(weighted.unsafe_get(1), 5)
    assert_equal(model.inverse_transform(weighted)[0], words)
    var returned = model.idf_()
    returned.unsafe_set(0, 99)
    assert_equal(model.idf_().unsafe_get(0), 7)
    assert_equal(model.set_idf_(returned).transform(docs).unsafe_get(0), 198)
    _ = model.fit(docs)
    assert_equal(model.idf_().unsafe_get(0), 1)
    var disabled = TfidfVectorizer(vocabulary=words^, use_idf=False)
    with assert_raises(contains="use_idf=True"):
        _ = disabled.set_idf_(idf)


def test_pipeline() raises:
    var docs: List[String] = ["tea tea coffee"]
    var vectorizer = TfidfVectorizer[DType.float32](norm=None, use_idf=False)
    var pipeline = TransformerPipeline(
        vectorizer^, Normalizer[DType.float32](norm="l1")
    )
    var result: NDArray[DType.float32] = pipeline.fit_transform(docs)
    assert_equal(result.unsafe_get(1), Float32(2) / 3)
    assert_equal(pipeline.transform(docs).unsafe_get(1), result.unsafe_get(1))


def main() raises:
    test_fit_and_refit()
    test_fixed_vocabulary_and_idf()
    test_pipeline()
    print("PASS: dense TfidfVectorizer lifecycle, IDF transfer and Pipeline composition")
