from std.testing import assert_equal, assert_false, assert_raises, assert_true

from numojo import NDArray, NDArrayStrides, Shape, zeros
from moxie.base import clone
from moxie.feature_extraction.text import CountVectorizer, CountVectorizerParams, strip_tags
from moxie.feature_extraction.text import TfidfTransformer
from moxie.pipeline import TransformerPipeline


def split_on_comma(document: String) raises -> List[String]:
    var parts = List[String]()
    for part in document.split(","):
        parts.append(String(part))
    return parts^


def uppercase_only(document: String) raises -> String:
    return String("PRE,", document)


def test_configuration_and_lifecycle() raises:
    var model = CountVectorizer()
    var train: List[String] = ["New York new", "York City", "new place"]
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(train)
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()
    _ = model.fit(train)
    assert_true(model.__sklearn_is_fitted__())
    assert_false(model.fixed_vocabulary_())
    assert_equal(model.get_feature_names_out(), ["city", "new", "place", "york"])
    var vocabulary = model.vocabulary_()
    assert_equal(vocabulary["york"], 3)
    vocabulary["new"] = 0
    assert_equal(model.vocabulary_()["new"], 1)
    var fitted_copy = model.copy()
    var cloned = clone(model)
    assert_false(cloned.__sklearn_is_fitted__())
    var params = model.get_params()
    params.binary = True
    params.ngram_range = (2, 2)
    var refit: List[String] = ["New York new york"]
    var counts = model.set_params(params).fit_transform(refit)
    assert_equal(counts.shape, Shape(1, 2))
    assert_equal(counts.unsafe_get(0), 1)
    assert_equal(model.get_feature_names_out(), ["new york", "york new"])
    assert_equal(fitted_copy.get_feature_names_out(), ["city", "new", "place", "york"])
    var empty: List[String] = []
    with assert_raises(contains="empty vocabulary"):
        _ = model.fit(empty)
    assert_equal(model.transform(empty).shape, Shape(0, 2))
    assert_equal(model.get_feature_names_out(), ["new york", "york new"])


def test_fixed_vocabulary_and_strided_inverse() raises:
    var dictionary: Dict[String, Int] = {"tea": 1, "coffee": 0, "cocoa": 2}
    var model = CountVectorizer(vocabulary=dictionary^)
    var docs: List[String] = ["tea tea Other", "coffee cocoa"]
    var got = model.transform(docs)
    assert_equal(got.shape, Shape(2, 3))
    assert_equal(got.unsafe_get(1), 2)
    assert_equal(got.unsafe_get(3), 1)
    assert_equal(model.get_feature_names_out(), ["coffee", "tea", "cocoa"])
    assert_true(model.fixed_vocabulary_())
    assert_false(model.__sklearn_is_fitted__())
    var base = zeros[DType.int64](Shape(10))
    base.unsafe_set(3, 1)
    base.unsafe_set(5, 2)
    base.unsafe_set(6, 3)
    var view = base.view_with_layout(Shape(2, 3), NDArrayStrides(4, -1), 3)
    var terms = model.inverse_transform(view)
    assert_equal(terms[0], ["coffee"])
    assert_equal(terms[1], ["tea", "cocoa"])
    with assert_raises(contains="vocabulary width"):
        _ = model.inverse_transform(zeros[DType.int64](Shape(1, 4)))
    var duplicates: List[String] = ["a", "a"]
    var invalid = CountVectorizer(vocabulary=duplicates^)
    with assert_raises(contains="duplicate term"):
        _ = invalid.fit(docs)
    var invalid_map: Dict[String, Int] = {"a": 1}
    invalid = CountVectorizer(vocabulary=invalid_map^)
    with assert_raises(contains="unique and contiguous"):
        _ = invalid.fit(docs)


def test_typed_analysis_hooks() raises:
    var model = CountVectorizer(preprocessor=uppercase_only, tokenizer=split_on_comma,
                                token_pattern=None, ngram_range=(1, 2))
    assert_equal(model.build_preprocessor()("A,B"), "PRE,A,B")
    assert_equal(model.build_tokenizer()("x,y"), ["x", "y"])
    var build = model.build_analyzer()
    assert_equal(build("A,B"), ["PRE", "A", "B", "PRE A", "A B"])
    var callable = CountVectorizer(analyzer=split_on_comma, preprocessor=uppercase_only,
                                   token_pattern="unsupported", ngram_range=(2, 3))
    var docs: List[String] = ["A,B,A"]
    var counts = callable.fit_transform(docs)
    assert_equal(callable.get_feature_names_out(), ["A", "B"])
    assert_equal(counts.unsafe_get(0), 2)
    var stop: List[String] = ["remove"]
    var custom = CountVectorizer(stop_words=stop^)
    assert_true("remove" in custom.get_stop_words())
    assert_equal(custom.build_analyzer()("Remove retain"), ["retain"])


def test_accent_preprocessing() raises:
    var docs: List[String] = ["CAFÉ Café ßß", "cafe"]
    var model = CountVectorizer(strip_accents="ascii")
    var counts = model.fit_transform(docs)
    assert_equal(model.get_feature_names_out(), ["cafe"])
    assert_equal(counts.unsafe_get(0), 2)
    var snapshot = model.build_preprocessor()
    assert_equal(snapshot("ᴬÉ"), "Ae")

    var params = model.get_params()
    params.strip_accents = String("unicode")
    params.lowercase = False
    _ = model.set_params(params).fit(docs)
    assert_equal(model.get_feature_names_out(), ["CAFE", "Cafe", "cafe", "ßß"])
    var cloned = clone(model)
    assert_false(cloned.__sklearn_is_fitted__())
    assert_equal(cloned.build_preprocessor()("CAFÉ"), "CAFE")
    assert_equal(snapshot("CAFÉ"), "cafe")

    with assert_raises(contains="ValueError: strip_accents"):
        _ = CountVectorizer(strip_accents="unknown").build_preprocessor()
    var custom = CountVectorizer(strip_accents="ascii", preprocessor=uppercase_only)
    assert_equal(custom.build_preprocessor()("É"), "PRE,É")
    var analyzer = CountVectorizer(strip_accents="ascii", analyzer=split_on_comma)
    assert_equal(analyzer.build_analyzer()("CAFÉ,ßß"), ["CAFÉ", "ßß"])
    var invalid = CountVectorizer(strip_accents="unknown", preprocessor=uppercase_only)
    assert_equal(invalid.build_preprocessor()("É"), "PRE,É")
    with assert_raises(contains="ValueError: strip_accents"):
        _ = invalid.fit(docs)


def test_validation_and_decode() raises:
    var docs: List[String] = ["one two", "one three"]
    var model = CountVectorizer(token_pattern="[a-z]+")
    with assert_raises(contains="default pattern"):
        _ = model.fit(docs)
    _ = model.set_params(CountVectorizerParams(min_df=2, max_df=Float64(0.5)))
    with assert_raises(contains="max_df corresponds"):
        _ = model.fit(docs)
    _ = model.set_params(CountVectorizerParams(min_df=2))
    var rare: List[String] = ["two", "three"]
    with assert_raises(contains="no terms remain"):
        _ = model.fit(rare)
    _ = model.set_params(CountVectorizerParams(ngram_range=(2, 1)))
    with assert_raises(contains="ngram bounds"):
        _ = model.fit(docs)
    var invalid: List[UInt8] = [65, 0xF0, 0x90, 66, 0xC0, 0x80, 0xC2, 0xA9]
    with assert_raises(contains="UnicodeDecodeError"):
        _ = CountVectorizer().decode(Span(invalid))
    assert_equal(CountVectorizer(decode_error="replace").decode(Span(invalid)), "A�B��©")
    assert_equal(CountVectorizer(decode_error="ignore").decode(Span(invalid)), "AB©")
    assert_equal(CountVectorizer(encoding="ascii", decode_error="ignore").decode(Span(invalid)), "AB")
    assert_equal(strip_tags("<a>é</a> <> <界"), " é  <> <界")


def test_lexical_ties() raises:
    var model = CountVectorizer(max_features=3)
    var docs: List[String] = [
        "v1 v2 v3 v4 v5 v6 v7 v8 v9 v0 u1 u2 u3 u4 u5 u6 u7 u8 u9 u0",
    ]
    _ = model.fit(docs)
    assert_equal(model.get_feature_names_out(), ["u0", "u1", "u2"])


def test_count_tfidf_pipeline() raises:
    var pipeline = TransformerPipeline(CountVectorizer(), TfidfTransformer[DType.int64](norm=None))
    var docs: List[String] = ["tea coffee", "coffee coffee"]
    var got: NDArray[DType.float64] = pipeline.fit_transform(docs)
    assert_equal(got.shape, Shape(2, 2))
    assert_equal(got.unsafe_get(0), 1.0)
    assert_true(got.unsafe_get(1) > 1.0)
    assert_equal(got.unsafe_get(2), 2.0)
    assert_equal(got.unsafe_get(3), 0.0)


def main() raises:
    test_configuration_and_lifecycle()
    test_fixed_vocabulary_and_strided_inverse()
    test_typed_analysis_hooks()
    test_accent_preprocessing()
    test_validation_and_decode()
    test_lexical_ties()
    test_count_tfidf_pipeline()
    print("PASS: CountVectorizer contracts, native callbacks, byte decoding and Pipeline composition")
