from std.testing import assert_equal, assert_raises
from numojo import zeros, Shape
from moxie.feature_extraction._mapping import OrderedVocabulary, category_key, FeatureNames
from moxie.feature_extraction._validation import require_dense_output, check_feature_matrix, require_dense_graph_output
from moxie.feature_extraction._image_contracts import PatchGeometry, GridGeometry, checked_extent_product
from moxie.feature_extraction._text_contracts import check_ngram_range, TextAnalyzer, DeterministicHasher


@fieldwise_init
struct AnalyzerProbe(TextAnalyzer):
    def analyze(self, document: String) raises -> List[String]:
        raise Error("Contract probe: analyzer deferred")


@fieldwise_init
struct HashProbe(DeterministicHasher):
    def hash_utf8(self, key: String, seed: UInt32 = 0) raises -> Int32:
        raise Error("Contract probe: hashing deferred")


@fieldwise_init
struct NamesProbe(FeatureNames):
    def get_feature_names_out(self) raises -> List[String]:
        raise Error("Contract probe: feature naming estimator deferred")


def arrays[dtype: DType]() raises:
    var x = zeros[dtype](Shape(2, 3))
    check_feature_matrix(x, 2, 3)
    var shared = x.view()
    var snapshot = x.copy()
    shared.unsafe_set(0, 7)
    assert_equal(x.unsafe_get(0), 7)
    assert_equal(snapshot.unsafe_get(0), 0)
    with assert_raises():
        check_feature_matrix(x, 3, 2)
    with assert_raises():
        check_feature_matrix(x, -1, 3)
    var empty = zeros[dtype](Shape(0, 0))
    check_feature_matrix(empty, 0, 0)
    var vector = zeros[dtype](Shape(2))
    with assert_raises():
        check_feature_matrix(vector, 1, 2)


def main() raises:
    assert_equal(category_key("city", "東京"), "city=東京")
    assert_equal(category_key("a=", "b", ""), category_key("a", "b"))
    var keys: List[String] = ["z", "city=東京", "a"]
    var vocabulary = OrderedVocabulary[String](keys)
    keys[0] = "caller changed"
    assert_equal(vocabulary.column("z"), 0)
    assert_equal(vocabulary.column("missing"), -1)
    assert_equal(vocabulary.size(), 3)
    var output_keys = vocabulary.keys()
    output_keys[0] = "output changed"
    assert_equal(vocabulary.column("z"), 0)
    var independent = vocabulary.copy()
    vocabulary = OrderedVocabulary[String](List[String]())
    assert_equal(vocabulary.size(), 0)
    assert_equal(independent.column("z"), 0)
    var integers = OrderedVocabulary[Int]([8, -2])
    assert_equal(integers.column(-2), 1)
    with assert_raises():
        var duplicate = OrderedVocabulary[String](["a", "a"])
    with assert_raises():
        require_dense_output()
    require_dense_output(False)
    require_dense_graph_output("dense")
    with assert_raises():
        require_dense_graph_output()
    with assert_raises():
        require_dense_graph_output("unknown")
    var patch = PatchGeometry(3, 4, 2, 2)
    assert_equal(patch.positions(), 6)
    var gray = patch.output_shape(6)
    assert_equal(gray[0], 6)
    assert_equal(gray[2], 2)
    var color = PatchGeometry(3, 4, 2, 2, 3).output_shape(0)
    assert_equal(color[0], 0)
    assert_equal(color[3], 3)
    assert_equal(PatchGeometry(2, 2, 2, 2).positions(), 1)
    with assert_raises():
        var bad = PatchGeometry(3, 4, 0, 2)
    with assert_raises():
        var bad = PatchGeometry(3, 4, 4, 2)
    with assert_raises():
        var bad = PatchGeometry(3, 4, 2, 2, 0)
    with assert_raises():
        _ = patch.output_shape(-1)
    with assert_raises():
        _ = checked_extent_product(Int.MAX, 2)
    var grid = GridGeometry(2, 3, 4)
    assert_equal(grid.vertices(), 24)
    assert_equal(grid.dense_shape(3)[1], 3)
    assert_equal(GridGeometry(0, 3).vertices(), 0)
    with assert_raises():
        _ = grid.dense_shape(25)
    with assert_raises():
        var bad = GridGeometry(-1, 3)
    with assert_raises():
        _ = GridGeometry(Int.MAX, 1).dense_shape(Int.MAX)
    check_ngram_range(1, 2)
    with assert_raises():
        check_ngram_range(0, 1)
    with assert_raises():
        check_ngram_range(2, 1)
    with assert_raises():
        _ = AnalyzerProbe().analyze("text")
    with assert_raises():
        _ = HashProbe().hash_utf8("text")
    with assert_raises():
        _ = NamesProbe().get_feature_names_out()
    arrays[DType.float32]()
    arrays[DType.float64]()
    print("PASS: family ownership, shapes and capability contracts")
