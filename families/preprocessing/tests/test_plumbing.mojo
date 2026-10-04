from numojo import NDArray, NDArrayStrides, Shape, full, zeros
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.preprocessing._validation import validate_dense, missing_at, require_dense_output, validate_category_code
from moxie.preprocessing._metadata import FeatureNames
from moxie.preprocessing._contracts import FeatureMoments, FeatureStatistics, CategoryVocabulary, DenseOutputBuilder, LabelTransformer

struct VocabularyProbe(CategoryVocabulary):
    def __init__(out self):
        pass
    comptime Value = String
    def size(self) -> Int:
        return 0
    def category(self, index: Int) raises -> String:
        raise Error("Probe only")
    def lookup(self, value: String, *, is_missing: Bool = False) raises -> Int:
        raise Error("Probe only")

struct StatisticsProbe(FeatureStatistics):
    def __init__(out self):
        pass
    def compute[dtype: DType](self, x: NDArray[dtype], weights: Optional[NDArray[DType.float64]] = None) raises -> FeatureMoments:
        raise Error("Probe only")
    def merge(self, left: FeatureMoments, right: FeatureMoments) raises -> FeatureMoments:
        raise Error("Probe only")

struct OutputProbe(DenseOutputBuilder):
    def __init__(out self):
        pass
    def build[dtype: DType](self, n_samples: Int, n_features_out: Int) raises -> NDArray[dtype]:
        raise Error("Probe only")

struct LabelProbe(LabelTransformer):
    comptime Params = Int
    comptime Labels = List[String]
    comptime Encoded = List[Int]
    def __init__(out self):
        pass
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return False
    def fit[
        origin: MutOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[y_origin] y: List[String],
    ) raises -> ref[self] Self:
        raise Error("Probe only")
    def transform(self, y: List[String]) raises -> List[Int]:
        raise Error("Probe only")
    def fit_transform(mut self, y: List[String]) raises -> List[Int]:
        raise Error("Probe only")
    def inverse_transform(self, y: List[Int]) raises -> List[String]:
        raise Error("Probe only")

def arrays[dtype: DType]() raises:
    var x = full[dtype](Shape(2, 3), 1)
    validate_dense(x)
    assert_false(missing_at(x, 0, 0))
    x.unsafe_set(1, SIMD[dtype, 1](Float64("nan")))
    assert_true(missing_at(x, 0, 1))
    validate_dense(x, allow_nan=True)
    with assert_raises():
        validate_dense(x)
    x.unsafe_set(1, SIMD[dtype, 1](Float64("inf")))
    with assert_raises():
        validate_dense(x, allow_nan=True)
    with assert_raises():
        _ = missing_at(x, -1, 0)
    with assert_raises():
        _ = missing_at(x, 0, 3)
    var empty = zeros[dtype](Shape(0, 3))
    with assert_raises():
        validate_dense(empty)
    validate_dense(empty, min_samples=0)
    var vector = zeros[dtype](Shape(3))
    with assert_raises():
        validate_dense(vector)
    with assert_raises():
        _ = missing_at(vector, 0, 0)
    var backing = full[dtype](Shape(3, 4), 1)
    backing.unsafe_set(0, SIMD[dtype, 1](Float64("inf")))
    var strided = backing.view_with_layout(Shape(2, 2), NDArrayStrides(-4, 2), 9)
    validate_dense(strided)
    backing.unsafe_set(5, SIMD[dtype, 1](Float64("nan")))
    assert_true(missing_at(strided, 1, 0))
    validate_dense(strided, allow_nan=True)
    with assert_raises():
        validate_dense(strided)

def main() raises:
    arrays[DType.float32]()
    arrays[DType.float64]()
    require_dense_output(False)
    with assert_raises():
        require_dense_output(True)
    validate_category_code(0, 1)
    validate_category_code(-1, 0, allow_unknown=True)
    with assert_raises():
        validate_category_code(-1, 1)
    with assert_raises():
        validate_category_code(1, 1)
    with assert_raises():
        validate_category_code(-1, -1, allow_unknown=True)
    var names = FeatureNames()
    with assert_raises():
        _ = names.get_feature_names_out()
    names.record_fit(2, 2)
    var generated = names.get_feature_names_out()
    assert_equal(generated[0], "x0")
    assert_equal(generated[1], "x1")
    var supplied: List[String] = ["city", "東京"]
    names.record_fit(2, 2, supplied.copy())
    supplied[0] = "changed"
    var output = names.get_feature_names_out()
    assert_equal(output[0], "city")
    output[0] = "also changed"
    with assert_raises():
        _ = names.get_feature_names_out(supplied.copy())
    with assert_raises():
        names.record_fit(0, 2)
    assert_equal(names.get_feature_names_out()[0], "city")
    names.record_fit(3, 1)
    assert_equal(len(names.get_feature_names_out()), 1)
    names.reset()
    with assert_raises():
        _ = names.get_feature_names_out()
    var count = full[DType.float64](Shape(2), 2)
    var mean = full[DType.float64](Shape(2), 3)
    var variance = zeros[DType.float64](Shape(2))
    var moments = FeatureMoments(count, mean, variance)
    mean.unsafe_set(0, 9)
    assert_equal(moments.mean.item(0), 3)
    var copied = moments.copy()
    copied.mean.unsafe_set(0, 7)
    assert_equal(moments.mean.item(0), 3)
    var wrong = zeros[DType.float64](Shape(3))
    with assert_raises():
        _ = FeatureMoments(wrong, mean, variance)
    var vocab = VocabularyProbe()
    with assert_raises():
        _ = vocab.lookup("unknown")
    var statistics = StatisticsProbe()
    with assert_raises():
        _ = statistics.compute(mean)
    with assert_raises():
        _ = statistics.merge(moments, moments)
    var builder = OutputProbe()
    with assert_raises():
        _ = builder.build[DType.float32](1, 2)
    var labels = LabelProbe()
    var y: List[String] = ["cat"]
    var codes: List[Int] = [0]
    with assert_raises():
        _ = labels.fit(y)
    with assert_raises():
        _ = labels.transform(y)
    with assert_raises():
        _ = labels.fit_transform(y)
    with assert_raises():
        _ = labels.inverse_transform(codes)
    print("PASS preprocessing plumbing")
