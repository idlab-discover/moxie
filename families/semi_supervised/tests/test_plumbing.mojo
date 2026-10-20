from std.testing import assert_equal, assert_raises
from numojo import NDArray, Shape, zeros, NDArrayStrides
from moxie._dense import sum_axis2d
from moxie.semi_supervised._contracts import (
    ClassEncoding, integer_targets, validate_codes, snapshot_features,
    validate_features, validate_probability_shape, DenseAffinity,
    ProbabilityNormalizer, ZeroMassPolicy, ProbabilisticEstimatorAdapter, ScoringEstimatorAdapter,
)

@fieldwise_init
struct AffinityProbe(DenseAffinity):
    comptime Input = NDArray[DType.float32]
    def affinity(self, train: Self.Input, query: Self.Input) raises -> NDArray[DType.float64]:
        raise Error("Contract probe: affinity engine is not implemented")

@fieldwise_init
struct NormalizerProbe(ProbabilityNormalizer):
    def normalize(self, values: NDArray[DType.float64], zero_mass: Int) raises -> NDArray[DType.float64]:
        raise Error("Contract probe: normalization engine is not implemented")

struct EstimatorProbe(ScoringEstimatorAdapter):
    comptime Input = NDArray[DType.float32]
    comptime Label = Int
    var fitted: Bool
    def __init__(out self):
        self.fitted = False
    def clone_unfitted(self) raises -> Self:
        return Self()
    def fit(mut self, X: Self.Input, y: List[Int]) raises:
        raise Error("Contract probe: estimator is not implemented")
    def classes(self) raises -> List[Int]:
        raise Error("NotFittedError: contract probe")
    def predict_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        raise Error("NotFittedError: contract probe")
    def score(self, X: Self.Input, y: List[Int]) raises -> Float64:
        raise Error("NotFittedError: contract probe")


struct ProbabilityOnlyProbe(ProbabilisticEstimatorAdapter):
    comptime Input = NDArray[DType.float32]
    comptime Label = Int
    var fitted: Bool
    def __init__(out self):
        self.fitted = False
    def clone_unfitted(self) raises -> Self:
        return Self()
    def fit(mut self, X: Self.Input, y: List[Int]) raises:
        raise Error("Contract probe: estimator is not implemented")
    def classes(self) raises -> List[Int]:
        raise Error("NotFittedError: contract probe")
    def predict_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        raise Error("NotFittedError: contract probe")


def arrays[dtype: DType]() raises:
    var x = zeros[dtype](Shape(2, 2))
    x.unsafe_set(0, 3)
    var snapshot = snapshot_features(x, 2)
    var view = x.view()
    view.unsafe_set(0, 7)
    assert_equal(Float64(x.unsafe_get(0)), 7.0)
    assert_equal(Float64(snapshot.unsafe_get(0)), 3.0)
    var strided = x.view_with_layout(Shape(2, 1), NDArrayStrides(2, 1), 1)
    var strided_snapshot = snapshot_features(strided, 2)
    strided.unsafe_set(0, 9)
    assert_equal(Float64(strided_snapshot.unsafe_get(0)), 0.0)
    var totals = sum_axis2d(snapshot, 1)
    assert_equal(Float64(totals.unsafe_get(0)), 3.0)
    validate_probability_shape(snapshot, 2, 2)
    with assert_raises():
        validate_features(x, 3)
    with assert_raises():
        validate_probability_shape(x, 2, 3)
    var vector = zeros[dtype](Shape(2))
    with assert_raises():
        validate_features(vector, 2)
    with assert_raises():
        validate_probability_shape(vector, 2, 1)
    var empty = zeros[dtype](Shape(0, 2))
    validate_probability_shape(empty, 0, 2)
    with assert_raises():
        validate_features(empty, 0)


def main() raises:
    arrays[DType.float32]()
    arrays[DType.float64]()
    var classes = List[Int]([-7, 2, 11])
    var codec = ClassEncoding(classes)
    classes[0] = 99
    var labels = integer_targets(List[Int]([11, -1, -7, 2]))
    var codes = codec.encode(labels)
    assert_equal(codes[0], 2)
    assert_equal(codes[1], -1)
    assert_equal(codes[2], 0)
    var decoded = codec.decode(List[Int]([2, 0]))
    assert_equal(decoded[0], 11)
    assert_equal(decoded[1], -7)
    var independent = codec.copy()
    independent.classes[0] = 100
    assert_equal(codec.classes[0], -7)
    var strings = ClassEncoding(List[String](["amsterdam", "paris", "東京"]))
    var tags: List[Optional[String]] = [Optional(String("東京")), None]
    var string_codes = strings.encode(tags)
    assert_equal(string_codes[0], 2)
    assert_equal(string_codes[1], -1)
    var sentinel_string = ClassEncoding(List[String](["-1"]))
    var sentinel_target: List[Optional[String]] = [Optional(String("-1"))]
    var sentinel_code = sentinel_string.encode(sentinel_target)
    assert_equal(sentinel_code[0], 0)
    var empty_codes = codec.encode(List[Optional[Int]]())
    assert_equal(len(empty_codes), 0)
    with assert_raises():
        _ = ClassEncoding(List[Int]())
    with assert_raises():
        _ = ClassEncoding(List[Int]([1, 1]))
    with assert_raises():
        _ = codec.encode(integer_targets(List[Int]([123])))
    with assert_raises():
        _ = codec.decode(List[Int]([-1]))
    with assert_raises():
        validate_codes(List[Int]([3]), 3)
    with assert_raises():
        validate_codes(List[Int]([-2]), 3)
    with assert_raises():
        validate_codes(List[Int](), 0)
    var x = zeros[DType.float32](Shape(1, 1))
    var p = zeros[DType.float64](Shape(1, 1))
    with assert_raises():
        _ = AffinityProbe().affinity(x, x)
    with assert_raises():
        _ = NormalizerProbe().normalize(p, ZeroMassPolicy.PRESERVE)
    var probability_only = ProbabilityOnlyProbe()
    with assert_raises():
        _ = probability_only.predict_proba(x)
    var estimator = EstimatorProbe()
    estimator.fitted = True
    var clone = estimator.clone_unfitted()
    assert_equal(clone.fitted, False)
    with assert_raises():
        clone.fit(x, List[Int]([2]))
    with assert_raises():
        _ = clone.predict_proba(x)
    print("Dense family plumbing contracts passed")
