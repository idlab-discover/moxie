"""Dense classification contracts and native output for the sklearn oracle."""
from std.testing import assert_equal, assert_raises

from moxie._labels import StringTarget
from moxie.metrics import accuracy_score, hamming_loss, zero_one_loss
from numojo import Item, NDArray, NDArrayStrides, Shape, ones, zeros

comptime Weights = Optional[NDArray[DType.float64]]


def vector[dtype: DType](values: List[Scalar[dtype]]) raises -> NDArray[dtype]:
    var result = NDArray[dtype](Shape(len(values)))
    for i in range(len(values)):
        result[Item(i)] = values[i]
    return result^


def numeric[dtype: DType](
    name: String, truth: NDArray[dtype], prediction: NDArray[dtype],
    weights: Weights = None,
) raises:
    print(name, accuracy_score(truth, prediction, sample_weight=weights),
          accuracy_score(truth, prediction, normalize=False, sample_weight=weights),
          zero_one_loss(truth, prediction, sample_weight=weights),
          zero_one_loss(truth, prediction, normalize=False, sample_weight=weights),
          hamming_loss(truth, prediction, sample_weight=weights))


def main() raises:
    var weights = vector[DType.float64]([0.5, 1.0, 2.0, 4.25])
    var truth = vector[DType.int64]([0, 1, 2, 3])
    var prediction = vector[DType.int64]([0, 2, 1, 3])
    numeric("multiclass", truth, prediction)
    numeric("weighted_multiclass", truth, prediction, weights.view())
    var extremes = vector[DType.int64]([9223372036854775806, 9223372036854775807])
    var swapped = vector[DType.int64]([9223372036854775807, 9223372036854775806])
    numeric("int64_identity", extremes, swapped)

    var column = NDArray[DType.float32](Shape(4, 1))
    for i in range(4):
        column[Item(i, 0)] = Float32(i)
    numeric("column", column, vector[DType.float32]([0, 1, 3, 3]))
    var backing = vector[DType.float64]([4, 99, 2, 99, 1, 99, 0, 99])
    var reversed = backing.view_with_layout(Shape(4), NDArrayStrides(-2), 6)
    var reversed_weights = weights.view_with_layout(Shape(4), NDArrayStrides(-1), 3)
    numeric("strided", reversed, vector[DType.float64]([1, 1, 3, 4]), reversed_weights^)

    var indicators = zeros[DType.int64](Shape(4, 3))
    var predicted_indicators = zeros[DType.int64](Shape(4, 3))
    indicators[Item(0, 0)] = 1
    indicators[Item(1, 1)] = 1
    indicators[Item(3, 0)] = 1
    predicted_indicators[Item(0, 0)] = 1
    predicted_indicators[Item(1, 2)] = 1
    predicted_indicators[Item(3, 0)] = 1
    predicted_indicators[Item(3, 2)] = 1
    numeric("multilabel", indicators, predicted_indicators)
    numeric("weighted_multilabel", indicators, predicted_indicators, weights.view())

    var strings = List[String](["é", "", "a", "Ω"])
    var predicted_strings = List[String](["é", "x", "b", "Ω"])
    print("strings", accuracy_score(strings, predicted_strings, sample_weight=weights.view()),
          accuracy_score(strings, predicted_strings, normalize=False, sample_weight=weights.view()),
          zero_one_loss(strings, predicted_strings, sample_weight=weights.view()),
          zero_one_loss(strings, predicted_strings, normalize=False, sample_weight=weights.view()),
          hamming_loss(strings, predicted_strings, sample_weight=weights.view()))
    var string_column = StringTarget(strings, 4, 1)
    var string_vector = StringTarget(predicted_strings)
    print("string_target", accuracy_score(string_column, string_vector),
          accuracy_score(string_column, string_vector, normalize=False),
          zero_one_loss(string_column, string_vector),
          zero_one_loss(string_column, string_vector, normalize=False),
          hamming_loss(string_column, string_vector))
    var ints = List[Int]([0, 1, 2, 3])
    var predicted_ints = List[Int]([0, 2, 1, 3])
    print("list_int", accuracy_score(ints, predicted_ints),
          accuracy_score(ints, predicted_ints, normalize=False),
          zero_one_loss(ints, predicted_ints),
          zero_one_loss(ints, predicted_ints, normalize=False),
          hamming_loss(ints, predicted_ints))

    var two = List[Int]([0, 1])
    var both_zero = List[Int]([0, 0])
    var signed = vector[DType.float64]([4.0, -4.0])
    assert_equal(
        accuracy_score(two, both_zero, normalize=False, sample_weight=signed.view()),
        4.0,
    )
    assert_equal(
        zero_one_loss(two, both_zero, normalize=False, sample_weight=signed.view()),
        -4.0,
    )
    with assert_raises(contains="ZeroDivisionError"):
        _ = accuracy_score(two, two, sample_weight=signed.view())
    with assert_raises(contains="ZeroDivisionError"):
        _ = zero_one_loss(two, two, sample_weight=signed.view())
    with assert_raises(contains="ZeroDivisionError"):
        _ = hamming_loss(two, two, sample_weight=signed.view())
    with assert_raises(contains="ValueError"):
        _ = accuracy_score(List[Int](), List[Int]())
    with assert_raises(contains="ValueError"):
        _ = accuracy_score(two, ints)
    with assert_raises(contains="ValueError"):
        _ = accuracy_score(truth, indicators)
    with assert_raises(contains="ValueError"):
        _ = accuracy_score(indicators, zeros[DType.int64](Shape(4, 2)))
    with assert_raises(contains="ValueError"):
        _ = accuracy_score(truth, zeros[DType.int64](Shape(4, 1, 1)))
    var continuous = vector[DType.float64]([0.5, 1.5])
    with assert_raises(contains="ValueError"):
        _ = accuracy_score(continuous, continuous)
    var nonfinite = vector[DType.float64]([Float64("nan"), 1.0])
    with assert_raises(contains="ValueError"):
        _ = accuracy_score(nonfinite, nonfinite)
    with assert_raises(contains="ValueError"):
        _ = hamming_loss(two, two, sample_weight=nonfinite.view())
    with assert_raises(contains="ValueError"):
        _ = zero_one_loss(
            two, two, normalize=False, sample_weight=zeros[DType.float64](Shape(2))
        )
    with assert_raises(contains="ValueError"):
        _ = accuracy_score(two, two, sample_weight=weights.view())
    with assert_raises(contains="ValueError"):
        _ = accuracy_score(two, two, sample_weight=ones[DType.float64](Shape(2, 1)))
    var string_multioutput = StringTarget(strings, 2, 2)
    with assert_raises(contains="ValueError"):
        _ = hamming_loss(string_multioutput, string_multioutput)

    var signed_multi = vector[DType.float64]([1.0, -1.0, 1.0, -1.0])
    print(
        "signed_hamming",
        hamming_loss(indicators, predicted_indicators, sample_weight=signed_multi.view()),
        hamming_loss(indicators, indicators, sample_weight=signed_multi.view()),
    )
