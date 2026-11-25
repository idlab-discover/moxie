"""Dense precision/recall contracts and native output for the sklearn oracle."""
from std.math import inf, nan
from std.testing import assert_raises

from moxie._labels import StringTarget
from moxie.metrics import AveragedClassificationScores, ClassificationScores
from moxie.metrics import f1_score, fbeta_score, precision_recall_fscore_support
from moxie.metrics import precision_score, recall_score
from numojo import Item, NDArray, NDArrayStrides, Shape, zeros


def vector[dtype: DType](values: List[Scalar[dtype]]) raises -> NDArray[dtype]:
    var result = NDArray[dtype](Shape(len(values)))
    for i in range(len(values)):
        result[Item(i)] = values[i]
    return result^


def emit(name: String, scores: ClassificationScores):
    var values = String(name)
    for i in range(len(scores.precision)):
        values += " " + String(scores.precision[i])
    for i in range(len(scores.recall)):
        values += " " + String(scores.recall[i])
    for i in range(len(scores.fscore)):
        values += " " + String(scores.fscore[i])
    for i in range(len(scores.support)):
        values += " " + String(scores.support[i])
    print(values)


def emit(name: String, scores: AveragedClassificationScores):
    print(name, scores.precision, scores.recall, scores.fscore)


def single_label_cases() raises:
    var truth = List[Int]([0, 1, 2, 2, 0, 3])
    var predicted = List[Int]([0, 2, 1, 0, 0, 2])
    var weights = vector[DType.float64]([0.5, 1.0, 2.0, 4.25, 0.0, -0.25])
    var labels = List[Int]([3, 0, 4, 2, 0])
    var missing = nan[DType.float64]()
    emit("default", precision_recall_fscore_support(truth, predicted))
    emit("selected", precision_recall_fscore_support(
        truth, predicted, labels=labels.copy(), beta=2.0,
        sample_weight=weights.view(), zero_division=missing,
    ))
    for average in List[String](["micro", "macro", "weighted"]):
        emit("selected_" + average, precision_recall_fscore_support(
            truth, predicted, labels=labels.copy(), beta=0.5, average=average,
            sample_weight=weights.view(), zero_division=missing,
        ))

    var binary = List[Int]([0, 1, 1, 0, 1, 1])
    var binary_prediction = List[Int]([0, 0, 1, 1, 1, 0])
    print("binary", precision_score(binary, binary_prediction, sample_weight=weights.view()),
          recall_score(binary, binary_prediction, sample_weight=weights.view()),
          f1_score(binary, binary_prediction, sample_weight=weights.view()),
          fbeta_score(binary, binary_prediction, beta=0, sample_weight=weights.view()),
          fbeta_score(binary, binary_prediction, beta=inf[DType.float64](),
                      sample_weight=weights.view()))
    emit("binary_zero", precision_recall_fscore_support(
        binary, binary_prediction, average="binary", pos_label=0,
        labels=List[Int]([99]), sample_weight=weights.view(),
    ))
    var one = List[Int]([0, 0])
    print("absent_positive", f1_score(one, one, zero_division=1.0))
    emit("empty_selection", precision_recall_fscore_support(
        truth, predicted, labels=List[Int](),
    ))
    print("cancelled_support", f1_score(
        List[Int]([0, 1]), List[Int]([0, 0]), average="weighted",
        sample_weight=vector[DType.float64]([1, -1]), zero_division=1.0,
    ))
    print("precision_warning", precision_score(binary, List[Int]([0, 0, 0, 0, 0, 0])))
    print("recall_warning", recall_score(List[Int]([0, 0, 0, 0, 0, 0]), binary_prediction))
    print("fscore_warning", f1_score(one, one))
    print("all_undefined", recall_score(one, one, zero_division=missing))


def typed_targets() raises:
    var strings = List[String](["Ω", "cat", "Ω", "cat"])
    var predictions = List[String](["cat", "cat", "Ω", "Ω"])
    var column = StringTarget(strings, 4, 1)
    var string_vector = StringTarget(predictions)
    emit("string_column", precision_recall_fscore_support(column, string_vector))
    emit("string_list", precision_recall_fscore_support(
        strings, predictions, labels=List[String](["Ω", "absent", "cat"]),
        zero_division=1.0,
    ))
    print("string_binary", precision_score(column, string_vector, pos_label="Ω"),
          recall_score(strings, predictions, pos_label="Ω"),
          f1_score(column, string_vector, pos_label="Ω"),
          fbeta_score(strings, predictions, beta=2.0, pos_label="Ω"))
    print("string_averages", precision_score(strings, predictions, average="macro"),
          recall_score(column, string_vector, average="macro"),
          f1_score(strings, predictions, average="micro"),
          fbeta_score(column, string_vector, beta=2.0, average="weighted"))

    var backing = vector[DType.float64]([1, 99, -0.0, 99, 2, 99, 0.0])
    var reversed = backing.view_with_layout(Shape(4), NDArrayStrides(-2), 6)
    var numeric = vector[DType.float64]([1, 2, 0, 1])
    var weights = vector[DType.float64]([0.5, 1.0, 2.0, 4.0])
    var reversed_weights = weights.view_with_layout(Shape(4), NDArrayStrides(-1), 3)
    emit("strided_numeric", precision_recall_fscore_support(
        reversed, numeric, labels=List[Float64]([-0.0, 2, 1]), sample_weight=reversed_weights^,
    ))
    var extremes = vector[DType.int64]([Int64.MAX - 1, Int64.MAX, Int64.MAX - 1])
    var swapped = vector[DType.int64]([Int64.MAX, Int64.MAX - 1, Int64.MAX - 1])
    emit("int64_identity", precision_recall_fscore_support(extremes, swapped))
    var float_column = NDArray[DType.float32](Shape(3, 1))
    float_column[Item(0, 0)] = 0
    float_column[Item(1, 0)] = 1
    float_column[Item(2, 0)] = 1
    var float_vector = vector[DType.float32]([0, 1, 0])
    print("float_column", precision_score(float_column, float_vector),
          recall_score(float_column, float_vector),
          f1_score(float_column, float_vector),
          fbeta_score(float_column, float_vector, beta=2.0))


def multilabel_cases() raises:
    var truth = zeros[DType.int64](Shape(4, 3))
    truth[Item(0, 0)] = 1
    truth[Item(0, 2)] = 1
    truth[Item(1, 1)] = 1
    truth[Item(3, 0)] = 1
    var predicted = zeros[DType.int64](Shape(4, 3))
    predicted[Item(0, 0)] = 1
    predicted[Item(0, 1)] = 1
    predicted[Item(0, 2)] = 1
    predicted[Item(1, 1)] = 1
    predicted[Item(1, 2)] = 1

    var weights = vector[DType.float64]([1, 0, 2, -1])
    var signed_labels = List[Int64]([Int64(2), Int64(0)])
    emit("multilabel", precision_recall_fscore_support(truth, predicted))
    emit("multilabel_signed", precision_recall_fscore_support(
        truth, predicted, labels=signed_labels.copy(), sample_weight=weights.view(),
        zero_division=1.0,
    ))
    for average in List[String](["micro", "macro", "weighted", "samples"]):
        emit("multilabel_" + average, precision_recall_fscore_support(
            truth, predicted, average=average, labels=List[Int64]([Int64(2), Int64(0), Int64(2)]),
            sample_weight=weights.view(), zero_division=nan[DType.float64](),
        ))
    emit("empty_multilabel_samples", precision_recall_fscore_support(
        truth, predicted, labels=List[Int64](), average="samples", zero_division=1.0,
    ))
    emit("signed_samples", precision_recall_fscore_support(
        truth, predicted, labels=signed_labels.copy(), average="samples",
        sample_weight=weights.view(), zero_division=1.0,
    ))

    with assert_raises(contains="ValueError"):
        _ = f1_score(truth, predicted)
    with assert_raises(contains="ValueError"):
        _ = f1_score(truth, predicted, average="macro", labels=List[Int64]([Int64(-1)]))
    with assert_raises(contains="ValueError"):
        _ = f1_score(truth, predicted, average="macro", labels=List[Int64]([Int64(3)]))
    with assert_raises(contains="ValueError"):
        _ = f1_score(truth, zeros[DType.int64](Shape(4, 2)), average="macro")
    with assert_raises(contains="ValueError"):
        _ = f1_score(truth, zeros[DType.int64](Shape(4)), average="macro")


def invalid_inputs() raises:
    var binary = List[Int]([0, 1])
    with assert_raises(contains="ValueError"):
        _ = f1_score(binary, List[Int]([2, 3]))
    with assert_raises(contains="ValueError"):
        _ = f1_score(binary, binary, pos_label=3)
    with assert_raises(contains="ValueError"):
        _ = f1_score(binary, binary, average="samples")
    with assert_raises(contains="ValueError"):
        _ = f1_score(binary, binary, average="unknown")
    with assert_raises(contains="ValueError"):
        _ = f1_score(binary, binary, zero_division=2.0)
    with assert_raises(contains="ValueError"):
        _ = fbeta_score(binary, binary, beta=-1.0)
    with assert_raises(contains="OverflowError"):
        _ = fbeta_score(binary, binary, beta=1e200)
    with assert_raises(contains="ValueError"):
        _ = f1_score(binary, binary, sample_weight=zeros[DType.float64](Shape(2)))
    with assert_raises(contains="ValueError"):
        _ = f1_score(binary, binary, sample_weight=vector[DType.float64]([1]))
    with assert_raises(contains="ValueError"):
        _ = f1_score(binary, binary, sample_weight=vector[DType.float64]([1, inf[DType.float64]()]))
    with assert_raises(contains="ValueError"):
        _ = f1_score(List[Int](), List[Int]())
    with assert_raises(contains="ValueError"):
        _ = f1_score(binary, List[Int]([0]))
    var texts = List[String](["cat", "dog"])
    with assert_raises(contains="ValueError"):
        _ = f1_score(texts, texts)
    var continuous = vector[DType.float64]([0.5, 0.75])
    with assert_raises(contains="ValueError"):
        _ = f1_score(continuous, continuous, average="macro")


def main() raises:
    single_label_cases()
    typed_targets()
    multilabel_cases()
    invalid_inputs()
