"""Dense classification report contracts and samples for sklearn comparison."""
from std.math import inf, nan
from std.testing import assert_raises

from moxie._labels import StringTarget
from moxie.metrics import (
    ClassificationReport,
    classification_report,
    classification_report_dict,
)
from numojo import NDArray, NDArrayStrides, Shape


def dense[dtype: DType](
    values: List[Scalar[dtype]], dimensions: List[Int],
) raises -> NDArray[dtype]:
    var result = NDArray[dtype](Shape(dimensions))
    for i in range(len(values)):
        result.unsafe_set(i, values[i])
    return result^


def emit_text(name: String, report: String):
    print("TEXT", name)
    print(report, end="")
    print("END")


def emit_dict(name: String, report: ClassificationReport) raises:
    print("DICT", name)
    for key in report.rows.keys():
        var scores = report.rows[key]
        print(String(key, "\t", scores.precision, "\t", scores.recall, "\t",
                     scores.f1_score, "\t", scores.support))
    if report.accuracy:
        print(String("accuracy\t", report.accuracy.value()))
    print("END")


def single_label_cases() raises:
    var truth = List[Int]([0, 1, 2, 2, 0, 1])
    var predicted = List[Int]([0, 0, 2, 1, 0, 1])
    var weights = dense[DType.float64]([0.5, 1.0, 2.0, 4.25, 0.0, -0.25], [6])
    emit_text("multiclass", classification_report(truth, predicted))
    emit_dict("multiclass", classification_report_dict(truth, predicted))

    var labels = List[Int]([2, 0, 2, 3])
    var names = List[String](["left", "right", "left", "unseen"])
    var missing = nan[DType.float64]()
    emit_text("subset", classification_report(
        truth, predicted, labels=labels.copy(), target_names=names.copy(),
        sample_weight=weights.view(), zero_division=missing, digits=5,
    ))
    emit_dict("subset", classification_report_dict(
        truth, predicted, labels=labels^, target_names=names^,
        sample_weight=weights.view(), zero_division=missing,
    ))

    var superset = List[Int]([3, 2, 1, 0])
    var short_names = List[String](["unseen", "second"])
    emit_text("short_names", classification_report(
        truth, predicted, labels=superset.copy(), target_names=short_names^,
        zero_division=1.0,
    ))
    var long_names = List[String](["a", "b", "c", "d", "a long unused heading"])
    emit_text("long_names", classification_report(
        truth, predicted, labels=superset.copy(), target_names=long_names^,
        zero_division=0.0,
    ))
    emit_dict("superset", classification_report_dict(
        truth, predicted, labels=superset^, zero_division=0.0,
    ))
    emit_dict("empty_selection", classification_report_dict(
        truth, predicted, labels=List[Int](), zero_division=0.0,
    ))
    emit_dict("warnings", classification_report_dict(
        List[Int]([0, 1]), List[Int]([0, 2]),
    ))
    emit_text("ties", classification_report(
        List[Int]([0, 0]), List[Int]([0, 1]), digits=0, zero_division=0.0,
    ))
    emit_dict("cancelled_weights", classification_report_dict(
        List[Int]([0, 1]), List[Int]([0, 0]),
        sample_weight=dense[DType.float64]([1.0, -1.0], [2]), zero_division=1.0,
    ))


def typed_cases() raises:
    var strings = List[String](["Ω", "chat", "Ω", "chat"])
    var predictions = List[String](["chat", "chat", "Ω", "Ω"])
    var column = StringTarget(strings, 4, 1)
    var string_vector = StringTarget(predictions)
    emit_text("string_column", classification_report(
        column, string_vector, target_names=List[String](["🙂é very_long_label", "Ω"]),
    ))
    emit_dict("strings", classification_report_dict(strings, predictions))
    emit_dict("string_target", classification_report_dict(column, string_vector))
    emit_dict("reserved_name", classification_report_dict(
        strings, predictions, labels=List[String](["Ω"]),
        target_names=List[String](["accuracy"]),
    ))

    var backing = dense[DType.float64]([1, 99, -0.0, 99, 2, 99, 0.0], [7])
    var reversed = backing.view_with_layout(Shape(4), NDArrayStrides(-2), 6)
    var numeric = dense[DType.float64]([1, 2, 0, 1], [4])
    var weights = dense[DType.float64]([0.5, 1.0, 2.0, 4.0], [4])
    var reversed_weights = weights.view_with_layout(Shape(4), NDArrayStrides(-1), 3)
    emit_dict("strided_numeric", classification_report_dict(
        reversed, numeric, labels=List[Float64]([-0.0, 2, 1]),
        sample_weight=reversed_weights.view(), zero_division=0.0,
    ))
    emit_text("float32_column", classification_report(
        dense[DType.float32]([1, 2, 1], [3, 1]), dense[DType.float32]([2, 2, 1], [3]),
    ))
    var extremes = dense[DType.int64]([Int64.MAX - 1, Int64.MAX, Int64.MAX - 1], [3])
    var swapped = dense[DType.int64]([Int64.MAX, Int64.MAX - 1, Int64.MAX - 1], [3])
    emit_dict("int64_identity", classification_report_dict(extremes, swapped))


def multilabel_cases() raises:
    var truth = dense[DType.int64]([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], [4, 3])
    var predicted = dense[DType.int64]([1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1], [4, 3])
    var weights = dense[DType.float64]([0.5, 1.0, 2.0, 4.25], [4])
    emit_text("multilabel", classification_report(truth, predicted, zero_division=0.0))
    emit_dict("multilabel", classification_report_dict(
        truth, predicted, zero_division=0.0,
    ))
    emit_dict("multilabel_selected", classification_report_dict(
        truth, predicted, labels=List[Int64]([Int64(2), Int64(0), Int64(2)]),
        sample_weight=weights.view(), zero_division=nan[DType.float64](),
    ))
    emit_dict("multilabel_empty", classification_report_dict(
        truth, predicted, labels=List[Int64](), zero_division=0.0,
    ))


def errors() raises:
    var binary = List[Int]([0, 1])
    with assert_raises(contains="ValueError:"):
        _ = classification_report(binary, binary, target_names=List[String](["only"]))
    with assert_raises(contains="ValueError:"):
        _ = classification_report(binary, binary, labels=List[Int](), zero_division=0.0)
    with assert_raises(contains="ValueError:"):
        _ = classification_report(binary, binary, digits=-1)
    with assert_raises(contains="ValueError:"):
        _ = classification_report_dict(binary, binary, zero_division=2.0)
    with assert_raises(contains="ValueError:"):
        _ = classification_report_dict(List[Int](), List[Int]())
    with assert_raises(contains="ValueError:"):
        _ = classification_report_dict(binary, List[Int]([0]))
    with assert_raises(contains="ValueError:"):
        _ = classification_report_dict(
            binary, binary, sample_weight=dense[DType.float64]([1], [1]),
        )
    with assert_raises(contains="ValueError:"):
        _ = classification_report_dict(
            binary, binary, sample_weight=dense[DType.float64]([0, 0], [2]),
        )
    with assert_raises(contains="ValueError:"):
        _ = classification_report_dict(
            binary, binary,
            sample_weight=dense[DType.float64]([1, inf[DType.float64]()], [2]),
        )
    var indicators = dense[DType.int64]([0, 1, 1, 0], [2, 2])
    with assert_raises(contains="ValueError:"):
        _ = classification_report_dict(
            indicators, indicators, labels=List[Int64]([Int64(2)]),
        )
    with assert_raises(contains="ValueError:"):
        _ = classification_report_dict(
            indicators, dense[DType.int64]([0, 1, 0, 1, 0, 0], [2, 3]),
        )
    with assert_raises(contains="ValueError:"):
        _ = classification_report_dict(indicators, dense[DType.int64]([0, 1], [2]))
    with assert_raises(contains="ValueError:"):
        var multioutput = dense[DType.int64]([0, 1, 1, 2], [2, 2])
        _ = classification_report_dict(multioutput, multioutput)
    with assert_raises(contains="ValueError:"):
        var continuous = dense[DType.float64]([0.1, 1.5], [2])
        _ = classification_report_dict(continuous, continuous)


def main() raises:
    errors()
    single_label_cases()
    typed_cases()
    multilabel_cases()
