# Classification semantics from scikit-learn (BSD-3-Clause).
# See LICENSES/scikit-learn.txt.
"""Text and typed dictionary reports for dense classification."""
from std.sys import stderr

from moxie._labels import StringTarget
from moxie.metrics._classification import _Weights
from moxie.metrics._confusion import _Label, _numeric_labels
from moxie.metrics._precision_recall import (
    AveragedClassificationScores,
    _averaged_scores,
    _Counts,
    _numeric_counts,
    _per_label_scores,
    _single_label_tallies,
    _sorted_labels,
)
from moxie.utils._format import format_fixed
from numojo import NDArray


@fieldwise_init
struct ClassificationReportRow(ImplicitlyCopyable, Movable):
    """Unrounded metrics and support for a class or average."""

    var precision: Float64
    var recall: Float64
    var f1_score: Float64
    var support: Float64


@fieldwise_init
struct ClassificationReport(Movable):
    """Named class and average rows, with the scalar accuracy if present."""

    var rows: Dict[String, ClassificationReportRow]
    var accuracy: Optional[Float64]


@fieldwise_init
struct _ReportLine(Movable):
    var name: String
    var scores: ClassificationReportRow
    var accuracy: Bool


@fieldwise_init
struct _ReportLines(Movable):
    var rows: List[_ReportLine]
    var class_count: Int


def _single_label_report[Label: _Label & Writable](
    y_true: List[Label],
    y_pred: List[Label],
    labels: Optional[List[Label]],
    target_names: Optional[List[String]],
    sample_weight: _Weights,
    zero_division: Optional[Float64],
    *,
    zero: Optional[Label] = None,
) raises -> _ReportLines:
    var tallies = _single_label_tallies(y_true, y_pred, sample_weight)
    var order = labels.value().copy() if labels else _sorted_labels(tallies)
    var counts = List[_Counts](capacity=len(order))
    var names = List[String](capacity=len(order))
    var included = Dict[Label, Bool]()
    for label in order:
        var key = label.copy()
        # Preserve the display spelling while using the canonical lookup key.
        if zero and label == zero.value():
            key = zero.value().copy()
        counts.append(tallies.get(key, _Counts()))
        if labels and key in tallies:
            included[key^] = True
        names.append(String(label))
    return _report_lines(
        counts, names, target_names, Bool(labels),
        not labels or len(included) == len(tallies), None,
        sample_weight, zero_division,
    )


def _numeric_report[dtype: DType](
    y_true: NDArray[dtype],
    y_pred: NDArray[dtype],
    labels: Optional[List[Scalar[dtype]]],
    target_names: Optional[List[String]],
    sample_weight: _Weights,
    zero_division: Optional[Float64],
) raises -> _ReportLines:
    if (y_true.ndim == 2 and y_true.shape[1] > 1) or (
        y_pred.ndim == 2 and y_pred.shape[1] > 1
    ):
        var counts = _numeric_counts(y_true, y_pred, labels, 1, "", sample_weight)
        var names = List[String](capacity=len(counts))
        if labels:
            for label in labels.value():
                names.append(String(label))
        else:
            for column in range(len(counts)):
                names.append(String(column))
        var samples = _numeric_counts(
            y_true, y_pred, labels, 1, "samples", sample_weight
        )
        return _report_lines(
            counts, names, target_names, Bool(labels), False, samples^,
            sample_weight, zero_division,
        )
    return _single_label_report(
        _numeric_labels(y_true), _numeric_labels(y_pred), labels, target_names,
        sample_weight, zero_division, zero=Scalar[dtype](0),
    )


def _average_line(
    name: String, scores: AveragedClassificationScores, support: Float64,
    *, accuracy: Bool = False,
) -> _ReportLine:
    return _ReportLine(
        name,
        ClassificationReportRow(
            scores.precision, scores.recall, scores.fscore, support
        ),
        accuracy,
    )


def _report_lines(
    counts: List[_Counts],
    names: List[String],
    target_names: Optional[List[String]],
    labels_given: Bool,
    micro_is_accuracy: Bool,
    sample_counts: Optional[List[_Counts]],
    sample_weight: _Weights,
    zero_division: Optional[Float64],
) raises -> _ReportLines:
    if target_names and len(target_names.value()) != len(names):
        if not labels_given:
            raise Error(
                String(
                    "ValueError: Number of classes, ", len(names),
                    ", does not match size of target_names, ", len(target_names.value()),
                )
            )
        print(
            String(
                "UserWarning: labels size, ", len(names),
                ", does not match size of target_names, ", len(target_names.value()),
            ),
            file=stderr,
        )

    var per_label = _per_label_scores(counts, 1.0, zero_division)
    var report = List[_ReportLine](capacity=len(names) + 4)
    var support: Float64 = 0
    for value in per_label.support:
        support += value
    var class_count = len(names)
    if target_names:
        class_count = min(len(target_names.value()), class_count)
    for i in range(class_count):
        var heading = (
            target_names.value()[i].copy() if target_names else names[i].copy()
        )
        report.append(
            _ReportLine(
                heading^,
                ClassificationReportRow(
                    per_label.precision[i], per_label.recall[i],
                    per_label.fscore[i], per_label.support[i],
                ),
                False,
            )
        )

    var micro = _averaged_scores(counts, 1.0, "micro", sample_weight, zero_division)
    report.append(
        _average_line(
            "accuracy" if micro_is_accuracy else "micro avg",
            micro, support, accuracy=micro_is_accuracy,
        )
    )
    for average in List[String](["macro", "weighted"]):
        var scores = _averaged_scores(
            counts, 1.0, average, sample_weight, zero_division
        )
        report.append(_average_line(average + " avg", scores, support))
    if sample_counts:
        var samples = _averaged_scores(
            sample_counts.value(), 1.0, "samples", sample_weight, zero_division,
        )
        report.append(_average_line("samples avg", samples, support))
    return _ReportLines(report^, class_count)


def _right_aligned(value: String, width: Int) -> String:
    var padding = String()
    for _ in range(width - len(value.codepoints())):
        padding += " "
    return padding + value


def _report_text(
    lines: _ReportLines,
    target_names: Optional[List[String]],
    digits: Int,
    weighted: Bool,
) raises -> String:
    var width = max(len(String("weighted avg").codepoints()), digits)
    if target_names:
        if len(target_names.value()) == 0:
            raise Error("ValueError: a text report needs at least one target name")
        for heading in target_names.value():
            width = max(width, len(heading.codepoints()))
    else:
        if lines.class_count == 0:
            raise Error("ValueError: a text report needs at least one target name")
        for i in range(lines.class_count):
            width = max(width, len(lines.rows[i].name.codepoints()))
    var report = _right_aligned("", width)
    report += "  precision    recall  f1-score   support\n\n"
    for i in range(len(lines.rows)):
        if i == lines.class_count:
            report += "\n"
        ref line = lines.rows[i]
        report += _right_aligned(line.name, width) + " "
        if line.accuracy:
            report += _right_aligned("", 20)
        else:
            var precision = format_fixed(line.scores.precision, digits)
            var recall = format_fixed(line.scores.recall, digits)
            report += " " + _right_aligned(precision, 9) + " " + _right_aligned(recall, 9)
        report += " " + _right_aligned(format_fixed(line.scores.f1_score, digits), 9)
        var support = (
            String(line.scores.support) if weighted else String(Int(line.scores.support))
        )
        report += " " + _right_aligned(support, 9) + "\n"
    return report


def _report_dictionary(lines: _ReportLines) -> ClassificationReport:
    var report = ClassificationReport(Dict[String, ClassificationReportRow](), None)
    for i in range(len(lines.rows)):
        ref line = lines.rows[i]
        if line.name == "accuracy":
            report.accuracy = line.scores.precision
        else:
            report.rows[line.name.copy()] = line.scores
    return report^


def classification_report[dtype: DType](
    y_true: NDArray[dtype],
    y_pred: NDArray[dtype],
    *,
    labels: Optional[List[Scalar[dtype]]] = None,
    target_names: Optional[List[String]] = None,
    sample_weight: _Weights = None,
    digits: Int = 2,
    zero_division: Optional[Float64] = None,
) raises -> String:
    """Report scores for dense numeric labels or multilabel indicators."""
    if digits < 0:
        raise Error("ValueError: digits must be nonnegative")
    var lines = _numeric_report(
        y_true, y_pred, labels, target_names, sample_weight, zero_division
    )
    return _report_text(lines, target_names, digits, Bool(sample_weight))


def classification_report(
    y_true: StringTarget,
    y_pred: StringTarget,
    *,
    labels: Optional[List[String]] = None,
    target_names: Optional[List[String]] = None,
    sample_weight: _Weights = None,
    digits: Int = 2,
    zero_division: Optional[Float64] = None,
) raises -> String:
    """Report scores for a single string label per sample."""
    if y_true.columns() != 1 or y_pred.columns() != 1:
        raise Error("ValueError: string targets must have one label per sample")
    return classification_report(
        y_true.values(), y_pred.values(), labels=labels, target_names=target_names,
        sample_weight=sample_weight,
        digits=digits, zero_division=zero_division,
    )


def classification_report[Label: _Label & Writable](
    y_true: List[Label],
    y_pred: List[Label],
    *,
    labels: Optional[List[Label]] = None,
    target_names: Optional[List[String]] = None,
    sample_weight: _Weights = None,
    digits: Int = 2,
    zero_division: Optional[Float64] = None,
) raises -> String:
    """Report scores for a list of integer or string labels."""
    comptime assert Label == Int or Label == String, (
        "List targets require Int or String labels"
    )
    if digits < 0:
        raise Error("ValueError: digits must be nonnegative")
    var lines = _single_label_report(
        y_true, y_pred, labels, target_names, sample_weight, zero_division,
    )
    return _report_text(lines, target_names, digits, Bool(sample_weight))


def classification_report_dict[dtype: DType](
    y_true: NDArray[dtype],
    y_pred: NDArray[dtype],
    *,
    labels: Optional[List[Scalar[dtype]]] = None,
    target_names: Optional[List[String]] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> ClassificationReport:
    """Report scores for dense numeric labels or multilabel indicators."""
    var lines = _numeric_report(
        y_true, y_pred, labels, target_names, sample_weight, zero_division
    )
    return _report_dictionary(lines)


def classification_report_dict(
    y_true: StringTarget,
    y_pred: StringTarget,
    *,
    labels: Optional[List[String]] = None,
    target_names: Optional[List[String]] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> ClassificationReport:
    """Report scores for a single string label per sample."""
    if y_true.columns() != 1 or y_pred.columns() != 1:
        raise Error("ValueError: string targets must have one label per sample")
    return classification_report_dict(
        y_true.values(), y_pred.values(), labels=labels, target_names=target_names,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def classification_report_dict[Label: _Label & Writable](
    y_true: List[Label],
    y_pred: List[Label],
    *,
    labels: Optional[List[Label]] = None,
    target_names: Optional[List[String]] = None,
    sample_weight: _Weights = None,
    zero_division: Optional[Float64] = None,
) raises -> ClassificationReport:
    """Report scores for a list of integer or string labels."""
    comptime assert Label == Int or Label == String, (
        "List targets require Int or String labels"
    )
    var lines = _single_label_report(
        y_true, y_pred, labels, target_names, sample_weight, zero_division,
    )
    return _report_dictionary(lines)
