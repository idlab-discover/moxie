# SPDX-License-Identifier: BSD-3-Clause
"""Adapters for production dense classifiers."""
import numojo as nm
from ._calibration_algorithms import Array
from ._calibration_estimator import (
    ClassifierCalibrationAdapter,
    MAX_INTEGRAL_LABEL,
    _adapt_linear_svc,
    validate_integral_labels,
)
from .base import TransformerMixin
from .discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from .frozen import FrozenEstimator
from .linear_model import LogisticRegression
from .naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from .pipeline import Pipeline
from .svm import SVC
from .svm._linear import LinearSVC


def _numeric_classes(labels: List[Int]) raises -> Array:
    var result = Array(nm.Shape(len(labels)))
    for i in range(len(labels)):
        if labels[i] < -MAX_INTEGRAL_LABEL or labels[i] > MAX_INTEGRAL_LABEL:
            raise Error(
                "ValueError: classifier class is outside the supported integral label range"
            )
        result[nm.Item(i)] = Float64(labels[i])
    return result^


def adapt_classifier(
    var estimator: GaussianNB[],
) -> ClassifierCalibrationAdapter[GaussianNB[]]:
    def classes(model: GaussianNB[]) raises -> Array:
        return _numeric_classes(model.classes_)

    var adapter = ClassifierCalibrationAdapter(estimator^, classes, "GaussianNB")
    adapter._builtin_reader = True
    return adapter^


def adapt_classifier(
    var estimator: FrozenEstimator[GaussianNB[]],
) -> ClassifierCalibrationAdapter[FrozenEstimator[GaussianNB[]]]:
    def classes(model: FrozenEstimator[GaussianNB[]]) raises -> Array:
        return _numeric_classes(model.estimator().classes_)

    return ClassifierCalibrationAdapter(estimator^, classes, "GaussianNB")


def adapt_classifier(
    var estimator: MultinomialNB[],
) -> ClassifierCalibrationAdapter[MultinomialNB[]]:
    def classes(model: MultinomialNB[]) raises -> Array:
        return _numeric_classes(model.classes_)

    var adapter = ClassifierCalibrationAdapter(estimator^, classes, "MultinomialNB")
    adapter._builtin_reader = True
    return adapter^


def adapt_classifier(
    var estimator: FrozenEstimator[MultinomialNB[]],
) -> ClassifierCalibrationAdapter[FrozenEstimator[MultinomialNB[]]]:
    def classes(model: FrozenEstimator[MultinomialNB[]]) raises -> Array:
        return _numeric_classes(model.estimator().classes_)

    return ClassifierCalibrationAdapter(estimator^, classes, "MultinomialNB")


def adapt_classifier(
    var estimator: ComplementNB[],
) -> ClassifierCalibrationAdapter[ComplementNB[]]:
    def classes(model: ComplementNB[]) raises -> Array:
        return _numeric_classes(model.classes_)

    var adapter = ClassifierCalibrationAdapter(estimator^, classes, "ComplementNB")
    adapter._builtin_reader = True
    return adapter^


def adapt_classifier(
    var estimator: FrozenEstimator[ComplementNB[]],
) -> ClassifierCalibrationAdapter[FrozenEstimator[ComplementNB[]]]:
    def classes(model: FrozenEstimator[ComplementNB[]]) raises -> Array:
        return _numeric_classes(model.estimator().classes_)

    return ClassifierCalibrationAdapter(estimator^, classes, "ComplementNB")


def adapt_classifier(
    var estimator: BernoulliNB[],
) -> ClassifierCalibrationAdapter[BernoulliNB[]]:
    def classes(model: BernoulliNB[]) raises -> Array:
        return _numeric_classes(model.classes_)

    var adapter = ClassifierCalibrationAdapter(estimator^, classes, "BernoulliNB")
    adapter._builtin_reader = True
    return adapter^


def adapt_classifier(
    var estimator: FrozenEstimator[BernoulliNB[]],
) -> ClassifierCalibrationAdapter[FrozenEstimator[BernoulliNB[]]]:
    def classes(model: FrozenEstimator[BernoulliNB[]]) raises -> Array:
        return _numeric_classes(model.estimator().classes_)

    return ClassifierCalibrationAdapter(estimator^, classes, "BernoulliNB")


def adapt_classifier(
    var estimator: CategoricalNB[],
) -> ClassifierCalibrationAdapter[CategoricalNB[]]:
    def classes(model: CategoricalNB[]) raises -> Array:
        return _numeric_classes(model.classes_)

    var adapter = ClassifierCalibrationAdapter(estimator^, classes, "CategoricalNB")
    adapter._builtin_reader = True
    return adapter^


def adapt_classifier(
    var estimator: FrozenEstimator[CategoricalNB[]],
) -> ClassifierCalibrationAdapter[FrozenEstimator[CategoricalNB[]]]:
    def classes(model: FrozenEstimator[CategoricalNB[]]) raises -> Array:
        return _numeric_classes(model.estimator().classes_)

    return ClassifierCalibrationAdapter(estimator^, classes, "CategoricalNB")


def adapt_classifier(
    var estimator: LinearDiscriminantAnalysis[],
) -> ClassifierCalibrationAdapter[LinearDiscriminantAnalysis[]]:
    def classes(model: LinearDiscriminantAnalysis[]) raises -> Array:
        return _numeric_classes(model.classes_)

    var adapter = ClassifierCalibrationAdapter(estimator^, classes, "LinearDiscriminantAnalysis")
    adapter._builtin_reader = True
    return adapter^


def adapt_classifier(
    var estimator: FrozenEstimator[LinearDiscriminantAnalysis[]],
) -> ClassifierCalibrationAdapter[FrozenEstimator[LinearDiscriminantAnalysis[]]]:
    def classes(model: FrozenEstimator[LinearDiscriminantAnalysis[]]) raises -> Array:
        return _numeric_classes(model.estimator().classes_)

    return ClassifierCalibrationAdapter(estimator^, classes, "LinearDiscriminantAnalysis")


def adapt_classifier(
    var estimator: QuadraticDiscriminantAnalysis[],
) -> ClassifierCalibrationAdapter[QuadraticDiscriminantAnalysis[]]:
    def classes(model: QuadraticDiscriminantAnalysis[]) raises -> Array:
        return _numeric_classes(model.classes_)

    var adapter = ClassifierCalibrationAdapter(estimator^, classes, "QuadraticDiscriminantAnalysis")
    adapter._builtin_reader = True
    return adapter^


def adapt_classifier(
    var estimator: FrozenEstimator[QuadraticDiscriminantAnalysis[]],
) -> ClassifierCalibrationAdapter[FrozenEstimator[QuadraticDiscriminantAnalysis[]]]:
    def classes(model: FrozenEstimator[QuadraticDiscriminantAnalysis[]]) raises -> Array:
        return _numeric_classes(model.estimator().classes_)

    return ClassifierCalibrationAdapter(estimator^, classes, "QuadraticDiscriminantAnalysis")


def _logistic_classes(model: LogisticRegression[]) raises -> Array:
    var result = Array(model.classes_.shape)
    for i in range(model.classes_.size):
        var label = model.classes_.item(i)
        if label < -MAX_INTEGRAL_LABEL or label > MAX_INTEGRAL_LABEL:
            raise Error(
                "ValueError: classifier class is outside the supported integral label range"
            )
        result[nm.Item(i)] = label.cast[DType.float64]()
    return result^


def adapt_classifier(
    var estimator: LogisticRegression[],
) -> ClassifierCalibrationAdapter[LogisticRegression[]]:
    return ClassifierCalibrationAdapter(estimator^, _logistic_classes, "LogisticRegression")


def adapt_classifier(
    var estimator: FrozenEstimator[LogisticRegression[]],
) -> ClassifierCalibrationAdapter[FrozenEstimator[LogisticRegression[]]]:
    def classes(model: FrozenEstimator[LogisticRegression[]]) raises -> Array:
        return _logistic_classes(model.estimator())

    return ClassifierCalibrationAdapter(estimator^, classes, "LogisticRegression")


def adapt_classifier[Label: Copyable & Comparable & Deinitable](
    var estimator: LinearSVC[Label],
) -> ClassifierCalibrationAdapter[LinearSVC[Label]]:
    return _adapt_linear_svc(estimator^)


def _svc_classes(labels: List[Float64]) raises -> Array:
    var result = Array(nm.Shape(len(labels)))
    for i in range(result.size):
        result[nm.Item(i)] = labels[i]
    validate_integral_labels(result)
    return result^


def adapt_classifier(var estimator: SVC[]) -> ClassifierCalibrationAdapter[SVC[]]:
    """Calibrate decision scores from an SVC with a built-in feature kernel."""
    def classes(model: SVC[]) raises -> Array:
        if model.kernel == "precomputed":
            raise Error("ValueError: calibration requires SVC feature input")
        if len(model.classes_) > 2 and model.decision_function_shape == "ovo":
            raise Error("ValueError: multiclass SVC calibration requires ovr scores")
        return _svc_classes(model.classes_)

    var adapter = ClassifierCalibrationAdapter(estimator^, classes, "SVC")
    adapter._builtin_reader = True
    return adapter^


def adapt_classifier[P: TransformerMixin & Deinitable](
    var estimator: Pipeline[P, SVC[]],
) -> ClassifierCalibrationAdapter[Pipeline[P, SVC[]]]:
    """Calibrate the decision scores of a whole dense SVC pipeline."""
    def classes(model: Pipeline[P, SVC[]]) raises -> Array:
        return _svc_classes(model.estimator.classes_)

    return ClassifierCalibrationAdapter(estimator^, classes, "Pipeline")
