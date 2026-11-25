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
from .naive_bayes import GaussianNB
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

    return ClassifierCalibrationAdapter(estimator^, classes, "GaussianNB")


def adapt_classifier(
    var estimator: FrozenEstimator[GaussianNB[]],
) -> ClassifierCalibrationAdapter[FrozenEstimator[GaussianNB[]]]:
    def classes(model: FrozenEstimator[GaussianNB[]]) raises -> Array:
        return _numeric_classes(model.estimator().classes_)

    return ClassifierCalibrationAdapter(estimator^, classes, "GaussianNB")


def adapt_classifier(
    var estimator: LinearDiscriminantAnalysis[],
) -> ClassifierCalibrationAdapter[LinearDiscriminantAnalysis[]]:
    def classes(model: LinearDiscriminantAnalysis[]) raises -> Array:
        return _numeric_classes(model.classes_)

    return ClassifierCalibrationAdapter(estimator^, classes, "LinearDiscriminantAnalysis")


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

    return ClassifierCalibrationAdapter(estimator^, classes, "QuadraticDiscriminantAnalysis")


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


def adapt_classifier(
    var estimator: LinearSVC[],
) -> ClassifierCalibrationAdapter[LinearSVC[]]:
    return _adapt_linear_svc(estimator^)


def adapt_classifier[P: TransformerMixin & Deinitable](
    var estimator: Pipeline[P, SVC[]],
) -> ClassifierCalibrationAdapter[Pipeline[P, SVC[]]]:
    """Calibrate the decision scores of a whole dense SVC pipeline."""
    def classes(model: Pipeline[P, SVC[]]) raises -> Array:
        ref labels = model.estimator.classes_
        var result = Array(nm.Shape(len(labels)))
        for i in range(result.size):
            result[nm.Item(i)] = labels[i]
        validate_integral_labels(result)
        return result^

    return ClassifierCalibrationAdapter(estimator^, classes, "Pipeline")
