"""Compose native calibrated classifiers with dense self-training."""

from numojo import NDArray, Shape
from moxie._calibration_cv import (
    CalibratedClassifierCV,
    CalibrationEstimator,
    MAX_INTEGRAL_LABEL,
)
from ._contracts import ScoringEstimatorAdapter
from ._self_training import PredictingEstimatorAdapter


def _calibration_targets(labels: List[Int]) raises -> NDArray[DType.float64]:
    var values = NDArray[DType.float64](Shape(len(labels)))
    for i in range(len(labels)):
        var label = labels[i]
        if label < -MAX_INTEGRAL_LABEL or label > MAX_INTEGRAL_LABEL:
            raise Error("ValueError: class label is outside calibration's integral label range")
        values.unsafe_set(i, Float64(label))
    return values^


def _integer_labels(values: NDArray[DType.float64]) raises -> List[Int]:
    var labels = List[Int](capacity=values.size)
    for i in range(values.size):
        labels.append(Int(values.item(i)))
    return labels^


@fieldwise_init
struct CalibratedClassifierAdapter[E: CalibrationEstimator](
    PredictingEstimatorAdapter, ScoringEstimatorAdapter,
):
    """Expose a calibrated classifier with Float64 features and integer labels."""

    comptime Input = NDArray[DType.float64]
    comptime Label = Int

    var estimator: CalibratedClassifierCV[Self.E]

    def clone_unfitted(self) raises -> Self:
        return Self(self.estimator.__sklearn_clone__())

    def fit(mut self, X: Self.Input, y: List[Int]) raises:
        _ = self.estimator.fit(X, _calibration_targets(y))

    def classes(self) raises -> List[Int]:
        return _integer_labels(self.estimator.class_labels())

    def predict(self, X: Self.Input) raises -> List[Int]:
        return _integer_labels(self.estimator.predict(X))

    def predict_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        return self.estimator.predict_proba(X)

    def score(self, X: Self.Input, y: List[Int]) raises -> Float64:
        return self.estimator.score(X, _calibration_targets(y))
