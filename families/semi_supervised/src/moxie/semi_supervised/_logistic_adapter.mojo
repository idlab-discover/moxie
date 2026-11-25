"""Compose native logistic regression with dense self-training."""

from numojo import NDArray, Shape, zeros
from moxie.linear_model._logistic import LogisticRegression
from moxie.utils.validation import check_is_fitted
from ._contracts import ScoringEstimatorAdapter
from ._self_training import (
    DecisionEstimatorAdapter,
    LogProbabilityEstimatorAdapter,
    PredictingEstimatorAdapter,
)


def _label_array(labels: List[Int]) raises -> NDArray[DType.int64]:
    var values = zeros[DType.int64](Shape(len(labels)))
    for i in range(len(labels)):
        values.unsafe_set(i, Int64(labels[i]))
    return values^


def _label_list(values: NDArray[DType.int64]) raises -> List[Int]:
    var labels = List[Int](capacity=values.size)
    var stride = Int(values.strides[0])
    for i in range(values.size):
        labels.append(Int(values.unsafe_get(i * stride)))
    return labels^


@fieldwise_init
struct LogisticRegressionAdapter[dtype: DType = DType.float64](
    DecisionEstimatorAdapter,
    LogProbabilityEstimatorAdapter,
    PredictingEstimatorAdapter,
    ScoringEstimatorAdapter,
):
    """Adapt Float32/Float64 logistic regression with signed 64-bit labels."""

    comptime Input = NDArray[Self.dtype]
    comptime Label = Int
    comptime Estimator = LogisticRegression[Self.dtype, DType.int64]

    var estimator: Self.Estimator

    def clone_unfitted(self) raises -> Self:
        return Self(self.estimator.__sklearn_clone__())

    def fit(mut self, X: Self.Input, y: List[Int]) raises:
        _ = self.estimator.fit(X, _label_array(y))

    def classes(self) raises -> List[Int]:
        check_is_fitted(self.estimator)
        return _label_list(self.estimator.classes_)

    def predict(self, X: Self.Input) raises -> List[Int]:
        return _label_list(self.estimator.predict(X))

    def predict_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        return self.estimator.predict_proba(X).astype[DType.float64]()

    def predict_log_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        return self.estimator.predict_log_proba(X).astype[DType.float64]()

    def decision_function(self, X: Self.Input) raises -> NDArray[DType.float64]:
        return self.estimator.decision_function(X).astype[DType.float64]()

    def score(self, X: Self.Input, y: List[Int]) raises -> Float64:
        return self.estimator.score(X, _label_array(y))
