"""Compose native SVC with dense self-training."""

from numojo import NDArray
from moxie.svm._c_svm import SVC
from moxie.utils.validation import check_is_fitted
from ._contracts import ScoringEstimatorAdapter
from ._self_training import (
    DecisionEstimatorAdapter,
    LogProbabilityEstimatorAdapter,
    PredictingEstimatorAdapter,
)


struct SVCAdapter[
    L: Copyable & Comparable & Deinitable, dtype: DType = DType.float64,
](
    DecisionEstimatorAdapter,
    LogProbabilityEstimatorAdapter,
    PredictingEstimatorAdapter,
    ScoringEstimatorAdapter,
):
    """Adapt an SVC configured with probability=True and a built-in feature kernel."""

    comptime Input = NDArray[Self.dtype]
    comptime Label = Self.L
    comptime Estimator = SVC[Self.Label]

    var estimator: Self.Estimator

    def __init__(out self, var estimator: Self.Estimator):
        comptime assert Self.dtype in (DType.float32, DType.float64)
        self.estimator = estimator^

    def clone_unfitted(self) raises -> Self:
        return Self(self.estimator.__sklearn_clone__())

    def fit(mut self, X: Self.Input, y: List[Self.Label]) raises:
        if self.estimator.pairwise_input():
            raise Error(
                "ValueError: self-training subsets feature rows and requires a feature kernel"
            )
        comptime if Self.dtype == DType.float32:
            _ = self.estimator.fit(rebind[NDArray[DType.float32]](X), y)
        else:
            _ = self.estimator.fit(rebind[NDArray[DType.float64]](X), y)

    def classes(self) raises -> List[Self.Label]:
        check_is_fitted(self.estimator)
        return self.estimator.classes_.copy()

    def predict(self, X: Self.Input) raises -> List[Self.Label]:
        comptime if Self.dtype == DType.float32:
            return self.estimator.predict(rebind[NDArray[DType.float32]](X))
        else:
            return self.estimator.predict(rebind[NDArray[DType.float64]](X))

    def predict_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        comptime if Self.dtype == DType.float32:
            return self.estimator.predict_proba(rebind[NDArray[DType.float32]](X))
        else:
            return self.estimator.predict_proba(rebind[NDArray[DType.float64]](X))

    def predict_log_proba(self, X: Self.Input) raises -> NDArray[DType.float64]:
        comptime if Self.dtype == DType.float32:
            return self.estimator.predict_log_proba(rebind[NDArray[DType.float32]](X))
        else:
            return self.estimator.predict_log_proba(rebind[NDArray[DType.float64]](X))

    def decision_function(self, X: Self.Input) raises -> NDArray[DType.float64]:
        comptime if Self.dtype == DType.float32:
            return self.estimator.decision_function(rebind[NDArray[DType.float32]](X))
        else:
            return self.estimator.decision_function(rebind[NDArray[DType.float64]](X))

    def score(self, X: Self.Input, y: List[Self.Label]) raises -> Float64:
        comptime if Self.dtype == DType.float32:
            return self.estimator.score(rebind[NDArray[DType.float32]](X), y)
        else:
            return self.estimator.score(rebind[NDArray[DType.float64]](X), y)
