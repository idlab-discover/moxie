"""Compile-only conformance probes. All operation bodies deliberately raise."""
from std.testing import assert_false, assert_raises
from moxie.base import (
    BaseEstimator,
    InputEstimator,
    SupervisedEstimator,
    UnsupervisedEstimator,
    Predictor,
    SupervisedScorer,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    FitTransformer,
    InverseTransformer,
    ProbabilityPredictor,
    LogProbabilityPredictor,
    DecisionFunction,
    SampleScorer,
    DensityMixin,
    ClusterMixin,
    OutlierMixin,
    WeightedSupervisedEstimator,
    IncrementalRegressor,
    IncrementalClassifier,
    IncrementalTransformer,
)
from moxie.compose import ColumnComposer, TransformedTargetRegressorAPI


struct ContractProbe(
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    ColumnComposer,
    DecisionFunction,
    DensityMixin,
    FitTransformer,
    IncrementalClassifier,
    IncrementalRegressor,
    IncrementalTransformer,
    InputEstimator,
    InverseTransformer,
    LogProbabilityPredictor,
    OutlierMixin,
    Predictor,
    ProbabilityPredictor,
    RegressorMixin,
    SampleScorer,
    SupervisedEstimator,
    SupervisedScorer,
    TransformedTargetRegressorAPI,
    TransformerMixin,
    UnsupervisedEstimator,
    WeightedSupervisedEstimator,
):
    comptime Classes = Int
    comptime Decision = Int
    comptime Input = Int
    comptime Labels = Int
    comptime Params = Int
    comptime Prediction = Int
    comptime Probabilities = Int
    comptime SampleScores = Int
    comptime SampleWeight = Int
    comptime Target = Int
    comptime Transformed = Int
    var marker: Int

    def __init__(out self):
        self.marker = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return 0

    def set_params(mut self, params: Self.Params) raises:
        raise Error("Contract probe: no algorithm implemented")

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return False

    def fit(mut self, X: Self.Input, y: Self.Target) raises:
        raise Error("Contract probe: no algorithm implemented")

    def fit(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises:
        raise Error("Contract probe: no algorithm implemented")

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        raise Error("Contract probe: no algorithm implemented")

    def score(
        self,
        X: Self.Input,
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        raise Error("Contract probe: no algorithm implemented")

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        raise Error("Contract probe: no algorithm implemented")

    def fit_transform(
        mut self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> Self.Transformed:
        raise Error("Contract probe: no algorithm implemented")

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        raise Error("Contract probe: no algorithm implemented")

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        raise Error("Contract probe: no algorithm implemented")

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        raise Error("Contract probe: no algorithm implemented")

    def decision_function(self, X: Self.Input) raises -> Self.Decision:
        raise Error("Contract probe: no algorithm implemented")

    def score_samples(self, X: Self.Input) raises -> Self.SampleScores:
        raise Error("Contract probe: no algorithm implemented")

    def score(
        self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> Float64:
        raise Error("Contract probe: no algorithm implemented")

    def fit_predict(
        mut self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> Self.Labels:
        raise Error("Contract probe: no algorithm implemented")

    def fit(
        mut self,
        X: Self.Input,
        y: Self.Target,
        *,
        sample_weight: Self.SampleWeight,
    ) raises:
        raise Error("Contract probe: no algorithm implemented")

    def partial_fit(mut self, X: Self.Input, y: Self.Target) raises:
        raise Error("Contract probe: no algorithm implemented")

    def partial_fit(
        mut self,
        X: Self.Input,
        y: Self.Target,
        classes: Optional[Self.Classes] = None,
    ) raises:
        raise Error("Contract probe: no algorithm implemented")

    def partial_fit(
        mut self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises:
        raise Error("Contract probe: no algorithm implemented")


def main() raises:
    var probe = ContractProbe()
    assert_false(probe.__sklearn_is_fitted__())
    with assert_raises():
        probe.fit(1, 2)
    with assert_raises():
        _ = probe.predict(1)
    print("Capability conformance checks passed")
