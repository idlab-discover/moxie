"""Interfaces for the ensemble family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import (
    ClassifierMixin,
    DecisionFunction,
    InputEstimator,
    OutlierMixin,
    Predictor,
    ProbabilityPredictor,
    RegressorMixin,
    SampleScorer,
)


trait EnsembleClassifier(ClassifierMixin):
    pass


trait EnsembleRegressor(RegressorMixin):
    pass


trait ProbabilisticEnsemble(EnsembleClassifier, ProbabilityPredictor):
    pass


trait DecisionEnsemble(DecisionFunction, EnsembleClassifier):
    pass


trait ForestInspection(InputEstimator):
    comptime Leaves: Movable
    comptime DecisionPath: Movable

    def apply(self, X: Self.Input) raises -> Self.Leaves:
        ...

    def decision_path(self, X: Self.Input) raises -> Self.DecisionPath:
        ...


trait StagedPredictor(Predictor):
    comptime Stages: Movable

    def staged_predict(self, X: Self.Input) raises -> Self.Stages:
        ...


trait IsolationDetector(OutlierMixin, SampleScorer):
    pass
