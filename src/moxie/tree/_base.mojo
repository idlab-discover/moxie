"""Interfaces for the tree family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import (
    ClassifierMixin,
    InputEstimator,
    LogProbabilityPredictor,
    RegressorMixin,
)


trait TreeInspection(InputEstimator):
    comptime Leaves: Movable
    comptime DecisionPath: Movable

    def apply(
        self, X: Self.Input, check_input: Bool = True
    ) raises -> Self.Leaves:
        ...

    def decision_path(
        self, X: Self.Input, check_input: Bool = True
    ) raises -> Self.DecisionPath:
        ...


trait DecisionTreeClassifierAPI(
    ClassifierMixin, LogProbabilityPredictor, TreeInspection
):
    pass


trait DecisionTreeRegressorAPI(RegressorMixin, TreeInspection):
    pass
