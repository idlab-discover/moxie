"""Dense capabilities; legacy sparse inspection contracts remain unchanged."""
from moxie.base import InputEstimator, ClassifierMixin, LogProbabilityPredictor, RegressorMixin

trait DenseTreeInspection(InputEstimator):
    comptime Leaves: Movable
    def apply(self, X: Self.Input, check_input: Bool = True) raises -> Self.Leaves: ...

trait DenseDecisionTreeClassifierAPI(ClassifierMixin, LogProbabilityPredictor, DenseTreeInspection):
    pass

trait DenseDecisionTreeRegressorAPI(RegressorMixin, DenseTreeInspection):
    pass
