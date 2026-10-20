"""Interfaces for the linear_model family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import (
    ClassifierMixin,
    DecisionFunction,
    IncrementalClassifier,
    IncrementalRegressor,
    LogProbabilityPredictor,
    RegressorMixin,
)


trait LinearRegressor(RegressorMixin):
    pass


trait LinearClassifier(ClassifierMixin, DecisionFunction):
    pass


trait ProbabilisticLinearClassifier(LinearClassifier, LogProbabilityPredictor):
    pass


trait OnlineLinearRegressor(IncrementalRegressor):
    pass


trait OnlineLinearClassifier(DecisionFunction, IncrementalClassifier):
    pass
