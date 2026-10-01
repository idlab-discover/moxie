"""Shared capabilities of the dense Dummy estimators."""
from moxie.base import ClassifierMixin, LogProbabilityPredictor, RegressorMixin


trait BaselineClassifier(ClassifierMixin, LogProbabilityPredictor):
    pass


trait BaselineRegressor(RegressorMixin):
    pass
