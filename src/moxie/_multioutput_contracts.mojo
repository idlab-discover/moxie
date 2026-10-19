"""Capability contracts for dense multioutput estimators."""
from moxie.base import ClassifierMixin, ProbabilityPredictor, RegressorMixin


trait MultiOutputClassifierAPI(ClassifierMixin):
    pass


trait MultiOutputRegressorAPI(RegressorMixin):
    pass


trait MultiOutputProbability(MultiOutputClassifierAPI, ProbabilityPredictor):
    pass
