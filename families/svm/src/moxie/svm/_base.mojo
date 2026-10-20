"""Interfaces for the svm family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import (
    ClassifierMixin,
    DecisionFunction,
    LogProbabilityPredictor,
    OutlierMixin,
    RegressorMixin,
    SampleScorer,
)


trait SupportVectorClassifier(ClassifierMixin, DecisionFunction):
    pass


trait ProbabilisticSVC(LogProbabilityPredictor, SupportVectorClassifier):
    pass


trait SupportVectorRegressor(RegressorMixin):
    pass


trait OneClassSupportVector(OutlierMixin, SampleScorer):
    pass
