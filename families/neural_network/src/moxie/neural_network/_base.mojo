"""Interfaces for the neural_network family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import (
    FitTransformer,
    IncrementalClassifier,
    IncrementalRegressor,
    LogProbabilityPredictor,
    SampleScorer,
    TransformerMixin,
    UnsupervisedEstimator,
)


trait MLPClassifierAPI(IncrementalClassifier, LogProbabilityPredictor):
    pass


trait MLPRegressorAPI(IncrementalRegressor):
    pass


trait RBMTransformer(
    FitTransformer, SampleScorer, TransformerMixin, UnsupervisedEstimator
):
    pass
