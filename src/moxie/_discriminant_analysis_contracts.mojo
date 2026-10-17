"""Shared discriminant-analysis capability contracts."""
from moxie.base import (
    ClassifierMixin,
    DecisionFunction,
    FitTransformer,
    LogProbabilityPredictor,
    TransformerMixin,
)


trait DiscriminantClassifier(
    ClassifierMixin, DecisionFunction, LogProbabilityPredictor
):
    pass


trait LinearDiscriminantTransformer(
    DiscriminantClassifier, FitTransformer, TransformerMixin
):
    pass
