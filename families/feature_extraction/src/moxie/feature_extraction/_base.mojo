"""Interfaces for the feature_extraction family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import FitTransformer, TransformerMixin, UnsupervisedEstimator


trait FeatureExtractor(FitTransformer, TransformerMixin, UnsupervisedEstimator):
    pass
