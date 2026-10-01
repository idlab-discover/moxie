"""Stable pipeline capability traits, re-exported by moxie.pipeline."""
from moxie.base import BaseEstimator, Predictor, TransformerMixin


trait PipelineEstimator(BaseEstimator):
    pass


trait PredictivePipeline(PipelineEstimator, Predictor):
    pass


trait TransformingPipeline(PipelineEstimator, TransformerMixin):
    pass
