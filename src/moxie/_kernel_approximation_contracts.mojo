"""Shared contract for dense kernel feature maps."""
from moxie.base import FitTransformer, TransformerMixin, UnsupervisedEstimator


trait KernelFeatureMap(FitTransformer, TransformerMixin, UnsupervisedEstimator):
    pass
