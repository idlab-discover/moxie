"""Interfaces for the impute family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import FitTransformer, TransformerMixin, UnsupervisedEstimator


trait Imputer(FitTransformer, TransformerMixin, UnsupervisedEstimator):
    pass
