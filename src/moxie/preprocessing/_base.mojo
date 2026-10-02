"""Interfaces for the preprocessing family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import (
    FitTransformer,
    IncrementalTransformer,
    InverseTransformer,
    SupervisedEstimator,
    TransformerMixin,
    UnsupervisedEstimator,
)


trait Preprocessor(FitTransformer, TransformerMixin, UnsupervisedEstimator):
    pass


trait InvertiblePreprocessor(InverseTransformer, Preprocessor):
    pass


trait OnlinePreprocessor(IncrementalTransformer):
    pass


trait TargetPreprocessor(SupervisedEstimator, TransformerMixin):
    pass
