"""Interfaces for the decomposition family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import (
    FitTransformer,
    IncrementalTransformer,
    InverseTransformer,
    TransformerMixin,
    UnsupervisedEstimator,
)


trait DecompositionTransformer(
    FitTransformer, TransformerMixin, UnsupervisedEstimator
):
    pass


trait InvertibleDecomposition(DecompositionTransformer, InverseTransformer):
    pass


trait IncrementalDecomposition(IncrementalTransformer):
    pass
