"""Interfaces for the manifold family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import FitTransformer, TransformerMixin, UnsupervisedEstimator


trait EmbeddingEstimator(FitTransformer, UnsupervisedEstimator):
    pass


trait OutOfSampleEmbedding(EmbeddingEstimator, TransformerMixin):
    pass
