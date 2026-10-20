"""Contracts for the manifold family."""
from ._base import EmbeddingEstimator, OutOfSampleEmbedding

# Public scikit-learn API scaffold.
from ._api import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
    locally_linear_embedding,
    smacof,
    spectral_embedding,
)
from ._t_sne import trustworthiness
from ._classical_mds import ClassicalMDS
