"""Contracts for the manifold family."""
from ._base import EmbeddingEstimator, OutOfSampleEmbedding

# Public scikit-learn API scaffold.
from ._api import (
    MDS,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
    locally_linear_embedding,
    smacof,
    spectral_embedding,
)
from ._t_sne import TSNE, TSNEParams, TSNEMetricParams, trustworthiness
from ._classical_mds import ClassicalMDS
