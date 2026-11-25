"""Contracts for the manifold family."""
from ._base import EmbeddingEstimator, OutOfSampleEmbedding

from ._locally_linear import (
    LocallyLinearEmbedding,
    LocallyLinearEmbeddingParams,
    LocallyLinearEmbeddingResult,
    locally_linear_embedding,
)
from ._t_sne import TSNE, TSNEParams, TSNEMetricParams, trustworthiness
from ._classical_mds import ClassicalMDS
from ._isomap import Isomap, IsomapParams
from ._mds import MDS, MDSMetric, MDSMetricParams, MDSParams
from ._smacof import NormalizedStress, SMACOFResult, smacof
from ._spectral_embedding import (
    SpectralEmbedding,
    SpectralEmbeddingParams,
    spectral_embedding,
)
