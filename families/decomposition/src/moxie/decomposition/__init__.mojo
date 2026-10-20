"""Contracts for the decomposition family."""
from ._base import (
    DecompositionTransformer,
    InvertibleDecomposition,
    IncrementalDecomposition,
)

# Public scikit-learn API scaffold.
from ._api import (
    DictionaryLearning,
    MiniBatchDictionaryLearning,
    MiniBatchSparsePCA,
    SparsePCA,
    dict_learning,
    dict_learning_online,
)

from ._pca import PCA, PCAParams

from ._incremental_pca import IncrementalPCA, IncrementalPCAParams

from ._svd_engine import randomized_svd

from ._truncated_svd import TruncatedSVD, TruncatedSVDParams

from ._factor_analysis import FactorAnalysis, FactorAnalysisParams

from ._fastica import FastICA, FastICAParams, FastICAResult, fastica

from ._nmf import NMF, NMFParams, MiniBatchNMF, MiniBatchNMFParams, NMFResult, non_negative_factorization

from ._lda import LatentDirichletAllocation, LatentDirichletAllocationParams

from ._kernel_pca import KernelPCA, KernelPCAParams

from ._sparse_coding import SparseCoder, SparseCoderParams, sparse_encode
