"""Gaussian-process estimators, kernels and native extension contracts."""
from ._base import (
    GaussianProcessClassifierAPI,
    GaussianProcessRegressorAPI,
    GaussianProcessPosterior,
    GaussianUncertainty,
)
from ._gpc import (
    GaussianProcessClassifier,
    GaussianProcessClassifierParams,
    StructuredGaussianProcessClassifier,
)
from ._gpr import (
    DenseGaussianProcessRegressor,
    FminLBFGSB,
    GPROptimizer,
    GaussianProcessRegressor,
    GenericGaussianProcessRegressor,
    StructuredGaussianProcessRegressor,
)
from . import kernels

from ._gpr import GaussianProcessRegressorParams, LikelihoodResult, LogMarginalLikelihood
from ._plumbing import KernelContract, KernelEvaluation, LatentMoments, PredictionResult, OptimizationResult
from ._search import GaussianProcessSearchAdapter
