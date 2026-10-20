"""Contracts for the neighbors family."""
from ._base import (
    KNeighborsQuery,
    RadiusNeighborsQuery,
    KNeighborsClassifierAPI,
    KNeighborsRegressorAPI,
    RadiusNeighborsClassifierAPI,
    RadiusNeighborsRegressorAPI,
    NeighborDensity,
    NoveltyDetector,
)

from ._contracts import (
    DenseMetric,
    DenseQueryBackend,
    TopKResult,
    RadiusResult,
    validate_matrix,
    fit_snapshot,
    validate_k,
    validate_radius_options,
)

# Implementations are composed from their algorithm owners by foundation.assembly.
from ._unsupervised import NearestNeighbors, NeighborParams
from ._classification import KNeighborsClassifier
from ._regression import KNeighborsRegressor
from ._radius_classification import RadiusNeighborsClassifier
from ._radius_regression import RadiusNeighborsRegressor
from ._query_engine import NeighborWeights, BuiltinWeights
from ._lof import LocalOutlierFactor
from ._kde import KernelDensity
from ._kd_tree import KDTree
from ._ball_tree import BallTree
from ._metrics import VALID_METRICS
