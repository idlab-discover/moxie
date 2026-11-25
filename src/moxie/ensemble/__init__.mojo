"""Contracts for the ensemble family."""
from ._base import (
    EnsembleClassifier,
    EnsembleRegressor,
    ProbabilisticEnsemble,
    DecisionEnsemble,
    ForestInspection,
    StagedPredictor,
    IsolationDetector,
)

from ._engine_api import DenseForestInspection
from ._forest_engine import BaseEnsemble

# Public scikit-learn API scaffold.
from ._api import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomTreesEmbedding,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)

from ._forest import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor

from ._hist_gradient_boosting.regressor import HistGradientBoostingRegressor, HistGradientBoostingRegressorParams
