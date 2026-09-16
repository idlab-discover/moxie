"""Reusable tabular fitted-state checks, independent of array storage."""
from ..base import BaseEstimator


def check_is_fitted[E: BaseEstimator](estimator: E) raises:
    if not estimator.__sklearn_is_fitted__():
        raise Error("NotFittedError: call fit before inference")


struct FittedState(Copyable, Movable):
    """Compose into tabular estimators; record dimensions after successful fit.

    An estimator exposes its own public n_features_in_ from this state.
    Failed validation leaves existing state intact. Algorithm fit code decides
    whether to reset state before learning or commit a new state on success.
    """

    var n_features_in_: Optional[Int]

    def __init__(out self):
        self.n_features_in_ = None

    def is_fitted(self) -> Bool:
        return Bool(self.n_features_in_)

    def reset(mut self):
        self.n_features_in_ = None

    def record_fit(
        mut self,
        n_samples: Int,
        n_features: Int,
        *,
        n_targets: Optional[Int] = None,
    ) raises:
        if n_samples <= 0 or n_features <= 0:
            raise Error(
                "ValueError: training data must have samples and features"
            )
        if n_targets:
            if n_targets.value() != n_samples:
                raise Error("ValueError: X and y have different sample counts")
        self.n_features_in_ = n_features

    def check_features(self, n_features: Int) raises:
        if not self.n_features_in_:
            raise Error("NotFittedError: call fit before inference")
        if n_features != self.n_features_in_.value():
            raise Error("ValueError: feature count differs from fit")
