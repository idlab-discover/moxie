"""Estimator-independent lifecycle seam; no CV evaluation engine."""
from moxie.base import BaseEstimator, clone

trait TypedScorer(Movable):
    comptime Estimator: BaseEstimator & Deinitable
    comptime Input: Movable
    comptime Target: Movable
    comptime Metadata: Movable

    def score(self, estimator: Self.Estimator, X: Self.Input,
              y: Self.Target, metadata: Self.Metadata) raises -> Float64:
        ...


trait FitAdapter(Movable):
    """Explicit fit capability, including unsupervised/weighted adapters."""
    comptime Scorer: TypedScorer

    def fit(self, mut estimator: Self.Scorer.Estimator,
            X: Self.Scorer.Input, y: Self.Scorer.Target,
            metadata: Self.Scorer.Metadata) raises:
        ...


trait ParameterAdapter(FitAdapter):
    """Apply typed partial configuration; clone nested estimator values."""
    comptime Value: Copyable & Deinitable

    def configure(self, mut estimator: Self.Scorer.Estimator,
                  params: Dict[String, Self.Value]) raises:
        """Reject unknown names and retain no fitted state or caller aliases."""
        ...


def clone_fit_score[A: FitAdapter](
    estimator: A.Scorer.Estimator, adapter: A, scorer: A.Scorer,
    X_train: A.Scorer.Input, y_train: A.Scorer.Target,
    X_test: A.Scorer.Input, y_test: A.Scorer.Target,
    fit_metadata: A.Scorer.Metadata, score_metadata: A.Scorer.Metadata,
) raises -> Float64:
    """Borrow inputs; fit an independent clone; propagate all errors."""
    var fitted = clone(estimator)
    adapter.fit(fitted, X_train, y_train, fit_metadata)
    return scorer.score(fitted, X_test, y_test, score_metadata)
