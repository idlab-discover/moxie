"""Interfaces for the covariance family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import OutlierMixin, SampleScorer, UnsupervisedEstimator


trait CovarianceEstimator(UnsupervisedEstimator):
    comptime Precision: Movable
    comptime Distances: Movable

    def get_precision(self) raises -> Self.Precision:
        ...

    def mahalanobis(self, X: Self.Input) raises -> Self.Distances:
        ...

    def score(
        self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> Float64:
        ...


trait RobustCovarianceDetector(OutlierMixin, SampleScorer):
    pass
