"""Interfaces for the covariance family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import OutlierMixin, SampleScorer, UnsupervisedScorer


trait CovarianceEstimator(UnsupervisedScorer):
    comptime Precision: Movable
    comptime Distances: Movable

    def covariance_(self) raises -> Self.Input:
        """Return an independent snapshot of the fitted covariance matrix."""
        ...

    def get_precision(self) raises -> Self.Precision:
        ...

    def mahalanobis(self, X: Self.Input) raises -> Self.Distances:
        ...


trait RobustCovarianceDetector(OutlierMixin, SampleScorer):
    pass
