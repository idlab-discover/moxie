"""Optional capabilities shared by the concrete mixture estimators."""
from moxie.base import (
    DensityMixin,
    InputEstimator,
    Predictor,
    ProbabilityPredictor,
)


trait MixtureDensity(DensityMixin, Predictor, ProbabilityPredictor):
    comptime Samples: Movable

    def sample(self, n_samples: Int = 1) raises -> Self.Samples:
        ...


trait InformationCriteria(InputEstimator):
    def aic(self, X: Self.Input) raises -> Float64:
        ...

    def bic(self, X: Self.Input) raises -> Float64:
        ...
