"""Capability contracts for Gaussian-process estimators."""
from moxie.base import (
    ClassifierMixin,
    InputEstimator,
    ProbabilityPredictor,
    RegressorMixin,
)


trait GaussianProcessClassifierAPI(ClassifierMixin, ProbabilityPredictor):
    pass


trait GaussianProcessRegressorAPI(RegressorMixin):
    pass


trait GaussianProcessPosterior(InputEstimator):
    comptime Draws: Movable
    comptime Theta: Movable
    comptime Likelihood: Movable

    def sample_y(
        self, X: Self.Input, n_samples: Int = 1, random_state: Int = 0
    ) raises -> Self.Draws:
        ...

    def log_marginal_likelihood(
        mut self,
        theta: Optional[Self.Theta] = None,
        eval_gradient: Bool = False,
        clone_kernel: Bool = True,
    ) raises -> Self.Likelihood:
        ...


trait GaussianUncertainty(InputEstimator):
    comptime Uncertainty: Movable

    def predict(
        self, X: Self.Input, *, return_std: Bool, return_cov: Bool = False
    ) raises -> Self.Uncertainty:
        ...
