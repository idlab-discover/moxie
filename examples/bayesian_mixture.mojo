"""Fit an over-specified Dirichlet-process mixture: unused components shrink."""
from numojo import Shape, zeros
from moxie._random import RandomState
from moxie.mixture import BayesianGaussianMixture


def main() raises:
    # Two 2-D blobs, but the model is allowed five components.
    var rng = RandomState(0)
    var X = zeros[DType.float64](Shape(200, 2))
    for i in range(200):
        var center = 0.0 if i < 100 else 6.0
        X.unsafe_set(i * 2, center + rng.standard_normal())
        X.unsafe_set(i * 2 + 1, center + rng.standard_normal())
    var model = BayesianGaussianMixture[DType.float64](n_components=5, covariance_type="full",
                                                       weight_concentration_prior=0.01, max_iter=500, random_state=0)
    _ = model.fit(X)
    var active = 0
    for j in range(5):
        print("weight", j, model.weights_.unsafe_get(j))
        if model.weights_.unsafe_get(j) > 0.05:
            active += 1
    print("converged:", model.converged_, "active components:", active, "lower bound:", model.lower_bound_)
    if active != 2:
        raise Error("example: expected two active components")
