"""Fit a two-component Gaussian mixture, then predict, score and sample."""
from numojo import Shape, zeros
from moxie.mixture import GaussianMixture


def main() raises:
    # Two well separated 2-D blobs around (0, 0) and (5, 5).
    var X = zeros[DType.float64](Shape(8, 2))
    var points: List[Float64] = [0.0, 0.1, 0.2, -0.1, -0.1, 0.0, 0.1, 0.2,
                                 5.0, 5.1, 5.2, 4.9, 4.9, 5.0, 5.1, 5.2]
    for i in range(16):
        X.unsafe_set(i, points[i])
    var model = GaussianMixture[DType.float64](2, covariance_type="diag", random_state=0)
    _ = model.fit(X)
    var labels = model.predict(X)
    print("converged:", model.converged_, "iterations:", model.n_iter_)
    print("weights:", model.weights_.unsafe_get(0), model.weights_.unsafe_get(1))
    print("first/last label:", labels.unsafe_get(0), labels.unsafe_get(7))
    print("mean log-likelihood:", model.score(X), "BIC:", model.bic(X))
    var drawn = model.sample(4)
    print("sampled labels:", drawn[1].unsafe_get(0), drawn[1].unsafe_get(3))
    if labels.unsafe_get(0) == labels.unsafe_get(7):
        raise Error("example: the two blobs must receive different labels")
