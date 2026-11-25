"""Print diagnostics from a reproducible fit of either public mixture."""
from numojo import Shape, zeros
from std.sys import argv
from moxie.mixture import BayesianGaussianMixture, GaussianMixture


def main() raises:
    var args = argv()
    var verbose = Int(String(args[2]))
    var max_iter = Int(String(args[3]))
    var interval = Int(String(args[4]))
    var values: List[Float64] = [-4, -1, 0, 2, 6]
    var X = zeros[DType.float64](Shape(len(values), 1))
    for i in range(len(values)):
        X.unsafe_set(i, values[i])
    if String(args[1]) == "gaussian":
        var model = GaussianMixture[DType.float64](
            n_components=1, init_params="random", random_state=0, n_init=2,
            verbose=verbose, max_iter=max_iter, verbose_interval=interval,
        )
        _ = model.fit(X)
    else:
        var model = BayesianGaussianMixture[DType.float64](
            n_components=1, init_params="random", random_state=0, n_init=2,
            verbose=verbose, max_iter=max_iter, verbose_interval=interval,
        )
        _ = model.fit(X)
