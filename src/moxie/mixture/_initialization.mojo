"""Initial responsibilities for mixture EM (sklearn BaseMixture._initialize_parameters).

| init_params | responsibilities (n, k) | RNG draws |
| --- | --- | --- |
| kmeans | one-hot KMeans(n_init=1) labels | KMeans k-means++ seeding |
| k-means++ | one at each selected seed row (column j = j-th seed) | k-means++ seeding on X |
| random | uniform(size=(n, k)) rows divided by their sums | n*k uniforms, row-major |
| random_from_data | one at choice(n, size=k, replace=False) rows | one permutation of n |

Seed matrices from kmeans++/random_from_data are not row-normalized (most rows
are zero), exactly as in the reference; do not impose that invariant.
"""
from numojo import NDArray, Shape, zeros
from moxie._random import RandomState
from moxie.mixture._contracts import InitializationEngine
from moxie.mixture._kmeans_init import kmeans_labels, kmeans_plusplus_indices


def initial_responsibilities[dtype: DType](X: NDArray[dtype], n_components: Int, method: String, mut rng: RandomState) raises -> NDArray[dtype]:
    """Return the (n_samples, n_components) initial responsibility matrix."""
    if X.ndim != 2:
        raise Error("ValueError: initialization expects an (n_samples, n_features) matrix")
    var n = X.shape[0]
    var k = n_components
    if k < 1 or n < k:
        raise Error(String("ValueError: Expected n_samples >= n_components but got n_components = ", k, ", n_samples = ", n))
    var resp = zeros[dtype](Shape(n, k))
    if method == "kmeans":
        var labels = kmeans_labels(X, k, rng)
        for i in range(n):
            resp.unsafe_set(i * k + labels[i], 1)
    elif method == "k-means++":
        var seeds = kmeans_plusplus_indices(X, k, rng)
        for j in range(k):
            resp.unsafe_set(seeds[j] * k + j, 1)
    elif method == "random":
        for i in range(n):
            for j in range(k):
                resp.unsafe_set(i * k + j, rng.random_sample().cast[dtype]())
        for i in range(n):
            # xp.sum(resp, axis=1) in the dtype, then elementwise division.
            var total = Scalar[dtype](0)
            for j in range(k):
                total += resp.unsafe_get(i * k + j)
            for j in range(k):
                resp.unsafe_set(i * k + j, resp.unsafe_get(i * k + j) / total)
    elif method == "random_from_data":
        var indices = rng.choice(n, k, False)
        for j in range(k):
            resp.unsafe_set(Int(indices.unsafe_get(j)) * k + j, 1)
    else:
        raise Error(String("ValueError: The 'init_params' parameter must be one of 'kmeans', 'k-means++', 'random' or 'random_from_data'; got '", method, "'"))
    return resp^


@fieldwise_init
struct MixtureInitializer[dtype_: DType](InitializationEngine, Copyable):
    """InitializationEngine implementing all four sklearn initializers."""

    comptime dtype = Self.dtype_

    def responsibilities(self, x: NDArray[Self.dtype], components: Int, method: String, mut rng: RandomState) raises -> NDArray[Self.dtype]:
        return initial_responsibilities(x, components, method, rng)
