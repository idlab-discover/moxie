# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense neighborhood preservation diagnostics."""
from numojo import NDArray, zeros, Shape
from std.builtin.sort import sort
from std.math import abs, sqrt, isfinite
from moxie.manifold._plumbing import validate_matrix, validate_pairwise


def _distance[dtype: DType](x: NDArray[dtype], i: Int, j: Int, metric: String) raises -> Float64:
    if metric == "precomputed":
        return Float64(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))
    var result: Float64 = 0
    var aa: Float64 = 0
    var bb: Float64 = 0
    for c in range(x.shape[1]):
        var a = Float64(x.unsafe_get(i * Int(x.strides[0]) + c * Int(x.strides[1])))
        var b = Float64(x.unsafe_get(j * Int(x.strides[0]) + c * Int(x.strides[1])))
        if metric == "cosine":
            result += a * b
            aa += a * a
            bb += b * b
        elif metric == "manhattan" or metric == "cityblock" or metric == "l1":
            result += abs(a - b)
        else:
            result += (a - b) * (a - b)
    if metric == "cosine":
        if aa == 0 or bb == 0:
            return 1
        result = max(Float64(0), min(Float64(2), 1 - result / sqrt(aa) / sqrt(bb)))
    if not isfinite(result):
        raise Error("ValueError: pairwise distance overflow")
    return result


def _neighbor_order[dtype: DType](x: NDArray[dtype], row: Int, metric: String) raises -> NDArray[DType.int64]:
    var pairs = List[Tuple[Float64, Int]]()
    for j in range(x.shape[0]):
        if j != row:
            pairs.append((_distance(x, row, j, metric), j))
    sort(Span(pairs))
    var indices = zeros[DType.int64](Shape(len(pairs)))
    for j in range(len(pairs)):
        indices.unsafe_set(j, Int64(pairs[j][1]))
    return indices^


def trustworthiness[dtype: DType, embedded_dtype: DType](
    X: NDArray[dtype], X_embedded: NDArray[embedded_dtype], *,
    n_neighbors: Int = 5, metric: String = "euclidean",
) raises -> Float64:
    """Rank penalty for unexpected embedding neighbors; fewer than n/2 neighbors."""
    validate_matrix(X)
    validate_matrix(X_embedded)
    var n = X.shape[0]
    if X_embedded.shape[0] != n:
        raise Error("ValueError: X and X_embedded must have the same sample count")
    if n_neighbors < 1 or n_neighbors >= (n + 1) // 2:
        raise Error("ValueError: n_neighbors must be positive and less than n_samples / 2")
    if metric == "precomputed":
        validate_pairwise(X, nonnegative=True, symmetric=False)
    elif metric != "euclidean" and metric != "l2" and metric != "sqeuclidean" and metric != "manhattan" and metric != "cityblock" and metric != "l1" and metric != "cosine":
        raise Error("ValueError: unsupported dense trustworthiness metric")
    var penalty: Float64 = 0
    for i in range(n):
        var original = _neighbor_order(X, i, metric)
        var embedded = _neighbor_order(X_embedded, i, "euclidean")
        var ranks = zeros[DType.int64](Shape(n))
        for j in range(n - 1):
            ranks.unsafe_set(Int(original.unsafe_get(j)), Int64(j + 1))
        for j in range(n_neighbors):
            penalty += Float64(max(0, Int(ranks.unsafe_get(Int(embedded.unsafe_get(j)))) - n_neighbors))
    return 1 - penalty * 2 / (Float64(n) * Float64(n_neighbors) * (2 * Float64(n) - 3 * Float64(n_neighbors) - 1))
