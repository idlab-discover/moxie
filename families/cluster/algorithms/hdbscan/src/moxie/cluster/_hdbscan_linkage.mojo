"""Dense HDBSCAN mutual reachability and single linkage."""
# Adapted from scikit-learn (BSD-3-Clause). See LICENSES/scikit-learn.txt.

import numojo as nm
from std.builtin.sort import sort
from std.math import abs, isfinite, isnan, max, min
from moxie.cluster._agglomerative import _root
from moxie.cluster._centroid_selection import count_order
from moxie.cluster._contracts import MergeTree, validate_matrix


@fieldwise_init
struct _Edge(ImplicitlyCopyable):
    var source: Int
    var target: Int
    var distance: Float64


def _validate_distances[dtype: DType](
    distances: nm.NDArray[dtype], min_samples: Int, alpha: Float64
) raises:
    validate_matrix(distances)
    var n = distances.shape[0]
    if n != distances.shape[1]:
        raise Error("ValueError: precomputed distances must be square")
    if n < 2:
        raise Error("ValueError: HDBSCAN requires more than one sample")
    if min_samples < 1 or min_samples > n:
        raise Error("ValueError: min_samples must lie between 1 and the sample count")
    if not isfinite(alpha) or alpha <= 0:
        raise Error("ValueError: alpha must be finite and positive")
    for i in range(n):
        for j in range(i + 1):
            var left = Float64(distances.item(i, j))
            var right = Float64(distances.item(j, i))
            if isnan(left) or isnan(right):
                raise Error("ValueError: NaN values found in precomputed distances")
            # Check both orientations, as in allclose(distances, distances.T).
            if left != right and (
                not isfinite(left) or not isfinite(right)
                or abs(left - right) > 1e-9 + 1e-7 * min(abs(left), abs(right))
            ):
                raise Error("ValueError: precomputed distances must be symmetric")


def _core_distances[dtype: DType](
    distances: nm.NDArray[dtype], min_samples: Int, alpha: Float64
) raises -> List[Float64]:
    var n = distances.shape[0]
    var core = List[Float64](capacity=n)
    var row = List[Float64](capacity=n)
    for _ in range(n):
        row.append(0)
    for i in range(n):
        for j in range(n):
            row[j] = Float64(distances.item(i, j)) / alpha
        sort(row)
        core.append(row[min_samples - 1])
    return core^


def _brute_edges[dtype: DType](
    distances: nm.NDArray[dtype], core: List[Float64], alpha: Float64
) raises -> List[_Edge]:
    var n = len(core)
    var edges = List[_Edge](capacity=n - 1)
    var visited = List[Bool](capacity=n)
    var best = List[Float64](capacity=n)
    for _ in range(n):
        visited.append(False)
        best.append(Float64("inf"))
    var current = 0
    for _ in range(n - 1):
        visited[current] = True
        var next_node = -1
        var closest = Float64("inf")
        for j in range(n):
            if visited[j]:
                continue
            var distance = Float64(distances.item(current, j)) / alpha
            var reachability = max(core[current], max(core[j], distance))
            best[j] = min(best[j], reachability)
            if next_node == -1 or best[j] < closest:
                next_node = j
                closest = best[j]
        # The sklearn brute builder records the last visited sample as the
        # source, even when an earlier sample set the target's best weight.
        edges.append(_Edge(current, next_node, closest))
        current = next_node
    return edges^


def _single_linkage(edges: List[_Edge]) raises -> MergeTree:
    var n = len(edges) + 1
    var weights = nm.zeros[DType.float64](nm.Shape(n - 1))
    for i in range(n - 1):
        weights.unsafe_set(i, edges[i].distance)
    var order = count_order(weights)
    var parents = List[Int](capacity=2 * n - 1)
    for i in range(2 * n - 1):
        parents.append(i)
    var children = nm.zeros[DType.int64](nm.Shape(n - 1, 2))
    var heights = nm.zeros[DType.float64](nm.Shape(n - 1))
    for i in range(n - 1):
        var edge = edges[Int(order.item(i))]
        var left = _root(parents, edge.source)
        var right = _root(parents, edge.target)
        children[nm.Item(i, 0)] = Int64(left)
        children[nm.Item(i, 1)] = Int64(right)
        heights.unsafe_set(i, edge.distance)
        parents[left] = n + i
        parents[right] = n + i
    return MergeTree(children^, n, 1, None, heights^)


def dense_brute_hierarchy[dtype: DType](
    distances: nm.NDArray[dtype], min_samples: Int, alpha: Float64 = 1
) raises -> MergeTree:
    """Build the dense precomputed brute hierarchy without mutating distances."""
    _validate_distances(distances, min_samples, alpha)
    var core = _core_distances(distances, min_samples, alpha)
    return _single_linkage(_brute_edges(distances, core, alpha))
