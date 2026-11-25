"""Dense HDBSCAN mutual reachability and single linkage."""
# Adapted from scikit-learn (BSD-3-Clause). See LICENSES/scikit-learn.txt.

import numojo as nm
from std.builtin.sort import sort
from std.math import abs, isfinite, isnan, max, min
from moxie.cluster._agglomerative import _root
from moxie.cluster._centroid_selection import count_order
from moxie.cluster._contracts import MergeTree, validate_matrix
from moxie.neighbors._distance import Metric
from moxie.neighbors._metrics import ball_eligible, kd_eligible
from moxie.neighbors._unsupervised import NearestNeighbors


@fieldwise_init
struct _Edge(ImplicitlyCopyable):
    var source: Int
    var target: Int
    var distance: Float64


def _validate_linkage_options(n: Int, min_samples: Int, alpha: Float64) raises:
    if n < 2:
        raise Error("ValueError: HDBSCAN requires more than one sample")
    if min_samples < 1 or min_samples > n:
        raise Error("ValueError: min_samples must lie between 1 and the sample count")
    if not isfinite(alpha) or alpha <= 0:
        raise Error("ValueError: alpha must be finite and positive")


def _validate_distances[dtype: DType](
    distances: nm.NDArray[dtype], min_samples: Int, alpha: Float64
) raises:
    validate_matrix(distances)
    var n = distances.shape[0]
    if n != distances.shape[1]:
        raise Error("ValueError: precomputed distances must be square")
    _validate_linkage_options(n, min_samples, alpha)
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


def _core_distances[Distance: def(Int, Int) raises -> Float64](
    n: Int, min_samples: Int, distance: Distance
) raises -> List[Float64]:
    var core = List[Float64](capacity=n)
    var row = List[Float64](capacity=n)
    for _ in range(n):
        row.append(0)
    for i in range(n):
        for j in range(n):
            row[j] = distance(i, j)
        sort(row)
        core.append(row[min_samples - 1])
    return core^


def _prim_edges[track_source: Bool, Distance: def(Int, Int) raises -> Float64](
    core: List[Float64], distance: Distance
) raises -> List[_Edge]:
    var n = len(core)
    var edges = List[_Edge](capacity=n - 1)
    var visited = List[Bool](capacity=n)
    var best = List[Float64](capacity=n)
    var sources = List[Int]()
    for _ in range(n):
        visited.append(False)
        best.append(Float64("inf"))
        comptime if track_source:
            sources.append(0)
    var current = 0
    for _ in range(n - 1):
        visited[current] = True
        var next_node = -1
        var closest = Float64("inf")
        for j in range(n):
            if visited[j]:
                continue
            var reachability = max(core[current], max(core[j], distance(current, j)))
            if reachability < best[j]:
                best[j] = reachability
                comptime if track_source:
                    sources[j] = current
            if next_node == -1 or best[j] < closest:
                next_node = j
                closest = best[j]
        var source = current
        comptime if track_source:
            source = sources[next_node]
        # The brute reference records the last visited sample as the source.
        edges.append(_Edge(source, next_node, closest))
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
    def distance(i: Int, j: Int) raises {imm distances, imm alpha} -> Float64:
        return Float64(distances.item(i, j)) / alpha

    var core = _core_distances(distances.shape[0], min_samples, distance)
    return _single_linkage(_prim_edges[False](core, distance))


def _tree_core_distances(
    X: nm.NDArray[DType.float64],
    min_samples: Int,
    metric: Metric[DType.float64],
    algorithm: String,
    leaf_size: Int,
) raises -> List[Float64]:
    var neighbors = NearestNeighbors(
        min_samples, algorithm=algorithm, leaf_size=leaf_size, metric_params=metric.copy()
    )
    _ = neighbors.fit(X)
    var n = X.shape[0]
    var core = List[Float64](capacity=n)
    # Bound top-k storage even when min_samples is the entire training set.
    comptime batch_size = 16
    for start in range(0, n, batch_size):
        var stop = min(start + batch_size, n)
        var block = X[start:stop, 0:X.shape[1]]
        var result = neighbors.kneighbors(block)
        ref distances = result.distances.value()
        for i in range(stop - start):
            var value = distances.item(i, min_samples - 1)
            if not isfinite(value):
                raise Error("ValueError: nonfinite HDBSCAN core distance")
            core.append(value)
    return core^


def _brute_braycurtis(X: nm.NDArray[DType.float64], i: Int, j: Int) raises -> Float64:
    # scipy.pdist uses |x + y|. The neighbors tree metric uses |x| + |y|.
    var numerator = Float64(0)
    var denominator = Float64(0)
    for feature in range(X.shape[1]):
        var left = X.item(i, feature)
        var right = X.item(j, feature)
        numerator += abs(left - right)
        denominator += abs(left + right)
    return numerator / denominator


def _feature_hierarchy(
    X: nm.NDArray[DType.float64],
    min_samples: Int,
    alpha: Float64,
    metric: Metric[DType.float64],
    algorithm: String,
    leaf_size: Int,
) raises -> MergeTree:
    metric.validate_features(X.shape[1])
    if metric.name == "precomputed":
        raise Error("ValueError: feature linkage requires a feature metric")

    var backend = algorithm
    if algorithm == "auto":
        if kd_eligible(metric):
            backend = "kd_tree"
        elif ball_eligible(metric):
            backend = "ball_tree"
        else:
            backend = "brute"
    elif algorithm != "brute" and algorithm != "kd_tree" and algorithm != "ball_tree":
        raise Error("ValueError: invalid HDBSCAN algorithm")

    if backend == "brute":
        var prepared = metric.self_query_metric(X)
        def brute_distance(i: Int, j: Int) raises {imm X, imm prepared, imm alpha} -> Float64:
            if i == j:
                return 0
            var value = (
                _brute_braycurtis(X, i, j)
                if prepared.name == "braycurtis"
                else prepared.distance(X, i, X, j)
            )
            if isnan(value) or value < 0:
                raise Error("ValueError: HDBSCAN metric produced an undefined or negative distance")
            return value / alpha

        var core = _core_distances(X.shape[0], min_samples, brute_distance)
        return _single_linkage(_prim_edges[False](core, brute_distance))

    # Tree queries find core distances in the original metric. Only the pair
    # distance is scaled by alpha during the Prim scan.
    metric.validate_cross_query()
    var core = _tree_core_distances(X, min_samples, metric, backend, leaf_size)
    def scaled_pair(i: Int, j: Int) raises {imm X, imm metric, imm alpha} -> Float64:
        var value = metric.distance(X, i, X, j) / alpha
        if not isfinite(value) or value < 0:
            raise Error("ValueError: HDBSCAN tree metric produced a nonfinite or negative distance")
        return value

    return _single_linkage(_prim_edges[True](core, scaled_pair))


def dense_feature_hierarchy[dtype: DType](
    X: nm.NDArray[dtype],
    min_samples: Int,
    metric: Metric[DType.float64],
    *,
    alpha: Float64 = 1,
    algorithm: String = "auto",
    leaf_size: Int = 40,
) raises -> MergeTree:
    """Build a dense hierarchy from finite, borrowed Float32 or Float64 features."""
    validate_matrix(X)
    _validate_linkage_options(X.shape[0], min_samples, alpha)
    if leaf_size < 1:
        raise Error("ValueError: leaf_size must be positive")

    # Materialize and widen in one pass when the original layout cannot be
    # borrowed as contiguous Float64. astype alone uses linear loads.
    comptime if dtype == DType.float64:
        if X.is_c_contiguous():
            for i in range(X.size):
                if not isfinite(X.unsafe_get(i)):
                    raise Error("ValueError: HDBSCAN linkage requires finite features")
            ref features = rebind[nm.NDArray[DType.float64]](X)
            return _feature_hierarchy(features, min_samples, alpha, metric, algorithm, leaf_size)
    var features = nm.empty[DType.float64](X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            var value = Float64(X.item(i, j))
            if not isfinite(value):
                raise Error("ValueError: HDBSCAN linkage requires finite features")
            features.unsafe_set(i * X.shape[1] + j, value)
    return _feature_hierarchy(features, min_samples, alpha, metric, algorithm, leaf_size)
