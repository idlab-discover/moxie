"""Dense agglomerative clustering; scikit-learn/SciPy BSD-3-Clause adaptations."""
import numojo as nm
from std.math import isfinite, sqrt, max, min
from moxie.cluster._contracts import MergeTree, Labels, validate_matrix
from moxie.cluster._base import Clusterer
from moxie.base import UnsupervisedEstimator, TransformerMixin, FitTransformer
from moxie.neighbors._distance import Metric
from moxie._dense import row_squared_norms


struct FullTree(ImplicitlyCopyable):
    """Typed auto/boolean compute_full_tree setting."""
    var mode: Int

    def __init__(out self):
        self.mode = 2

    @implicit
    def __init__(out self, value: Bool):
        self.mode = 1 if value else 0

    @implicit
    def __init__(out self, value: String) raises:
        if value != "auto": raise Error("ValueError: compute_full_tree must be auto or boolean")
        self.mode = 2


def _finite[dtype: DType](X: nm.NDArray[dtype]) raises:
    validate_matrix(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not isfinite(X.item(i, j)):
                raise Error("ValueError: X must contain only finite values")


def _root(mut parents: List[Int], node: Int) -> Int:
    var result = node
    while parents[result] != result: result = parents[result]
    var current = node
    while parents[current] != result:
        var next_node = parents[current]
        parents[current] = result
        current = next_node
    return result


@no_inline
def _tree[dtype: DType](X: nm.NDArray[dtype], linkage: String, metric: String,
                       return_distance: Bool) raises -> MergeTree:
    if X.ndim == 1:
        if X.size < 2: raise Error("ValueError: linkage requires at least two samples")
    else: validate_matrix(X)
    # NuMojo astype reads physical order; materialize strided logical order first.
    var data = X.astype[DType.float64]() if X.is_c_contiguous() else X.contiguous().astype[DType.float64]()
    if data.ndim == 1: data = data.reshape(nm.Shape(data.shape[0], 1))
    if metric == "precomputed": validate_matrix(data)
    else: _finite(data)
    var n = data.shape[0]
    if n < 2: raise Error("ValueError: linkage requires at least two samples")
    if linkage != "ward" and linkage != "complete" and linkage != "average" and linkage != "single":
        raise Error("ValueError: linkage must be ward, complete, average or single")
    var distance = Metric[DType.float64](metric)
    if linkage == "ward" and distance.name != "euclidean":
        raise Error("ValueError: Ward requires Euclidean distances")
    if distance.name != "euclidean" and distance.name != "manhattan" and distance.name != "cosine" and distance.name != "chebyshev" and distance.name != "precomputed":
        raise Error("NotImplementedError: supported metrics are euclidean/l2, manhattan/l1/cityblock, cosine, chebyshev and precomputed")
    if distance.name == "precomputed" and data.shape[1] != n:
        raise Error("ValueError: precomputed distance matrix must be square")
    if distance.name == "cosine":
        var norms = row_squared_norms(data)
        for i in range(n):
            # The neighbor metric clamps cosine, so reject overflow beforehand.
            if not isfinite(norms.item(i)) or norms.item(i) == 0:
                raise Error("ValueError: cosine requires finite, nonzero squared row norms")
    # Single linkage uses linear scratch and evaluates distances on demand.
    var pairwise = nm.zeros[DType.float64](nm.Shape(0, 0))
    if linkage != "single":
        pairwise = nm.zeros[DType.float64](nm.Shape(n, n))
        for i in range(n):
            for j in range(i + 1, n):
                var value = data.item(i, j) if distance.name == "precomputed" else distance.distance(data, i, data, j)
                if not isfinite(value): raise Error("ValueError: nonfinite pairwise distance")
                pairwise[nm.Item(i, j)] = value
                pairwise[nm.Item(j, i)] = value
    var edges = nm.zeros[DType.int64](nm.Shape(n - 1, 2))
    var heights = nm.zeros[DType.float64](nm.Shape(n - 1))
    var sizes = List[Int]()
    for _ in range(n): sizes.append(1)
    if linkage == "single":
        var best = nm.full[DType.float64](nm.Shape(n), Float64("inf"))
        var current = 0
        for k in range(n - 1):
            sizes[current] = 0
            var next_node = -1
            var closest = Float64("inf")
            for i in range(n):
                if sizes[i] == 0: continue
                var value: Float64
                if distance.name == "precomputed":
                    value = data.item(min(current, i), max(current, i))
                else: value = distance.distance(data, current, data, i)
                if not isfinite(value): raise Error("ValueError: nonfinite pairwise distance")
                if value < best.item(i): best.unsafe_set(i, value)
                if best.item(i) < closest:
                    closest = best.item(i)
                    next_node = i
            if next_node < 0: raise Error("ValueError: distances exceed supported finite range")
            edges[nm.Item(k, 0)] = Int64(current)
            edges[nm.Item(k, 1)] = Int64(next_node)
            heights.unsafe_set(k, closest)
            current = next_node
    else:
        var chain = List[Int]()
        for k in range(n - 1):
            if len(chain) == 0:
                for i in range(n):
                    if sizes[i] > 0:
                        chain.append(i)
                        break
            var left: Int
            var right = 0
            var closest: Float64
            while True:
                left = chain[len(chain) - 1]
                closest = Float64("inf")
                if len(chain) > 1:
                    right = chain[len(chain) - 2]
                    closest = pairwise.item(left, right)
                for i in range(n):
                    if sizes[i] == 0 or i == left: continue
                    var value = pairwise.item(left, i)
                    if value < closest:
                        closest = value
                        right = i
                if len(chain) > 1 and right == chain[len(chain) - 2]: break
                chain.append(right)
            _ = chain.pop()
            _ = chain.pop()
            if left > right: left, right = right, left
            var nl = sizes[left]
            var nr = sizes[right]
            edges[nm.Item(k, 0)] = Int64(left)
            edges[nm.Item(k, 1)] = Int64(right)
            heights.unsafe_set(k, closest)
            sizes[left] = 0
            sizes[right] = nl + nr
            for i in range(n):
                if sizes[i] == 0 or i == right: continue
                var a = pairwise.item(i, left)
                var b = pairwise.item(i, right)
                var updated = max(a, b)
                if linkage == "average": updated = (Float64(nl)*a + Float64(nr)*b) / Float64(nl + nr)
                elif linkage == "ward":
                    var ni = Float64(sizes[i])
                    var t = 1 / (ni + Float64(nl + nr))
                    updated = sqrt((ni + Float64(nl))*t*a*a + (ni + Float64(nr))*t*b*b - ni*t*closest*closest)
                if not isfinite(updated): raise Error("ValueError: linkage distance overflow")
                pairwise[nm.Item(i, right)] = updated
                pairwise[nm.Item(right, i)] = updated
    # Stable insertion sorting of n-1 edges preserves reference merge ties.
    var order = List[Int]()
    for i in range(n - 1):
        order.append(i)
        var j = i
        while j > 0 and heights.item(order[j - 1]) > heights.item(i):
            order[j] = order[j - 1]
            j -= 1
        order[j] = i
    var parents = List[Int]()
    for i in range(2*n - 1): parents.append(i)
    var children = nm.zeros[DType.int64](nm.Shape(n - 1, 2))
    var distances = nm.zeros[DType.float64](nm.Shape(n - 1))
    # sklearn's vector MST preserves edge orientation; SciPy sorts children.
    var oriented = linkage == "single" and distance.name != "cosine" and distance.name != "precomputed"
    for k in range(n - 1):
        var edge = order[k]
        var left = _root(parents, Int(edges.item(edge, 0)))
        var right = _root(parents, Int(edges.item(edge, 1)))
        if not oriented and left > right: left, right = right, left
        children[nm.Item(k, 0)] = Int64(left)
        children[nm.Item(k, 1)] = Int64(right)
        distances.unsafe_set(k, heights.item(edge))
        parents[left] = n + k
        parents[right] = n + k
    var requested = Optional[nm.NDArray[DType.float64]](None)
    if return_distance: requested = distances^
    return MergeTree(children^, n, 1, None, requested^)


def linkage_tree[dtype: DType](X: nm.NDArray[dtype], connectivity: Optional[nm.NDArray[dtype]] = None,
        n_clusters: Optional[Int] = None, linkage: String = "complete", affinity: String = "euclidean",
        return_distance: Bool = False) raises -> MergeTree:
    if connectivity: raise Error("NotImplementedError: connectivity-constrained linkage is deferred")
    if linkage == "ward": raise Error("ValueError: use ward_tree for Ward linkage")
    # As in sklearn, unstructured n_clusters does not truncate the tree.
    return _tree(X, linkage, affinity, return_distance)


def ward_tree[dtype: DType](X: nm.NDArray[dtype], *, connectivity: Optional[nm.NDArray[dtype]] = None,
        n_clusters: Optional[Int] = None, return_distance: Bool = False) raises -> MergeTree:
    # sklearn validates n_clusters as Interval(Integral, 1, None) even though the
    # unstructured tree ignores it.
    if n_clusters and n_clusters.value() < 1: raise Error("ValueError: ward_tree n_clusters must be at least 1")
    if connectivity: raise Error("NotImplementedError: connectivity-constrained linkage is deferred")
    return _tree(X, "ward", "euclidean", return_distance)


@no_inline
def _cut(children: Labels, n: Int, count: Int) raises -> Labels:
    if count < 1 or count > n: raise Error("ValueError: n_clusters must lie between 1 and n_leaves")
    # Python heapq ordering is observable in sklearn label numbering.
    var nodes: List[Int] = [2*n - 2]
    for _ in range(count - 1):
        var parent = nodes[0] - n
        var first = Int(children.item(parent, 0))
        var second = Int(children.item(parent, 1))
        nodes.append(first)
        var pos = len(nodes) - 1
        while pos > 0:
            var above = (pos - 1) // 2
            if first <= nodes[above]: break
            nodes[pos] = nodes[above]
            pos = above
        nodes[pos] = first
        nodes[0] = second
        pos = 0
        # heapq siftup descends fully, then sifts the replacement back up.
        var child = 1
        while child < len(nodes):
            if child + 1 < len(nodes) and nodes[child] <= nodes[child + 1]: child += 1
            nodes[pos] = nodes[child]
            pos = child
            child = 2*pos + 1
        nodes[pos] = second
        while pos > 0:
            var above = (pos - 1) // 2
            if second <= nodes[above]: break
            nodes[pos] = nodes[above]
            pos = above
        nodes[pos] = second
    var labels = nm.zeros[DType.int64](nm.Shape(n))
    for label in range(len(nodes)):
        var stack: List[Int] = [nodes[label]]
        while len(stack) > 0:
            var node = stack.pop()
            if node < n: labels.unsafe_set(node, Int64(label))
            else:
                stack.append(Int(children.item(node - n, 0)))
                stack.append(Int(children.item(node - n, 1)))
    return labels^


@fieldwise_init
struct AgglomerativeClusteringParams[dtype: DType](Copyable, Movable):
    var n_clusters: Optional[Int]
    var metric: String
    var memory: Optional[String]
    var connectivity: Optional[nm.NDArray[Self.dtype]]
    var compute_full_tree: FullTree
    var linkage: String
    var distance_threshold: Optional[Float64]
    var compute_distances: Bool


struct AgglomerativeClustering[dtype: DType = DType.float64](Clusterer, Copyable, Movable):
    comptime Params = AgglomerativeClusteringParams[Self.dtype]
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime Labels = nm.NDArray[DType.int64]
    var params: Self.Params
    var labels_: Self.Labels
    var children_: Self.Labels
    var distances_: Optional[nm.NDArray[DType.float64]]
    var n_clusters_: Int
    var n_leaves_: Int
    var n_connected_components_: Int
    var n_features_in_: Int

    def __init__(out self, n_clusters: Optional[Int] = 2, *, metric: String = "euclidean",
                 memory: Optional[String] = None, connectivity: Optional[Self.Input] = None,
                 compute_full_tree: FullTree = FullTree(), linkage: String = "ward",
                 distance_threshold: Optional[Float64] = None, compute_distances: Bool = False) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.params = Self.Params(n_clusters, metric, memory.copy(), connectivity.copy(), compute_full_tree,
            linkage, distance_threshold, compute_distances)
        self.labels_ = nm.zeros[DType.int64](nm.Shape(0))
        self.children_ = nm.zeros[DType.int64](nm.Shape(0, 2))
        self.distances_ = None
        self.n_clusters_ = 0
        self.n_leaves_ = 0
        self.n_connected_components_ = 0
        self.n_features_in_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, n_clusters: Optional[Optional[Int]] = None,
                   metric: Optional[String] = None, memory: Optional[Optional[String]] = None,
                   connectivity: Optional[Optional[Self.Input]] = None,
                   compute_full_tree: Optional[FullTree] = None, linkage: Optional[String] = None,
                   distance_threshold: Optional[Optional[Float64]] = None,
                   compute_distances: Optional[Bool] = None) raises -> ref[self] Self:
        if n_clusters: self.params.n_clusters = n_clusters.value()
        if metric: self.params.metric = metric.value().copy()
        if memory: self.params.memory = memory.value().copy()
        if connectivity: self.params.connectivity = connectivity.value().copy()
        if compute_full_tree: self.params.compute_full_tree = compute_full_tree.value()
        if linkage: self.params.linkage = linkage.value().copy()
        if distance_threshold: self.params.distance_threshold = distance_threshold.value()
        if compute_distances: self.params.compute_distances = compute_distances.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input,
            y: Optional[Self.Target] = None) raises -> ref[self] Self:
        _finite(X)
        ref data = X
        var result = _fit_tree(data, self.params.n_clusters, self.params.metric, self.params.memory,
            self.params.connectivity, self.params.compute_full_tree, self.params.linkage,
            self.params.distance_threshold, self.params.compute_distances)
        var count: Int
        if self.params.n_clusters: count = self.params.n_clusters.value()
        else:
            count = 1
            for i in range(result.n_leaves - 1):
                if result.distances.value().item(i) >= self.params.distance_threshold.value(): count += 1
        var labels = _cut(result.children, result.n_leaves, count)
        self.labels_ = labels^
        self.n_clusters_ = count
        self.n_leaves_ = result.n_leaves
        self.n_connected_components_ = result.n_connected_components
        self.n_features_in_ = X.shape[1]
        self.children_ = result.children.copy()
        self.distances_ = result.distances.copy()
        return self

    def fit_predict(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Labels:
        _ = self.fit(X, y)
        return self.labels_.copy()


@fieldwise_init
struct FeatureAgglomerationParams[dtype: DType](Copyable, Movable):
    var n_clusters: Optional[Int]
    var metric: String
    var memory: Optional[String]
    var connectivity: Optional[nm.NDArray[Self.dtype]]
    var compute_full_tree: FullTree
    var linkage: String
    var distance_threshold: Optional[Float64]
    var compute_distances: Bool
    var pooling_func: String


struct FeatureAgglomeration[dtype: DType = DType.float64](UnsupervisedEstimator, TransformerMixin, FitTransformer, Copyable, Movable):
    comptime Params = FeatureAgglomerationParams[Self.dtype]
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime Labels = nm.NDArray[DType.int64]
    comptime Transformed = nm.NDArray[DType.float64]
    var params: Self.Params
    var labels_: Self.Labels
    var children_: Self.Labels
    var distances_: Optional[nm.NDArray[DType.float64]]
    var n_clusters_: Int
    var n_leaves_: Int
    var n_connected_components_: Int
    var n_features_in_: Int

    def __init__(out self, n_clusters: Optional[Int] = 2, *, metric: String = "euclidean",
                 memory: Optional[String] = None, connectivity: Optional[Self.Input] = None,
                 compute_full_tree: FullTree = FullTree(), linkage: String = "ward",
                 pooling_func: String = "mean",
                 distance_threshold: Optional[Float64] = None, compute_distances: Bool = False) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.params = Self.Params(n_clusters, metric, memory.copy(), connectivity.copy(), compute_full_tree,
            linkage, distance_threshold, compute_distances, pooling_func)
        self.labels_ = nm.zeros[DType.int64](nm.Shape(0))
        self.children_ = nm.zeros[DType.int64](nm.Shape(0, 2))
        self.distances_ = None
        self.n_clusters_ = 0
        self.n_leaves_ = 0
        self.n_connected_components_ = 0
        self.n_features_in_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, n_clusters: Optional[Optional[Int]] = None,
                   metric: Optional[String] = None, memory: Optional[Optional[String]] = None,
                   connectivity: Optional[Optional[Self.Input]] = None,
                   compute_full_tree: Optional[FullTree] = None, linkage: Optional[String] = None,
                   pooling_func: Optional[String] = None,
                   distance_threshold: Optional[Optional[Float64]] = None,
                   compute_distances: Optional[Bool] = None) raises -> ref[self] Self:
        if n_clusters: self.params.n_clusters = n_clusters.value()
        if metric: self.params.metric = metric.value().copy()
        if memory: self.params.memory = memory.value().copy()
        if connectivity: self.params.connectivity = connectivity.value().copy()
        if compute_full_tree: self.params.compute_full_tree = compute_full_tree.value()
        if linkage: self.params.linkage = linkage.value().copy()
        if distance_threshold: self.params.distance_threshold = distance_threshold.value()
        if compute_distances: self.params.compute_distances = compute_distances.value()
        if pooling_func: self.params.pooling_func = pooling_func.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input,
            y: Optional[Self.Target] = None) raises -> ref[self] Self:
        _finite(X)
        if X.shape[1] < 2: raise Error("ValueError: feature agglomeration requires at least two features")
        if self.params.pooling_func != "mean":
            raise Error("NotImplementedError: pooling_func currently supports mean only")
        var data = X.T()
        var result = _fit_tree(data, self.params.n_clusters, self.params.metric, self.params.memory,
            self.params.connectivity, self.params.compute_full_tree, self.params.linkage,
            self.params.distance_threshold, self.params.compute_distances)
        var count: Int
        if self.params.n_clusters: count = self.params.n_clusters.value()
        else:
            count = 1
            for i in range(result.n_leaves - 1):
                if result.distances.value().item(i) >= self.params.distance_threshold.value(): count += 1
        var labels = _cut(result.children, result.n_leaves, count)
        self.labels_ = labels^
        self.n_clusters_ = count
        self.n_leaves_ = result.n_leaves
        self.n_connected_components_ = result.n_connected_components
        self.n_features_in_ = X.shape[1]
        self.children_ = result.children.copy()
        self.distances_ = result.distances.copy()
        return self

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        if not self.__sklearn_is_fitted__(): raise Error("NotFittedError: FeatureAgglomeration is not fitted")
        _finite(X)
        if X.shape[1] != self.n_features_in_: raise Error("ValueError: feature count differs from fit")
        if self.params.pooling_func != "mean": raise Error("NotImplementedError: pooling_func currently supports mean only")
        var output = nm.zeros[DType.float64](nm.Shape(X.shape[0], self.n_clusters_))
        var counts = nm.zeros[DType.int64](nm.Shape(self.n_clusters_))
        for j in range(self.n_features_in_):
            var label = Int(self.labels_.item(j))
            counts.unsafe_set(label, counts.item(label) + 1)
            for i in range(X.shape[0]):
                output[nm.Item(i, label)] = output.item(i, label) + Float64(X.item(i, j))
        for i in range(X.shape[0]):
            for j in range(self.n_clusters_): output[nm.Item(i, j)] = output.item(i, j) / Float64(counts.item(j))
        return output^

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def inverse_transform[value_dtype: DType](self, X: nm.NDArray[value_dtype]) raises -> nm.NDArray[value_dtype]:
        if not self.__sklearn_is_fitted__(): raise Error("NotFittedError: FeatureAgglomeration is not fitted")
        if X.ndim != 1 and X.ndim != 2: raise Error("ValueError: inverse input must have rank 1 or 2")
        if X.shape[X.ndim - 1] != self.n_clusters_: raise Error("ValueError: inverse input must have n_clusters columns")
        var output = nm.zeros[value_dtype](nm.Shape(self.n_features_in_))
        if X.ndim == 2: output = nm.zeros[value_dtype](nm.Shape(X.shape[0], self.n_features_in_))
        for j in range(self.n_features_in_):
            var label = Int(self.labels_.item(j))
            if X.ndim == 1: output.unsafe_set(j, X.item(label))
            else:
                for i in range(X.shape[0]): output[nm.Item(i, j)] = X.item(i, label)
        return output^

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if not self.__sklearn_is_fitted__(): raise Error("NotFittedError: FeatureAgglomeration is not fitted")
        if input_features and len(input_features.value()) != self.n_features_in_:
            raise Error("ValueError: input_features must match n_features_in_")
        var names = List[String]()
        for i in range(self.n_clusters_): names.append("featureagglomeration" + String(i))
        return names^


def _fit_tree[dtype: DType](X: nm.NDArray[dtype], n_clusters: Optional[Int], metric: String,
        memory: Optional[String], connectivity: Optional[nm.NDArray[dtype]], compute_full_tree: FullTree,
        linkage: String, threshold: Optional[Float64], compute_distances: Bool) raises -> MergeTree:
    if memory: raise Error("NotImplementedError: joblib memory caching is unavailable")
    if connectivity: raise Error("NotImplementedError: connectivity-constrained linkage is deferred")
    if Bool(n_clusters) == Bool(threshold):
        raise Error("ValueError: exactly one of n_clusters and distance_threshold must be set")
    if n_clusters and (n_clusters.value() < 1 or n_clusters.value() > X.shape[0]):
        raise Error("ValueError: n_clusters must lie between 1 and n_leaves")
    if threshold:
        if not isfinite(threshold.value()) or threshold.value() < 0:
            raise Error("ValueError: distance_threshold must be finite and nonnegative")
        if compute_full_tree.mode == 0:
            raise Error("ValueError: compute_full_tree must be true with distance_threshold")
    return _tree(X, linkage, metric, compute_distances or Bool(threshold))
