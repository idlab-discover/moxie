"""Dense divisive clustering, adapted from sklearn (BSD-3-Clause)."""
import numojo as nm
from std.math import isfinite
from moxie._random import RandomState, check_random_state
from moxie.utils.validation import FittedState
from moxie.cluster._base import PredictiveClusterer, ClusterDistanceTransformer
from moxie.cluster._contracts import Labels, CentroidResult, validate_matrix
from moxie.cluster._centroid_engine import _at, _weight, _finite, assign, distances, inertia
from moxie.cluster._kmeans import _sample_weights, _probabilities, _gather, _single, kmeans_plusplus


@fieldwise_init
struct _BisectingNode[dtype: DType](Copyable, Movable):
    var center: nm.NDArray[Self.dtype]
    var indices: Labels
    var score: Float64
    var weight: Float64
    var inertia: Float64
    var left: Int
    var right: Int
    var label: Int


@fieldwise_init
struct BisectingKMeansParams(Copyable, Movable):
    var n_clusters: Int
    var init: String
    var n_init: Int
    var max_iter: Int
    var tol: Float64
    var verbose: Int
    var random_state: Optional[Int]
    var copy_x: Bool
    var algorithm: String
    var bisecting_strategy: String


struct BisectingKMeans[dtype: DType = DType.float64](PredictiveClusterer, ClusterDistanceTransformer, Copyable, Movable):
    comptime Params = BisectingKMeansParams
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime Prediction = Labels
    comptime Labels = nm.NDArray[DType.int64]
    comptime Transformed = nm.NDArray[Self.dtype]
    var params: Self.Params
    var cluster_centers_: Self.Input
    var labels_: Self.Labels
    var inertia_: Float64
    var n_features_in_: Int
    var _nodes: List[_BisectingNode[Self.dtype]]
    var _mean: Self.Input
    var _state: FittedState

    def __init__(out self, n_clusters: Int = 8, *, init: String = "random", n_init: Int = 1, max_iter: Int = 300, tol: Float64 = 1e-4, verbose: Int = 0, random_state: Optional[Int] = None, copy_x: Bool = True, algorithm: String = "lloyd", bisecting_strategy: String = "biggest_inertia") raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.params = Self.Params(n_clusters, init, n_init, max_iter, tol, verbose, random_state, copy_x, algorithm, bisecting_strategy)
        self.cluster_centers_ = nm.zeros[Self.dtype](nm.Shape(0, 0))
        self.labels_ = nm.zeros[DType.int64](nm.Shape(0))
        self.inertia_ = 0
        self.n_features_in_ = 0
        self._nodes = List[_BisectingNode[Self.dtype]]()
        self._mean = nm.zeros[Self.dtype](nm.Shape(0))
        self._state = FittedState()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, n_clusters: Optional[Int] = None, init: Optional[String] = None, n_init: Optional[Int] = None, max_iter: Optional[Int] = None, tol: Optional[Float64] = None, verbose: Optional[Int] = None, random_state: Optional[Optional[Int]] = None, copy_x: Optional[Bool] = None, algorithm: Optional[String] = None, bisecting_strategy: Optional[String] = None) raises -> ref[self] Self:
        if n_clusters: self.params.n_clusters = n_clusters.value()
        if init: self.params.init = init.value().copy()
        if n_init: self.params.n_init = n_init.value().copy()
        if max_iter: self.params.max_iter = max_iter.value()
        if tol: self.params.tol = tol.value()
        if verbose: self.params.verbose = verbose.value()
        if random_state: self.params.random_state = random_state.value()
        if copy_x: self.params.copy_x = copy_x.value()
        if algorithm: self.params.algorithm = algorithm.value()
        if bisecting_strategy: self.params.bisecting_strategy = bisecting_strategy.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.is_fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        _ = self.fit(X, y, sample_weight=None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None, *, sample_weight: Optional[Self.Input]) raises -> ref[self] Self:
        with check_random_state(self.params.random_state) as rng:
            _ = self.fit(X, y, random_state=rng[], sample_weight=sample_weight)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None, *, mut random_state: RandomState, sample_weight: Optional[Self.Input] = None) raises -> ref[self] Self:
        validate_matrix(X)
        _finite(X)
        ref p = self.params
        var k = p.n_clusters
        if k < 1 or k > X.shape[0] or p.n_init < 1 or p.max_iter < 1 or p.verbose < 0 or p.tol < 0 or not isfinite(p.tol):
            raise Error("ValueError: invalid clustering parameters")
        if p.init != "random" and p.init != "k-means++":
            raise Error("ValueError: init must be random or k-means++")
        if p.algorithm != "lloyd" and p.algorithm != "elkan":
            raise Error("ValueError: algorithm must be lloyd or elkan")
        if p.bisecting_strategy != "biggest_inertia" and p.bisecting_strategy != "largest_cluster":
            raise Error("ValueError: unknown bisecting_strategy")
        var weights = _sample_weights(X, sample_weight)
        var mean = nm.zeros[Self.dtype](nm.Shape(X.shape[1]))
        var centered = nm.zeros[Self.dtype](nm.Shape(X.shape[0], X.shape[1]))
        var root_center = nm.zeros[Self.dtype](nm.Shape(1, X.shape[1]))
        # Same-dtype sums preserve the existing KMeans centering policy.
        for f in range(X.shape[1]):
            var total = SIMD[Self.dtype, 1](0)
            for i in range(X.shape[0]): total += _at(X, i, f)
            var average = total / SIMD[Self.dtype, 1](X.shape[0])
            mean.unsafe_set(f, average)
            total = 0
            for i in range(X.shape[0]):
                var delta = _at(X, i, f) - average
                centered.unsafe_set(i * X.shape[1] + f, delta)
                total += delta
            root_center.unsafe_set(f, total / SIMD[Self.dtype, 1](X.shape[0]))
        var indices = nm.zeros[DType.int64](nm.Shape(X.shape[0]))
        for i in range(X.shape[0]): indices.unsafe_set(i, Int64(i))
        var nodes = List[_BisectingNode[Self.dtype]]()
        nodes.append(_BisectingNode[Self.dtype](root_center^, indices^, 0, 0, 0, -1, -1, -1))
        var leaves: List[Int] = [0]
        for _ in range(k - 1):
            var selected = 0
            for j in range(1, len(leaves)):
                if nodes[leaves[j]].score > nodes[leaves[selected]].score: selected = j
            var parent = leaves[selected]
            var subset = _gather(centered, nodes[parent].indices)
            var subset_weights = nm.zeros[Self.dtype](nm.Shape(subset.shape[0]))
            for i in range(subset.shape[0]):
                subset_weights.unsafe_set(i, _weight(weights, Int(nodes[parent].indices.item(i))))
            var best = Optional[CentroidResult[Self.dtype]]()
            for _ in range(p.n_init):
                var initial: Self.Input
                if p.init == "random":
                    var probabilities = _probabilities(subset_weights)
                    var chosen = random_state.choice(subset.shape[0], probabilities, 2, False)
                    initial = _gather(subset, chosen)
                elif subset.shape[0] == 1:
                    # sklearn's private seeder permits repeated centers on a
                    # singleton leaf. Preserve its categorical + two trial draws.
                    var seeded = kmeans_plusplus(subset, 1, random_state=random_state, sample_weight=subset_weights.view())
                    for _ in range(2): _ = random_state.random_sample()
                    initial = nm.zeros[Self.dtype](nm.Shape(2, X.shape[1]))
                    for child in range(2):
                        for f in range(X.shape[1]): initial.unsafe_set(child * X.shape[1] + f, seeded.centers.item(0, f))
                else:
                    var seeded = kmeans_plusplus(subset, 2, random_state=random_state, sample_weight=subset_weights.view())
                    initial = seeded.centers.view()
                var result = _single(subset, subset_weights, initial^, p.max_iter, p.tol, p.algorithm == "elkan", p.verbose)
                if not best:
                    best = result^
                elif result.inertia < best.value().inertia * (1 - 1e-6):
                    best = result^
            var first_child = len(nodes)
            for child in range(2):
                var count = 0
                var mass = SIMD[Self.dtype, 1](0)
                for i in range(subset.shape[0]):
                    if best.value().labels.item(i) == Int64(child):
                        count += 1
                        mass += subset_weights.item(i)
                var child_indices = nm.zeros[DType.int64](nm.Shape(count))
                var position = 0
                for i in range(subset.shape[0]):
                    if best.value().labels.item(i) == Int64(child):
                        child_indices.unsafe_set(position, nodes[parent].indices.item(i))
                        position += 1
                var center = nm.zeros[Self.dtype](nm.Shape(1, X.shape[1]))
                for f in range(X.shape[1]): center.unsafe_set(f, best.value().centers.item(child, f))
                var loss = inertia(subset, best.value().centers, subset_weights, best.value().labels, child)
                var score = loss if p.bisecting_strategy == "biggest_inertia" else Float64(count)
                nodes.append(_BisectingNode[Self.dtype](center^, child_indices^, score, Float64(mass), loss, -1, -1, -1))
            nodes[parent].left = first_child
            nodes[parent].right = first_child + 1
            nodes[parent].indices = nm.zeros[DType.int64](nm.Shape(0))
            var next_leaves = List[Int]()
            for j in range(len(leaves)):
                if j == selected:
                    next_leaves.append(first_child)
                    next_leaves.append(first_child + 1)
                else: next_leaves.append(leaves[j])
            leaves = next_leaves^
        var labels = nm.zeros[DType.int64](nm.Shape(X.shape[0]))
        var centers = nm.zeros[Self.dtype](nm.Shape(k, X.shape[1]))
        for j in range(k):
            var node = leaves[j]
            nodes[node].label = j
            for i in range(nodes[node].indices.size): labels.unsafe_set(Int(nodes[node].indices.item(i)), Int64(j))
            for f in range(X.shape[1]): centers.unsafe_set(j * X.shape[1] + f, nodes[node].center.item(0, f) + mean.item(f))
            nodes[node].indices = nm.zeros[DType.int64](nm.Shape(0))
        # sklearn computes final inertia after undoing centering (including rounding).
        for i in range(X.shape[0]):
            for f in range(X.shape[1]): centered.unsafe_set(i * X.shape[1] + f, centered.item(i, f) + mean.item(f))
        var loss = inertia(centered, centers, weights, labels)
        self.cluster_centers_ = centers^
        self.labels_ = labels^
        self.inertia_ = loss
        self.n_features_in_ = X.shape[1]
        self._nodes = nodes^
        self._mean = mean^
        self._state.record_fit(X.shape[0], X.shape[1])
        return self

    def _check(self, X: Self.Input) raises:
        self._state.check_features(X.shape[1] if X.ndim == 2 else 0)
        validate_matrix(X)
        _finite(X)

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        self._check(X)
        var result = nm.zeros[DType.int64](nm.Shape(X.shape[0]))
        var route = nm.zeros[DType.int64](nm.Shape(X.shape[0]))
        for node in range(len(self._nodes)):
            if self._nodes[node].left < 0:
                for i in range(X.shape[0]):
                    if route.item(i) == Int64(node): result.unsafe_set(i, Int64(self._nodes[node].label))
                continue
            var count = 0
            for i in range(X.shape[0]):
                if route.item(i) == Int64(node): count += 1
            if count == 0: continue
            var indices = nm.zeros[DType.int64](nm.Shape(count))
            var position = 0
            for i in range(X.shape[0]):
                if route.item(i) == Int64(node):
                    indices.unsafe_set(position, Int64(i))
                    position += 1
            var subset = _gather(X, indices)
            var centers = nm.zeros[Self.dtype](nm.Shape(2, X.shape[1]))
            for child in range(2):
                var child_node = self._nodes[node].left if child == 0 else self._nodes[node].right
                for f in range(X.shape[1]): centers.unsafe_set(child * X.shape[1] + f, self._nodes[child_node].center.item(0, f) + self._mean.item(f))
            var weights = nm.ones[Self.dtype](nm.Shape(count))
            var assigned = assign(subset, centers, weights)
            for i in range(count):
                route.unsafe_set(Int(indices.item(i)), Int64(self._nodes[node].left if assigned.labels.item(i) == 0 else self._nodes[node].right))
        return result^

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: BisectingKMeans is not fitted")
        if input_features and len(input_features.value()) != self.n_features_in_:
            raise Error("ValueError: input_features must match n_features_in_")
        var names = List[String]()
        for i in range(self.cluster_centers_.shape[0]):
            names.append("bisectingkmeans" + String(i))
        return names^

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check(X)
        return distances(X, self.cluster_centers_, False)

    def score(self, X: Self.Input, y: Optional[Self.Target] = None, *, sample_weight: Optional[Self.Input] = None) raises -> Float64:
        self._check(X)
        var weights = _sample_weights(X, sample_weight)
        var result = assign(X, self.cluster_centers_, weights)
        return -result.inertia

    def fit_predict(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Labels:
        return self.fit_predict(X, y, sample_weight=None)

    def fit_predict(mut self, X: Self.Input, y: Optional[Self.Target] = None, *, sample_weight: Optional[Self.Input]) raises -> Self.Labels:
        _ = self.fit(X, y, sample_weight=sample_weight)
        return self.labels_.copy()

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None, *, sample_weight: Optional[Self.Input] = None) raises -> Self.Transformed:
        _ = self.fit(X, y, sample_weight=sample_weight)
        return self.transform(X)
