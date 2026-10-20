"""Dense BIRCH CF tree, adapted from scikit-learn (BSD-3-Clause)."""
import numojo as nm
from std.math import isfinite, sqrt
from moxie.cluster._base import PredictiveClusterer, ClusterDistanceTransformer, IncrementalClusterer
from moxie.cluster._contracts import validate_matrix
from moxie.cluster._centroid_engine import distances
from moxie.cluster._agglomerative import AgglomerativeClustering
from moxie.neighbors._unsupervised import NearestNeighbors


struct CFSubcluster[dtype: DType](Copyable, Movable):
    var n_samples_: Int
    var linear_sum_: nm.NDArray[Self.dtype]
    var squared_sum_: Scalar[Self.dtype]
    var centroid_: nm.NDArray[Self.dtype]
    var sq_norm_: Scalar[Self.dtype]
    var child_: Optional[Int]

    def __init__(out self, n_features: Int) raises:
        self.n_samples_ = 0
        self.linear_sum_ = nm.zeros[Self.dtype](nm.Shape(n_features))
        self.centroid_ = nm.zeros[Self.dtype](nm.Shape(n_features))
        self.squared_sum_ = 0
        self.sq_norm_ = 0
        self.child_ = None

    def update(mut self, other: Self) raises:
        self.n_samples_ += other.n_samples_
        self.squared_sum_ += other.squared_sum_
        self.sq_norm_ = 0
        for j in range(self.linear_sum_.size):
            var total = self.linear_sum_.item(j) + other.linear_sum_.item(j)
            var center = total / Scalar[Self.dtype](self.n_samples_)
            self.linear_sum_.unsafe_set(j, total)
            self.centroid_.unsafe_set(j, center)
            self.sq_norm_ += center * center
        self._finite()

    def merge(mut self, other: Self, threshold: Float64) raises -> Bool:
        var count = self.n_samples_ + other.n_samples_
        var squared_sum = self.squared_sum_ + other.squared_sum_
        var norm = Scalar[Self.dtype](0)
        var inverse = Scalar[Self.dtype](1.0 / Float64(count))
        var center = nm.zeros[Self.dtype](nm.Shape(self.linear_sum_.size))
        for j in range(center.size):
            var value = (self.linear_sum_.item(j) + other.linear_sum_.item(j)) * inverse
            center.unsafe_set(j, value)
            norm += value * value
        if not isfinite(norm) or not isfinite(squared_sum):
            raise Error("ValueError: Birch CF statistics overflow")
        var radius_squared = squared_sum / Scalar[Self.dtype](count) - norm
        if radius_squared > Scalar[Self.dtype](threshold * threshold): return False
        self.n_samples_ = count
        self.squared_sum_ = squared_sum
        self.sq_norm_ = norm
        self.centroid_ = center^
        for j in range(self.linear_sum_.size):
            self.linear_sum_.unsafe_set(j, self.linear_sum_.item(j) + other.linear_sum_.item(j))
        self._finite()
        return True

    def radius(self) raises -> Float64:
        if self.n_samples_ == 0: return 0
        var value = self.squared_sum_ / Scalar[Self.dtype](self.n_samples_) - self.sq_norm_
        return sqrt(max(Float64(0), Float64(value)))

    def _finite(self) raises:
        if not isfinite(self.squared_sum_) or not isfinite(self.sq_norm_):
            raise Error("ValueError: Birch CF statistics overflow")
        for j in range(self.linear_sum_.size):
            if not isfinite(self.linear_sum_.item(j)) or not isfinite(self.centroid_.item(j)):
                raise Error("ValueError: Birch CF statistics overflow")


struct CFNode[dtype: DType](Copyable, Movable):
    var subclusters_: List[CFSubcluster[Self.dtype]]
    var is_leaf: Bool
    var threshold: Float64
    var branching_factor: Int
    var prev_leaf_: Optional[Int]
    var next_leaf_: Optional[Int]

    def __init__(out self, threshold: Float64, branching_factor: Int, is_leaf: Bool):
        self.subclusters_ = List[CFSubcluster[Self.dtype]]()
        self.is_leaf = is_leaf
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.prev_leaf_ = None
        self.next_leaf_ = None


@fieldwise_init
struct _Split[dtype: DType](Movable):
    var left: CFSubcluster[Self.dtype]
    var right: CFSubcluster[Self.dtype]


@fieldwise_init
struct BirchParams(Copyable, Movable):
    var threshold: Float64
    var branching_factor: Int
    var n_clusters: Optional[Int]
    var compute_labels: Bool


def _finite_matrix[dtype: DType](X: nm.NDArray[dtype]) raises:
    validate_matrix(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not isfinite(X.item(i, j)): raise Error("ValueError: Birch requires finite X")


def _split_distance[dtype: DType](a: CFSubcluster[dtype], b: CFSubcluster[dtype]) raises -> Scalar[dtype]:
    var dot = Float64(0)
    var norm_a = Float64(0)
    var norm_b = Float64(0)
    for j in range(a.centroid_.size):
        var x = Float64(a.centroid_.item(j))
        var y = Float64(b.centroid_.item(j))
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    comptime if dtype == DType.float64:
        norm_a = Float64(a.sq_norm_)
        norm_b = Float64(b.sq_norm_)
    return Scalar[dtype](max(Float64(0), (-2 * dot + norm_a) + norm_b))


struct Birch[dtype: DType = DType.float64](PredictiveClusterer, ClusterDistanceTransformer, IncrementalClusterer, Copyable, Movable):
    comptime Params = BirchParams
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Labels = nm.NDArray[DType.int64]
    comptime Prediction = Self.Labels
    comptime Transformed = Self.Input
    var params: Self.Params
    var subcluster_centers_: Self.Input
    var subcluster_labels_: Self.Labels
    var labels_: Optional[Self.Labels]
    var n_features_in_: Int
    var root_: Optional[Int]
    var dummy_leaf_: Optional[Int]
    var n_clusters_warning_: Bool
    var _nodes: List[CFNode[Self.dtype]]

    def __init__(out self, *, threshold: Float64 = 0.5, branching_factor: Int = 50,
            n_clusters: Optional[Int] = 3, compute_labels: Bool = True) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.params = Self.Params(threshold, branching_factor, n_clusters, compute_labels)
        self.subcluster_centers_ = nm.zeros[Self.dtype](nm.Shape(0, 0))
        self.subcluster_labels_ = nm.zeros[DType.int64](nm.Shape(0))
        self.labels_ = None
        self.n_features_in_ = 0
        self.root_ = None
        self.dummy_leaf_ = None
        self.n_clusters_warning_ = False
        self._nodes = List[CFNode[Self.dtype]]()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, threshold: Optional[Float64] = None,
            branching_factor: Optional[Int] = None, n_clusters: Optional[Optional[Int]] = None,
            compute_labels: Optional[Bool] = None) raises -> ref[self] Self:
        if threshold: self.params.threshold = threshold.value()
        if branching_factor: self.params.branching_factor = branching_factor.value()
        if n_clusters: self.params.n_clusters = n_clusters.value()
        if compute_labels: self.params.compute_labels = compute_labels.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.root_)

    def _validate_params(self) raises:
        if not isfinite(self.params.threshold) or self.params.threshold <= 0:
            raise Error("ValueError: threshold must be finite and positive")
        if self.params.branching_factor < 2:
            raise Error("ValueError: branching_factor must be at least 2")
        if self.params.n_clusters and self.params.n_clusters.value() < 1:
            raise Error("ValueError: n_clusters must be positive or None")

    def get_node(self, index: Int) raises -> CFNode[Self.dtype]:
        if index < 0 or index >= len(self._nodes): raise Error("ValueError: invalid CF node handle")
        return self._nodes[index].copy()

    def _split(mut self, index: Int, threshold: Float64, branching_factor: Int) raises -> _Split[Self.dtype]:
        var old = self._nodes[index].copy()
        var first = 0
        var second = 0
        var farthest = Scalar[Self.dtype](-1)
        for i in range(len(old.subclusters_)):
            for j in range(len(old.subclusters_)):
                var distance = Scalar[Self.dtype](0)
                if i != j: distance = _split_distance(old.subclusters_[i], old.subclusters_[j])
                if distance > farthest:
                    first = i
                    second = j
                    farthest = distance
        var left = CFNode[Self.dtype](threshold, branching_factor, old.is_leaf)
        var right = CFNode[Self.dtype](threshold, branching_factor, old.is_leaf)
        var right_index = len(self._nodes)
        var left_cf = CFSubcluster[Self.dtype](self.n_features_in_)
        var right_cf = CFSubcluster[Self.dtype](self.n_features_in_)
        left_cf.child_ = index
        right_cf.child_ = right_index
        for i in range(len(old.subclusters_)):
            var d_left = Scalar[Self.dtype](0)
            var d_right = Scalar[Self.dtype](0)
            if i != first: d_left = _split_distance(old.subclusters_[first], old.subclusters_[i])
            if i != second: d_right = _split_distance(old.subclusters_[second], old.subclusters_[i])
            if i == first or d_left < d_right:
                left_cf.update(old.subclusters_[i])
                left.subclusters_.append(old.subclusters_[i].copy())
            else:
                right_cf.update(old.subclusters_[i])
                right.subclusters_.append(old.subclusters_[i].copy())
        if old.is_leaf:
            left.prev_leaf_ = old.prev_leaf_
            left.next_leaf_ = right_index
            right.prev_leaf_ = index
            right.next_leaf_ = old.next_leaf_
            if old.next_leaf_: self._nodes[old.next_leaf_.value()].prev_leaf_ = right_index
        self._nodes[index] = left^
        self._nodes.append(right^)
        return _Split[Self.dtype](left_cf^, right_cf^)

    def _insert(mut self, index: Int, point: CFSubcluster[Self.dtype]) raises -> Bool:
        if len(self._nodes[index].subclusters_) == 0:
            self._nodes[index].subclusters_.append(point.copy())
            return False
        var closest = 0
        var best = Scalar[Self.dtype](0)
        for i in range(len(self._nodes[index].subclusters_)):
            ref candidate = self._nodes[index].subclusters_[i]
            var dot = Scalar[Self.dtype](0)
            for j in range(self.n_features_in_): dot += candidate.centroid_.item(j) * point.centroid_.item(j)
            var score = -2 * dot + candidate.sq_norm_
            if i == 0 or score < best:
                closest = i
                best = score
        var child = self._nodes[index].subclusters_[closest].child_
        if child:
            if not self._insert(child.value(), point):
                self._nodes[index].subclusters_[closest].update(point)
                return False
            var threshold = self._nodes[index].threshold
            var branching = self._nodes[index].branching_factor
            var split = self._split(child.value(), threshold, branching)
            self._nodes[index].subclusters_[closest] = split.left.copy()
            self._nodes[index].subclusters_.append(split.right.copy())
        else:
            var threshold = self._nodes[index].threshold
            if self._nodes[index].subclusters_[closest].merge(point, threshold): return False
            self._nodes[index].subclusters_.append(point.copy())
        return len(self._nodes[index].subclusters_) > self._nodes[index].branching_factor

    def _grow(mut self, X: Self.Input) raises:
        if not self.root_:
            self.n_features_in_ = X.shape[1]
            self._nodes.append(CFNode[Self.dtype](self.params.threshold, self.params.branching_factor, True))
            self._nodes.append(CFNode[Self.dtype](self.params.threshold, self.params.branching_factor, True))
            self.dummy_leaf_ = 0
            self.root_ = 1
            self._nodes[0].next_leaf_ = 1
            self._nodes[1].prev_leaf_ = 0
        for i in range(X.shape[0]):
            var point = CFSubcluster[Self.dtype](self.n_features_in_)
            point.n_samples_ = 1
            for j in range(self.n_features_in_):
                var value = X.item(i, j)
                point.linear_sum_.unsafe_set(j, value)
                point.centroid_.unsafe_set(j, value)
                point.squared_sum_ += value * value
            point.sq_norm_ = point.squared_sum_
            point._finite()
            if self._insert(self.root_.value(), point):
                var split = self._split(self.root_.value(), self.params.threshold, self.params.branching_factor)
                var root = CFNode[Self.dtype](self.params.threshold, self.params.branching_factor, False)
                root.subclusters_.append(split.left.copy())
                root.subclusters_.append(split.right.copy())
                self.root_ = len(self._nodes)
                self._nodes.append(root^)
        var count = 0
        var leaf = self._nodes[self.dummy_leaf_.value()].next_leaf_
        while leaf:
            count += len(self._nodes[leaf.value()].subclusters_)
            leaf = self._nodes[leaf.value()].next_leaf_
        var centers = nm.zeros[Self.dtype](nm.Shape(count, self.n_features_in_))
        var row = 0
        leaf = self._nodes[self.dummy_leaf_.value()].next_leaf_
        while leaf:
            for cluster in self._nodes[leaf.value()].subclusters_:
                for j in range(self.n_features_in_): centers[nm.Item(row, j)] = cluster.centroid_.item(j)
                row += 1
            leaf = self._nodes[leaf.value()].next_leaf_
        self.subcluster_centers_ = centers^

    def _global_clustering(mut self) raises:
        var count = self.subcluster_centers_.shape[0]
        var labels = nm.zeros[DType.int64](nm.Shape(count))
        self.n_clusters_warning_ = self.params.n_clusters and count < self.params.n_clusters.value()
        if self.params.n_clusters and not self.n_clusters_warning_:
            var global_model = AgglomerativeClustering[Self.dtype](self.params.n_clusters.value())
            labels = global_model.fit_predict(self.subcluster_centers_)
        else:
            for i in range(count): labels.unsafe_set(i, Int64(i))
        self.subcluster_labels_ = labels^

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input,
            y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self._validate_params()
        _finite_matrix(X)
        var next = self.__sklearn_clone__()
        next.labels_ = self.labels_.copy()
        next._grow(X)
        next._global_clustering()
        if next.params.compute_labels: next.labels_ = next.predict(X)
        self = next^
        return self

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input,
            y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self._validate_params()
        _finite_matrix(X)
        if self.root_ and X.shape[1] != self.n_features_in_:
            raise Error("ValueError: feature count differs from fit")
        var next = self.copy()
        next._grow(X)
        next._global_clustering()
        if next.params.compute_labels: next.labels_ = next.predict(X)
        self = next^
        return self

    def partial_fit[origin: MutOrigin](ref[origin] self, X: Optional[Self.Input] = None,
            y: Optional[Self.Target] = None) raises -> ref[self] Self:
        if X:
            _ = self.partial_fit(X.value(), y)
            return self
        self._validate_params()
        self._check_fitted()
        self._global_clustering()
        return self

    def _check_fitted(self) raises:
        if not self.__sklearn_is_fitted__(): raise Error("NotFittedError: Birch is not fitted")

    def _check(self, X: Self.Input) raises:
        self._check_fitted()
        _finite_matrix(X)
        if X.shape[1] != self.n_features_in_: raise Error("ValueError: feature count differs from fit")

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        self._check(X)
        var query = NearestNeighbors[Self.dtype](n_neighbors=1, algorithm="brute")
        _ = query.fit(self.subcluster_centers_)
        var nearest = query.kneighbors(X.copy())
        var labels = nm.zeros[DType.int64](nm.Shape(X.shape[0]))
        for i in range(X.shape[0]): labels.unsafe_set(i, self.subcluster_labels_.item(nearest.indices[i]))
        return labels^

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check(X)
        return distances(X, self.subcluster_centers_, False)

    def fit_predict(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Labels:
        _ = self.fit(X, y)
        if not self.labels_: raise Error("ValueError: labels_ is unavailable with compute_labels=False")
        return self.labels_.value().copy()

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        self._check_fitted()
        if input_features and len(input_features.value()) != self.n_features_in_:
            raise Error("ValueError: input_features must match n_features_in_")
        var names = List[String]()
        for i in range(self.subcluster_centers_.shape[0]): names.append("birch" + String(i))
        return names^
