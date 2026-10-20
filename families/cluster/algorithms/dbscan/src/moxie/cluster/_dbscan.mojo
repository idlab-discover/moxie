"""Dense DBSCAN, adapted from scikit-learn (BSD-3-Clause)."""
import numojo as nm
from std.math import isfinite
from moxie.cluster._base import Clusterer
from moxie.cluster._contracts import Labels
from moxie.neighbors._contracts import validate_matrix
from moxie.neighbors._distance import Metric
from moxie.neighbors._unsupervised import NearestNeighbors


def _neighbor_weight_sum[dtype: DType](weights: nm.NDArray[dtype], indices: List[Int], start: Int, end: Int) raises -> SIMD[dtype, 1]:
    """NumPy's pairwise order over one gathered neighborhood (BSD-3-Clause)."""
    var n = end - start
    if n < 8:
        var total = SIMD[dtype, 1](-0.0)
        for i in range(start, end): total += weights.unsafe_get(indices[i] * Int(weights.strides[0]))
        return total
    if n <= 128:
        var lanes = SIMD[dtype, 8](0)
        comptime for lane in range(8): lanes[lane] = weights.unsafe_get(indices[start + lane] * Int(weights.strides[0]))
        for i in range(8, n - n % 8, 8):
            comptime for lane in range(8): lanes[lane] += weights.unsafe_get(indices[start + i + lane] * Int(weights.strides[0]))
        var total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
        for i in range(n - n % 8, n): total += weights.unsafe_get(indices[start + i] * Int(weights.strides[0]))
        return total
    var half = n // 2
    half -= half % 8
    return _neighbor_weight_sum(weights, indices, start, start + half) + _neighbor_weight_sum(weights, indices, start + half, end)


def _core_weight_sum[dtype: DType](weights: nm.NDArray[dtype], indices: List[Int], start: Int, end: Int) raises -> SIMD[dtype, 1]:
    return SIMD[dtype, 1](0) + _neighbor_weight_sum(weights, indices, start, end)


@fieldwise_init
struct DBSCANParams[dtype: DType](Copyable, Movable):
    var eps: Float64
    var min_samples: Int
    var metric: String
    var metric_params: Optional[Metric[Self.dtype]]
    var algorithm: String
    var leaf_size: Int
    var p: Optional[Float64]
    var n_jobs: Optional[Int]


@fieldwise_init
struct DBSCANResult(Copyable, Movable):
    var core_sample_indices: Labels
    var labels: Labels


struct DBSCAN[dtype: DType = DType.float64](Clusterer, Copyable, Movable):
    comptime Params = DBSCANParams[Self.dtype]
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime Labels = nm.NDArray[DType.int64]
    var params: Self.Params
    var labels_: Self.Labels
    var core_sample_indices_: Self.Labels
    var components_: Self.Input
    var n_features_in_: Int

    def __init__(out self, eps: Float64 = 0.5, *, min_samples: Int = 5,
                 metric: String = "euclidean", metric_params: Optional[Metric[Self.dtype]] = None,
                 algorithm: String = "auto", leaf_size: Int = 30,
                 p: Optional[Float64] = None, n_jobs: Optional[Int] = None) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.params = Self.Params(eps, min_samples, metric, metric_params.copy(), algorithm, leaf_size, p, n_jobs)
        self.labels_ = nm.zeros[DType.int64](nm.Shape(0))
        self.core_sample_indices_ = nm.zeros[DType.int64](nm.Shape(0))
        self.components_ = nm.zeros[Self.dtype](nm.Shape(0, 0))
        self.n_features_in_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, eps: Optional[Float64] = None, min_samples: Optional[Int] = None,
                   metric: Optional[String] = None, metric_params: Optional[Optional[Metric[Self.dtype]]] = None,
                   algorithm: Optional[String] = None, leaf_size: Optional[Int] = None,
                   p: Optional[Optional[Float64]] = None, n_jobs: Optional[Optional[Int]] = None) raises -> ref[self] Self:
        if eps: self.params.eps = eps.value()
        if min_samples: self.params.min_samples = min_samples.value()
        if metric: self.params.metric = metric.value().copy()
        if metric_params: self.params.metric_params = metric_params.value().copy()
        if algorithm: self.params.algorithm = algorithm.value().copy()
        if leaf_size: self.params.leaf_size = leaf_size.value()
        if p: self.params.p = p.value()
        if n_jobs: self.params.n_jobs = n_jobs.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        _ = self.fit(X, y, sample_weight=None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None, *, sample_weight: Optional[Self.Input]) raises -> ref[self] Self:
        ref config = self.params
        if not isfinite(config.eps) or config.eps <= 0:
            raise Error("ValueError: eps must be finite and strictly positive")
        if config.min_samples <= 0:
            raise Error("ValueError: min_samples must be positive")
        if config.p and (not isfinite(config.p.value()) or config.p.value() < 0):
            raise Error("ValueError: p must be finite and nonnegative")
        var power = Float64(2)
        if config.p and (config.metric == "minkowski" or config.metric == "p"):
            power = config.p.value()
        var effective = Metric[Self.dtype]("minkowski" if config.metric == "p" else config.metric, power)
        if config.metric_params:
            effective = config.metric_params.value().copy()
            var declared = Metric[Self.dtype]("minkowski" if config.metric == "p" else config.metric,
                effective.p, w=effective.w.copy(), V=effective.V.copy(), VI=effective.VI.copy())
            if declared.name != effective.name:
                raise Error("ValueError: metric_params name must agree with metric")
        if effective.name != "euclidean" and effective.name != "manhattan" and effective.name != "chebyshev" and effective.name != "minkowski" and effective.name != "precomputed":
            raise Error("NotImplementedError: DBSCAN supports Euclidean, Manhattan, Chebyshev, Minkowski and dense precomputed metrics")
        validate_matrix(X, precomputed=effective.name == "precomputed", training=True)
        if sample_weight:
            ref weights = sample_weight.value()
            if weights.ndim != 1 or weights.shape[0] != X.shape[0]:
                raise Error("ValueError: sample_weight must have shape (n_samples,)")
            for i in range(weights.shape[0]):
                if not isfinite(weights.unsafe_get(i * Int(weights.strides[0]))):
                    raise Error("ValueError: sample_weight must be finite")
        var query = NearestNeighbors[Self.dtype](radius=config.eps, algorithm=config.algorithm,
            leaf_size=config.leaf_size, metric=config.metric, p=power,
            metric_params=effective.copy(), n_jobs=config.n_jobs)
        _ = query.fit(X)
        # Explicit X keeps the training point; a no-argument query excludes it.
        var neighbors = query.radius_neighbors(X.copy(), return_distance=False)
        var n = X.shape[0]
        var core = List[Bool]()
        var core_indices = List[Int]()
        for i in range(n):
            var start = neighbors.offsets[i]
            var end = neighbors.offsets[i + 1]
            var is_core = end - start >= config.min_samples
            if sample_weight:
                var total = _core_weight_sum(sample_weight.value(), neighbors.indices, start, end)
                is_core = total >= SIMD[Self.dtype, 1](config.min_samples)
            core.append(is_core)
            if is_core: core_indices.append(i)
        var labels = nm.full[DType.int64](nm.Shape(n), -1)
        var stack = List[Int]()
        var label_num = 0
        for seed in range(n):
            if labels.item(seed) != -1 or not core[seed]: continue
            var i = seed
            while True:
                if labels.item(i) == -1:
                    labels.unsafe_set(i, Int64(label_num))
                    if core[i]:
                        for j in range(neighbors.offsets[i], neighbors.offsets[i + 1]):
                            var neighbor = neighbors.indices[j]
                            if labels.item(neighbor) == -1: stack.append(neighbor)
                if len(stack) == 0: break
                i = stack.pop()
            label_num += 1
        var indices = nm.zeros[DType.int64](nm.Shape(len(core_indices)))
        var components = nm.zeros[Self.dtype](nm.Shape(len(core_indices), X.shape[1]))
        for i in range(len(core_indices)):
            indices.unsafe_set(i, Int64(core_indices[i]))
            for j in range(X.shape[1]):
                components[nm.Item(i, j)] = X.item(core_indices[i], j)
        self.labels_ = labels^
        self.core_sample_indices_ = indices^
        self.components_ = components^
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Labels:
        _ = self.fit(X, y)
        return self.labels_.copy()

    def fit_predict(mut self, X: Self.Input, y: Optional[Self.Target] = None, *, sample_weight: Optional[Self.Input]) raises -> Self.Labels:
        _ = self.fit(X, y, sample_weight=sample_weight)
        return self.labels_.copy()


def dbscan[dtype: DType](X: nm.NDArray[dtype], eps: Float64 = 0.5, *, min_samples: Int = 5,
                        metric: String = "minkowski", metric_params: Optional[Metric[dtype]] = None,
                        algorithm: String = "auto", leaf_size: Int = 30, p: Float64 = 2,
                        sample_weight: Optional[nm.NDArray[dtype]] = None,
                        n_jobs: Optional[Int] = None) raises -> DBSCANResult:
    var model = DBSCAN[dtype](eps, min_samples=min_samples, metric=metric,
        metric_params=metric_params.copy(), algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs)
    _ = model.fit(X, sample_weight=sample_weight)
    return DBSCANResult(model.core_sample_indices_.copy(), model.labels_.copy())
