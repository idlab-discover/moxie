"""Dense OPTICS and Xi extraction, adapted from scikit-learn (BSD-3-Clause)."""
import numojo as nm
from std.math import isfinite, isnan, max, round, inf
from moxie.cluster._base import Clusterer
from moxie.cluster._contracts import Labels
from moxie.neighbors._contracts import validate_matrix
from moxie.neighbors._distance import Metric
from moxie.neighbors._unsupervised import NearestNeighbors

comptime Distances = nm.NDArray[DType.float64]


struct OPTICSSize(Copyable, Movable):
    """Distinguish integer counts from fractions, including integer versus float 1."""
    var value: Float64
    var fraction: Bool

    @implicit
    def __init__(out self, count: Int):
        self.value = Float64(count)
        self.fraction = False

    @implicit
    def __init__(out self, fraction: Float64):
        self.value = fraction
        self.fraction = True

    def validate(self, allow_zero: Bool = True) raises:
        if not isfinite(self.value):
            raise Error("ValueError: OPTICS size must be finite")
        if self.fraction:
            if self.value < 0 or self.value > 1 or (not allow_zero and self.value == 0):
                raise Error("ValueError: OPTICS fraction must lie in [0, 1] (min_cluster_size excludes 0)")
        elif self.value < 2:
            raise Error("ValueError: OPTICS count must be at least 2")

    def resolve(self, n: Int, allow_zero: Bool = True) raises -> Int:
        self.validate(allow_zero)
        if self.fraction: return max(2, Int(self.value * Float64(n)))
        if self.value > Float64(n):
            raise Error("ValueError: OPTICS count must not exceed n_samples")
        return Int(self.value)



@fieldwise_init
struct OPTICSGraph(Copyable, Movable):
    var ordering: Labels
    var core_distances: Distances
    var reachability: Distances
    var predecessor: Labels


@fieldwise_init
struct OPTICSXiResult(Copyable, Movable):
    var labels: Labels
    var clusters: Labels


def _round_distance(value: Float64) -> Float64:
    if not isfinite(value): return value
    return round(value * 1e15) / 1e15


def compute_optics_graph[dtype: DType](X: nm.NDArray[dtype], *,
        min_samples: OPTICSSize, max_eps: Float64, metric: String, p: Float64,
        metric_params: Optional[Metric[DType.float64]], algorithm: String,
        leaf_size: Int, n_jobs: Optional[Int]) raises -> OPTICSGraph:
    validate_matrix(X, precomputed=metric == "precomputed", training=True)
    var n = X.shape[0]
    var k = min_samples.resolve(n)
    if isnan(max_eps) or max_eps < 0:
        raise Error("ValueError: max_eps must be nonnegative")
    if isnan(p) or p <= 0:
        raise Error("ValueError: p must be positive")
    var effective = Metric[DType.float64](metric, p)
    if metric_params:
        effective = metric_params.value().copy()
        var declared = Metric[DType.float64](metric, effective.p, w=effective.w.copy(), V=effective.V.copy(), VI=effective.VI.copy())
        if declared.name != effective.name:
            raise Error("ValueError: metric_params name must agree with metric")
    if effective.name != "euclidean" and effective.name != "manhattan" and effective.name != "chebyshev" and effective.name != "minkowski" and effective.name != "precomputed":
        raise Error("NotImplementedError: OPTICS supports Euclidean, Manhattan, Chebyshev, Minkowski and dense precomputed metrics")
    # sklearn OPTICS.fit explicitly requests float (Float64), even for Float32 X.
    var data = nm.zeros[DType.float64](nm.Shape(n, X.shape[1]))
    for i in range(n):
        for j in range(X.shape[1]): data[nm.Item(i, j)] = Float64(X.item(i, j))
    var query = NearestNeighbors[DType.float64](n_neighbors=k, algorithm=algorithm,
        leaf_size=leaf_size, metric=metric, p=p, metric_params=effective.copy(), n_jobs=n_jobs)
    _ = query.fit(data)
    var core = nm.full[DType.float64](nm.Shape(n), inf[DType.float64]())
    var reach = nm.full[DType.float64](nm.Shape(n), inf[DType.float64]())
    var predecessor = nm.full[DType.int64](nm.Shape(n), -1)
    var ordering = nm.zeros[DType.int64](nm.Shape(n))
    var processed = nm.zeros[DType.bool](nm.Shape(n))
    var row = nm.zeros[DType.float64](nm.Shape(1, X.shape[1]))
    for i in range(n):
        for j in range(X.shape[1]): row[nm.Item(0, j)] = data.item(i, j)
        var neighbors = query.kneighbors(row.copy())
        var distance = neighbors.distances.value().item(0, k - 1)
        if distance <= max_eps: core.unsafe_set(i, _round_distance(distance))
    for position in range(n):
        var point = -1
        for i in range(n):
            if not processed.item(i):
                if point == -1 or reach.item(i) < reach.item(point): point = i
        processed.unsafe_set(point, True)
        ordering.unsafe_set(position, Int64(point))
        if not isfinite(core.item(point)): continue
        for j in range(X.shape[1]): row[nm.Item(0, j)] = data.item(point, j)
        var neighbors = query.radius_neighbors(row.copy(), radius=max_eps)
        for offset in range(len(neighbors.indices)):
            var candidate = neighbors.indices[offset]
            if processed.item(candidate): continue
            var distance = neighbors.distances.value().item(offset)
            var proposed = _round_distance(max(distance, core.item(point)))
            if proposed < reach.item(candidate):
                reach.unsafe_set(candidate, proposed)
                predecessor.unsafe_set(candidate, Int64(point))
    return OPTICSGraph(ordering^, core^, reach^, predecessor^)


def _validate_plot(reachability: Distances, ordering: Labels) raises:
    if reachability.ndim != 1 or ordering.ndim != 1 or reachability.shape[0] != ordering.shape[0]:
        raise Error("ValueError: reachability and ordering must be equally sized vectors")
    var n = ordering.shape[0]
    var seen = nm.zeros[DType.bool](nm.Shape(n))
    for i in range(n):
        var index = Int(ordering.item(i))
        if index < 0 or index >= n: raise Error("ValueError: ordering index out of range")
        if seen.item(index): raise Error("ValueError: ordering must be a permutation")
        seen.unsafe_set(index, True)
        if isnan(reachability.item(i)) or reachability.item(i) < 0:
            raise Error("ValueError: reachability must be nonnegative and not NaN")


def cluster_optics_dbscan(*, reachability: Distances, core_distances: Distances,
                          ordering: Labels, eps: Float64) raises -> Labels:
    _validate_plot(reachability, ordering)
    if isnan(eps) or eps < 0: raise Error("ValueError: eps must be nonnegative")
    var n = ordering.shape[0]
    if core_distances.ndim != 1 or core_distances.shape[0] != n:
        raise Error("ValueError: core_distances must match reachability")
    var labels = nm.full[DType.int64](nm.Shape(n), -1)
    var label = -1
    for i in range(n):
        var point = Int(ordering.item(i))
        var core = core_distances.item(point)
        if isnan(core) or core < 0: raise Error("ValueError: invalid core distance")
        var far = reachability.item(point) > eps
        var near = core <= eps
        if far and near: label += 1
        if not (far and not near): labels.unsafe_set(point, Int64(label))
    return labels^


@fieldwise_init
struct _SteepArea(Copyable, Movable):
    var start: Int
    var end: Int
    var mib: Float64


def _extend_region(steep: nm.NDArray[DType.bool], opposite: nm.NDArray[DType.bool], start: Int, min_samples: Int) raises -> Int:
    var end = start
    var nonsteep = 0
    for i in range(start, steep.shape[0]):
        if steep.item(i):
            nonsteep = 0
            end = i
        elif not opposite.item(i):
            nonsteep += 1
            if nonsteep > min_samples: break
        else: break
    return end


def cluster_optics_xi(*, reachability: Distances, predecessor: Labels, ordering: Labels,
        min_samples: OPTICSSize, min_cluster_size: Optional[OPTICSSize] = None,
        xi: Float64 = 0.05, predecessor_correction: Bool = True) raises -> OPTICSXiResult:
    _validate_plot(reachability, ordering)
    var n = ordering.shape[0]
    var minimum = min_samples.resolve(n)
    var cluster_size = minimum
    if min_cluster_size: cluster_size = min_cluster_size.value().resolve(n, False)
    if not isfinite(xi) or xi < 0 or xi >= 1: raise Error("ValueError: Xi extraction requires 0 <= xi < 1; xi=1 is undefined")
    if predecessor.ndim != 1 or predecessor.shape[0] != n:
        raise Error("ValueError: predecessor must match reachability")
    var plot = nm.full[DType.float64](nm.Shape(n + 1), inf[DType.float64]())
    var predecessors = nm.zeros[DType.int64](nm.Shape(n))
    for i in range(n):
        var point = Int(ordering.item(i))
        plot.unsafe_set(i, reachability.item(point))
        var pred = predecessor.item(point)
        if pred < -1 or pred >= Int64(n): raise Error("ValueError: predecessor index out of range")
        predecessors.unsafe_set(i, pred)
    var steep_up = nm.zeros[DType.bool](nm.Shape(n))
    var steep_down = nm.zeros[DType.bool](nm.Shape(n))
    var upward = nm.zeros[DType.bool](nm.Shape(n))
    var downward = nm.zeros[DType.bool](nm.Shape(n))
    var complement = 1 - xi
    var inverse = 1 / complement
    for i in range(n):
        var ratio = plot.item(i) / plot.item(i + 1)
        steep_up.unsafe_set(i, ratio <= complement)
        steep_down.unsafe_set(i, ratio >= inverse)
        upward.unsafe_set(i, ratio < 1)
        downward.unsafe_set(i, ratio > 1)
    var areas = List[_SteepArea]()
    var clusters = List[Tuple[Int, Int]]()
    var index = 0
    var mib = Float64(0)
    for steep_index in range(n):
        if steep_index < index or not (steep_up.item(steep_index) or steep_down.item(steep_index)): continue
        for i in range(index, steep_index + 1): mib = max(mib, plot.item(i))
        var retained = List[_SteepArea]()
        if isfinite(mib):
            for area in areas:
                if mib <= plot.item(area.start) * complement:
                    retained.append(_SteepArea(area.start, area.end, max(area.mib, mib)))
        areas = retained^
        if steep_down.item(steep_index):
            var end = _extend_region(steep_down, upward, steep_index, minimum)
            areas.append(_SteepArea(steep_index, end, 0))
            index = end + 1
            mib = plot.item(index)
            continue
        var up_end = _extend_region(steep_up, downward, steep_index, minimum)
        index = up_end + 1
        mib = plot.item(index)
        var found = List[Tuple[Int, Int]]()
        for area in areas:
            var start = area.start
            var end = up_end
            if plot.item(end + 1) * complement < area.mib: continue
            var down_max = plot.item(area.start)
            if down_max * complement >= plot.item(end + 1):
                while start < area.end and plot.item(start + 1) > plot.item(end + 1): start += 1
            elif plot.item(end + 1) * complement >= down_max:
                while end > steep_index and plot.item(end - 1) > down_max: end -= 1
            if predecessor_correction:
                while start < end:
                    if plot.item(start) > plot.item(end): break
                    var contains = False
                    for i in range(start, end):
                        if predecessors.item(end) == ordering.item(i):
                            contains = True
                            break
                    if contains: break
                    end -= 1
                if start == end: continue
            if end - start + 1 < cluster_size or start > area.end or end < steep_index: continue
            found.append((start, end))
        for i in range(len(found) - 1, -1, -1): clusters.append(found[i])
    var hierarchy = nm.zeros[DType.int64](nm.Shape(len(clusters), 2))
    var ordered_labels = nm.full[DType.int64](nm.Shape(n), -1)
    var label = 0
    for i in range(len(clusters)):
        var start = clusters[i][0]
        var end = clusters[i][1]
        hierarchy[nm.Item(i, 0)] = Int64(start)
        hierarchy[nm.Item(i, 1)] = Int64(end)
        var available = True
        for j in range(start, end + 1):
            if ordered_labels.item(j) != -1: available = False
        if available:
            for j in range(start, end + 1): ordered_labels.unsafe_set(j, Int64(label))
            label += 1
    var labels = nm.full[DType.int64](nm.Shape(n), -1)
    for i in range(n): labels.unsafe_set(Int(ordering.item(i)), ordered_labels.item(i))
    return OPTICSXiResult(labels^, hierarchy^)


@fieldwise_init
struct OPTICSParams(Copyable, Movable):
    var min_samples: OPTICSSize
    var max_eps: Float64
    var metric: String
    var p: Float64
    var metric_params: Optional[Metric[DType.float64]]
    var cluster_method: String
    var eps: Optional[Float64]
    var xi: Float64
    var predecessor_correction: Bool
    var min_cluster_size: Optional[OPTICSSize]
    var algorithm: String
    var leaf_size: Int
    var memory: Optional[String]
    var n_jobs: Optional[Int]


struct OPTICS[dtype: DType = DType.float64](Clusterer, Copyable, Movable):
    comptime Params = OPTICSParams
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Labels = nm.NDArray[DType.int64]
    var params: Self.Params
    var labels_: Self.Labels
    var ordering_: Self.Labels
    var core_distances_: Distances
    var reachability_: Distances
    var predecessor_: Self.Labels
    var cluster_hierarchy_: Optional[Self.Labels]
    var n_features_in_: Int

    def __init__(out self, *, min_samples: OPTICSSize = OPTICSSize(5),
                 max_eps: Float64 = inf[DType.float64](),
                 metric: String = "minkowski",
                 p: Float64 = 2,
                 metric_params: Optional[Metric[DType.float64]] = None,
                 cluster_method: String = "xi",
                 eps: Optional[Float64] = None,
                 xi: Float64 = 0.05,
                 predecessor_correction: Bool = True,
                 min_cluster_size: Optional[OPTICSSize] = None,
                 algorithm: String = "auto",
                 leaf_size: Int = 30,
                 memory: Optional[String] = None,
                 n_jobs: Optional[Int] = None) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.params = Self.Params(min_samples.copy(), max_eps, metric.copy(), p,
            metric_params.copy(), cluster_method.copy(), eps, xi, predecessor_correction,
            min_cluster_size.copy(), algorithm.copy(), leaf_size, memory.copy(), n_jobs)
        self.labels_ = nm.zeros[DType.int64](nm.Shape(0))
        self.ordering_ = nm.zeros[DType.int64](nm.Shape(0))
        self.predecessor_ = nm.zeros[DType.int64](nm.Shape(0))
        self.core_distances_ = nm.zeros[DType.float64](nm.Shape(0))
        self.reachability_ = nm.zeros[DType.float64](nm.Shape(0))
        self.cluster_hierarchy_ = None
        self.n_features_in_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, min_samples: Optional[OPTICSSize] = None,
                   max_eps: Optional[Float64] = None,
                   metric: Optional[String] = None,
                   p: Optional[Float64] = None,
                   metric_params: Optional[Optional[Metric[DType.float64]]] = None,
                   cluster_method: Optional[String] = None,
                   eps: Optional[Optional[Float64]] = None,
                   xi: Optional[Float64] = None,
                   predecessor_correction: Optional[Bool] = None,
                   min_cluster_size: Optional[Optional[OPTICSSize]] = None,
                   algorithm: Optional[String] = None,
                   leaf_size: Optional[Int] = None,
                   memory: Optional[Optional[String]] = None,
                   n_jobs: Optional[Optional[Int]] = None) raises -> ref[self] Self:
        if min_samples: self.params.min_samples = min_samples.value().copy()
        if max_eps: self.params.max_eps = max_eps.value()
        if metric: self.params.metric = metric.value().copy()
        if p: self.params.p = p.value()
        if metric_params: self.params.metric_params = metric_params.value().copy()
        if cluster_method: self.params.cluster_method = cluster_method.value().copy()
        if eps: self.params.eps = eps.value()
        if xi: self.params.xi = xi.value()
        if predecessor_correction: self.params.predecessor_correction = predecessor_correction.value()
        if min_cluster_size: self.params.min_cluster_size = min_cluster_size.value().copy()
        if algorithm: self.params.algorithm = algorithm.value().copy()
        if leaf_size: self.params.leaf_size = leaf_size.value()
        if memory: self.params.memory = memory.value().copy()
        if n_jobs: self.params.n_jobs = n_jobs.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        ref config = self.params
        if config.memory: raise Error("NotImplementedError: OPTICS memory caching is not supported")
        if config.cluster_method != "xi" and config.cluster_method != "dbscan":
            raise Error("ValueError: cluster_method must be xi or dbscan")
        if isnan(config.p) or config.p < 1: raise Error("ValueError: OPTICS p must be at least 1")
        if not isfinite(config.xi) or config.xi < 0 or config.xi > 1:
            raise Error("ValueError: xi must lie in [0, 1]")
        if config.eps and (isnan(config.eps.value()) or config.eps.value() < 0):
            raise Error("ValueError: eps must be nonnegative")
        validate_matrix(X, precomputed=config.metric == "precomputed", training=True)
        if config.min_cluster_size:
            config.min_cluster_size.value().validate(False)
        var graph = compute_optics_graph(X, min_samples=config.min_samples.copy(),
            max_eps=config.max_eps, metric=config.metric, p=config.p,
            metric_params=config.metric_params.copy(), algorithm=config.algorithm,
            leaf_size=config.leaf_size, n_jobs=config.n_jobs)
        var labels: Self.Labels
        var hierarchy = Optional[Self.Labels](None)
        if config.cluster_method == "xi":
            var extracted = cluster_optics_xi(reachability=graph.reachability,
                predecessor=graph.predecessor, ordering=graph.ordering,
                min_samples=config.min_samples.copy(), min_cluster_size=config.min_cluster_size.copy(),
                xi=config.xi, predecessor_correction=config.predecessor_correction)
            labels = extracted.labels.copy()
            hierarchy = extracted.clusters.copy()
        else:
            var eps = config.max_eps
            if config.eps: eps = config.eps.value()
            if eps > config.max_eps: raise Error("ValueError: eps must not exceed max_eps")
            labels = cluster_optics_dbscan(reachability=graph.reachability,
                core_distances=graph.core_distances, ordering=graph.ordering, eps=eps)
        self.labels_ = labels^
        self.ordering_ = graph.ordering.copy()
        self.core_distances_ = graph.core_distances.copy()
        self.reachability_ = graph.reachability.copy()
        self.predecessor_ = graph.predecessor.copy()
        self.cluster_hierarchy_ = hierarchy^
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Labels:
        _ = self.fit(X, y)
        return self.labels_.copy()
