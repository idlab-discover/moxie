"""Dense flat-kernel MeanShift, adapted from scikit-learn (BSD-3-Clause)."""
import numojo as nm
from std.math import round, isfinite, sqrt
from moxie._dense import sum_axis2d, row_squared_norms
from moxie._random import RandomState, check_random_state
from moxie.cluster._base import PredictiveClusterer
from moxie.neighbors._contracts import validate_matrix
from moxie.neighbors._unsupervised import NearestNeighbors


def estimate_bandwidth[dtype: DType](X: nm.NDArray[dtype], *, quantile: Float64 = 0.3,
        n_samples: Optional[Int] = None, mut random_state: RandomState,
        n_jobs: Optional[Int] = None) raises -> Float64:
    validate_matrix(X)
    if not isfinite(quantile) or quantile < 0 or quantile > 1:
        raise Error("ValueError: quantile must be in [0, 1]")
    if n_samples and n_samples.value() <= 0:
        raise Error("ValueError: n_samples must be positive")
    var data = X.copy()
    if n_samples:
        var indices = random_state.permutation(X.shape[0])
        var count = min(n_samples.value(), X.shape[0])
        data = nm.zeros[dtype](nm.Shape(count, X.shape[1]))
        for i in range(count):
            for j in range(X.shape[1]): data[nm.Item(i, j)] = X.item(Int(indices.item(i)), j)
    var k = max(1, Int(Float64(data.shape[0]) * quantile))
    var query = NearestNeighbors[dtype](n_neighbors=k, n_jobs=n_jobs)
    _ = query.fit(data)
    var total = Float64(0)
    for start in range(0, data.shape[0], 500):
        var count = min(500, data.shape[0] - start)
        var batch = nm.zeros[dtype](nm.Shape(count, data.shape[1]))
        for i in range(count):
            for j in range(data.shape[1]): batch[nm.Item(i, j)] = data.item(start + i, j)
        var neighbors = query.kneighbors(batch^)
        for i in range(count): total += Float64(neighbors.distances.value().item(i, k - 1))
    return total / Float64(data.shape[0])


def estimate_bandwidth[dtype: DType](X: nm.NDArray[dtype], *, quantile: Float64 = 0.3,
        n_samples: Optional[Int] = None, random_state: Optional[Int] = 0,
        n_jobs: Optional[Int] = None) raises -> Float64:
    with check_random_state(random_state) as rng:
        return estimate_bandwidth(X, quantile=quantile, n_samples=n_samples, random_state=rng[], n_jobs=n_jobs)


def get_bin_seeds[dtype: DType](X: nm.NDArray[dtype], bin_size: Float64,
        min_bin_freq: Int = 1) raises -> nm.NDArray[dtype]:
    validate_matrix(X)
    if not isfinite(bin_size) or bin_size < 0 or min_bin_freq < 1:
        raise Error("ValueError: bin_size must be finite and nonnegative; min_bin_freq must be positive")
    if bin_size == 0: return X.copy()
    var bins = List[nm.NDArray[dtype]]()
    var counts = List[Int]()
    for i in range(X.shape[0]):
        var point = nm.zeros[dtype](nm.Shape(X.shape[1]))
        for j in range(X.shape[1]):
            var scaled = X.item(i, j) / SIMD[dtype, 1](bin_size)
            if not isfinite(scaled): raise Error("ValueError: bin coordinates overflow")
            point.unsafe_set(j, round(scaled))
        var found = -1
        for b in range(len(bins)):
            var equal = True
            for j in range(X.shape[1]): equal = equal and point.item(j) == bins[b].item(j)
            if equal:
                found = b
                break
        if found < 0:
            bins.append(point^)
            counts.append(1)
        else: counts[found] += 1
    var count = 0
    for frequency in counts:
        if frequency >= min_bin_freq: count += 1
    if count == X.shape[0]: return X.copy()
    var seeds = nm.zeros[dtype](nm.Shape(count, X.shape[1]))
    var row = 0
    for b in range(len(bins)):
        if counts[b] < min_bin_freq: continue
        for j in range(X.shape[1]):
            var value = Float32(bins[b].item(j)) * Float32(bin_size)
            if not isfinite(value): raise Error("ValueError: bin seeds overflow Float32")
            seeds[nm.Item(row, j)] = value.cast[dtype]()
        row += 1
    return seeds^


@fieldwise_init
struct MeanShiftParams[dtype: DType](Copyable, Movable):
    var bandwidth: Optional[Float64]
    var seeds: Optional[nm.NDArray[Self.dtype]]
    var bin_seeding: Bool
    var min_bin_freq: Int
    var cluster_all: Bool
    var n_jobs: Optional[Int]
    var max_iter: Int


@fieldwise_init
struct MeanShiftResult[dtype: DType](Copyable, Movable):
    var cluster_centers: nm.NDArray[Self.dtype]
    var labels: nm.NDArray[DType.int64]


def _labels[dtype: DType](X: nm.NDArray[dtype], centers: nm.NDArray[dtype],
        cluster_all: Bool, bandwidth: Float64, n_jobs: Optional[Int]) raises -> nm.NDArray[DType.int64]:
    var query = NearestNeighbors[dtype](n_neighbors=1, n_jobs=n_jobs)
    _ = query.fit(centers)
    var neighbors = query.kneighbors(X.copy())
    var labels = nm.full[DType.int64](nm.Shape(X.shape[0]), -1)
    for i in range(X.shape[0]):
        if cluster_all or Float64(neighbors.distances.value().item(i, 0)) <= bandwidth:
            labels.unsafe_set(i, Int64(neighbors.indices[i]))
    return labels^


struct MeanShift[dtype: DType = DType.float64](PredictiveClusterer, Copyable, Movable):
    comptime Params = MeanShiftParams[Self.dtype]
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime Labels = nm.NDArray[DType.int64]
    comptime Prediction = Self.Labels
    var params: Self.Params
    var cluster_centers_: Self.Input
    var labels_: Self.Labels
    var n_features_in_: Int
    var n_iter_: Int

    def __init__(out self, *, bandwidth: Optional[Float64] = None,
            seeds: Optional[Self.Input] = None, bin_seeding: Bool = False,
            min_bin_freq: Int = 1, cluster_all: Bool = True,
            n_jobs: Optional[Int] = None, max_iter: Int = 300) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.params = Self.Params(bandwidth, seeds.copy(), bin_seeding, min_bin_freq, cluster_all, n_jobs, max_iter)
        self.cluster_centers_ = nm.zeros[Self.dtype](nm.Shape(0, 0))
        self.labels_ = nm.zeros[DType.int64](nm.Shape(0))
        self.n_features_in_ = 0
        self.n_iter_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, bandwidth: Optional[Optional[Float64]] = None,
            seeds: Optional[Optional[Self.Input]] = None, bin_seeding: Optional[Bool] = None,
            min_bin_freq: Optional[Int] = None, cluster_all: Optional[Bool] = None,
            n_jobs: Optional[Optional[Int]] = None, max_iter: Optional[Int] = None) raises -> ref[self] Self:
        if bandwidth: self.params.bandwidth = bandwidth.value()
        if seeds: self.params.seeds = seeds.value().copy()
        if bin_seeding: self.params.bin_seeding = bin_seeding.value()
        if min_bin_freq: self.params.min_bin_freq = min_bin_freq.value()
        if cluster_all: self.params.cluster_all = cluster_all.value()
        if n_jobs: self.params.n_jobs = n_jobs.value()
        if max_iter: self.params.max_iter = max_iter.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        validate_matrix(X)
        ref config = self.params
        if config.max_iter < 0 or config.min_bin_freq < 1:
            raise Error("ValueError: max_iter must be nonnegative and min_bin_freq positive")
        if config.n_jobs and config.n_jobs.value() == 0: raise Error("ValueError: n_jobs cannot be zero")
        var bandwidth: Float64
        if config.bandwidth:
            bandwidth = config.bandwidth.value()
            if not isfinite(bandwidth) or bandwidth <= 0:
                raise Error("ValueError: bandwidth must be finite and strictly positive")
        else: bandwidth = estimate_bandwidth(X, n_jobs=config.n_jobs)
        var seeds = X.copy()
        if config.seeds: seeds = config.seeds.value().copy()
        elif config.bin_seeding: seeds = get_bin_seeds(X, bandwidth, config.min_bin_freq)
        validate_matrix(seeds)
        if seeds.shape[1] != X.shape[1]: raise Error("ValueError: seeds must match X features")
        var query = NearestNeighbors[Self.dtype](radius=bandwidth, n_jobs=1)
        _ = query.fit(X)
        var candidates = List[Self.Input]()
        var intensities = List[Int]()
        var n_iter = 0
        for seed in range(seeds.shape[0]):
            var center = nm.zeros[Self.dtype](nm.Shape(1, X.shape[1]))
            for j in range(X.shape[1]): center[nm.Item(0, j)] = seeds.item(seed, j)
            var completed = 0
            var count: Int
            while True:
                var neighbors = query.radius_neighbors(center.copy(), return_distance=False)
                count = len(neighbors.indices)
                if count == 0: break
                var points = nm.zeros[Self.dtype](nm.Shape(count, X.shape[1]))
                for i in range(count):
                    for j in range(X.shape[1]): points[nm.Item(i, j)] = X.item(neighbors.indices[i], j)
                var sums = sum_axis2d(points, 0)
                var delta = nm.zeros[Self.dtype](nm.Shape(1, X.shape[1]))
                for j in range(X.shape[1]):
                    var value = sums.item(j) / SIMD[Self.dtype, 1](count)
                    delta[nm.Item(0, j)] = value - center.item(0, j)
                    center[nm.Item(0, j)] = value
                validate_matrix(center)
                var norm = row_squared_norms(delta)
                if sqrt(Float64(norm.item(0))) <= 1e-3 * bandwidth or completed == config.max_iter: break
                completed += 1
            n_iter = max(n_iter, completed)
            if count == 0: continue
            var duplicate = -1
            for i in range(len(candidates)):
                var equal = True
                for j in range(X.shape[1]): equal = equal and candidates[i].item(0, j) == center.item(0, j)
                if equal:
                    duplicate = i
                    break
            if duplicate >= 0: intensities[duplicate] = count
            else:
                candidates.append(center^)
                intensities.append(count)
        if len(candidates) == 0: raise Error("ValueError: No point was within bandwidth of any seed")
        # Stable insertion order, sorted by descending (intensity, center tuple).
        var order = List[Int]()
        for i in range(len(candidates)):
            order.append(i)
            var pos = len(order) - 1
            while pos > 0:
                var previous = order[pos - 1]
                var before = intensities[i] > intensities[previous]
                if intensities[i] == intensities[previous]:
                    for j in range(X.shape[1]):
                        if candidates[i].item(0, j) != candidates[previous].item(0, j):
                            before = candidates[i].item(0, j) > candidates[previous].item(0, j)
                            break
                if not before: break
                order[pos] = previous
                pos -= 1
            order[pos] = i
        var sorted_centers = nm.zeros[Self.dtype](nm.Shape(len(order), X.shape[1]))
        for i in range(len(order)):
            for j in range(X.shape[1]): sorted_centers[nm.Item(i, j)] = candidates[order[i]].item(0, j)
        var merger = NearestNeighbors[Self.dtype](radius=bandwidth, n_jobs=config.n_jobs)
        _ = merger.fit(sorted_centers)
        var unique = List[Bool]()
        for _ in range(len(order)): unique.append(True)
        var kept = List[Int]()
        for i in range(len(order)):
            if not unique[i]: continue
            var neighbors = merger.radius_neighbors(candidates[order[i]].copy(), return_distance=False)
            for idx in neighbors.indices: unique[idx] = False
            unique[i] = True
        for i in range(len(order)):
            if unique[i]: kept.append(i)
        var centers = nm.zeros[Self.dtype](nm.Shape(len(kept), X.shape[1]))
        for i in range(len(kept)):
            for j in range(X.shape[1]): centers[nm.Item(i, j)] = sorted_centers.item(kept[i], j)
        var labels = _labels(X, centers, config.cluster_all, bandwidth, config.n_jobs)
        self.cluster_centers_ = centers^
        self.labels_ = labels^
        self.n_iter_ = n_iter
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Labels:
        _ = self.fit(X, y)
        return self.labels_.copy()

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        if not self.__sklearn_is_fitted__(): raise Error("NotFittedError: MeanShift is not fitted")
        validate_matrix(X)
        if X.shape[1] != self.n_features_in_: raise Error("ValueError: X features do not match fitted MeanShift")
        return _labels(X, self.cluster_centers_, True, 0, self.params.n_jobs)


def mean_shift[dtype: DType](X: nm.NDArray[dtype], *, bandwidth: Optional[Float64] = None,
        seeds: Optional[nm.NDArray[dtype]] = None, bin_seeding: Bool = False,
        min_bin_freq: Int = 1, cluster_all: Bool = True, max_iter: Int = 300,
        n_jobs: Optional[Int] = None) raises -> MeanShiftResult[dtype]:
    var model = MeanShift[dtype](bandwidth=bandwidth, seeds=seeds.copy(), bin_seeding=bin_seeding,
        min_bin_freq=min_bin_freq, cluster_all=cluster_all, max_iter=max_iter, n_jobs=n_jobs)
    _ = model.fit(X)
    return MeanShiftResult(model.cluster_centers_.copy(), model.labels_.copy())
