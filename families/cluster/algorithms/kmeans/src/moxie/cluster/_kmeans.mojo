"""Dense KMeans and weighted greedy seeding, adapted from sklearn (BSD-3-Clause)."""
import numojo as nm
from std.math import log, max, min, isfinite
from moxie._dense import row_squared_norms
from moxie._random import RandomState, check_random_state
from moxie.utils.validation import FittedState
from moxie.cluster._base import PredictiveClusterer, ClusterDistanceTransformer
from moxie.cluster._contracts import Labels, CentroidResult, validate_matrix
from moxie.cluster._centroid_engine import (
    _at, _weight, _finite, _weights, assign, update, distances, inertia,
    center_shift, ElkanBounds,
)


@fieldwise_init
struct KMeansPlusPlusResult[dtype: DType](Copyable, Movable):
    var centers: nm.NDArray[Self.dtype]
    var indices: Labels


def _sample_weights[dtype: DType](x: nm.NDArray[dtype], weights: Optional[nm.NDArray[dtype]]) raises -> nm.NDArray[dtype]:
    if weights:
        _weights(weights.value(), x.shape[0])
        return weights.value().copy()
    return nm.ones[dtype](nm.Shape(x.shape[0]))


def _probabilities[dtype: DType](weights: nm.NDArray[dtype]) raises -> nm.NDArray[DType.float64]:
    var total = nm.sum(weights)
    if not isfinite(total) or total <= 0:
        raise Error("ValueError: sampling weights must have positive finite sum")
    var p = nm.zeros[DType.float64](nm.Shape(weights.size))
    var norm = Float64(0)
    for i in range(weights.size):
        var w = _weight(weights, i)
        if w < 0:
            raise Error("ValueError: sampling weights must be nonnegative")
        var value = Float64(w / total)
        p.unsafe_set(i, value)
        norm += value
    # Shared choice accepts Float64 probabilities; sklearn accepts Float32 p
    # and normalizes its Float64 CDF, with a looser input-sum tolerance.
    for i in range(p.size):
        p.unsafe_set(i, p.unsafe_get(i) / norm)
    return p^


def _gather[dtype: DType](x: nm.NDArray[dtype], indices: Labels) raises -> nm.NDArray[dtype]:
    var result = nm.zeros[dtype](nm.Shape(indices.size, x.shape[1]))
    for i in range(indices.size):
        for j in range(x.shape[1]):
            result.unsafe_set(i * x.shape[1] + j, _at(x, Int(indices.item(i)), j))
    return result^


def _seed_distances[dtype: DType](wide: nm.NDArray[DType.float64], transposed: nm.NDArray[DType.float64], norms: nm.NDArray[DType.float64], index: Int) raises -> nm.NDArray[dtype]:
    var row = nm.zeros[DType.float64](nm.Shape(1, wide.shape[1]))
    for f in range(wide.shape[1]):
        row.unsafe_set(f, _at(wide, index, f))
    var own_norm = row_squared_norms(row)
    var product = row @ transposed
    var result = nm.zeros[dtype](nm.Shape(wide.shape[0]))
    for i in range(result.size):
        var value = max((-2 * product.unsafe_get(i) + own_norm.item(0)) + norms.item(i), Float64(0))
        if not isfinite(value):
            raise Error("ValueError: squared distance overflow in initialization")
        result.unsafe_set(i, value.cast[dtype]())
    return result^


def _potential[dtype: DType](closest: nm.NDArray[dtype], weights: nm.NDArray[dtype]) raises -> SIMD[dtype, 1]:
    var result = SIMD[dtype, 1](0)
    for i in range(closest.size):
        result += closest.item(i) * _weight(weights, i)
    if not isfinite(result):
        raise Error("ValueError: weighted potential overflow")
    return result


def kmeans_plusplus[dtype: DType](X: nm.NDArray[dtype], n_clusters: Int, *, mut random_state: RandomState, sample_weight: Optional[nm.NDArray[dtype]] = None, x_squared_norms: Optional[nm.NDArray[dtype]] = None, n_local_trials: Optional[Int] = None) raises -> KMeansPlusPlusResult[dtype]:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    validate_matrix(X)
    _finite(X)
    if n_clusters < 1 or n_clusters > X.shape[0]:
        raise Error("ValueError: require 1 <= n_clusters <= n_samples")
    var trials = 2 + Int(log(Float64(n_clusters)))
    if n_local_trials:
        trials = n_local_trials.value()
    if trials < 1:
        raise Error("ValueError: n_local_trials must be positive")
    var weights = _sample_weights(X, sample_weight)
    var p = _probabilities(weights)
    var wide = X.astype[DType.float64]()
    var transposed = wide.T()
    var norms = row_squared_norms(wide)
    if x_squared_norms:
        ref supplied = x_squared_norms.value()
        _weights(supplied, X.shape[0])
        # sklearn recomputes Float32 norms in its upcast distance path.
        comptime if dtype == DType.float64:
            norms = supplied.astype[DType.float64]()
    var indices = nm.zeros[DType.int64](nm.Shape(n_clusters))
    indices.unsafe_set(0, Int64(random_state.categorical(p)))
    var closest = _seed_distances[dtype](wide, transposed, norms, Int(indices.item(0)))
    var potential = _potential(closest, weights)
    var cdf = nm.zeros[dtype](nm.Shape(X.shape[0]))
    for c in range(1, n_clusters):
        var cumulative = SIMD[dtype, 1](0)
        for i in range(X.shape[0]):
            cumulative += _weight(weights, i) * closest.item(i)
            cdf.unsafe_set(i, cumulative)
        var best_index = 0
        var best_potential = SIMD[dtype, 1](0)
        var best_distances = closest.copy()
        for trial in range(trials):
            var draw = random_state.random_sample() * Float64(potential)
            var candidate = 0
            while candidate < X.shape[0] - 1 and Float64(cdf.item(candidate)) < draw:
                candidate += 1
            var candidate_distances = _seed_distances[dtype](wide, transposed, norms, candidate)
            for i in range(X.shape[0]):
                candidate_distances.unsafe_set(i, min(candidate_distances.item(i), closest.item(i)))
            var candidate_potential = _potential(candidate_distances, weights)
            if trial == 0 or candidate_potential < best_potential:
                best_index = candidate
                best_potential = candidate_potential
                best_distances = candidate_distances^
        indices.unsafe_set(c, Int64(best_index))
        closest = best_distances^
        potential = best_potential
    var centers = _gather(X, indices)
    return KMeansPlusPlusResult[dtype](centers^, indices^)


def kmeans_plusplus[dtype: DType](X: nm.NDArray[dtype], n_clusters: Int, *, random_state: Optional[Int] = None, sample_weight: Optional[nm.NDArray[dtype]] = None, x_squared_norms: Optional[nm.NDArray[dtype]] = None, n_local_trials: Optional[Int] = None) raises -> KMeansPlusPlusResult[dtype]:
    with check_random_state(random_state) as rng:
        return kmeans_plusplus(X, n_clusters, random_state=rng[], sample_weight=sample_weight, x_squared_norms=x_squared_norms, n_local_trials=n_local_trials)


struct KMeansInit[dtype: DType](Copyable, Movable):
    var method: String
    var centers: Optional[nm.NDArray[Self.dtype]]

    @implicit
    def __init__(out self, method: String = "k-means++"):
        self.method = method
        self.centers = None

    @implicit
    def __init__(out self, centers: nm.NDArray[Self.dtype]):
        self.method = "array"
        self.centers = centers.copy()


struct KMeansNInit(Copyable, Movable):
    var automatic: Bool
    var count: Int
    var name: String

    @implicit
    def __init__(out self, name: String = "auto"):
        self.automatic = True
        self.count = 0
        self.name = name

    @implicit
    def __init__(out self, count: Int):
        self.automatic = False
        self.count = count
        self.name = ""


@fieldwise_init
struct KMeansParams[dtype: DType](Copyable, Movable):
    var n_clusters: Int
    var init: KMeansInit[Self.dtype]
    var n_init: KMeansNInit
    var max_iter: Int
    var tol: Float64
    var verbose: Int
    var random_state: Optional[Int]
    var copy_x: Bool
    var algorithm: String


def _same_partition(a: Labels, b: Labels, k: Int) raises -> Bool:
    var mapping = nm.full[DType.int64](nm.Shape(k), -1)
    for i in range(a.size):
        var label = Int(a.item(i))
        if mapping.item(label) == -1:
            mapping.unsafe_set(label, b.item(i))
        elif mapping.item(label) != b.item(i):
            return False
    return True


def _single[dtype: DType](x: nm.NDArray[dtype], weights: nm.NDArray[dtype], var centers: nm.NDArray[dtype], maximum: Int, tolerance: Float64, elkan: Bool, verbose: Int) raises -> CentroidResult[dtype]:
    var previous = nm.full[DType.int64](nm.Shape(x.shape[0]), -1)
    var labels = previous.copy()
    var bounds = Optional[ElkanBounds[dtype]]()
    if elkan:
        bounds = ElkanBounds[dtype](x, centers)
    var iterations = 0
    var strict = False
    for iteration in range(maximum):
        var assigned = assign(x, centers, weights) if not elkan else bounds.value().assign(x, weights)
        labels = assigned.labels.view()
        # BisectingKMeans can request two coincident centers on one row.
        var updated = update(x, labels, weights, centers, allow_singleton=x.shape[0] == 1 and centers.shape[0] == 2)
        var shifts = center_shift(centers, updated.centers)
        var total_shift = SIMD[dtype, 1](0)
        for j in range(shifts.size):
            total_shift += shifts.item(j) * shifts.item(j)
        strict = True
        for i in range(labels.size):
            if labels.item(i) != previous.item(i):
                strict = False
        centers = updated.centers.view()
        if elkan:
            bounds.value().advance(centers)
        iterations = iteration + 1
        if verbose:
            print("Iteration", iteration, "inertia", assigned.inertia)
        if strict or Float64(total_shift) <= tolerance:
            break
        previous = labels.copy()
    if not strict:
        var final_assignment = assign(x, centers, weights) if not elkan else bounds.value().assign(x, weights)
        labels = final_assignment.labels.view()
    var loss = inertia(x, centers, weights, labels)
    return CentroidResult[dtype](labels^, centers^, loss, iterations)


struct KMeans[dtype: DType = DType.float64](PredictiveClusterer, ClusterDistanceTransformer, Copyable, Movable):
    comptime Params = KMeansParams[Self.dtype]
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime Prediction = Labels
    comptime Labels = nm.NDArray[DType.int64]
    comptime Transformed = nm.NDArray[Self.dtype]
    var params: Self.Params
    var cluster_centers_: Self.Input
    var labels_: Self.Labels
    var inertia_: Float64
    var n_iter_: Int
    var n_features_in_: Int
    var n_distinct_clusters_: Int
    var _state: FittedState

    def __init__(out self, n_clusters: Int = 8, *, var init: KMeansInit[Self.dtype] = KMeansInit[Self.dtype](), var n_init: KMeansNInit = KMeansNInit(), max_iter: Int = 300, tol: Float64 = 1e-4, verbose: Int = 0, random_state: Optional[Int] = None, copy_x: Bool = True, algorithm: String = "lloyd") raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.params = Self.Params(n_clusters, init^, n_init^, max_iter, tol, verbose, random_state, copy_x, algorithm)
        self.cluster_centers_ = nm.zeros[Self.dtype](nm.Shape(0, 0))
        self.labels_ = nm.zeros[DType.int64](nm.Shape(0))
        self.inertia_ = 0
        self.n_iter_ = 0
        self.n_features_in_ = 0
        self.n_distinct_clusters_ = 0
        self._state = FittedState()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, n_clusters: Optional[Int] = None, init: Optional[KMeansInit[Self.dtype]] = None, n_init: Optional[KMeansNInit] = None, max_iter: Optional[Int] = None, tol: Optional[Float64] = None, verbose: Optional[Int] = None, random_state: Optional[Optional[Int]] = None, copy_x: Optional[Bool] = None, algorithm: Optional[String] = None) raises -> ref[self] Self:
        if n_clusters: self.params.n_clusters = n_clusters.value()
        if init: self.params.init = init.value().copy()
        if n_init: self.params.n_init = n_init.value().copy()
        if max_iter: self.params.max_iter = max_iter.value()
        if tol: self.params.tol = tol.value()
        if verbose: self.params.verbose = verbose.value()
        if random_state: self.params.random_state = random_state.value()
        if copy_x: self.params.copy_x = copy_x.value()
        if algorithm: self.params.algorithm = algorithm.value()
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
        if k < 1 or k > X.shape[0] or p.max_iter < 1 or p.tol < 0 or not isfinite(p.tol) or p.verbose < 0:
            raise Error("ValueError: invalid n_clusters, max_iter, tol or verbose")
        if p.algorithm != "lloyd" and p.algorithm != "elkan":
            raise Error("ValueError: algorithm must be lloyd or elkan")
        if p.init.method != "k-means++" and p.init.method != "random" and not p.init.centers:
            raise Error("ValueError: init must be k-means++, random or a center array")
        var runs = p.n_init.count
        if p.n_init.automatic:
            if p.n_init.name != "auto":
                raise Error("ValueError: n_init must be auto or a positive integer")
            runs = 10 if p.init.method == "random" else 1
        if runs < 1:
            raise Error("ValueError: n_init must be positive")
        var weights = _sample_weights(X, sample_weight)
        var explicit_centers = Optional[Self.Input]()
        if p.init.centers:
            ref initial = p.init.centers.value()
            validate_matrix(initial)
            _finite(initial)
            if initial.shape[0] != k or initial.shape[1] != X.shape[1]:
                raise Error("ValueError: initial centers must have shape (n_clusters, n_features)")
            explicit_centers = initial.copy()
            runs = 1
        var mean = nm.zeros[Self.dtype](nm.Shape(X.shape[1]))
        var centered = nm.zeros[Self.dtype](nm.Shape(X.shape[0], X.shape[1]))
        var variance_sum = SIMD[Self.dtype, 1](0)
        for f in range(X.shape[1]):
            var total = SIMD[Self.dtype, 1](0)
            for i in range(X.shape[0]):
                total += _at(X, i, f)
            var average = total / SIMD[Self.dtype, 1](X.shape[0])
            mean.unsafe_set(f, average)
            var variance = SIMD[Self.dtype, 1](0)
            for i in range(X.shape[0]):
                var delta = _at(X, i, f) - average
                centered.unsafe_set(i * X.shape[1] + f, delta)
                variance += delta * delta
            variance_sum += variance / SIMD[Self.dtype, 1](X.shape[0])
            if explicit_centers:
                for j in range(k):
                    explicit_centers.value()[nm.Item(j, f)] = _at(explicit_centers.value(), j, f) - average
        var tolerance = Float64(variance_sum / SIMD[Self.dtype, 1](X.shape[1])) * p.tol
        var best = Optional[CentroidResult[Self.dtype]]()
        for _ in range(runs):
            var initial: Self.Input
            if explicit_centers:
                initial = explicit_centers.value().copy()
            elif p.init.method == "random":
                var probabilities = _probabilities(weights)
                var indices = random_state.choice(X.shape[0], probabilities, k, False)
                initial = _gather(centered, indices)
            else:
                var seeded = kmeans_plusplus(centered, k, random_state=random_state, sample_weight=weights.view())
                initial = seeded.centers.view()
            var result = _single(centered, weights, initial^, p.max_iter, tolerance, p.algorithm == "elkan" and k > 1, p.verbose)
            if not best:
                best = result^
            elif result.inertia < best.value().inertia and not _same_partition(result.labels, best.value().labels, k):
                best = result^
        var chosen = best.value().copy()
        for j in range(k):
            for f in range(X.shape[1]):
                chosen.centers.unsafe_set(j * X.shape[1] + f, chosen.centers.item(j, f) + mean.item(f))
        self.cluster_centers_ = chosen.centers.view()
        self.labels_ = chosen.labels.view()
        self.inertia_ = chosen.inertia
        self.n_iter_ = chosen.n_iter
        self.n_features_in_ = X.shape[1]
        self.n_distinct_clusters_ = 0
        var seen = nm.zeros[DType.bool](nm.Shape(k))
        for i in range(self.labels_.size):
            var label = Int(self.labels_.item(i))
            if not seen.item(label):
                seen.unsafe_set(label, True)
                self.n_distinct_clusters_ += 1
        self._state.record_fit(X.shape[0], X.shape[1])
        return self

    def _check(self, X: Self.Input) raises:
        self._state.check_features(X.shape[1] if X.ndim == 2 else 0)
        validate_matrix(X)
        _finite(X)

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        self._check(X)
        var weights = nm.ones[Self.dtype](nm.Shape(X.shape[0]))
        var result = assign(X, self.cluster_centers_, weights)
        return result.labels.view()

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: KMeans is not fitted")
        if input_features and len(input_features.value()) != self.n_features_in_:
            raise Error("ValueError: input_features must match n_features_in_")
        var names = List[String]()
        for i in range(self.cluster_centers_.shape[0]):
            names.append("kmeans" + String(i))
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


@fieldwise_init
struct KMeansResult[dtype: DType](Copyable, Movable):
    var centers: nm.NDArray[Self.dtype]
    var labels: Labels
    var inertia: Float64
    var n_iter: Optional[Int]


def k_means[dtype: DType](X: nm.NDArray[dtype], n_clusters: Int, *, sample_weight: Optional[nm.NDArray[dtype]] = None, var init: KMeansInit[dtype] = KMeansInit[dtype](), var n_init: KMeansNInit = KMeansNInit(), max_iter: Int = 300, verbose: Int = 0, tol: Float64 = 1e-4, random_state: Optional[Int] = None, copy_x: Bool = True, algorithm: String = "lloyd", return_n_iter: Bool = False) raises -> KMeansResult[dtype]:
    var model = KMeans[dtype](n_clusters, init=init^, n_init=n_init^, max_iter=max_iter, verbose=verbose, tol=tol, random_state=random_state, copy_x=copy_x, algorithm=algorithm)
    _ = model.fit(X, sample_weight=sample_weight)
    var iterations = Optional[Int]()
    if return_n_iter: iterations = model.n_iter_
    return KMeansResult[dtype](model.cluster_centers_.view(), model.labels_.view(), model.inertia_, iterations)


def k_means[dtype: DType](X: nm.NDArray[dtype], n_clusters: Int, *, sample_weight: Optional[nm.NDArray[dtype]] = None, var init: KMeansInit[dtype] = KMeansInit[dtype](), var n_init: KMeansNInit = KMeansNInit(), max_iter: Int = 300, verbose: Int = 0, tol: Float64 = 1e-4, mut random_state: RandomState, copy_x: Bool = True, algorithm: String = "lloyd", return_n_iter: Bool = False) raises -> KMeansResult[dtype]:
    var model = KMeans[dtype](n_clusters, init=init^, n_init=n_init^, max_iter=max_iter, verbose=verbose, tol=tol, copy_x=copy_x, algorithm=algorithm)
    _ = model.fit(X, random_state=random_state, sample_weight=sample_weight)
    var iterations = Optional[Int]()
    if return_n_iter: iterations = model.n_iter_
    return KMeansResult[dtype](model.cluster_centers_.view(), model.labels_.view(), model.inertia_, iterations)
