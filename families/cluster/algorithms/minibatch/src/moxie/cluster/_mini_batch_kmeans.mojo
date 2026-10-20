"""Dense incremental clustering, adapted from sklearn (BSD-3-Clause)."""
import numojo as nm
from std.math import min, max, isfinite, ceildiv
from moxie._random import RandomState, check_random_state
from moxie.utils.validation import FittedState
from moxie.cluster._base import PredictiveClusterer, ClusterDistanceTransformer, IncrementalClusterer
from moxie.cluster._contracts import Labels, validate_matrix
from moxie.cluster._centroid_engine import _at, _weight, _finite, assign, distances, minibatch_update
from moxie.cluster._centroid_selection import count_order
from moxie.cluster._kmeans import KMeansInit, KMeansNInit, _sample_weights, _probabilities, _gather, kmeans_plusplus


@fieldwise_init
struct MiniBatchKMeansParams[dtype: DType](Copyable, Movable):
    var n_clusters: Int
    var init: KMeansInit[Self.dtype]
    var max_iter: Int
    var batch_size: Int
    var verbose: Int
    var compute_labels: Bool
    var random_state: Optional[Int]
    var tol: Float64
    var max_no_improvement: Optional[Int]
    var init_size: Optional[Int]
    var n_init: KMeansNInit
    var reassignment_ratio: Float64


struct MiniBatchKMeans[dtype: DType = DType.float64](PredictiveClusterer, ClusterDistanceTransformer, IncrementalClusterer, Copyable, Movable):
    comptime Params = MiniBatchKMeansParams[Self.dtype]
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime Prediction = Labels
    comptime Labels = nm.NDArray[DType.int64]
    comptime Transformed = nm.NDArray[Self.dtype]
    var params: Self.Params
    var cluster_centers_: Self.Input
    var labels_: Self.Labels
    var inertia_: Optional[Float64]
    var n_iter_: Optional[Int]
    var n_steps_: Int
    var n_features_in_: Int
    var _counts: Self.Input
    var _state: FittedState
    var _random_state: Optional[RandomState]
    var _uses_global_rng: Bool
    var _batch_size: Int
    var _init_size: Int
    var _n_init: Int
    var _tol: Float64
    var _n_since_last_reassign: Int
    var _ewa_inertia: Optional[Float64]
    var _ewa_inertia_min: Optional[Float64]
    var _no_improvement: Int

    def __init__(out self, n_clusters: Int = 8, *, var init: KMeansInit[Self.dtype] = KMeansInit[Self.dtype](), max_iter: Int = 100, batch_size: Int = 1024, verbose: Int = 0, compute_labels: Bool = True, random_state: Optional[Int] = None, tol: Float64 = 0.0, max_no_improvement: Optional[Int] = 10, init_size: Optional[Int] = None, var n_init: KMeansNInit = KMeansNInit(), reassignment_ratio: Float64 = 0.01) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.params = Self.Params(n_clusters, init^, max_iter, batch_size, verbose, compute_labels, random_state, tol, max_no_improvement, init_size, n_init^, reassignment_ratio)
        self.cluster_centers_ = nm.zeros[Self.dtype](nm.Shape(0, 0))
        self.labels_ = nm.zeros[DType.int64](nm.Shape(0))
        self._counts = nm.zeros[Self.dtype](nm.Shape(0))
        self.inertia_ = None
        self.n_iter_ = None
        self.n_steps_ = 0
        self.n_features_in_ = 0
        self._state = FittedState()
        self._random_state = None
        self._uses_global_rng = False
        self._batch_size = 0
        self._init_size = 0
        self._n_init = 0
        self._tol = 0
        self._n_since_last_reassign = 0
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, n_clusters: Optional[Int] = None, init: Optional[KMeansInit[Self.dtype]] = None, max_iter: Optional[Int] = None, batch_size: Optional[Int] = None, verbose: Optional[Int] = None, compute_labels: Optional[Bool] = None, random_state: Optional[Optional[Int]] = None, tol: Optional[Float64] = None, max_no_improvement: Optional[Optional[Int]] = None, init_size: Optional[Optional[Int]] = None, n_init: Optional[KMeansNInit] = None, reassignment_ratio: Optional[Float64] = None) raises -> ref[self] Self:
        if n_clusters: self.params.n_clusters = n_clusters.value()
        if init: self.params.init = init.value().copy()
        if max_iter: self.params.max_iter = max_iter.value()
        if batch_size: self.params.batch_size = batch_size.value()
        if verbose: self.params.verbose = verbose.value()
        if compute_labels: self.params.compute_labels = compute_labels.value()
        if random_state: self.params.random_state = random_state.value()
        if tol: self.params.tol = tol.value()
        if max_no_improvement: self.params.max_no_improvement = max_no_improvement.value()
        if init_size: self.params.init_size = init_size.value()
        if n_init: self.params.n_init = n_init.value().copy()
        if reassignment_ratio: self.params.reassignment_ratio = reassignment_ratio.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.is_fitted()

    def _validate_params(mut self, X: Self.Input) raises:
        ref p = self.params
        if p.n_clusters < 1 or p.max_iter < 1 or p.batch_size < 1 or p.verbose < 0 or p.tol < 0 or not isfinite(p.tol) or p.reassignment_ratio < 0 or not isfinite(p.reassignment_ratio):
            raise Error("ValueError: invalid MiniBatchKMeans parameters")
        if p.max_no_improvement and p.max_no_improvement.value() < 0:
            raise Error("ValueError: max_no_improvement must be nonnegative or None")
        if p.init_size and p.init_size.value() < 1:
            raise Error("ValueError: init_size must be positive or None")
        self._n_init = p.n_init.count
        if p.n_init.automatic:
            if p.n_init.name != "auto":
                raise Error("ValueError: n_init must be auto or a positive integer")
            self._n_init = 3 if p.init.method == "random" else 1
        if self._n_init < 1:
            raise Error("ValueError: n_init must be positive")
        if p.init.method != "random" and p.init.method != "k-means++" and not p.init.centers:
            raise Error("ValueError: init must be k-means++, random or a center array")

    def _configure(mut self, X: Self.Input) raises:
        if self.params.n_clusters > X.shape[0]:
            raise Error("ValueError: n_samples must be >= n_clusters on initialization")
        self._batch_size = min(self.params.batch_size, X.shape[0])
        self._init_size = self.params.init_size.value() if self.params.init_size else 3 * self._batch_size
        if self._init_size < self.params.n_clusters:
            self._init_size = 3 * self.params.n_clusters
        self._init_size = min(self._init_size, X.shape[0])
        if self.params.init.centers:
            ref centers = self.params.init.centers.value()
            validate_matrix(centers)
            _finite(centers)
            if centers.shape[0] != self.params.n_clusters or centers.shape[1] != X.shape[1]:
                raise Error("ValueError: initial centers must have shape (n_clusters, n_features)")
            self._n_init = 1
        var variance = SIMD[Self.dtype, 1](0)
        if self.params.tol > 0:
            for f in range(X.shape[1]):
                var average = SIMD[Self.dtype, 1](0)
                for i in range(X.shape[0]): average += _at(X, i, f)
                average /= SIMD[Self.dtype, 1](X.shape[0])
                var column = SIMD[Self.dtype, 1](0)
                for i in range(X.shape[0]):
                    var delta = _at(X, i, f) - average
                    column += delta * delta
                variance += column / SIMD[Self.dtype, 1](X.shape[0])
        self._tol = Float64(variance / SIMD[Self.dtype, 1](X.shape[1])) * self.params.tol

    def _initialize(self, X: Self.Input, weights: Self.Input, mut rng: RandomState) raises -> Self.Input:
        if self._init_size < X.shape[0]:
            var indices = rng.choice(X.shape[0], self._init_size)
            var data = _gather(X, indices)
            var mass = nm.zeros[Self.dtype](nm.Shape(indices.size))
            for i in range(indices.size): mass.unsafe_set(i, _weight(weights, Int(indices.item(i))))
            return self._seed(data, mass, rng)
        return self._seed(X, weights, rng)

    def _seed(self, X: Self.Input, weights: Self.Input, mut rng: RandomState) raises -> Self.Input:
        if self.params.init.centers:
            return self.params.init.centers.value().copy()
        if self.params.init.method == "random":
            var probabilities = _probabilities(weights)
            var indices = rng.choice(X.shape[0], probabilities, self.params.n_clusters, False)
            return _gather(X, indices)
        var seeded = kmeans_plusplus(X, self.params.n_clusters, random_state=rng, sample_weight=weights.copy())
        return seeded.centers.view()

    def _step(mut self, X: Self.Input, weights: Self.Input, mut rng: RandomState) raises -> Float64:
        self._n_since_last_reassign += self._batch_size
        var reassign = self._n_since_last_reassign >= 10 * self.params.n_clusters
        for j in range(self._counts.size):
            if self._counts.item(j) == 0: reassign = True
        if reassign: self._n_since_last_reassign = 0
        var assigned = assign(X, self.cluster_centers_, weights)
        var updated = minibatch_update(X, assigned.labels, weights, self.cluster_centers_, self._counts)
        self.cluster_centers_ = updated.centers.view()
        self._counts = updated.masses.view()
        if reassign and self.params.reassignment_ratio > 0:
            var largest = SIMD[Self.dtype, 1](0)
            for j in range(self._counts.size): largest = max(largest, self._counts.item(j))
            var selected = nm.zeros[DType.bool](nm.Shape(self._counts.size))
            var count = 0
            var threshold = SIMD[Self.dtype, 1](self.params.reassignment_ratio) * largest
            for j in range(self._counts.size):
                if self._counts.item(j) < threshold:
                    selected.unsafe_set(j, True)
                    count += 1
            if count > X.shape[0] // 2:
                # NumPy argsort tie order is observable at the reassignment cap.
                var order = count_order(self._counts)
                for i in range(X.shape[0] // 2, order.size): selected.unsafe_set(Int(order.item(i)), False)
                count = X.shape[0] // 2
            if count == self._counts.size:
                raise Error("ValueError: reassignment leaves no retained center to set minimum weight")
            if count:
                var rows = rng.choice(X.shape[0], count, False)
                var minimum = largest
                for j in range(self._counts.size):
                    if not selected.item(j): minimum = min(minimum, self._counts.item(j))
                var index = 0
                for j in range(self._counts.size):
                    if selected.item(j):
                        for f in range(X.shape[1]): self.cluster_centers_.unsafe_set(j * X.shape[1] + f, _at(X, Int(rows.item(index)), f))
                        self._counts.unsafe_set(j, minimum)
                        index += 1
                if self.params.verbose: print("[MiniBatchKMeans] Reassigning", count, "cluster centers.")
        return assigned.inertia

    def _convergence(mut self, step: Int, n_samples: Int, change: Float64, loss: Float64) -> Bool:
        var batch_inertia = loss / Float64(self._batch_size)
        if self.params.verbose: print("Minibatch step", step + 1, "mean batch inertia:", batch_inertia)
        if step == 0: return False
        if not self._ewa_inertia:
            self._ewa_inertia = batch_inertia
        else:
            var alpha = min(Float64(self._batch_size) * 2 / Float64(n_samples + 1), Float64(1))
            self._ewa_inertia = self._ewa_inertia.value() * (1 - alpha) + batch_inertia * alpha
        if self._tol > 0 and change <= self._tol:
            if self.params.verbose: print("Converged (small centers change)")
            return True
        if not self._ewa_inertia_min or self._ewa_inertia.value() < self._ewa_inertia_min.value():
            self._ewa_inertia_min = self._ewa_inertia
            self._no_improvement = 0
        else:
            self._no_improvement += 1
        if self.params.max_no_improvement and self._no_improvement >= self.params.max_no_improvement.value():
            if self.params.verbose: print("Converged (lack of improvement in inertia)")
            return True
        return False

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
        self._validate_params(X)
        self._configure(X)
        var weights = _sample_weights(X, sample_weight)
        var probabilities = _probabilities(weights)
        var validation = random_state.choice(X.shape[0], self._init_size)
        var valid = _gather(X, validation)
        var valid_weights = nm.zeros[Self.dtype](nm.Shape(validation.size))
        for i in range(validation.size): valid_weights.unsafe_set(i, _weight(weights, Int(validation.item(i))))
        var best_loss = Optional[Float64]()
        for _ in range(self._n_init):
            var initial = self._initialize(X, weights, random_state)
            var result = assign(valid, initial, valid_weights)
            if not best_loss or result.inertia < best_loss.value():
                best_loss = result.inertia
                self.cluster_centers_ = initial.view()
        self._counts = nm.zeros[Self.dtype](nm.Shape(self.params.n_clusters))
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0
        self._n_since_last_reassign = 0
        var unit = nm.ones[Self.dtype](nm.Shape(self._batch_size))
        for step in range(self.params.max_iter * X.shape[0] // self._batch_size):
            var indices = random_state.choice(X.shape[0], probabilities, self._batch_size, True)
            var batch = _gather(X, indices)
            var previous = self.cluster_centers_.view()
            var loss = self._step(batch, unit, random_state)
            var change = SIMD[Self.dtype, 1](0)
            if self._tol > 0:
                for j in range(previous.size):
                    var delta = self.cluster_centers_.item(j) - previous.item(j)
                    change += delta * delta
            self.n_steps_ = step + 1
            if self._convergence(step, X.shape[0], Float64(change), loss): break
        self.n_iter_ = ceildiv(self.n_steps_ * self._batch_size, X.shape[0])
        self.n_features_in_ = X.shape[1]
        self._state.record_fit(X.shape[0], X.shape[1])
        if self.params.compute_labels:
            self._label(X, weights)
        else:
            if not self._ewa_inertia:
                raise Error("ValueError: compute_labels=False requires at least two fit steps for smoothed inertia")
            self.inertia_ = self._ewa_inertia.value() * Float64(nm.sum(weights))
        return self

    def _label(mut self, X: Self.Input, weights: Self.Input) raises:
        var result = assign(X, self.cluster_centers_, weights)
        self.labels_ = result.labels.view()
        self.inertia_ = result.inertia

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        _ = self.partial_fit(X, y, sample_weight=None)
        return self

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None, *, sample_weight: Optional[Self.Input]) raises -> ref[self] Self:
        if not self._random_state:
            self._uses_global_rng = not self.params.random_state
        if self._uses_global_rng:
            with check_random_state(None) as rng:
                _ = self.partial_fit(X, y, random_state=rng[], sample_weight=sample_weight)
                self._random_state = rng[].copy()
        elif not self._random_state:
            with check_random_state(self.params.random_state) as rng:
                _ = self.partial_fit(X, y, random_state=rng[], sample_weight=sample_weight)
                self._random_state = rng[].copy()
        else:
            var rng = self._random_state.value().copy()
            _ = self.partial_fit(X, y, random_state=rng, sample_weight=sample_weight)
            self._random_state = rng^
        return self

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None, *, mut random_state: RandomState, sample_weight: Optional[Self.Input] = None) raises -> ref[self] Self:
        validate_matrix(X)
        _finite(X)
        self._validate_params(X)
        var weights = _sample_weights(X, sample_weight)
        if not self._state.is_fitted():
            self._configure(X)
            self.cluster_centers_ = self._initialize(X, weights, random_state)
            self._counts = nm.zeros[Self.dtype](nm.Shape(self.params.n_clusters))
            self._n_since_last_reassign = 0
            self.n_features_in_ = X.shape[1]
        else:
            self._state.check_features(X.shape[1])
        _ = self._step(X, weights, random_state)
        if self.params.compute_labels: self._label(X, weights)
        self.n_steps_ += 1
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
            raise Error("NotFittedError: MiniBatchKMeans is not fitted")
        if input_features and len(input_features.value()) != self.n_features_in_:
            raise Error("ValueError: input_features must match n_features_in_")
        var names = List[String]()
        for i in range(self.cluster_centers_.shape[0]):
            names.append("minibatchkmeans" + String(i))
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
        if not self.params.compute_labels:
            raise Error("ValueError: fit_predict requires compute_labels=True")
        return self.labels_.copy()

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None, *, sample_weight: Optional[Self.Input] = None) raises -> Self.Transformed:
        _ = self.fit(X, y, sample_weight=sample_weight)
        return self.transform(X)
