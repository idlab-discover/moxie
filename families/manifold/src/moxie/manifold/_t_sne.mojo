# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense t-SNE and neighborhood preservation diagnostics."""
from numojo import NDArray, zeros, Shape
from std.builtin.sort import sort
from std.math import abs, sqrt, isfinite, inf, nan
from std.utils import Variant
from numax.core.libm import exp, log
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount
from moxie._random import check_random_state
from moxie.decomposition._pca import PCA
from moxie.manifold._base import EmbeddingEstimator
from moxie.manifold._barnes_hut_tsne import _Pair, _merge_neighbor_probabilities, _barnes_hut_objective
from moxie.neighbors._contracts import DenseMetric
from moxie.neighbors._distance import Metric, at, metric_power
from moxie.neighbors._query_engine import BruteForce
from moxie.manifold._plumbing import validate_matrix, validate_pairwise


def _distance[dtype: DType](x: NDArray[dtype], i: Int, j: Int, metric: String) raises -> Float64:
    if metric == "precomputed":
        return Float64(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))
    var result: Float64 = 0
    var aa: Float64 = 0
    var bb: Float64 = 0
    for c in range(x.shape[1]):
        var a = Float64(x.unsafe_get(i * Int(x.strides[0]) + c * Int(x.strides[1])))
        var b = Float64(x.unsafe_get(j * Int(x.strides[0]) + c * Int(x.strides[1])))
        if metric == "cosine":
            result += a * b
            aa += a * a
            bb += b * b
        elif metric == "manhattan" or metric == "cityblock" or metric == "l1":
            result += abs(a - b)
        else:
            result += (a - b) * (a - b)
    if metric == "cosine":
        if aa == 0 or bb == 0:
            return 1
        result = max(Float64(0), min(Float64(2), 1 - result / sqrt(aa) / sqrt(bb)))
    if not isfinite(result):
        raise Error("ValueError: pairwise distance overflow")
    return result


def _neighbor_order[dtype: DType](x: NDArray[dtype], row: Int, metric: String) raises -> NDArray[DType.int64]:
    var pairs = List[Tuple[Float64, Int]]()
    for j in range(x.shape[0]):
        if j != row:
            pairs.append((_distance(x, row, j, metric), j))
    sort(Span(pairs))
    var indices = zeros[DType.int64](Shape(len(pairs)))
    for j in range(len(pairs)):
        indices.unsafe_set(j, Int64(pairs[j][1]))
    return indices^


def trustworthiness[dtype: DType, embedded_dtype: DType](
    X: NDArray[dtype], X_embedded: NDArray[embedded_dtype], *,
    n_neighbors: Int = 5, metric: String = "euclidean",
) raises -> Float64:
    """Rank penalty for unexpected embedding neighbors; fewer than n/2 neighbors."""
    validate_matrix(X)
    validate_matrix(X_embedded)
    var n = X.shape[0]
    if X_embedded.shape[0] != n:
        raise Error("ValueError: X and X_embedded must have the same sample count")
    if n_neighbors < 1 or n_neighbors >= (n + 1) // 2:
        raise Error("ValueError: n_neighbors must be positive and less than n_samples / 2")
    if metric == "precomputed":
        validate_pairwise(X, nonnegative=True, symmetric=False)
    elif metric != "euclidean" and metric != "l2" and metric != "sqeuclidean" and metric != "manhattan" and metric != "cityblock" and metric != "l1" and metric != "cosine":
        raise Error("ValueError: unsupported dense trustworthiness metric")
    var penalty: Float64 = 0
    for i in range(n):
        var original = _neighbor_order(X, i, metric)
        var embedded = _neighbor_order(X_embedded, i, "euclidean")
        var ranks = zeros[DType.int64](Shape(n))
        for j in range(n - 1):
            ranks.unsafe_set(Int(original.unsafe_get(j)), Int64(j + 1))
        for j in range(n_neighbors):
            penalty += Float64(max(0, Int(ranks.unsafe_get(Int(embedded.unsafe_get(j)))) - n_neighbors))
    return 1 - penalty * 2 / (Float64(n) * Float64(n_neighbors) * (2 * Float64(n) - 3 * Float64(n_neighbors) - 1))


comptime _EPS = 2.220446049250313e-16
comptime _EXPLORATION = 250
comptime _CHECK_INTERVAL = 50
comptime _MAX_FLOAT64 = 1.7976931348623157e308
comptime _LearningRate = Variant[String, Float64]
comptime _Init[dtype: DType] = Variant[String, NDArray[dtype]]
comptime _DistanceCallback[dtype: DType] = def(
    NDArray[dtype], Int, NDArray[dtype], Int,
) thin raises -> Float64


@fieldwise_init
struct _CallableDistance[input_dtype: DType](DenseMetric):
    comptime dtype = Self.input_dtype
    var callback: _DistanceCallback[Self.dtype]

    def validate_features(self, n_features: Int) raises:
        pass

    def distance(
        self, x: NDArray[Self.dtype], i: Int, y: NDArray[Self.dtype], j: Int,
    ) raises -> Float64:
        var distance = self.callback(x, i, y, j)
        if not isfinite(distance) or distance < 0:
            raise Error("ValueError: TSNE metric callback must return a finite nonnegative distance")
        return distance


struct _WeightedHamming[input_dtype: DType](DenseMetric):
    comptime dtype = Self.input_dtype
    var _weights: List[Float64]

    def __init__(out self, weights: NDArray[Self.dtype]) raises:
        if weights.ndim != 1:
            raise Error("ValueError: Hamming weights must be a vector")
        var normalized = List[Float64](capacity=weights.shape[0])
        var total: Float64 = 0
        for feature in range(weights.shape[0]):
            var weight = Float64(weights.unsafe_get(feature * Int(weights.strides[0])))
            if not isfinite(weight) or weight < 0:
                raise Error("ValueError: Hamming weights must be finite and nonnegative")
            total += weight
            normalized.append(weight)
        if not isfinite(total) or total == 0:
            raise Error("ValueError: Hamming weights must have a finite positive sum")
        for feature in range(len(normalized)):
            normalized[feature] /= total
        self._weights = normalized^

    def validate_features(self, n_features: Int) raises:
        if len(self._weights) != n_features:
            raise Error("ValueError: Hamming weights must match features")

    def distance(
        self, x: NDArray[Self.dtype], i: Int, y: NDArray[Self.dtype], j: Int,
    ) raises -> Float64:
        var mismatch: Float64 = 0
        for feature in range(len(self._weights)):
            if at(x, i, feature) != at(y, j, feature):
                mismatch += self._weights[feature]
        return mismatch


struct TSNEMetricParams[dtype: DType](Copyable, Movable):
    """Typed options for dense t-SNE metrics."""

    var p: Optional[Float64]
    var w: Optional[NDArray[Self.dtype]]
    var V: Optional[NDArray[Self.dtype]]
    var VI: Optional[NDArray[Self.dtype]]
    var missing_values: Optional[Float64]
    var distance: Optional[_DistanceCallback[Self.dtype]]

    def __init__(out self, *, p: Optional[Float64] = None,
                 w: Optional[NDArray[Self.dtype]] = None,
                 V: Optional[NDArray[Self.dtype]] = None,
                 VI: Optional[NDArray[Self.dtype]] = None,
                 missing_values: Optional[Float64] = None,
                 distance: Optional[_DistanceCallback[Self.dtype]] = None):
        self.p = p
        self.w = w.copy()
        self.V = V.copy()
        self.VI = VI.copy()
        self.missing_values = missing_values
        self.distance = distance


@fieldwise_init
struct TSNEParams[dtype: DType, embedding_dtype: DType](Copyable, Movable):
    var n_components: Int
    var perplexity: Float64
    var early_exaggeration: Float64
    var learning_rate: _LearningRate
    var max_iter: Int
    var n_iter_without_progress: Int
    var min_grad_norm: Float64
    var metric: String
    var metric_params: Optional[TSNEMetricParams[Self.dtype]]
    var init: _Init[Self.embedding_dtype]
    var verbose: Int
    var random_state: Optional[Int]
    var method: String
    var angle: Float64
    var n_jobs: Optional[Int]


struct TSNE[dtype: DType = DType.float64, embedding_dtype: DType = DType.float32](
    EmbeddingEstimator, FeatureNamesOut, InputFeatureCount,
):
    """Dense t-SNE with Float32 coordinates by default."""

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.embedding_dtype]
    comptime Params = TSNEParams[Self.dtype, Self.embedding_dtype]

    var n_components: Int
    var perplexity: Float64
    var early_exaggeration: Float64
    var learning_rate: _LearningRate
    var max_iter: Int
    var n_iter_without_progress: Int
    var min_grad_norm: Float64
    var metric: String
    var metric_params: Optional[TSNEMetricParams[Self.dtype]]
    var init: _Init[Self.embedding_dtype]
    var verbose: Int
    var random_state: Optional[Int]
    var method: String
    var angle: Float64
    var n_jobs: Optional[Int]
    var embedding_: Optional[Self.Transformed]
    var kl_divergence_: Float64
    var n_iter_: Int
    var learning_rate_: Float64
    var n_features_in_: Int

    def __init__(
        out self, n_components: Int = 2, *, perplexity: Float64 = 30.0,
        early_exaggeration: Float64 = 12.0, learning_rate: _LearningRate = "auto",
        max_iter: Int = 1000, n_iter_without_progress: Int = 300,
        min_grad_norm: Float64 = 1e-7, metric: String = "euclidean",
        metric_params: Optional[TSNEMetricParams[Self.dtype]] = None,
        init: _Init[Self.embedding_dtype] = "pca", verbose: Int = 0,
        random_state: Optional[Int] = None, method: String = "barnes_hut",
        angle: Float64 = 0.5, n_jobs: Optional[Int] = None,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        comptime assert Self.embedding_dtype == DType.float32 or Self.embedding_dtype == DType.float64
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate.copy()
        self.max_iter = max_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params.copy()
        self.init = init.copy()
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.embedding_ = None
        self.kl_divergence_ = 0
        self.n_iter_ = 0
        self.learning_rate_ = 0
        self.n_features_in_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            self.n_components, self.perplexity, self.early_exaggeration,
            self.learning_rate.copy(), self.max_iter, self.n_iter_without_progress,
            self.min_grad_norm, self.metric, self.metric_params.copy(), self.init.copy(),
            self.verbose, self.random_state, self.method, self.angle, self.n_jobs,
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.n_components = params.n_components
        self.perplexity = params.perplexity
        self.early_exaggeration = params.early_exaggeration
        self.learning_rate = params.learning_rate.copy()
        self.max_iter = params.max_iter
        self.n_iter_without_progress = params.n_iter_without_progress
        self.min_grad_norm = params.min_grad_norm
        self.metric = params.metric
        self.metric_params = params.metric_params.copy()
        self.init = params.init.copy()
        self.verbose = params.verbose
        self.random_state = params.random_state
        self.method = params.method
        self.angle = params.angle
        self.n_jobs = params.n_jobs
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, n_components: Optional[Int] = None,
        perplexity: Optional[Float64] = None, early_exaggeration: Optional[Float64] = None,
        learning_rate: Optional[_LearningRate] = None, max_iter: Optional[Int] = None,
        n_iter_without_progress: Optional[Int] = None, min_grad_norm: Optional[Float64] = None,
        metric: Optional[String] = None,
        metric_params: Optional[Optional[TSNEMetricParams[Self.dtype]]] = None,
        init: Optional[_Init[Self.embedding_dtype]] = None, verbose: Optional[Int] = None,
        random_state: Optional[Optional[Int]] = None, method: Optional[String] = None,
        angle: Optional[Float64] = None, n_jobs: Optional[Optional[Int]] = None,
    ) raises -> ref[self] Self:
        if n_components:
            self.n_components = n_components.value()
        if perplexity:
            self.perplexity = perplexity.value()
        if early_exaggeration:
            self.early_exaggeration = early_exaggeration.value()
        if learning_rate:
            self.learning_rate = learning_rate.value().copy()
        if max_iter:
            self.max_iter = max_iter.value()
        if n_iter_without_progress:
            self.n_iter_without_progress = n_iter_without_progress.value()
        if min_grad_norm:
            self.min_grad_norm = min_grad_norm.value()
        if metric:
            self.metric = metric.value()
        if metric_params:
            self.metric_params = metric_params.value().copy()
        if init:
            self.init = init.value().copy()
        if verbose:
            self.verbose = verbose.value()
        if random_state:
            self.random_state = random_state.value()
        if method:
            self.method = method.value()
        if angle:
            self.angle = angle.value()
        if n_jobs:
            self.n_jobs = n_jobs.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            self.n_components, perplexity=self.perplexity, early_exaggeration=self.early_exaggeration,
            learning_rate=self.learning_rate, max_iter=self.max_iter,
            n_iter_without_progress=self.n_iter_without_progress, min_grad_norm=self.min_grad_norm,
            metric=self.metric, metric_params=self.metric_params, init=self.init, verbose=self.verbose,
            random_state=self.random_state, method=self.method, angle=self.angle, n_jobs=self.n_jobs,
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.embedding_)

    def get_n_features_in(self) raises -> Int:
        if not self.__sklearn_is_fitted__():
            raise Error("AttributeError: TSNE has no n_features_in_ before fit")
        return self.n_features_in_

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if not self.embedding_:
            raise Error("NotFittedError: TSNE is not fitted")
        if input_features:
            if len(input_features.value()) != self.n_features_in_:
                raise Error("ValueError: input_features length differs from n_features_in_")
        var names = List[String]()
        for component in range(self.embedding_.value().shape[1]):
            names.append("tsne" + String(component))
        return names^

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default":
            raise Error("NotImplementedError: TSNE supports native array output only")

    def get_metadata_routing(self) raises:
        raise Error("NotImplementedError: Python metadata routing is not available")

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self._fit(X)
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self._fit(X)
        return self.embedding_.value().copy()

    def _validate_params(self, X: Self.Input) raises -> Float64:
        if self.method != "exact" and self.method != "barnes_hut":
            raise Error("ValueError: method must be exact or barnes_hut")
        validate_matrix(X)
        if self.n_components < 1:
            raise Error("ValueError: n_components must be positive")
        if self.method == "barnes_hut" and self.n_components > 3:
            raise Error("ValueError: n_components must be at most 3 for Barnes–Hut t-SNE")
        if not (self.perplexity > 0 and self.perplexity < Float64(X.shape[0])):
            raise Error("ValueError: perplexity must be positive and less than n_samples")
        if not isfinite(self.early_exaggeration) or self.early_exaggeration < 1:
            raise Error("ValueError: early_exaggeration must be finite and at least 1")
        if self.max_iter < 250 or self.n_iter_without_progress < -1:
            raise Error("ValueError: max_iter must be at least 250 and n_iter_without_progress at least -1")
        if not isfinite(self.min_grad_norm) or self.min_grad_norm < 0:
            raise Error("ValueError: min_grad_norm must be finite and nonnegative")
        if not (self.angle >= 0 and self.angle <= 1) or self.verbose < 0:
            raise Error("ValueError: angle must be in [0, 1] and verbose nonnegative")
        if self.n_jobs:
            if self.n_jobs.value() == 0 and self.metric != "euclidean" and self.metric != "precomputed":
                raise Error("ValueError: n_jobs cannot be zero")
        if self.random_state:
            var seed = self.random_state.value()
            if seed < 0 or UInt64(seed) > 4294967295:
                raise Error("ValueError: random_state must be a uint32 seed")
        if self.init.isa[String]():
            var name = self.init[String]
            if name != "pca" and name != "random":
                raise Error("ValueError: init must be pca, random, or a coordinate array")
            if name == "pca" and self.metric == "precomputed":
                raise Error('ValueError: init="pca" cannot be used with metric="precomputed"')
        else:
            ref coordinates = self.init[Self.Transformed]
            validate_matrix(coordinates)
            if coordinates.shape[0] != X.shape[0] or coordinates.shape[1] != self.n_components:
                raise Error("ValueError: init shape must be (n_samples, n_components)")
        if self.learning_rate.isa[String]():
            if self.learning_rate[String] != "auto":
                raise Error('ValueError: learning_rate must be positive or "auto"')
            return max(Float64(X.shape[0]) / self.early_exaggeration / 4, 50)
        var rate = self.learning_rate[Float64]
        if not isfinite(rate) or rate <= 0:
            raise Error("ValueError: learning_rate must be finite and positive")
        return rate

    def _fit(mut self, X: Self.Input) raises:
        var rate = self._validate_params(X)
        var pairs: Optional[List[_Pair]] = None
        var probabilities: NDArray[DType.float64]
        if self.method == "barnes_hut":
            pairs = _neighbor_probabilities(X, self.metric, self.metric_params, self.perplexity, self.verbose)
            probabilities = zeros[DType.float64](Shape(0))
        else:
            var distances = _squared_distances(X, self.metric, self.metric_params)
            probabilities = _joint_probabilities(distances^, self.perplexity, self.verbose)
        var coordinates = self._initial_coordinates(X)
        var weights = zeros[DType.float64](Shape(probabilities.size))
        var gradient = zeros[DType.float64](coordinates.shape)
        var dof = Float64(max(self.n_components - 1, 1))
        for i in range(probabilities.size):
            probabilities.unsafe_set(i, probabilities.unsafe_get(i) * self.early_exaggeration)
        var loss, iteration = _gradient_descent(
            coordinates, probabilities, weights, gradient, dof, 0,
            _EXPLORATION, _EXPLORATION, 0.5, rate, self.min_grad_norm, self.verbose,
            pairs, Float32(self.angle), self.early_exaggeration,
        )
        if self.verbose:
            print("[t-SNE] KL divergence after", iteration + 1, "iterations with early exaggeration:", loss)
        for i in range(probabilities.size):
            probabilities.unsafe_set(i, probabilities.unsafe_get(i) / self.early_exaggeration)
        loss, iteration = _gradient_descent(
            coordinates, probabilities, weights, gradient, dof, iteration + 1,
            self.max_iter, self.n_iter_without_progress, 0.8, rate, self.min_grad_norm, self.verbose,
            pairs, Float32(self.angle),
        )
        if self.verbose:
            print("[t-SNE] KL divergence after", iteration + 1, "iterations:", loss)
        self.n_features_in_ = X.shape[1]
        self.learning_rate_ = rate
        self.kl_divergence_ = loss
        self.n_iter_ = iteration
        self.embedding_ = coordinates^

    def _initial_coordinates(self, X: Self.Input) raises -> Self.Transformed:
        var count = X.shape[0]
        var components = self.n_components
        var coordinates = zeros[Self.embedding_dtype](Shape(count, components))
        if self.init.isa[Self.Transformed]():
            # Pack logical entries once so the optimizer uses contiguous indexing.
            ref init = self.init[Self.Transformed]
            for i in range(count):
                for j in range(components):
                    coordinates.unsafe_set(
                        i * components + j,
                        init.unsafe_get(i * Int(init.strides[0]) + j * Int(init.strides[1])),
                    )
        elif self.init[String] == "random":
            var rng = check_random_state(self.random_state)
            for i in range(coordinates.size):
                var draw = Float32(rng[].standard_normal()) * Float32(1e-4)
                coordinates.unsafe_set(i, draw.cast[Self.embedding_dtype]())
        else:
            var pca = PCA[Self.dtype](components, random_state=self.random_state)
            var scores = pca.fit_transform(X).astype[DType.float32]()
            var average: Float32 = 0
            for i in range(count):
                average += scores.unsafe_get(i * components)
            average /= Float32(count)
            var variance: Float32 = 0
            for i in range(count):
                var centered = scores.unsafe_get(i * components) - average
                variance += centered * centered
            var scale = sqrt(variance / Float32(count))
            if not isfinite(scale) or scale == 0:
                raise Error("ValueError: PCA initialization requires nonzero finite variance in the first component")
            for i in range(coordinates.size):
                var value = scores.unsafe_get(i) / scale * Float32(1e-4)
                coordinates.unsafe_set(i, value.cast[Self.embedding_dtype]())
        return coordinates^


def _feature_weights[dtype: DType](weights: NDArray[dtype]) raises -> List[Float64]:
    if weights.ndim != 1:
        raise Error("ValueError: TSNE feature weights must be a vector")
    var values = List[Float64](capacity=weights.shape[0])
    for feature in range(weights.shape[0]):
        var weight = Float64(weights.unsafe_get(feature * Int(weights.strides[0])))
        if not isfinite(weight) or weight < 0:
            raise Error("ValueError: TSNE feature weights must be finite and nonnegative")
        values.append(weight)
    return values^


@fieldwise_init
struct _WeightedSquaredEuclidean[input_dtype: DType](DenseMetric):
    comptime dtype = Self.input_dtype
    var weights: List[Float64]

    def validate_features(self, n_features: Int) raises:
        if len(self.weights) != n_features:
            raise Error("ValueError: TSNE weights must match features")

    def distance(
        self, x: NDArray[Self.dtype], i: Int, y: NDArray[Self.dtype], j: Int,
    ) raises -> Float64:
        var total: Float64 = 0
        for feature in range(len(self.weights)):
            var weight = self.weights[feature]
            if weight != 0:
                var delta = at(x, i, feature) - at(y, j, feature)
                total += (weight * delta) * delta
        return total


struct _ExactRatioDistance[input_dtype: DType](DenseMetric):
    comptime dtype = Self.input_dtype
    var _canberra: Bool
    var _weights: Optional[List[Float64]]

    def __init__(
        out self, name: String, weights: Optional[NDArray[Self.dtype]],
    ) raises:
        self._canberra = name == "canberra"
        self._weights = None
        if weights:
            self._weights = _feature_weights(weights.value())

    def validate_features(self, n_features: Int) raises:
        if self._weights and len(self._weights.value()) != n_features:
            raise Error("ValueError: TSNE weights must match features")

    def distance(
        self, x: NDArray[Self.dtype], i: Int, y: NDArray[Self.dtype], j: Int,
    ) raises -> Float64:
        var total: Float64 = 0
        var denominator: Float64 = 0
        for feature in range(x.shape[1]):
            var weight = self._weights.value()[feature] if self._weights else Float64(1)
            var left = at(x, i, feature)
            var right = at(y, j, feature)
            if self._canberra:
                var divisor = abs(left) + abs(right)
                if divisor != 0:
                    total += weight * (abs(left - right) / divisor)
            else:
                total += weight * abs(left - right)
                # SciPy uses |x + y|, while the neighbors kernel uses |x| + |y|.
                denominator += weight * abs(left + right)
        return total if self._canberra else total / denominator


def _metric[dtype: DType](
    name: String, options: Optional[TSNEMetricParams[dtype]], *, squared_euclidean: Bool = False,
) raises -> Metric[dtype]:
    # Use the neighbors owner's kernels where their distance conventions match
    # sklearn's dense pairwise distances.
    if name not in [
        "euclidean", "l2", "sqeuclidean", "manhattan", "l1", "cityblock",
        "cosine", "chebyshev", "minkowski", "braycurtis", "canberra", "correlation", "hamming",
        "jaccard", "dice", "rogerstanimoto", "russellrao", "sokalsneath", "yule",
        "seuclidean", "mahalanobis", "nan_euclidean", "haversine",
    ]:
        raise Error("ValueError: unsupported dense TSNE metric: " + name)
    var params = options.copy().or_else(TSNEMetricParams[dtype]())
    if ((params.p or params.w) and name != "minkowski"
        or params.V and name != "seuclidean"
        or params.VI and name != "mahalanobis"
        or params.missing_values and name != "nan_euclidean"):
        raise Error("ValueError: metric_params do not match the selected metric")
    return Metric[dtype](
        "sqeuclidean" if squared_euclidean and name == "euclidean" else name,
        params.p.or_else(2.0), params.w, params.V, params.VI,
        params.missing_values.or_else(nan[DType.float64]()),
    )


def _metric_callback[dtype: DType](
    name: String, options: Optional[TSNEMetricParams[dtype]],
) raises -> Optional[_DistanceCallback[dtype]]:
    if options:
        ref params = options.value()
        if params.distance:
            if name != "callable" or params.p or params.w or params.V or params.VI or params.missing_values:
                raise Error("ValueError: metric_params do not match the selected metric")
            return params.distance
    if name == "callable":
        raise Error("ValueError: metric='callable' requires metric_params.distance")
    return None


def _squared_distances[dtype: DType](
    X: NDArray[dtype], name: String, options: Optional[TSNEMetricParams[dtype]],
) raises -> NDArray[DType.float32]:
    var callback = _metric_callback(name, options)
    if callback:
        return _symmetric_squared_distances(
            _CallableDistance[dtype](callback.value()), X, name,
        )
    var n = X.shape[0]
    if name == "precomputed":
        var squared = zeros[DType.float32](Shape(n, n))
        validate_pairwise(X, nonnegative=True, symmetric=False)
        for i in range(n):
            for j in range(n):
                if i != j:
                    var value = at(X, i, j).cast[dtype]()
                    var converted = Float32(value * value)
                    if not isfinite(converted):
                        raise Error("ValueError: squared pairwise distance overflow")
                    squared.unsafe_set(i * n + j, converted)
        return squared^

    if name == "braycurtis" or (name == "canberra" and options and options.value().w):
        var ratio: _ExactRatioDistance[dtype]
        if options:
            ref params = options.value()
            if params.p or params.V or params.VI or params.missing_values:
                raise Error("ValueError: metric_params do not match the selected metric")
            ratio = _ExactRatioDistance[dtype](name, params.w)
        else:
            ratio = _ExactRatioDistance[dtype](name, None)
        ratio.validate_features(X.shape[1])
        return _symmetric_squared_distances(ratio, X, name)

    # SciPy's exact pairwise path accepts "matching" as an alias for Hamming.
    # NearestNeighbors does not accept the alias in the Barnes–Hut path.
    if (name == "hamming" or name == "matching") and options:
        ref params = options.value()
        if params.w:
            if params.p or params.V or params.VI or params.missing_values:
                raise Error("ValueError: metric_params do not match the selected metric")
            var weighted = _WeightedHamming(params.w.value())
            weighted.validate_features(X.shape[1])
            return _symmetric_squared_distances(weighted, X, name)
    if (name == "chebyshev" or name == "sqeuclidean") and options:
        ref params = options.value()
        if params.w:
            if params.p or params.V or params.VI or params.missing_values:
                raise Error("ValueError: metric_params do not match the selected metric")
            if name == "chebyshev":
                var chebyshev = Metric[dtype]("minkowski", p=inf[DType.float64](), w=params.w)
                chebyshev.validate_features(X.shape[1])
                return _symmetric_squared_distances(chebyshev, X, name)
            var squared_euclidean = _WeightedSquaredEuclidean[dtype](_feature_weights(params.w.value()))
            squared_euclidean.validate_features(X.shape[1])
            return _symmetric_squared_distances(squared_euclidean, X, name)
    var kernel = _metric("hamming" if name == "matching" else name, options, squared_euclidean=True)
    kernel.validate_features(X.shape[1])
    kernel = kernel.self_query_metric(X)
    return _symmetric_squared_distances(kernel, X, name)


def _symmetric_squared_distances[M: DenseMetric](
    kernel: M, X: NDArray[M.dtype], name: String,
) raises -> NDArray[DType.float32]:
    var n = X.shape[0]
    var squared = zeros[DType.float32](Shape(n, n))
    var callable_metric = name == "callable"
    var round_before_squaring = callable_metric or name == "l2" or name == "cosine" or name == "nan_euclidean"
    for i in range(n):
        for j in range(i if callable_metric else i + 1, n):
            var distance = kernel.distance(X, i, X, j)
            var value = Float32(distance)
            if round_before_squaring:
                # These sklearn kernels return Float32 for Float32 inputs.
                var rounded = distance.cast[M.dtype]()
                value = Float32(rounded * rounded)
            elif name != "euclidean":
                value = Float32(distance * distance)
            if not isfinite(value) or distance < 0:
                raise Error("ValueError: TSNE requires finite nonnegative squared distances")
            squared.unsafe_set(i * n + j, value)
            squared.unsafe_set(j * n + i, value)
    return squared^


def _conditional_probabilities(
    squared: NDArray[DType.float32], perplexity: Float64, verbose: Int,
) raises -> NDArray[DType.float64]:
    var n = squared.shape[0]
    var width = squared.shape[1]
    var conditional = zeros[DType.float64](squared.shape)
    var desired_entropy = log(Float64(Float32(perplexity)))
    var beta_sum: Float64 = 0
    for i in range(n):
        var beta: Float64 = 1
        var lower: Float64 = 0
        var upper = inf[DType.float64]()
        var row = i * width
        for _ in range(100):
            var total: Float64 = 0
            for j in range(width):
                if width < n or i != j:
                    var probability = exp(-beta * Float64(squared.unsafe_get(row + j)))
                    conditional.unsafe_set(row + j, probability)
                    total += probability
            if total == 0:
                total = Float64(Float32(1e-8))
            var weighted_distance: Float64 = 0
            for j in range(width):
                var probability = conditional.unsafe_get(row + j) / total
                conditional.unsafe_set(row + j, probability)
                weighted_distance += Float64(squared.unsafe_get(row + j)) * probability
            var difference = log(total) + beta * weighted_distance - desired_entropy
            if abs(difference) <= Float64(Float32(1e-5)):
                break
            if difference > 0:
                lower = beta
                beta = 2 * beta if upper == inf[DType.float64]() else (beta + upper) / 2
            else:
                upper = beta
                beta = (beta + lower) / 2
        beta_sum += beta
        if verbose and ((i + 1) % 1000 == 0 or i + 1 == n):
            print("[t-SNE] Computed conditional probabilities for sample", i + 1, "/", n)
    if verbose:
        print("[t-SNE] Mean sigma:", sqrt(Float64(n) / beta_sum))
    return conditional^


def _neighbor_probabilities[dtype: DType](
    X: NDArray[dtype], name: String, options: Optional[TSNEMetricParams[dtype]],
    perplexity: Float64, verbose: Int = 0,
) raises -> List[_Pair]:
    var callback = _metric_callback(name, options)
    if callback:
        return _metric_neighbor_probabilities(
            _CallableDistance[dtype](callback.value()), X, False, perplexity, verbose,
        )
    var precomputed = name == "precomputed"
    if precomputed:
        validate_pairwise(X, nonnegative=True, symmetric=False)
    var kernel = _metric("euclidean" if precomputed else name, options)
    return _metric_neighbor_probabilities(kernel, X, precomputed, perplexity, verbose)


def _metric_neighbor_probabilities[M: DenseMetric & Deinitable](
    kernel: M, X: NDArray[M.dtype], precomputed: Bool, perplexity: Float64, verbose: Int,
) raises -> List[_Pair]:
    var n = X.shape[0]
    var k = min(n - 1, Int(3 * perplexity + 1))
    var search = BruteForce(kernel, precomputed=precomputed)
    search.fit(X)
    if verbose:
        print("[t-SNE] Computing", k, "nearest neighbors per sample")
    var neighborhood = search.kneighbors(None, k)
    ref distances = neighborhood.distances.value()
    var squared = zeros[DType.float32](Shape(n, k))
    var indices = List[Int](capacity=n * k)
    # The reference's entropy search consumes each row in sample-index order.
    var row = List[Tuple[Int, Float32]](capacity=k)
    for i in range(n):
        row.clear()
        for j in range(k):
            var offset = i * k + j
            var distance = distances.unsafe_get(offset)
            var value = Float32(distance * distance)
            if not isfinite(value):
                raise Error("ValueError: TSNE requires finite squared neighbor distances")
            row.append((neighborhood.indices[offset], value))
        sort(Span(row))
        for j in range(k):
            indices.append(row[j][0])
            squared.unsafe_set(i * k + j, row[j][1])
    var conditional = _conditional_probabilities(squared, perplexity, verbose)
    var directed = List[_Pair](capacity=n * k)
    for i in range(n):
        for offset in range(i * k, (i + 1) * k):
            var j = indices[offset]
            directed.append((min(i, j), max(i, j), conditional.unsafe_get(offset)))
    return _merge_neighbor_probabilities(directed^)


def _joint_probabilities(
    var squared: NDArray[DType.float32], perplexity: Float64, verbose: Int = 0,
) raises -> NDArray[DType.float64]:
    """Symmetrize the row-wise Gaussian probabilities into condensed storage."""
    var n = squared.shape[0]
    var conditional = _conditional_probabilities(squared, perplexity, verbose)
    var joint = zeros[DType.float64](Shape(n * (n - 1) // 2))
    var k = 0
    var total: Float64 = 0
    for i in range(n):
        for j in range(i + 1, n):
            var value = conditional.unsafe_get(i * n + j) + conditional.unsafe_get(j * n + i)
            joint.unsafe_set(k, value)
            total += value
            k += 1
    var denominator = max(2 * total, _EPS)
    for index in range(joint.size):
        joint.unsafe_set(index, max(joint.unsafe_get(index) / denominator, _EPS))
    return joint^


def _kl_divergence[dtype: DType](
    coordinates: NDArray[dtype], probabilities: NDArray[DType.float64],
    mut weights: NDArray[DType.float64], mut gradient: NDArray[DType.float64],
    degrees_of_freedom: Float64, compute_error: Bool,
) raises -> Float64:
    """Evaluate the Student t objective and accumulate both ends of each pair."""
    var n = coordinates.shape[0]
    var components = coordinates.shape[1]
    var k = 0
    var weight_sum: Float64 = 0
    var exponent = -(degrees_of_freedom + 1) / 2
    for i in range(n):
        for j in range(i + 1, n):
            var squared: Float64 = 0
            for c in range(components):
                var delta = Float64(coordinates.unsafe_get(i * components + c)) - Float64(
                    coordinates.unsafe_get(j * components + c)
                )
                squared += delta * delta
            var base = 1 + squared / degrees_of_freedom
            var weight = 1 / base if degrees_of_freedom == 1 else metric_power(base, exponent)
            weights.unsafe_set(k, weight)
            weight_sum += weight
            k += 1
    if k > 0 and (not isfinite(weight_sum) or weight_sum <= 0):
        raise Error("ValueError: t-SNE Student t normalization is not finite and positive")
    for index in range(gradient.size):
        gradient.unsafe_set(index, 0)
    var normalizer = 2 * weight_sum
    var loss: Float64 = 0
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            var weight = weights.unsafe_get(k)
            var p = probabilities.unsafe_get(k)
            var q = max(weight / normalizer, _EPS)
            var coefficient = (p - q) * weight
            if compute_error:
                loss += p * log(max(p, _EPS) / q)
            for c in range(components):
                var left = i * components + c
                var right = j * components + c
                # sklearn forms the difference in the embedding dtype before
                # the Float64 dot, then rounds the gradient before scaling.
                var delta = Float64(coordinates.unsafe_get(left) - coordinates.unsafe_get(right))
                var contribution = coefficient * delta
                gradient.unsafe_set(left, gradient.unsafe_get(left) + contribution)
                gradient.unsafe_set(right, gradient.unsafe_get(right) - contribution)
            k += 1
    return 2 * loss


def _gradient_descent[dtype: DType](
    mut coordinates: NDArray[dtype], probabilities: NDArray[DType.float64],
    mut weights: NDArray[DType.float64], mut gradient: NDArray[DType.float64],
    degrees_of_freedom: Float64, start: Int, stop: Int, patience: Int,
    momentum: Float64, learning_rate: Float64, min_grad_norm: Float64, verbose: Int,
    pairs: Optional[List[_Pair]] = None, angle: Float32 = 0.5, probability_scale: Float64 = 1,
) raises -> Tuple[Float64, Int]:
    # The reference starts each phase with fresh velocity and gains.
    var update = zeros[dtype](coordinates.shape)
    var gains = zeros[dtype](coordinates.shape)
    for k in range(gains.size):
        gains.unsafe_set(k, 1)
    var best_loss: Float64 = _MAX_FLOAT64
    var loss: Float64 = _MAX_FLOAT64
    var best_iteration = start
    var last_iteration = start
    var gradient_scale = SIMD[dtype, 1](2 * (degrees_of_freedom + 1) / degrees_of_freedom)
    for iteration in range(start, stop):
        last_iteration = iteration
        var check = (iteration + 1) % _CHECK_INTERVAL == 0
        var compute_error = check or iteration == stop - 1
        if pairs:
            loss = _barnes_hut_objective(
                coordinates, pairs.value(), probability_scale, gradient, angle, compute_error,
            )
        else:
            loss = _kl_divergence(
                coordinates, probabilities, weights, gradient, degrees_of_freedom, compute_error,
            )
        var squared_norm: Float64 = 0
        for k in range(coordinates.size):
            var velocity = update.unsafe_get(k)
            var value = gradient.unsafe_get(k).cast[dtype]() * gradient_scale
            var gain = gains.unsafe_get(k)
            gain = gain + 0.2 if velocity * value < 0 else gain * 0.8
            gain = max(gain, SIMD[dtype, 1](0.01))
            gains.unsafe_set(k, gain)
            value *= gain
            velocity = SIMD[dtype, 1](momentum) * velocity - SIMD[dtype, 1](learning_rate) * value
            update.unsafe_set(k, velocity)
            var next = coordinates.unsafe_get(k) + velocity
            if not isfinite(next):
                raise Error("ValueError: t-SNE optimization produced nonfinite coordinates")
            coordinates.unsafe_set(k, next)
            if check:
                squared_norm += Float64(value) * Float64(value)
        if check:
            var norm = Float64(sqrt(squared_norm).cast[dtype]())
            if verbose >= 2:
                print("[t-SNE] Iteration", iteration + 1, "error =", loss, "gradient norm =", norm)
            if loss < best_loss:
                best_loss = loss
                best_iteration = iteration
            elif iteration - best_iteration > patience:
                if verbose >= 2:
                    print("[t-SNE] Iteration", iteration + 1, "did not make any progress")
                break
            if norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration", iteration + 1, "gradient norm", norm, "finished")
                break
    return loss, last_iteration
