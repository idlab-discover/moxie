"""Dense affinity propagation. Adapted from sklearn; see LICENSES/scikit-learn.txt."""
import numojo as nm
from std.math import isfinite, isnan
from moxie._random import RandomState
from moxie.utils.validation import FittedState
from moxie._dense import sum_axis2d, row_squared_norms
from moxie.cluster._contracts import Labels, validate_matrix, validate_labels


@fieldwise_init
struct AffinityResult(Copyable, Movable):
    var cluster_centers_indices: Labels
    var labels: Labels
    var n_iter: Optional[Int]
    var converged: Bool
    var equal_similarities: Bool


def _finite[dtype: DType](x: nm.NDArray[dtype]) raises:
    validate_matrix(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not isfinite(x.item(i, j)):
                raise Error("ValueError: input must contain only finite values")


def _get[dtype: DType](x: nm.NDArray[dtype], i: Int, j: Int) raises -> SIMD[dtype, 1]:
    return x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))


def _put[dtype: DType](mut x: nm.NDArray[dtype], i: Int, j: Int, v: SIMD[dtype, 1]) raises:
    x.unsafe_set(i * Int(x.strides[0]) + j * Int(x.strides[1]), v)


def _engine[dtype: DType](mut s: nm.NDArray[dtype], preference: nm.NDArray[DType.float64], mut rng: RandomState, damping: Float64, max_iter: Int, convergence_iter: Int, verbose: Bool) raises -> AffinityResult:
    var n = s.shape[0]
    var labels = nm.zeros[DType.int64](nm.Shape(n))
    var equal = True
    for i in range(n):
        if preference.item(i) != preference.item(0):
            equal = False
        for j in range(n):
            if i != j and _get(s, i, j) != _get(s, 0, n - 1):
                equal = False
    if n == 1 or equal:
        var k = n if preference.item(0) > Float64(_get(s, 0, n - 1)) else 1
        var indices = nm.zeros[DType.int64](nm.Shape(k))
        for i in range(k):
            indices.unsafe_set(i, Int64(i))
        if k == n:
            for i in range(n):
                labels.unsafe_set(i, Int64(i))
        if verbose:
            print("All samples have mutually equal similarities. Returning arbitrary cluster center(s).")
        return AffinityResult(indices^, labels^, Optional[Int](0), True, True)
    var eps = SIMD[dtype, 1](2.220446049250313e-16)
    var tiny = SIMD[dtype, 1](2.2250738585072014e-308)
    comptime if dtype == DType.float32:
        eps = SIMD[dtype, 1](1.1920928955078125e-7)
        tiny = SIMD[dtype, 1](1.1754943508222875e-38)
    for i in range(n):
        _put(s, i, i, preference.item(i).cast[dtype]())
    for i in range(n):
        for j in range(n):
            var v = _get(s, i, j)
            var scale = eps * v + tiny * 100
            _put(s, i, j, (Float64(v) + Float64(scale) * rng.standard_normal()).cast[dtype]())
    var a = nm.zeros[DType.float64](nm.Shape(n, n))
    var r = nm.zeros[DType.float64](nm.Shape(n, n))
    var positive = nm.zeros[DType.float64](nm.Shape(n, n))
    var history = nm.zeros[DType.int64](nm.Shape(n, convergence_iter))
    var exemplars = List[Int]()
    var converged = False
    var iterations = 0
    for it in range(max_iter):
        for i in range(n):
            var best = Float64(_get(s, i, 0)) + _get(a, i, 0)
            var second = -Float64.MAX
            var index = 0
            for j in range(1, n):
                var v = Float64(_get(s, i, j)) + _get(a, i, j)
                if v > best or (isnan(v) and not isnan(best)):
                    second = best
                    best = v
                    index = j
                elif v > second or isnan(v):
                    second = v
            for j in range(n):
                var value = Float64(_get(s, i, j)) - (second if j == index else best)
                _put(r, i, j, damping * _get(r, i, j) + (1 - damping) * value)
        for i in range(n):
            for j in range(n):
                _put(positive, i, j, _get(r, i, j) if i == j or isnan(_get(r, i, j)) else max(_get(r, i, j), Float64(0)))
        var totals = sum_axis2d(positive, 0)
        for i in range(n):
            for j in range(n):
                var v = _get(positive, i, j) - totals.item(j)
                if i != j and not isnan(v):
                    v = max(v, Float64(0))
                _put(a, i, j, damping * _get(a, i, j) - (1 - damping) * v)
        exemplars.clear()
        var stable = True
        for i in range(n):
            var present = _get(a, i, i) + _get(r, i, i) > 0
            _put(history, i, it % convergence_iter, Int64(present))
            if present:
                exemplars.append(i)
            if it >= convergence_iter:
                var total = Int64(0)
                for j in range(convergence_iter):
                    total += _get(history, i, j)
                if total != 0 and total != Int64(convergence_iter):
                    stable = False
        iterations = it + 1
        if it >= convergence_iter and stable and len(exemplars) > 0:
            converged = True
            break
    if verbose:
        if converged:
            print("Converged after", iterations - 1, "iterations.")
        else:
            print("Did not converge")
    var k = len(exemplars)
    if k == 0:
        for i in range(n):
            labels.unsafe_set(i, Int64(-1))
        return AffinityResult(nm.zeros[DType.int64](nm.Shape(0)), labels^, Optional[Int](iterations), False, False)
    # Initial assignment and medoid refinement use perturbed similarities.
    for i in range(n):
        var best = 0
        for j in range(1, k):
            if _get(s, i, exemplars[j]) > _get(s, i, exemplars[best]):
                best = j
        labels.unsafe_set(i, Int64(best))
    for j in range(k):
        labels.unsafe_set(exemplars[j], Int64(j))
    for j in range(k):
        var best = -1
        var best_sum = -SIMD[dtype, 1].MAX
        for candidate in range(n):
            if Int(labels.item(candidate)) != j:
                continue
            var total = SIMD[dtype, 1](0)
            for i in range(n):
                if Int(labels.item(i)) == j:
                    total += _get(s, i, candidate)
            if best == -1 or total > best_sum:
                best = candidate
                best_sum = total
        exemplars[j] = best
    for i in range(n):
        var best = 0
        for j in range(1, k):
            if _get(s, i, exemplars[j]) > _get(s, i, exemplars[best]):
                best = j
        labels.unsafe_set(i, Int64(exemplars[best]))
    for j in range(k):
        labels.unsafe_set(exemplars[j], Int64(exemplars[j]))
    var indices = nm.zeros[DType.int64](nm.Shape(k))
    # Scan sample order to obtain sorted, gapless exemplar numbering.
    var next_index = 0
    for i in range(n):
        var found = False
        for j in range(k):
            if exemplars[j] == i:
                found = True
        if found:
            indices.unsafe_set(next_index, Int64(i))
            next_index += 1
    for i in range(n):
        for j in range(k):
            if labels.item(i) == indices.item(j):
                labels.unsafe_set(i, Int64(j))
                break
    validate_labels(labels, n, k)
    return AffinityResult(indices^, labels^, Optional[Int](iterations), converged, False)


struct Preference(Copyable, Movable):
    """Typed None/scalar/vector preference; arrays are independently owned."""
    var scalar: Optional[Float64]
    var values: Optional[nm.NDArray[DType.float64]]

    def __init__(out self):
        self.scalar = None
        self.values = None

    @implicit
    def __init__(out self, value: Float64):
        self.scalar = value
        self.values = None

    @implicit
    def __init__(out self, value: Int):
        self.scalar = Float64(value)
        self.values = None

    @implicit
    def __init__(out self, values: nm.NDArray[DType.float64]):
        self.scalar = None
        self.values = values.copy()


@fieldwise_init
struct AffinityParams(Copyable, Movable):
    var damping: Float64
    var max_iter: Int
    var convergence_iter: Int
    var copy_input: Bool
    var preference: Preference
    var affinity: String
    var verbose: Bool
    var random_state: Optional[Int]


def _preferences[dtype: DType](s: nm.NDArray[dtype], pref: Preference) raises -> nm.NDArray[DType.float64]:
    var n = s.shape[0]
    var result = nm.zeros[DType.float64](nm.Shape(n))
    if pref.values:
        ref values = pref.values.value()
        if values.ndim != 1 or (values.size != n and values.size != 1):
            raise Error("ValueError: preference must be scalar or a vector with n_samples entries")
        for i in range(n):
            result.unsafe_set(i, values.item(0 if values.size == 1 else i))
    else:
        var value: Float64
        if pref.scalar:
            value = pref.scalar.value()
            if not isfinite(value):
                raise Error("ValueError: scalar preference must be finite")
        else:
            var flat = nm.zeros[dtype](nm.Shape(n * n))
            for i in range(n):
                for j in range(n):
                    flat.unsafe_set(i * n + j, _get(s, i, j))
            var ordered = nm.sort(flat)
            var middle = n * n // 2
            if n * n % 2:
                value = Float64(ordered.item(middle))
            else:
                value = Float64((ordered.item(middle - 1) + ordered.item(middle)) / 2)
        for i in range(n):
            result.unsafe_set(i, value)
    return result^


def _similarities[dtype: DType](x: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
    # sklearn computes Float32 pairwise distances in Float64, then casts back.
    var wide = x.astype[DType.float64]()
    var norms = row_squared_norms(wide)
    var transposed = wide.T()
    var gram = wide @ transposed
    var s = nm.zeros[dtype](nm.Shape(x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            var d = (-2 * _get(gram, i, j) + norms.item(i)) + norms.item(j)
            _put(s, i, j, (-max(d, Float64(0)) if i != j else Float64(0)).cast[dtype]())
    return s^


from moxie.cluster._base import PredictiveClusterer


struct AffinityPropagation[dtype: DType = DType.float64](PredictiveClusterer, Copyable, Movable):
    comptime Params = AffinityParams
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime Prediction = Labels
    comptime Labels = nm.NDArray[DType.int64]
    var params: AffinityParams
    var cluster_centers_indices_: Self.Labels
    var labels_: Self.Labels
    var cluster_centers_: Optional[Self.Input]
    var affinity_matrix_: Self.Input
    var n_iter_: Int
    var n_features_in_: Int
    var converged_: Bool
    var equal_similarities_: Bool
    var _state: FittedState

    def __init__(out self, *, damping: Float64 = 0.5, max_iter: Int = 200, convergence_iter: Int = 15, copy: Bool = True, var preference: Preference = Preference(), affinity: String = "euclidean", verbose: Bool = False, random_state: Optional[Int] = None) raises:
        self.params = AffinityParams(damping, max_iter, convergence_iter, copy, preference^, affinity, verbose, random_state)
        self.cluster_centers_indices_ = nm.zeros[DType.int64](nm.Shape(0))
        self.labels_ = nm.zeros[DType.int64](nm.Shape(0))
        self.cluster_centers_ = None
        self.affinity_matrix_ = nm.zeros[Self.dtype](nm.Shape(0, 0))
        self.n_iter_ = 0
        self.n_features_in_ = 0
        self.converged_ = False
        self.equal_similarities_ = False
        self._state = FittedState()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, damping: Optional[Float64] = None, max_iter: Optional[Int] = None, convergence_iter: Optional[Int] = None, copy: Optional[Bool] = None, preference: Optional[Preference] = None, affinity: Optional[String] = None, verbose: Optional[Bool] = None, random_state: Optional[Int] = None) raises -> ref[self] Self:
        if damping: self.params.damping = damping.value()
        if max_iter: self.params.max_iter = max_iter.value()
        if convergence_iter: self.params.convergence_iter = convergence_iter.value()
        if copy: self.params.copy_input = copy.value()
        if preference: self.params.preference = preference.value().copy()
        if affinity: self.params.affinity = affinity.value()
        if verbose: self.params.verbose = verbose.value()
        if random_state: self.params.random_state = random_state
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.is_fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        if not self.params.random_state:
            raise Error("ValueError: random_state=None requires fit(X, mutable_application_random_state)")
        var rng = RandomState(self.params.random_state.value())
        _ = self.fit(X, rng, y)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, mut random_state: RandomState, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self._validate_fit(X)
        var s: Self.Input
        if self.params.affinity == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise Error("ValueError: The matrix of similarities must be a square array")
            if not self.params.copy_input:
                raise Error("ValueError: copy=False precomputed input requires fit_inplace(mut X)")
            s = X.copy()
        else:
            s = _similarities(X)
        self._finish(X, s, random_state)
        return self

    def _validate_fit(self, X: Self.Input) raises:
        _finite(X)
        if not isfinite(self.params.damping) or self.params.damping < 0.5 or self.params.damping >= 1:
            raise Error("ValueError: damping must lie in [0.5, 1)")
        if self.params.max_iter < 1 or self.params.convergence_iter < 1:
            raise Error("ValueError: iteration limits must be positive")
        if self.params.affinity != "euclidean" and self.params.affinity != "precomputed":
            raise Error("ValueError: affinity must be euclidean or precomputed")
        if self.params.affinity == "precomputed" and X.shape[0] != X.shape[1]:
            raise Error("ValueError: The matrix of similarities must be a square array")

    def fit_inplace(mut self, mut X: Self.Input) raises:
        if not self.params.random_state:
            raise Error("ValueError: random_state=None requires mutable application context")
        var rng = RandomState(self.params.random_state.value())
        self.fit_inplace(X, rng)

    def fit_inplace(mut self, mut X: Self.Input, mut random_state: RandomState) raises:
        self._validate_fit(X)
        if self.params.affinity != "precomputed" or self.params.copy_input:
            _ = self.fit(X, random_state)
            return
        var s = X.view()
        self._finish(X, s, random_state)

    def _finish(mut self, X: Self.Input, mut s: Self.Input, mut random_state: RandomState) raises:
        var preferences = _preferences(s, self.params.preference)
        var result = _engine(s, preferences, random_state, self.params.damping, self.params.max_iter, self.params.convergence_iter, self.params.verbose)
        self.cluster_centers_ = None
        if self.params.affinity == "euclidean":
            var centers = nm.zeros[Self.dtype](nm.Shape(result.cluster_centers_indices.size, X.shape[1]))
            for i in range(result.cluster_centers_indices.size):
                for j in range(X.shape[1]):
                    _put(centers, i, j, _get(X, Int(result.cluster_centers_indices.item(i)), j))
            self.cluster_centers_ = centers^
        self.n_iter_ = result.n_iter.value()
        self.converged_ = result.converged
        self.equal_similarities_ = result.equal_similarities
        self.cluster_centers_indices_ = result.cluster_centers_indices.copy()
        self.labels_ = result.labels.copy()
        self.affinity_matrix_ = s.view()
        self.n_features_in_ = X.shape[1]
        self._state.record_fit(X.shape[0], X.shape[1])

    def fit_predict(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Labels:
        _ = self.fit(X, y)
        return self.labels_.copy()

    def fit_predict(mut self, X: Self.Input, mut random_state: RandomState, y: Optional[Self.Target] = None) raises -> Self.Labels:
        _ = self.fit(X, random_state, y)
        return self.labels_.copy()

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        if not self._state.is_fitted():
            raise Error("NotFittedError: AffinityPropagation is not fitted")
        _finite(X)
        self._state.check_features(X.shape[1])
        if not self.cluster_centers_:
            raise Error("ValueError: Predict method is not supported when affinity='precomputed'")
        ref centers = self.cluster_centers_.value()
        var output = nm.zeros[DType.int64](nm.Shape(X.shape[0]))
        for i in range(X.shape[0]):
            var best = 0 if centers.shape[0] > 0 else -1
            var best_distance = Float64.MAX
            for k in range(centers.shape[0]):
                var distance = Float64(0)
                for j in range(X.shape[1]):
                    var delta = Float64(_get(X, i, j)) - Float64(_get(centers, k, j))
                    distance += delta * delta
                if distance < best_distance:
                    best = k
                    best_distance = distance
            output.unsafe_set(i, Int64(best))
        return output^


def affinity_propagation[dtype: DType](mut S: nm.NDArray[dtype], *, var preference: Preference = Preference(), convergence_iter: Int = 15, max_iter: Int = 200, damping: Float64 = 0.5, copy: Bool = True, verbose: Bool = False, return_n_iter: Bool = False, random_state: Optional[Int] = None) raises -> AffinityResult:
    var model = AffinityPropagation[dtype](damping=damping, max_iter=max_iter, convergence_iter=convergence_iter, copy=copy, preference=preference^, affinity="precomputed", verbose=verbose, random_state=random_state)
    model.fit_inplace(S)
    return AffinityResult(model.cluster_centers_indices_.copy(), model.labels_.copy(), Optional[Int](model.n_iter_) if return_n_iter else Optional[Int](None), model.converged_, model.equal_similarities_)


def affinity_propagation[dtype: DType](mut S: nm.NDArray[dtype], mut random_state: RandomState, *, var preference: Preference = Preference(), convergence_iter: Int = 15, max_iter: Int = 200, damping: Float64 = 0.5, copy: Bool = True, verbose: Bool = False, return_n_iter: Bool = False) raises -> AffinityResult:
    var model = AffinityPropagation[dtype](damping=damping, max_iter=max_iter, convergence_iter=convergence_iter, copy=copy, preference=preference^, affinity="precomputed", verbose=verbose)
    model.fit_inplace(S, random_state)
    return AffinityResult(model.cluster_centers_indices_.copy(), model.labels_.copy(), Optional[Int](model.n_iter_) if return_n_iter else Optional[Int](None), model.converged_, model.equal_similarities_)
