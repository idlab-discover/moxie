# Adapted from scikit-learn sklearn/preprocessing/_discretization.py,
# sklearn/utils/_indexing.py (resample) and sklearn/cluster/_kmeans.py,
# _k_means_lloyd.pyx, _k_means_common.pyx (single-run Lloyd); authors:
# scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause.
# NumPy mean/var reduction rules: Copyright (c) 2005-2025, NumPy Developers
# (BSD-3-Clause, LICENSES/numpy.txt). Percentiles and linspace live in
# `_percentiles`, pairwise sums and sorting in `_special_math`.
"""KBinsDiscretizer: bin continuous features into intervals (dense scope).

Fitting follows sklearn 4358542a operation by operation, in the precision the
reference uses: uniform, k-means and weighted-quantile edges are computed in
the input dtype, unweighted quantile edges in Float64. See
docs/api/discretization.md for the native representation and adaptations.

The 1-D k-means below is a family-local stand-in for the future
`moxie.cluster.KMeans` dependency (docs/plumbing.md: bins depends on
cluster.kmeans). It reproduces only sklearn's call
`KMeans(n_clusters, init=<uniform midpoints>, n_init=1).fit(column[:, None],
sample_weight)`: explicit init, one run, algorithm='lloyd', max_iter=300,
tol=1e-4 scaled by the column variance, single-threaded accumulation.
"""
from numojo import NDArray, Shape
from std.math import sqrt, isinf, isnan, inf
from moxie._random import RandomState, check_random_state
from moxie.preprocessing._base import Preprocessor
from moxie.preprocessing._common import (
    at, at1, matrix, vector, check_matrix, check_fitted, check_n_features,
    check_set_output, check_names_length, feature_names_in, dtype_name,
)
from moxie.preprocessing._special_math import pairwise_sum, pairwise_sum_in, sort_prefix
from moxie.preprocessing._percentiles import (
    linspace_scalar, weighted_percentile_column, percentile_sorted, quantile_method_code,
)


comptime _OWNER = "KBinsDiscretizer"
comptime _MAX_ITER = 300
comptime _KMEANS_TOL = 1e-4


def _reduce_mean[T: DType](total: Scalar[T], count: Int) -> Scalar[T]:
    """`umr_sum(...) / rcount` in NumPy mean/var: the intp divisor promotes the
    division to Float64 and the result is cast back into the input dtype."""
    return Scalar[T](Float64(total) / Float64(count))


# ---------------------------------------------------------------------------
# 1-D Lloyd k-means (local stand-in for cluster.KMeans, see module docstring)


def _assign[T: DType](x: NDArray[T], n: Int, centers: List[Scalar[T]], mut labels: List[Int32]):
    """E-step with distances formed as the reference gemm does: ||c||^2 - 2 x c."""
    var k = len(centers)
    var norms = List[Scalar[T]](capacity=k)
    for j in range(k):
        norms.append(centers[j] * centers[j])
    for i in range(n):
        var xi = x.unsafe_get(i)
        var best = norms[0] + Scalar[T](-2) * (xi * centers[0])
        var label = 0
        for j in range(1, k):
            var d = norms[j] + Scalar[T](-2) * (xi * centers[j])
            if d < best:
                best = d
                label = j
        labels[i] = Int32(label)


def _lloyd_iter[T: DType](
    x: NDArray[T], w: NDArray[T], n: Int, centers_old: List[Scalar[T]],
    mut centers_new: List[Scalar[T]], mut weight_in_clusters: List[Scalar[T]],
    mut labels: List[Int32], mut center_shift: List[Scalar[T]],
):
    """One `lloyd_iter_chunked_dense` step with one thread: E-step, weighted
    M-step, empty-cluster relocation, averaging and center shifts."""
    var k = len(centers_old)
    _assign(x, n, centers_old, labels)
    for j in range(k):
        centers_new[j] = 0
        weight_in_clusters[j] = 0
    for i in range(n):
        var label = Int(labels[i])
        weight_in_clusters[label] += w.unsafe_get(i)
        centers_new[label] += x.unsafe_get(i) * w.unsafe_get(i)
    # _relocate_empty_clusters_dense
    var n_empty = 0
    for j in range(k):
        if weight_in_clusters[j] == 0:
            n_empty += 1
    if n_empty > 0:
        var max_distance = Scalar[T](0)
        for i in range(n):
            var diff = x.unsafe_get(i) - centers_old[Int(labels[i])]
            var dist = diff * diff
            if i == 0 or dist > max_distance:
                max_distance = dist
        if max_distance != 0:
            # The n_empty farthest samples by decreasing distance. NumPy's
            # argpartition leaves the order of tied distances unspecified;
            # ties here take the lowest sample index.
            var far = List[Int](capacity=n_empty)
            for _ in range(n_empty):
                var best = -1
                var best_distance = Scalar[T](0)
                for i in range(n):
                    var taken = False
                    for f in far:
                        if f == i:
                            taken = True
                            break
                    if taken:
                        continue
                    var diff = x.unsafe_get(i) - centers_old[Int(labels[i])]
                    var dist = diff * diff
                    if best == -1 or dist > best_distance:
                        best = i
                        best_distance = dist
                far.append(best)
            var idx = 0
            for j in range(k):
                if weight_in_clusters[j] != 0:
                    continue
                var far_idx = far[idx]
                idx += 1
                var weight = w.unsafe_get(far_idx)
                var old = Int(labels[far_idx])
                var xw = x.unsafe_get(far_idx) * weight
                centers_new[old] -= xw
                centers_new[j] = xw
                weight_in_clusters[j] = weight
                weight_in_clusters[old] -= weight
    # _average_centers: sequential and in place, exactly like the reference.
    var argmax_weight = 0
    for j in range(1, k):
        if weight_in_clusters[j] > weight_in_clusters[argmax_weight]:
            argmax_weight = j
    for j in range(k):
        if weight_in_clusters[j] > 0:
            # C: `alpha = 1.0 / w` divides in double, then stores into T.
            var alpha = Scalar[T](1.0 / Float64(weight_in_clusters[j]))
            centers_new[j] *= alpha
        else:
            centers_new[j] = centers_new[argmax_weight]
    # _center_shift (Euclidean distance of one feature; libm sqrt in double).
    for j in range(k):
        var diff = centers_new[j] - centers_old[j]
        center_shift[j] = Scalar[T](sqrt(Float64(diff * diff)))


def _kmeans_1d[T: DType](
    mut x: NDArray[T], w: NDArray[T], n: Int, init: List[Scalar[T]],
) raises -> List[Scalar[T]]:
    """Centers of `KMeans(len(init), init=init, n_init=1).fit(x[:n, None], w)`.

    `x` is scratch: it is centered in place, as the reference centers its copy.
    Returns unsorted centers in the original coordinates.
    """
    var k = len(init)
    if n < k:
        raise Error("ValueError: n_samples=" + String(n) + " should be >= n_clusters=" + String(k) + ".")
    # _tolerance: mean(np.var(X, axis=0)) * tol, evaluated in T.
    var mean = _reduce_mean(pairwise_sum_in(x, 0, n), n)
    var squares = vector[T](n)
    for i in range(n):
        var d = x.unsafe_get(i) - mean
        squares.unsafe_set(i, d * d)
    var variance = _reduce_mean(pairwise_sum_in(squares, 0, n), n)
    var tol = variance * Scalar[T](_KMEANS_TOL)
    # Center the data and the init on the column mean.
    for i in range(n):
        x.unsafe_set(i, x.unsafe_get(i) - mean)
    var centers = List[Scalar[T]](capacity=k)
    for j in range(k):
        centers.append(init[j] - mean)
    var centers_new = List[Scalar[T]](length=k, fill=Scalar[T](0))
    var weight_in_clusters = List[Scalar[T]](length=k, fill=Scalar[T](0))
    var center_shift = List[Scalar[T]](length=k, fill=Scalar[T](0))
    var labels = List[Int32](length=n, fill=Int32(-1))
    var labels_old = List[Int32](length=n, fill=Int32(-1))
    var shifts = vector[T](k)
    for _ in range(_MAX_ITER):
        _lloyd_iter(x, w, n, centers, centers_new, weight_in_clusters, labels, center_shift)
        var previous = centers^
        centers = centers_new^
        centers_new = previous^
        var same = True
        for i in range(n):
            if labels[i] != labels_old[i]:
                same = False
                break
        if same:
            break  # strict convergence
        for j in range(k):
            shifts.unsafe_set(j, center_shift[j] * center_shift[j])
        if pairwise_sum_in(shifts, 0, k) <= tol:
            break
        for i in range(n):
            labels_old[i] = labels[i]
    for j in range(k):
        centers[j] = centers[j] + mean
    return centers^


# ---------------------------------------------------------------------------
# Parameters and validation messages


@fieldwise_init
struct KBinsDiscretizerParams(Copyable, Movable):
    """Constructor configuration. `n_bins_per_feature`, when set, is sklearn's
    array-like `n_bins` and takes precedence over the scalar `n_bins`."""
    var n_bins: Int
    var n_bins_per_feature: Optional[List[Int]]
    var encode: String
    var strategy: String
    var quantile_method: String
    var subsample: Optional[Int]
    var random_state: Optional[Int]


def _bad_option(name: String, options: List[String], got: String) -> Error:
    var text = String("{")
    for k in range(len(options)):
        if k:
            text += ", "
        text += "'" + options[k] + "'"
    return Error("ValueError: The '" + name + "' parameter of KBinsDiscretizer must be a str among " + text + "}. Got '" + got + "' instead.")


def _contains(options: List[String], value: String) -> Bool:
    for option in options:
        if option == value:
            return True
    return False


# ---------------------------------------------------------------------------
# Estimator


struct KBinsDiscretizer[dtype: DType = DType.float64, out_dtype: DType = dtype](Preprocessor):
    """Bin continuous data into intervals.

    `dtype` is the input dtype. `out_dtype` is sklearn's `dtype` argument, the
    output dtype; its default (the input dtype) is sklearn's `dtype=None`.
    `encode='onehot'`, sklearn's default, needs sparse output and raises at
    fit: select `'onehot-dense'` or `'ordinal'`.
    """
    comptime Params = KBinsDiscretizerParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.out_dtype]
    var n_bins: Int
    var n_bins_per_feature: Optional[List[Int]]
    var encode: String
    var strategy: String
    var quantile_method: String
    var subsample: Optional[Int]
    var random_state: Optional[Int]
    var bin_edges_: List[NDArray[DType.float64]]
    var n_bins_: NDArray[DType.int64]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    # True where sklearn's bin_edges_[j] is Float32 (Float32 input with uniform,
    # k-means or weighted quantile); inverse_transform then averages in Float32.
    var _float32_edges: List[Bool]

    def __init__(out self, n_bins: Int = 5, *, encode: String = "onehot", strategy: String = "quantile",
        quantile_method: String = "averaged_inverted_cdf", subsample: Optional[Int] = 200000,
        random_state: Optional[Int] = None,
    ) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 input only"
        comptime assert Self.out_dtype == DType.float32 or Self.out_dtype == DType.float64, "Only np.float32 and np.float64 output dtypes are supported"
        self.n_bins = n_bins
        self.n_bins_per_feature = None
        self.encode = encode
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.subsample = subsample
        self.random_state = random_state
        self.bin_edges_ = List[NDArray[DType.float64]]()
        self.n_bins_ = vector[DType.int64](0)
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self._float32_edges = List[Bool]()

    def __init__(out self, n_bins: List[Int], *, encode: String = "onehot", strategy: String = "quantile",
        quantile_method: String = "averaged_inverted_cdf", subsample: Optional[Int] = 200000,
        random_state: Optional[Int] = None,
    ) raises:
        """Array-like `n_bins`: one bin count per feature (validated at fit)."""
        self = Self(5, encode=encode, strategy=strategy, quantile_method=quantile_method, subsample=subsample, random_state=random_state)
        self.n_bins_per_feature = n_bins.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.n_bins, self.n_bins_per_feature.copy(), self.encode, self.strategy, self.quantile_method, self.subsample, self.random_state)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.n_bins = params.n_bins
        self.n_bins_per_feature = params.n_bins_per_feature.copy()
        self.encode = params.encode
        self.strategy = params.strategy
        self.quantile_method = params.quantile_method
        self.subsample = params.subsample
        self.random_state = params.random_state
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        n_bins: Optional[Int] = None,
        encode: Optional[String] = None,
        strategy: Optional[String] = None,
        quantile_method: Optional[String] = None,
        subsample: Optional[Optional[Int]] = None,
        random_state: Optional[Optional[Int]] = None,
    ) raises -> ref[self] Self:
        """Keyword updates; a scalar `n_bins` replaces a per-feature list.
        `subsample=Optional[Int](None)` selects sklearn's `subsample=None`."""
        if n_bins:
            self.n_bins = n_bins.value()
            self.n_bins_per_feature = None
        if encode:
            self.encode = encode.value()
        if strategy:
            self.strategy = strategy.value()
        if quantile_method:
            self.quantile_method = quantile_method.value()
        if subsample:
            self.subsample = subsample.value()
        if random_state:
            self.random_state = random_state.value()
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        n_bins: List[Int],
    ) raises -> ref[self] Self:
        """Set a per-feature (array-like) `n_bins`."""
        self.n_bins_per_feature = n_bins.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        _ = result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    # -- validation ---------------------------------------------------------

    def _validate_params(self) raises:
        """sklearn `_parameter_constraints` at fit, in its sorted-name order."""
        var encodes: List[String] = ["onehot", "onehot-dense", "ordinal"]
        if not _contains(encodes, self.encode):
            raise _bad_option("encode", encodes, self.encode)
        if not self.n_bins_per_feature and self.n_bins < 2:
            raise Error("ValueError: The 'n_bins' parameter of KBinsDiscretizer must be an int in the range [2, inf) or an array-like. Got " + String(self.n_bins) + " instead.")
        if quantile_method_code(self.quantile_method) < 0:
            var methods: List[String] = [
                "averaged_inverted_cdf", "closest_observation", "hazen", "interpolated_inverted_cdf",
                "inverted_cdf", "linear", "median_unbiased", "normal_unbiased", "weibull",
            ]
            raise _bad_option("quantile_method", methods, self.quantile_method)
        if self.random_state:
            var seed = self.random_state.value()
            if seed < 0 or seed > 4294967295:
                raise Error("ValueError: The 'random_state' parameter of KBinsDiscretizer must be an int in the range [0, 4294967295], an instance of 'numpy.random.mtrand.RandomState' or None. Got " + String(seed) + " instead.")
        var strategies: List[String] = ["kmeans", "quantile", "uniform"]
        if not _contains(strategies, self.strategy):
            raise _bad_option("strategy", strategies, self.strategy)
        if self.subsample and self.subsample.value() < 1:
            raise Error("ValueError: The 'subsample' parameter of KBinsDiscretizer must be an int in the range [1, inf) or None. Got " + String(self.subsample.value()) + " instead.")
        if self.encode == "onehot":
            raise Error("ValueError: encode='onehot' returns a sparse matrix and sparse output is deferred; select encode='onehot-dense' or encode='ordinal'")

    def _validate_n_bins(self, n_features: Int) raises -> List[Int]:
        var result = List[Int]()
        if not self.n_bins_per_feature:
            for _ in range(n_features):
                result.append(self.n_bins)
            return result^
        var given = self.n_bins_per_feature.value().copy()
        if len(given) != n_features:
            raise Error("ValueError: n_bins must be a scalar or array of shape (n_features,).")
        var bad = String("")
        for j in range(n_features):
            if given[j] < 2:
                if bad.byte_length():
                    bad += ", "
                bad += String(j)
        if bad.byte_length():
            raise Error("ValueError: KBinsDiscretizer received an invalid number of bins at indices " + bad + ". Number of bins must be at least 2, and must be an int.")
        return given^

    # -- fitting ------------------------------------------------------------

    def _check_weights(self, X: Self.Input, sample_weight: NDArray[Self.dtype]) raises:
        """sklearn `_check_sample_weight(sample_weight, X, dtype=X.dtype)`."""
        var n = X.shape[0]
        if sample_weight.ndim != 1:
            raise Error("ValueError: Sample weights must be 1D array or scalar, got " + String(sample_weight.ndim) + "D array. Expected either a scalar value or a 1D array of length " + String(n) + ".")
        if sample_weight.shape[0] != n:
            raise Error("ValueError: sample_weight.shape == (" + String(sample_weight.shape[0]) + ",), expected (" + String(n) + ",)!")
        var all_zero = True
        for i in range(n):
            var value = at1(sample_weight, i)
            if isnan(value):
                raise Error("ValueError: Input sample_weight contains NaN.")
            if isinf(value):
                raise Error("ValueError: Input sample_weight contains infinity or a value too large for dtype('" + dtype_name[Self.dtype]() + "').")
            if value != 0:
                all_zero = False
        if all_zero:
            raise Error("ValueError: Sample weights must contain at least one non-zero number.")

    def _draw_rows(self, mut rng: RandomState, n: Int, sample_weight: NDArray[Self.dtype], weighted: Bool) raises -> NDArray[DType.int64]:
        """sklearn `resample(X, replace=True, n_samples=subsample, sample_weight=...)`.

        Rows are drawn with replacement by NumPy's legacy `choice`; weights are
        normalized as `w / w.sum()` in Float64 with NumPy's pairwise sum.
        """
        var size = self.subsample.value()
        if not weighted:
            return rng.choice(n, size=size)
        var prob = vector[DType.float64](n)
        for i in range(n):
            prob.unsafe_set(i, Float64(at1(sample_weight, i)))
        var total = pairwise_sum(prob, 0, n)
        for i in range(n):
            prob.unsafe_set(i, prob.unsafe_get(i) / total)
        return rng.choice(n, prob, size=size)

    def _fit(
        mut self, X: Self.Input, sample_weight: NDArray[Self.dtype], has_weights: Bool,
        mut external: RandomState, use_external: Bool,
    ) raises:
        """Learned state is replaced only after every check and feature succeeds."""
        comptime T = Self.dtype
        self._validate_params()
        check_matrix(X, allow_nan=False)
        var n = X.shape[0]
        var p = X.shape[1]
        var weighted = has_weights
        if weighted:
            self._check_weights(X, sample_weight)

        # Subsample rows with replacement; the weights are consumed by the draw.
        var use_rows = False
        var rows = vector[DType.int64](0)
        var n_eff = n
        if self.subsample and n > self.subsample.value():
            if use_external:
                rows = self._draw_rows(external, n, sample_weight, weighted)
            elif self.random_state:
                with check_random_state(self.random_state.value()) as rng:
                    rows = self._draw_rows(rng[], n, sample_weight, weighted)
            else:
                with check_random_state() as rng:
                    rows = self._draw_rows(rng[], n, sample_weight, weighted)
            use_rows = True
            n_eff = self.subsample.value()
            weighted = False

        var n_bins = self._validate_n_bins(p)
        var method = quantile_method_code(self.quantile_method)
        if self.strategy == "quantile" and weighted and method != 0 and method != 1:
            raise Error("ValueError: When fitting with strategy='quantile' and sample weights, quantile_method should either be set to 'averaged_inverted_cdf' or 'inverted_cdf', got quantile_method='" + self.quantile_method + "' instead.")

        # Bounded workspace: one column (k-means: values + weights in T;
        # unweighted quantile: values in Float64) reused for every feature.
        var kmeans = self.strategy == "kmeans"
        var plain_quantile = self.strategy == "quantile" and not weighted
        var weighted_quantile = self.strategy == "quantile" and weighted
        var column = vector[T](n_eff if kmeans else 0)
        var column_weights = vector[T](n_eff if kmeans else 0)
        var ordered = vector[DType.float64](n_eff if plain_quantile else 0)
        var weight_list = List[Scalar[T]]()
        if weighted_quantile:
            for i in range(n):
                weight_list.append(at1(sample_weight, i))
        var edges_out = List[NDArray[DType.float64]]()
        var float32_edges = List[Bool]()
        var counts = vector[DType.int64](p)
        for j in range(p):
            # Extremes, over nonzero-weight samples for the uniform/k-means strategies.
            var masked = weighted and not self.strategy == "quantile"
            var col_min = Scalar[T](0)
            var col_max = Scalar[T](0)
            var seen = False
            for i in range(n_eff):
                var r = Int(rows.unsafe_get(i)) if use_rows else i
                var value = at(X, r, j)
                if kmeans:
                    column.unsafe_set(i, value)
                    column_weights.unsafe_set(i, at1(sample_weight, r) if weighted else Scalar[T](1))
                elif plain_quantile:
                    ordered.unsafe_set(i, Float64(value))
                if masked and at1(sample_weight, r) == 0:
                    continue
                if not seen or value < col_min:
                    col_min = value
                if not seen or value > col_max:
                    col_max = value
                seen = True
            var nb = n_bins[j]
            if col_min == col_max:
                # Constant feature: a single bin with edges [-inf, inf] (Float64).
                var constant = vector[DType.float64](2)
                constant.unsafe_set(0, -inf[DType.float64]())
                constant.unsafe_set(1, inf[DType.float64]())
                edges_out.append(constant^)
                float32_edges.append(False)
                counts.unsafe_set(j, 1)
                continue
            var edges = List[Float64]()
            var input_precision = True
            if self.strategy == "uniform":
                for edge in linspace_scalar[T](col_min, col_max, nb + 1):
                    edges.append(Float64(edge))
            elif plain_quantile:
                input_precision = False
                sort_prefix(ordered, n_eff)
                for level in linspace_scalar[DType.float64](0, 100, nb + 1):
                    edges.append(percentile_sorted[T](ordered, n_eff, level / 100, method))
            elif weighted_quantile:
                var ranks = linspace_scalar[DType.float64](0, 100, nb + 1)
                for edge in weighted_percentile_column(X, j, weight_list, ranks, method == 1):
                    edges.append(Float64(edge))
            else:
                var uniform = linspace_scalar[T](col_min, col_max, nb + 1)
                var init = List[Scalar[T]](capacity=nb)
                for k in range(nb):
                    init.append((uniform[k + 1] + uniform[k]) * Scalar[T](0.5))
                var centers = _kmeans_1d(column, column_weights, n_eff, init)
                var sorted_centers = vector[DType.float64](nb)
                for k in range(nb):
                    sorted_centers.unsafe_set(k, Float64(centers[k]))
                sort_prefix(sorted_centers, nb)
                edges.append(Float64(col_min))
                for k in range(nb - 1):
                    var mid = (Scalar[T](sorted_centers.unsafe_get(k + 1)) + Scalar[T](sorted_centers.unsafe_get(k))) * Scalar[T](0.5)
                    edges.append(Float64(mid))
                edges.append(Float64(col_max))
            # Remove bins whose width is too small (<= 1e-8 in the edge dtype).
            if self.strategy == "quantile" or kmeans:
                var kept = List[Float64]()
                for k in range(len(edges)):
                    var keep = k == 0
                    if not keep:
                        if input_precision:
                            keep = (Scalar[T](edges[k]) - Scalar[T](edges[k - 1])) > Scalar[T](1e-8)
                        else:
                            keep = (edges[k] - edges[k - 1]) > 1e-8
                    if keep:
                        kept.append(edges[k])
                edges = kept^
            var array = vector[DType.float64](len(edges))
            for k in range(len(edges)):
                array.unsafe_set(k, edges[k])
            counts.unsafe_set(j, Int64(len(edges) - 1))
            edges_out.append(array^)
            float32_edges.append(input_precision and T == DType.float32)

        self.bin_edges_ = edges_out^
        self._float32_edges = float32_edges^
        self.n_bins_ = counts^
        self.n_features_in_ = p
        self.feature_names_in_ = None

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        var unused = RandomState(0)
        self._fit(X, vector[Self.dtype](0), False, unused, False)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target],
        ref[sample_weight_origin] sample_weight: NDArray[Self.dtype],
    ) raises -> ref[self] Self:
        var unused = RandomState(0)
        self._fit(X, sample_weight, True, unused, False)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        *,
        ref[sample_weight_origin] sample_weight: NDArray[Self.dtype],
    ) raises -> ref[self] Self:
        var unused = RandomState(0)
        self._fit(X, sample_weight, True, unused, False)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        *,
        ref[sample_weight_origin] sample_weight: Float64,
    ) raises -> ref[self] Self:
        """Scalar `sample_weight`: every sample gets that weight, cast to X's dtype."""
        var n = X.shape[0] if X.ndim == 2 else 0
        var weights = vector[Self.dtype](n)
        for i in range(n):
            weights.unsafe_set(i, Scalar[Self.dtype](sample_weight))
        var unused = RandomState(0)
        self._fit(X, weights, True, unused, False)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        mut random_state: RandomState,
    ) raises -> ref[self] Self:
        """Fit drawing any subsample from a borrowed RandomState (advanced in
        place); the configured `random_state` is ignored, like passing a
        RandomState instance to sklearn's constructor."""
        self._fit(X, vector[Self.dtype](0), False, random_state, True)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        mut random_state: RandomState,
        *,
        ref[sample_weight_origin] sample_weight: NDArray[Self.dtype],
    ) raises -> ref[self] Self:
        self._fit(X, sample_weight, True, random_state, True)
        return self

    def fit_named[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        feature_names: List[String],
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        check_matrix(X, allow_nan=False)
        check_names_length(feature_names, X.shape[1])
        _ = self.fit(X, y)
        self.feature_names_in_ = feature_names.copy()
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def fit_transform(mut self, X: Self.Input, *, sample_weight: NDArray[Self.dtype]) raises -> Self.Transformed:
        _ = self.fit(X, sample_weight=sample_weight)
        return self.transform(X)

    # -- transforms ---------------------------------------------------------

    def _bin(self, value: Scalar[Self.out_dtype], j: Int) raises -> Int:
        """`np.searchsorted(bin_edges_[j][1:-1], value, side='right')`."""
        ref edges = self.bin_edges_[j]
        var key = Float64(value)
        var low = 1
        var high = edges.shape[0] - 1
        while low < high:
            var mid = low + ((high - low) >> 1)
            if edges.unsafe_get(mid) <= key:
                low = mid + 1
            else:
                high = mid
        return low - 1

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """Bin indices (ordinal) or their dense one-hot expansion, in `out_dtype`.

        Values are first cast to `out_dtype`, as sklearn validates X with that dtype.
        """
        check_fitted(self.__sklearn_is_fitted__(), _OWNER)
        check_matrix(X, allow_nan=False)
        check_n_features(_OWNER, self.n_features_in_, X.shape[1])
        var n = X.shape[0]
        var p = X.shape[1]
        if self.encode == "ordinal":
            var result = matrix[Self.out_dtype](n, p)
            for i in range(n):
                for j in range(p):
                    var value = Scalar[Self.out_dtype](at(X, i, j))
                    result.unsafe_set(i * p + j, Scalar[Self.out_dtype](self._bin(value, j)))
            return result^
        if self.encode != "onehot-dense":
            raise Error("ValueError: encode='" + self.encode + "' changed after fit; refit the estimator")
        var offsets = List[Int](capacity=p)
        var width = 0
        for j in range(p):
            offsets.append(width)
            width += Int(self.n_bins_.unsafe_get(j))
        var result = matrix[Self.out_dtype](n, width)
        for i in range(n):
            for j in range(p):
                var value = Scalar[Self.out_dtype](at(X, i, j))
                result.unsafe_set(i * width + offsets[j] + self._bin(value, j), 1)
        return result^

    def _center(self, j: Int, b: Int) raises -> Float64:
        """Bin center `(edges[b + 1] + edges[b]) * 0.5` in the edge dtype."""
        var hi = self.bin_edges_[j].unsafe_get(b + 1)
        var lo = self.bin_edges_[j].unsafe_get(b)
        if self._float32_edges[j]:
            return Float64((Float32(hi) + Float32(lo)) * Float32(0.5))
        return (hi + lo) * 0.5

    def inverse_transform[d: DType](self, X: NDArray[d]) raises -> NDArray[d]:
        """Bin centers of binned data; the result has the dtype of `X`.

        Ordinal input truncates each value toward zero and indexes the centers
        as NumPy does (negative indices wrap, others raise IndexError). One-hot
        input picks the first maximal column of each feature block; an
        all-zero block raises. Constant features invert to NaN.
        """
        comptime assert d == DType.float32 or d == DType.float64, "Float32/Float64 only"
        check_fitted(self.__sklearn_is_fitted__(), _OWNER)
        check_matrix(X, allow_nan=False)
        var n = X.shape[0]
        var p = self.n_bins_.shape[0]
        var result = matrix[d](n, p)
        if self.encode == "onehot-dense":
            var width = 0
            for j in range(p):
                width += Int(self.n_bins_.unsafe_get(j))
            if X.shape[1] != width:
                raise Error("ValueError: Shape of the passed X data is not correct. Expected " + String(width) + " columns, got " + String(X.shape[1]) + ".")
            var offset = 0
            for j in range(p):
                var nb = Int(self.n_bins_.unsafe_get(j))
                var zero_rows = String("")
                var block = vector[d](nb)
                for i in range(n):
                    var best = 0
                    for b in range(nb):
                        var value = at(X, i, offset + b)
                        block.unsafe_set(b, value)
                        if value > at(X, i, offset + best):
                            best = b
                    if pairwise_sum_in(block, 0, nb) == 0:
                        if zero_rows.byte_length():
                            zero_rows += " "
                        zero_rows += String(i)
                    result.unsafe_set(i * p + j, Scalar[d](self._center(j, best)))
                if zero_rows.byte_length():
                    raise Error("ValueError: Samples [" + zero_rows + "] can not be inverted when drop=None and handle_unknown='error' because they contain all zeros")
                offset += nb
            return result^
        if X.shape[1] != p:
            raise Error("ValueError: Incorrect number of features. Expecting " + String(p) + ", received " + String(X.shape[1]) + ".")
        for j in range(p):
            var nb = Int(self.n_bins_.unsafe_get(j))
            for i in range(n):
                var value = Float64(at(X, i, j))
                # astype(int64): truncation; out-of-range values become INT64_MIN.
                var index = Int(value) if (value > -9.2e18 and value < 9.2e18) else -9223372036854775807 - 1
                if index < -nb or index >= nb:
                    raise Error("IndexError: index " + String(index) + " is out of bounds for axis 0 with size " + String(nb))
                if index < 0:
                    index += nb
                result.unsafe_set(i * p + j, Scalar[d](self._center(j, index)))
        return result^

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        """Ordinal: the input names. One-hot: `<name>_<bin as float>` (x0_0.0, x0_1.0, ...)."""
        check_fitted(self.__sklearn_is_fitted__(), _OWNER)
        var names = feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)
        if self.encode == "ordinal":
            return names^
        var result = List[String]()
        for j in range(len(names)):
            for b in range(Int(self.n_bins_.unsafe_get(j))):
                result.append(names[j] + "_" + String(Float64(b)))
        return result^
