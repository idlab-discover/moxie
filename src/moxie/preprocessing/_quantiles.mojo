# Adapted from scikit-learn sklearn/preprocessing/_data.py; authors: scikit-learn developers.
# Lambda estimation follows SciPy 1.18.1 `stats.yeojohnson`/`stats.boxcox`
# (Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers).
# SPDX-License-Identifier: BSD-3-Clause
"""Dense QuantileTransformer, PowerTransformer, quantile_transform, power_transform.

QuantileTransformer: NaN-ignoring NumPy linear percentiles of an optional
row subsample (NumPy legacy `RandomState.shuffle` order through
`moxie._random`), NumPy `interp` forward/backward averaging, sklearn's bound
handling and Cephes `ndtr`/`ndtri` for the normal output.

PowerTransformer: SciPy's Yeo-Johnson (`fminbound` with SciPy's overflow-safe
bounds) and Box-Cox (`brent` from bracket (-2, 2) plus the `ymax` overflow
guard) maximum-likelihood lambdas, then the family StandardScaler.

See docs/api/quantiles.md for adaptations, memory and tolerances.
"""
from moxie._feature_metadata import FeatureNamesOut as NativeFeatureNamesOut
from std.math import isnan, isinf, abs, inf
from numojo import NDArray, Shape, NDArrayStrides
from moxie._random import RandomState, check_random_state
from moxie.preprocessing._base import InvertiblePreprocessor
from moxie.preprocessing._common import (
    at, put, matrix, vector, check_matrix, check_fitted, check_n_features,
    check_set_output, check_names_length, feature_names_in, check_writable,
)
from moxie.preprocessing._special_math import (
    flog, fexp, fpow, flog1p, fexpm1, sign, ndtr, ndtri, boxcox, inv_boxcox,
    lambertw_m1, pairwise_sum, variance, logsumexp, logdiffexp, sort_prefix,
    linear_percentile, InterpTable, F64_EPS, F32_EPS,
)
from moxie.preprocessing._optimize import ScalarObjective, brent, fminbound
from moxie.preprocessing._scalers import StandardScaler

comptime BOUNDS_THRESHOLD = 1e-7
comptime _SEED_MAX = 4294967295


@always_inline
def _rd[d: DType](x: Float64) -> Float64:
    """Round a Float64 to the working dtype `d` (identity for Float64)."""
    return Float64(Scalar[d](x))


def _matrix_view[d: DType](X: NDArray[d], transpose: Bool) raises -> NDArray[d]:
    """Borrowed rank-2 view of X or of its transpose; no copy."""
    if X.ndim != 2:
        raise Error("ValueError: Expected 2D array, got " + String(X.ndim) + "D array instead")
    if transpose:
        return X.view_with_layout(Shape(X.shape[1], X.shape[0]), NDArrayStrides(Int(X.strides[1]), Int(X.strides[0])), X.offset)
    return X.view_with_layout(Shape(X.shape[0], X.shape[1]), NDArrayStrides(Int(X.strides[0]), Int(X.strides[1])), X.offset)


def _writable_transpose[d: DType](mut owner: NDArray[d]) raises -> NDArray[d]:
    """Writable transposed view of a freshly allocated, exclusively owned result."""
    var view = owner.view_with_layout(Shape(owner.shape[1], owner.shape[0]), NDArrayStrides(Int(owner.strides[1]), Int(owner.strides[0])), owner.offset)
    view.flags.WRITEABLE = True
    return view^


def _flat_view(mut owner: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
    """Writable alias of an owned contiguous Float64 workspace."""
    var view = owner.view_with_layout(Shape(owner.shape[0]), NDArrayStrides(1), owner.offset)
    view.flags.WRITEABLE = True
    return view^


# ---------------------------------------------------------------------------
# QuantileTransformer


@fieldwise_init
struct QuantileTransformerParams(Copyable, Movable):
    var n_quantiles: Int
    var output_distribution: String
    var ignore_implicit_zeros: Bool
    var subsample: Optional[Int]
    var random_state: Optional[Int]
    var copy_input: Bool


def _check_quantile_params(n_quantiles: Int, output_distribution: String, subsample: Optional[Int], random_state: Optional[Int], owner: String) raises:
    if n_quantiles < 1:
        raise Error("ValueError: The 'n_quantiles' parameter of " + owner + " must be an int in the range [1, inf). Got " + String(n_quantiles) + " instead.")
    if output_distribution != "uniform" and output_distribution != "normal":
        raise Error("ValueError: The 'output_distribution' parameter of " + owner + " must be a str among {'uniform', 'normal'}. Got '" + output_distribution + "' instead.")
    if subsample and subsample.value() < 1:
        raise Error("ValueError: The 'subsample' parameter of " + owner + " must be an int in the range [1, inf) or None. Got " + String(subsample.value()) + " instead.")
    if random_state and (random_state.value() < 0 or random_state.value() > _SEED_MAX):
        raise Error("ValueError: The 'random_state' parameter of " + owner + " must be an int in the range [0, 4294967295], an instance of 'numpy.random.mtrand.RandomState' or None. Got " + String(random_state.value()) + " instead.")


def _references(count: Int) raises -> NDArray[DType.float64]:
    """np.linspace(0, 1, count): k * (1 / (count - 1)) with an exact endpoint."""
    var refs = vector[DType.float64](count)
    if count > 1:
        var step = 1.0 / Float64(count - 1)
        for k in range(count):
            refs.unsafe_set(k, Float64(k) * step + 0.0)
        refs.unsafe_set(count - 1, 1.0)
    return refs^


def _fit_quantiles[d: DType](
    X: NDArray[d], references: NDArray[DType.float64], rows: Optional[NDArray[DType.int64]], used: Int,
) raises -> NDArray[DType.float64]:
    """np.nanpercentile(X[rows], references * 100, axis=0), one feature at a time.

    One Float64 workspace of `used` rows is reused for every feature.
    """
    var p = X.shape[1]
    var nq = references.shape[0]
    var quantiles = matrix[DType.float64](nq, p)
    var column = vector[DType.float64](used)
    var first_empty = False
    for j in range(p):
        var size = 0
        for k in range(used):
            var i = Int(rows.value().unsafe_get(k)) if rows else k
            var value = Float64(at(X, i, j))
            if not isnan(value):
                column.unsafe_set(size, value)
                size += 1
        if j == 0 and size == 0:
            first_empty = True
        sort_prefix(column, size)
        for r in range(nq):
            # nanpercentile divides the percentages back by 100.
            var q = (references.unsafe_get(r) * 100.0) / 100.0
            quantiles.unsafe_set(r * p + j, linear_percentile[d](column, size, q))
    comptime if d == DType.float32:
        # np.apply_along_axis sizes its output from the first feature's result,
        # which is a float32 NaN vector when that feature has no values.
        if first_empty:
            for k in range(nq * p):
                quantiles.unsafe_set(k, _rd[d](quantiles.unsafe_get(k)))
    return quantiles^


def _quantile_into[d: DType](
    X: NDArray[d], mut out: NDArray[d], quantiles: NDArray[DType.float64],
    references: NDArray[DType.float64], normal: Bool, inverse: Bool,
) raises:
    """sklearn `_transform_col` for every feature; `out` may alias X element-wise."""
    var nq = quantiles.shape[0]
    var p = quantiles.shape[1]
    var clip_min = ndtri(BOUNDS_THRESHOLD - F64_EPS)
    var clip_max = ndtri(1 - (BOUNDS_THRESHOLD - F64_EPS))
    for j in range(X.shape[1]):
        var q_low = quantiles.unsafe_get(j)
        var q_high = quantiles.unsafe_get((nq - 1) * p + j)
        if not inverse:
            var forward = InterpTable[False](quantiles, j, references, 0)
            var backward = InterpTable[True](quantiles, j, references, 0)
            for i in range(X.shape[0]):
                var x = at(X, i, j)
                var lower: Bool
                var upper: Bool
                if normal:
                    lower = Float64(x - Scalar[d](BOUNDS_THRESHOLD)) < q_low
                    upper = Float64(x + Scalar[d](BOUNDS_THRESHOLD)) > q_high
                else:
                    lower = Float64(x) == q_low
                    upper = Float64(x) == q_high
                var y = x
                if not isnan(x):
                    var xf = Float64(x)
                    y = Scalar[d](0.5 * (forward(xf) - backward(-xf)))
                if upper:
                    y = 1
                if lower:
                    y = 0
                if normal:
                    var z = ndtri(Float64(y))
                    if z < clip_min:
                        z = clip_min
                    if z > clip_max:
                        z = clip_max
                    y = Scalar[d](z)
                put(out, i, j, y)
        else:
            var table = InterpTable[False](references, 0, quantiles, j)
            for i in range(X.shape[0]):
                var u = Float64(at(X, i, j))
                var lower: Bool
                var upper: Bool
                if normal:
                    u = ndtr(u)
                    lower = u - BOUNDS_THRESHOLD < 0
                    upper = u + BOUNDS_THRESHOLD > 1
                else:
                    lower = u == 0
                    upper = u == 1
                var v = u
                if not isnan(u):
                    v = table(u)
                if upper:
                    v = q_high
                if lower:
                    v = q_low
                put(out, i, j, Scalar[d](v))


struct QuantileTransformer[dtype: DType = DType.float64](InvertiblePreprocessor, NativeFeatureNamesOut, Copyable):
    """Map each feature to a uniform or normal distribution through its quantiles.

    Learned `quantiles_` (n_quantiles_, n_features) and `references_` are Float64,
    as in sklearn. `random_state` is `Optional[Int]`: an integer reseeds a fresh
    NumPy-compatible RandomState on every fit; `None` draws from the process
    default stream. `fit(X, rng)` borrows and advances a caller's RandomState.
    """
    comptime Params = QuantileTransformerParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var n_quantiles: Int
    var output_distribution: String
    var ignore_implicit_zeros: Bool
    var subsample: Optional[Int]
    var random_state: Optional[Int]
    var copy_input: Bool
    var n_quantiles_: Int
    var quantiles_: NDArray[DType.float64]
    var references_: NDArray[DType.float64]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    var _fitted: Bool

    def __init__(out self, *, n_quantiles: Int = 1000, output_distribution: String = "uniform", ignore_implicit_zeros: Bool = False, subsample: Optional[Int] = 10000, random_state: Optional[Int] = None, copy: Bool = True) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 only"
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy_input = copy
        self.n_quantiles_ = 0
        self.quantiles_ = matrix[DType.float64](0, 0)
        self.references_ = vector[DType.float64](0)
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.n_quantiles, self.output_distribution, self.ignore_implicit_zeros, self.subsample, self.random_state, self.copy_input)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.n_quantiles = params.n_quantiles
        self.output_distribution = params.output_distribution
        self.ignore_implicit_zeros = params.ignore_implicit_zeros
        self.subsample = params.subsample
        self.random_state = params.random_state
        self.copy_input = params.copy_input
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        n_quantiles: Optional[Int] = None,
        output_distribution: Optional[String] = None,
        ignore_implicit_zeros: Optional[Bool] = None,
        subsample: Optional[Optional[Int]] = None,
        random_state: Optional[Optional[Int]] = None,
        copy: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        """Outer None leaves a parameter unchanged; an inner None sets it to None."""
        if n_quantiles:
            self.n_quantiles = n_quantiles.value()
        if output_distribution:
            self.output_distribution = output_distribution.value()
        if ignore_implicit_zeros:
            self.ignore_implicit_zeros = ignore_implicit_zeros.value()
        if subsample:
            self.subsample = subsample.value()
        if random_state:
            self.random_state = random_state.value()
        if copy:
            self.copy_input = copy.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(n_quantiles=self.n_quantiles, output_distribution=self.output_distribution, ignore_implicit_zeros=self.ignore_implicit_zeros, subsample=self.subsample, random_state=self.random_state, copy=self.copy_input)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    # -- fitting ------------------------------------------------------------

    def _validate(self, X: Self.Input) raises:
        _check_quantile_params(self.n_quantiles, self.output_distribution, self.subsample, self.random_state, "QuantileTransformer")
        if self.subsample and self.n_quantiles > self.subsample.value():
            raise Error("ValueError: The number of quantiles cannot be greater than the number of samples used. Got " + String(self.n_quantiles) + " quantiles and " + String(self.subsample.value()) + " samples.")
        check_matrix(X, allow_nan=True)

    def _needs_subsample(self, n_samples: Int) -> Bool:
        return Bool(self.subsample) and self.subsample.value() < n_samples

    def _commit(mut self, X: Self.Input, rows: Optional[NDArray[DType.int64]]) raises:
        var n = X.shape[0]
        var nq = max(1, min(self.n_quantiles, n))
        var refs = _references(nq)
        var used = self.subsample.value() if self._needs_subsample(n) else n
        var quantiles = _fit_quantiles(X, refs, rows, used)
        self.n_quantiles_ = nq
        self.references_ = refs^
        self.quantiles_ = quantiles^
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = None
        self._fitted = True

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self._validate(X)
        var n = X.shape[0]
        if self._needs_subsample(n):
            # sklearn resample(replace=False): arange(n), legacy shuffle, first `subsample`.
            var rows: NDArray[DType.int64]
            if self.random_state:
                with check_random_state(self.random_state.value()) as rng:
                    rows = rng[].permutation(n)
            else:
                with check_random_state() as rng:
                    rows = rng[].permutation(n)
            self._commit(X, rows^)
        else:
            self._commit(X, None)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        mut random_state: RandomState,
    ) raises -> ref[self] Self:
        """Fit drawing the subsample from a borrowed RandomState (advanced in place).

        The configured `random_state` is ignored, like passing a RandomState
        instance to sklearn's constructor.
        """
        self._validate(X)
        if self._needs_subsample(X.shape[0]):
            var rows = random_state.permutation(X.shape[0])
            self._commit(X, rows^)
        else:
            self._commit(X, None)
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
        check_matrix(X, allow_nan=True)
        check_names_length(feature_names, X.shape[1])
        _ = self.fit(X, y)
        self.feature_names_in_ = feature_names.copy()
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    # -- transformation -------------------------------------------------------

    def _check(self, X: Self.Input) raises:
        check_fitted(self._fitted, "QuantileTransformer")
        check_matrix(X, allow_nan=True)
        check_n_features("QuantileTransformer", self.n_features_in_, X.shape[1])
        _check_quantile_params(self.n_quantiles, self.output_distribution, None, None, "QuantileTransformer")

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check(X)
        var out = matrix[Self.dtype](X.shape[0], X.shape[1])
        _quantile_into(X, out, self.quantiles_, self.references_, self.output_distribution == "normal", False)
        return out^

    def transform_inplace(self, mut X: Self.Input) raises:
        """In-place copy=False: transform a writable array in place; allocates nothing."""
        self._check(X)
        check_writable(X)
        var target = _matrix_view(X, False)
        target.flags.WRITEABLE = True
        _quantile_into(X, target, self.quantiles_, self.references_, self.output_distribution == "normal", False)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        self._check(X)
        var out = matrix[Self.dtype](X.shape[0], X.shape[1])
        _quantile_into(X, out, self.quantiles_, self.references_, self.output_distribution == "normal", True)
        return out^

    def inverse_transform_inplace(self, mut X: Self.Transformed) raises:
        self._check(X)
        check_writable(X)
        var target = _matrix_view(X, False)
        target.flags.WRITEABLE = True
        _quantile_into(X, target, self.quantiles_, self.references_, self.output_distribution == "normal", True)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        check_fitted(self._fitted, "QuantileTransformer")
        return feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)


def quantile_transform[dtype: DType](
    X: NDArray[dtype], *, axis: Int = 0, n_quantiles: Int = 1000, output_distribution: String = "uniform",
    ignore_implicit_zeros: Bool = False, subsample: Optional[Int] = 100000, random_state: Optional[Int] = None,
    copy: Bool = True,
) raises -> NDArray[dtype]:
    """QuantileTransformer(...).fit_transform(X) along `axis` (1: per sample, via a transposed view)."""
    comptime assert dtype == DType.float32 or dtype == DType.float64, "Float32/Float64 only"
    if axis != 0 and axis != 1:
        raise Error("ValueError: The 'axis' parameter of quantile_transform must be a int among {0, 1}. Got " + String(axis) + " instead.")
    var model = QuantileTransformer[dtype](n_quantiles=n_quantiles, output_distribution=output_distribution, ignore_implicit_zeros=ignore_implicit_zeros, subsample=subsample, random_state=random_state, copy=copy)
    var view = _matrix_view(X, axis == 1)
    _ = model.fit(view)
    model._check(view)
    var result = matrix[dtype](X.shape[0], X.shape[1])
    var normal = output_distribution == "normal"
    if axis == 0:
        _quantile_into(view, result, model.quantiles_, model.references_, normal, False)
    else:
        var out = _writable_transpose(result)
        _quantile_into(view, out, model.quantiles_, model.references_, normal, False)
        _ = out^
    return result^


# ---------------------------------------------------------------------------
# Power transforms (SciPy special/stats semantics in the working dtype)


@always_inline
def _finfo_eps[d: DType]() -> Float64:
    comptime if d == DType.float32:
        return F32_EPS
    else:
        return F64_EPS


def yeo_johnson[d: DType](x: Scalar[d], lmbda: Scalar[d]) -> Scalar[d]:
    """SciPy `_yeojohnson_transform` for one value, with dtype-`d` ufunc rounding."""
    var eps = Scalar[d](_finfo_eps[d]())
    if x >= 0:
        var l1p = Scalar[d](flog1p(Float64(x)))
        if abs(lmbda) < eps:
            return l1p
        return Scalar[d](fexpm1(Float64(lmbda * l1p))) / lmbda
    var l1m = Scalar[d](flog1p(Float64(-x)))
    if abs(lmbda - 2) > eps:
        var two = Scalar[d](2) - lmbda
        return -Scalar[d](fexpm1(Float64(two * l1m))) / two
    return -l1m


def yeo_johnson_inverse[d: DType](x: Scalar[d], lmbda: Scalar[d]) -> Scalar[d]:
    """sklearn `_yeo_johnson_inverse_transform` for one value (NumPy dtype-`d` ops)."""
    if x >= 0:
        if abs(Float64(lmbda)) < F64_EPS:
            return Scalar[d](fexp(Float64(x))) - 1
        var base = x * lmbda + 1
        return Scalar[d](fpow(Float64(base), Float64(Scalar[d](1) / lmbda))) - 1
    var two = Scalar[d](2) - lmbda
    if abs(Float64(lmbda - 2)) > F64_EPS:
        var base = -two * x + 1
        return 1 - Scalar[d](fpow(Float64(base), Float64(Scalar[d](1) / two)))
    return 1 - Scalar[d](fexp(Float64(-x)))


def _log_var(mut logx: NDArray[DType.float64], count: Int, mut scratch: NDArray[DType.float64]) raises -> Float64:
    """SciPy `_log_var`: log of the variance of exp(logx), overwriting `logx`."""
    var log_n = flog(Float64(count))
    var logmean = logsumexp(logx, count, scratch) - log_n
    for i in range(count):
        logx.unsafe_set(i, 2 * logdiffexp(logx.unsafe_get(i), logmean))
    return logsumexp(logx, count, scratch) - log_n


struct _YeoJohnsonLLF(ScalarObjective):
    """Negative SciPy `yeojohnson_llf` on Float64 data (`_neg_llf`)."""
    var data: NDArray[DType.float64]
    var count: Int
    var work: NDArray[DType.float64]
    var scratch: NDArray[DType.float64]
    var all_pos: Bool
    var all_neg: Bool
    var signed_log: Float64

    def __init__(out self, mut data: NDArray[DType.float64], count: Int, mut work: NDArray[DType.float64], mut scratch: NDArray[DType.float64]) raises:
        self.data = _flat_view(data)
        self.count = count
        self.work = _flat_view(work)
        self.scratch = _flat_view(scratch)
        self.all_pos = True
        self.all_neg = True
        for i in range(count):
            var x = data.unsafe_get(i)
            if x >= 0:
                self.all_neg = False
            else:
                self.all_pos = False
        # sum(sign(x) * log1p(|x|)) does not depend on lambda.
        for i in range(count):
            var x = data.unsafe_get(i)
            self.work.unsafe_set(i, sign(x) * flog1p(abs(x)))
        self.signed_log = pairwise_sum(self.work, 0, count)

    def __call__(mut self, lmb: Float64) raises -> Float64:
        var n = self.count
        var logvar: Float64
        if self.all_pos:
            if abs(lmb) < F64_EPS:
                for i in range(n):
                    self.work.unsafe_set(i, flog1p(self.data.unsafe_get(i)))
                logvar = flog(variance(self.work, n, self.scratch))
            else:
                for i in range(n):
                    self.work.unsafe_set(i, lmb * flog1p(self.data.unsafe_get(i)))
                logvar = _log_var(self.work, n, self.scratch) - 2 * flog(abs(lmb))
        elif self.all_neg:
            if abs(lmb - 2) < F64_EPS:
                for i in range(n):
                    self.work.unsafe_set(i, flog1p(-self.data.unsafe_get(i)))
                logvar = flog(variance(self.work, n, self.scratch))
            else:
                for i in range(n):
                    self.work.unsafe_set(i, (2 - lmb) * flog1p(-self.data.unsafe_get(i)))
                logvar = _log_var(self.work, n, self.scratch) - 2 * flog(abs(2 - lmb))
        else:
            for i in range(n):
                self.work.unsafe_set(i, yeo_johnson[DType.float64](self.data.unsafe_get(i), lmb))
            var sigma = variance(self.work, n, self.scratch)
            if sigma >= 2.2250738585072014e-308:
                logvar = flog(sigma)
            else:
                logvar = -inf[DType.float64]()
        var llf = (-Float64(n) / 2) * logvar + (lmb - 1) * self.signed_log
        if isinf(llf):
            llf = -inf[DType.float64]()
        return -llf


struct _BoxCoxLLF(ScalarObjective):
    """Negative SciPy `boxcox_llf` on positive Float64 data (`_eval_mle`)."""
    var logdata: NDArray[DType.float64]
    var count: Int
    var work: NDArray[DType.float64]
    var scratch: NDArray[DType.float64]
    var log_sum: Float64

    def __init__(out self, data: NDArray[DType.float64], count: Int, mut logdata: NDArray[DType.float64], mut work: NDArray[DType.float64], mut scratch: NDArray[DType.float64]) raises:
        self.count = count
        self.logdata = _flat_view(logdata)
        self.work = _flat_view(work)
        self.scratch = _flat_view(scratch)
        for i in range(count):
            self.logdata.unsafe_set(i, flog(data.unsafe_get(i)))
        self.log_sum = pairwise_sum(self.logdata, 0, count)

    def __call__(mut self, lmb: Float64) raises -> Float64:
        var n = self.count
        var logvar: Float64
        if lmb == 0:
            logvar = flog(variance(self.logdata, n, self.scratch))
        else:
            for i in range(n):
                self.work.unsafe_set(i, lmb * self.logdata.unsafe_get(i))
            logvar = _log_var(self.work, n, self.scratch) - 2 * flog(abs(lmb))
        var res = (lmb - 1) * self.log_sum - Float64(n) / 2 * logvar
        return -res


def _yeo_johnson_lambda[d: DType](mut column: NDArray[DType.float64], count: Int, mut work: NDArray[DType.float64], mut scratch: NDArray[DType.float64]) raises -> Float64:
    """SciPy `yeojohnson_normmax(x)` (default brack=None) on the non-NaN values."""
    if count == 0:
        # stats.yeojohnson returns the empty input, so sklearn's unpacking fails.
        raise Error("ValueError: not enough values to unpack (expected 2, got 0)")
    var all_zero = True
    var any_neg = False
    var all_neg = True
    var max_abs = Float64(0)
    for i in range(count):
        var x = column.unsafe_get(i)
        if x != 0:
            all_zero = False
        if x < 0:
            any_neg = True
        else:
            all_neg = False
        if abs(x) > max_abs:
            max_abs = abs(x)
    if all_zero:
        return 1.0
    # SciPy's bounds are NumPy scalars of the data dtype.
    var log1p_max_x = _rd[d](flog1p(_rd[d](20 * max_abs)))
    var eps: Float64
    var tiny: Float64
    var big: Float64
    comptime if d == DType.float32:
        eps = F32_EPS
        tiny = 1.1754943508222875e-38
        big = 3.4028234663852886e38
    else:
        eps = F64_EPS
        tiny = 2.2250738585072014e-308
        big = 1.7976931348623157e308
    var log_eps = _rd[d](flog(eps))
    var log_tiny_float = _rd[d](_rd[d](_rd[d](flog(tiny)) - log_eps) / 2)
    var log_max_float = _rd[d](_rd[d](_rd[d](flog(big)) + log_eps) / 2)
    var lb = _rd[d](log_tiny_float / log1p_max_x)
    var ub = _rd[d](log_max_float / log1p_max_x)
    if all_neg:
        var new_lb = _rd[d](2 - ub)
        var new_ub = _rd[d](2 - lb)
        lb = new_lb
        ub = new_ub
    elif any_neg:
        var low = _rd[d](2 - ub)
        var high = _rd[d](2 - lb)
        lb = low if low > lb else lb
        ub = high if high < ub else ub
    var objective = _YeoJohnsonLLF(column, count, work, scratch)
    return fminbound(objective, lb, ub, xatol=1.48e-08, float32_bounds=d == DType.float32)


def _box_cox_lambda[d: DType](mut column: NDArray[DType.float64], count: Int, mut logdata: NDArray[DType.float64], mut work: NDArray[DType.float64], mut scratch: NDArray[DType.float64]) raises -> Float64:
    """sklearn `_box_cox_optimize` -> SciPy `boxcox_normmax(x, method='mle')`."""
    if count == 0:
        raise Error("ValueError: Column must not be all nan.")
    var first = column.unsafe_get(0)
    var constant = True
    var xmin = first
    var xmax = first
    for i in range(count):
        var x = column.unsafe_get(i)
        if x != first:
            constant = False
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
    if constant:
        raise Error("ValueError: Data must not be constant.")
    if xmin <= 0:
        raise Error("ValueError: Data must be positive.")
    var objective = _BoxCoxLLF(column, count, logdata, work, scratch)
    var res = brent(objective, -2.0, 2.0)
    # Overflow guard: constrain lambda so |boxcox(x_treme)| <= finfo(dtype).max / 10000.
    var ymax: Float64
    comptime if d == DType.float32:
        ymax = _rd[d](3.4028234663852886e38 / 10000)
    else:
        ymax = 1.7976931348623157e308 / 10000
    var x_treme: Float64
    if xmin >= 1:
        x_treme = xmax
    elif xmax <= 1:
        x_treme = xmin
    else:
        x_treme = xmax if boxcox(xmax, res) > abs(boxcox(xmin, res)) else xmin
    if abs(boxcox(x_treme, res)) > ymax:
        # SciPy _boxcox_inv_lmbda(x_treme, ymax * sign(x_treme - 1)) in dtype scalars.
        var y = _rd[d](ymax * sign(_rd[d](x_treme - 1)))
        var log_x = _rd[d](flog(x_treme))
        var inv_y = _rd[d](-1 / y)
        var z = _rd[d](_rd[d](-_rd[d](fpow(x_treme, inv_y)) * log_x) / y)
        var num = lambertw_m1(z)
        res = -num / log_x - _rd[d](1 / y)
    return res


@fieldwise_init
struct PowerTransformerParams(Copyable, Movable):
    var method: String
    var standardize: Bool
    var copy_input: Bool


struct PowerTransformer[dtype: DType = DType.float64](InvertiblePreprocessor, NativeFeatureNamesOut, Copyable):
    """Featurewise Yeo-Johnson / Box-Cox power transform, optionally standardized.

    `lambdas_` has X's dtype (as in sklearn); standardization is a fitted
    family `StandardScaler[dtype]`. Lambda estimation holds one gathered Float64
    column plus two (Yeo-Johnson) or three (Box-Cox) Float64 workspaces of
    n_samples values; fit with standardize also holds the transformed table.
    """
    comptime Params = PowerTransformerParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var method: String
    var standardize: Bool
    var copy_input: Bool
    var lambdas_: NDArray[Self.dtype]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    var _scaler: Optional[StandardScaler[Self.dtype]]
    var _fitted: Bool

    def __init__(out self, method: String = "yeo-johnson", *, standardize: Bool = True, copy: Bool = True) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 only"
        self.method = method
        self.standardize = standardize
        self.copy_input = copy
        self.lambdas_ = vector[Self.dtype](0)
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self._scaler = None
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.method, self.standardize, self.copy_input)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.method = params.method
        self.standardize = params.standardize
        self.copy_input = params.copy_input
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        method: Optional[String] = None,
        standardize: Optional[Bool] = None,
        copy: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        if method:
            self.method = method.value()
        if standardize:
            self.standardize = standardize.value()
        if copy:
            self.copy_input = copy.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.method, standardize=self.standardize, copy=self.copy_input)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    def _check_method(self) raises:
        if self.method != "yeo-johnson" and self.method != "box-cox":
            raise Error("ValueError: The 'method' parameter of PowerTransformer must be a str among {'yeo-johnson', 'box-cox'}. Got '" + self.method + "' instead.")

    def _check_positive(self, X: Self.Input) raises:
        """np.nanmin(X) <= 0 is rejected for Box-Cox (an all-NaN X passes)."""
        if self.method != "box-cox":
            return
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if at(X, i, j) <= 0:
                    raise Error("ValueError: The Box-Cox transformation can only be applied to strictly positive data")

    def _transform_column(self, mut out: NDArray[Self.dtype], j: Int, lmbda: Scalar[Self.dtype]) raises:
        var box_cox = self.method == "box-cox"
        for i in range(out.shape[0]):
            var x = at(out, i, j)
            var y: Scalar[Self.dtype]
            if box_cox:
                y = Scalar[Self.dtype](boxcox(Float64(x), Float64(lmbda)))
            else:
                y = yeo_johnson[Self.dtype](x, lmbda)
            put(out, i, j, y)

    def _fit(mut self, X: Self.Input, force_transform: Bool) raises -> Optional[NDArray[Self.dtype]]:
        self._check_method()
        check_matrix(X, allow_nan=True)
        self._check_positive(X)
        var n = X.shape[0]
        var p = X.shape[1]
        var yeo = self.method == "yeo-johnson"
        var keep = self.standardize or force_transform
        var table = matrix[Self.dtype](n if keep else 0, p if keep else 0)
        var lambdas = vector[Self.dtype](p)
        var column = vector[DType.float64](n)
        var work = vector[DType.float64](n)
        var scratch = vector[DType.float64](n)
        var logdata = vector[DType.float64](0 if yeo else n)
        for j in range(p):
            if keep:
                for i in range(n):
                    table.unsafe_set(i * p + j, at(X, i, j))
            if yeo:
                # np.mean / np.var(axis=0, dtype=float64): row-order accumulation.
                var total = Float64(0)
                for i in range(n):
                    total += Float64(at(X, i, j))
                var mean = total / Float64(n)
                var squares = Float64(0)
                for i in range(n):
                    var delta = Float64(at(X, i, j)) - mean
                    squares += delta * delta
                var var_ = squares / Float64(n)
                var shift = Float64(n) * mean * F64_EPS
                if var_ <= Float64(n) * F64_EPS * var_ + shift * shift:
                    # Constant feature: lambda 1 and left untransformed by fit.
                    lambdas.unsafe_set(j, 1)
                    continue
            var count = 0
            for i in range(n):
                var value = Float64(at(X, i, j))
                if not isnan(value):
                    column.unsafe_set(count, value)
                    count += 1
            var lmbda: Float64
            if yeo:
                lmbda = _yeo_johnson_lambda[Self.dtype](column, count, work, scratch)
            else:
                lmbda = _box_cox_lambda[Self.dtype](column, count, logdata, work, scratch)
            lambdas.unsafe_set(j, Scalar[Self.dtype](lmbda))
            if keep:
                self._transform_column(table, j, lambdas.unsafe_get(j))
        _ = column^
        _ = work^
        _ = scratch^
        _ = logdata^
        var scaler = Optional[StandardScaler[Self.dtype]](None)
        if self.standardize:
            var fitted = StandardScaler[Self.dtype](copy=False)
            _ = fitted.fit(table)
            if force_transform:
                fitted.transform_inplace(table)
            scaler = fitted^
        self.lambdas_ = lambdas^
        self._scaler = scaler^
        self.n_features_in_ = p
        self.feature_names_in_ = None
        self._fitted = True
        if force_transform:
            return table^
        return None

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        _ = self._fit(X, False)
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
        check_matrix(X, allow_nan=True)
        check_names_length(feature_names, X.shape[1])
        _ = self.fit(X, y)
        self.feature_names_in_ = feature_names.copy()
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        var result = self._fit(X, True)
        return result.take()

    def _check(self, X: Self.Input) raises:
        check_fitted(self._fitted, "PowerTransformer")
        self._check_method()
        check_matrix(X, allow_nan=True)
        check_n_features("PowerTransformer", self.n_features_in_, X.shape[1])

    def _forward(self, mut out: NDArray[Self.dtype]) raises:
        if self.lambdas_.ndim != 1 or self.lambdas_.shape[0] != out.shape[1]:
            raise Error("ValueError: Input data has a different number of features than fitting data. Should have " + String(self.lambdas_.shape[0]) + ", data has " + String(out.shape[1]))
        for j in range(out.shape[1]):
            self._transform_column(out, j, self.lambdas_.unsafe_get(j))
        if self.standardize:
            self._scaler.value().transform_inplace(out)

    def _backward(self, mut out: NDArray[Self.dtype]) raises:
        if self.standardize:
            self._scaler.value().inverse_transform_inplace(out)
        var box_cox = self.method == "box-cox"
        for j in range(out.shape[1]):
            var lmbda = self.lambdas_.unsafe_get(j)
            for i in range(out.shape[0]):
                var x = at(out, i, j)
                var y: Scalar[Self.dtype]
                if box_cox:
                    y = Scalar[Self.dtype](inv_boxcox(Float64(x), Float64(lmbda)))
                else:
                    y = yeo_johnson_inverse[Self.dtype](x, lmbda)
                put(out, i, j, y)

    def _copy_of(self, X: Self.Input) raises -> NDArray[Self.dtype]:
        var out = matrix[Self.dtype](X.shape[0], X.shape[1])
        var p = X.shape[1]
        for i in range(X.shape[0]):
            for j in range(p):
                out.unsafe_set(i * p + j, at(X, i, j))
        return out^

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check(X)
        self._check_positive(X)
        var out = self._copy_of(X)
        self._forward(out)
        return out^

    def transform_inplace(self, mut X: Self.Input) raises:
        """In-place copy=False: transform a writable array in place; allocates nothing."""
        self._check(X)
        self._check_positive(X)
        check_writable(X)
        self._forward(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        self._check(X)
        var out = self._copy_of(X)
        self._backward(out)
        return out^

    def inverse_transform_inplace(self, mut X: Self.Transformed) raises:
        self._check(X)
        check_writable(X)
        self._backward(X)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        check_fitted(self._fitted, "PowerTransformer")
        return feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)


def power_transform[dtype: DType](X: NDArray[dtype], method: String = "yeo-johnson", *, standardize: Bool = True, copy: Bool = True) raises -> NDArray[dtype]:
    """PowerTransformer(method, standardize=..., copy=...).fit_transform(X)."""
    var model = PowerTransformer[dtype](method, standardize=standardize, copy=copy)
    return model.fit_transform(X)
