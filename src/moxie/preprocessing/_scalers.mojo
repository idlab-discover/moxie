# Adapted from scikit-learn sklearn/preprocessing/_data.py; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense feature scalers: StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler.

Statistics come from the family's bounded-memory engines (`_moments`): fitting
keeps O(features) state and never materializes masks, casts or residual
tables. Affine application uses `_dense_transforms`. See docs/api/scalers.md.
"""
from std.math import sqrt, log, isnan, abs, inf, nan
from numojo import NDArray, Shape, NDArrayStrides
from moxie.preprocessing._base import InvertiblePreprocessor, OnlinePreprocessor
from moxie.preprocessing._common import (
    at, at1, matrix, vector, check_matrix, check_fitted, check_n_features,
    check_set_output, check_names_length, feature_names_in, check_writable,
)
from moxie.preprocessing._moments import (
    MomentsStateV1, update_moments, update_moments_weighted, update_moments_scalar,
    ExtremaStateV1, update_extrema, robust_quantiles, nanmean_at, nanmean_std_at,
)
from moxie.preprocessing import _dense_transforms as engine
from moxie.preprocessing._special_math import ndtri

comptime _F64_EPS = 2.220446049250313e-16


@always_inline
def _eps[d: DType]() -> Float64:
    comptime if d == DType.float32:
        return 1.1920928955078125e-07
    else:
        return _F64_EPS


def _norm_ppf(p: Float64) -> Float64:
    """Standard normal quantile: SciPy's Cephes `ndtri` (as `stats.norm.ppf`)."""
    return ndtri(p)


# ---------------------------------------------------------------------------
# Views and application helpers


def _matrix_view[d: DType](X: NDArray[d]) raises -> NDArray[d]:
    """Borrowed rank-2 view: a vector becomes one (n, 1) column; no copy."""
    if X.ndim == 1:
        return X.view_with_layout(Shape(X.shape[0], 1), NDArrayStrides(Int(X.strides[0]), 1), X.offset)
    if X.ndim != 2:
        raise Error("ValueError: Expected 1D or 2D array, got " + String(X.ndim) + "D array instead")
    return X.view_with_layout(Shape(X.shape[0], X.shape[1]), NDArrayStrides(Int(X.strides[0]), Int(X.strides[1])), X.offset)


def _transposed[d: DType](X: NDArray[d]) raises -> NDArray[d]:
    """Borrowed transpose of a rank-2 array; no copy."""
    return X.view_with_layout(Shape(X.shape[1], X.shape[0]), NDArrayStrides(Int(X.strides[1]), Int(X.strides[0])), X.offset)


def _writable_view[d: DType](mut owner: NDArray[d], rows: Int, cols: Int) raises -> NDArray[d]:
    """Writable (rows, cols) row-major view of a freshly allocated, owned result."""
    var view = owner.view_with_layout(Shape(rows, cols), NDArrayStrides(cols, 1), owner.offset)
    # The caller owns this allocation exclusively; writes through the view are
    # the only accesses until the owner is returned.
    view.flags.WRITEABLE = True
    return view^


def _check_axis(axis: Int, owner: String) raises:
    if axis != 0 and axis != 1:
        raise Error("ValueError: The 'axis' parameter of " + owner + " must be an int among {0, 1}. Got " + String(axis) + " instead.")


def _scaled_output[d: DType, s: DType](
    X: NDArray[d], shift: NDArray[s], scale: NDArray[s], operation: Int, axis: Int,
    lower: Scalar[d] = 0, upper: Scalar[d] = 1,
) raises -> NDArray[d]:
    """Apply per-feature (axis 0) or per-row (axis 1) parameters to a 1-D/2-D X."""
    if X.ndim == 2:
        return engine.transform(X, shift, scale, operation=operation, axis=axis, lower=lower, upper=upper)
    var result = vector[d](X.shape[0])
    var out = _writable_view(result, X.shape[0], 1)
    engine.transform_into(_matrix_view(X), out, shift, scale, operation=operation, axis=axis, lower=lower, upper=upper)
    return result^


def _all_close_zero[d: DType](values: NDArray[d]) -> Bool:
    """np.allclose(values, 0): |v| <= 1e-8 everywhere, NaN never close."""
    for k in range(values.size):
        if not (abs(Float64(values.unsafe_get(k))) <= 1e-8):
            return False
    return True


def _fill[d: DType](mut values: NDArray[d], value: Scalar[d]):
    for k in range(values.size):
        values.unsafe_set(k, value)


# Parameter math shared by the estimators and the axis functions. Each writes
# only the vectors the affine transform reads into caller-owned storage, so the
# functions never keep a fitted estimator (or owner state) alive beside the result.


def _minmax_affine[d: DType](state: ExtremaStateV1, lower: Scalar[d], upper: Scalar[d], mut shift: NDArray[d], mut scale: NDArray[d]):
    """sklearn MinMaxScaler: scale_ = span / handled(range), min_ = lower - data_min * scale_."""
    var span = upper - lower
    var tiny = Scalar[d](10 * _eps[d]())
    for j in range(scale.size):
        var lo = Scalar[d](state.minimum.unsafe_get(j))
        var width = Scalar[d](state.maximum.unsafe_get(j)) - lo
        var factor = span / (Scalar[d](1) if width < tiny else width)
        scale.unsafe_set(j, factor)
        shift.unsafe_set(j, lower - lo * factor)


def _maxabs_divisor[d: DType](state: ExtremaStateV1, mut scale: NDArray[d]):
    var tiny = Scalar[d](10 * _eps[d]())
    for j in range(scale.size):
        var magnitude = Scalar[d](state.max_abs.unsafe_get(j))
        scale.unsafe_set(j, Scalar[d](1) if magnitude < tiny else magnitude)


def _check_quantile_range(q_min: Float64, q_max: Float64) raises:
    if not (Float64(0) <= q_min and q_min <= q_max and q_max <= Float64(100)):
        raise Error("ValueError: Invalid quantile range: (" + String(q_min) + ", " + String(q_max) + ")")


def _robust_affine[d: DType](
    X: NDArray[d], with_centering: Bool, with_scaling: Bool, q_min: Float64, q_max: Float64,
    unit_variance: Bool, mut shift: NDArray[DType.float64], mut divisor: NDArray[DType.float64],
) raises:
    """Center (median, rounded to X's dtype) and Float64 quantile-range divisor."""
    # Exact quantiles gather one Float64 column at a time (8 * n_samples bytes).
    var quantiles = robust_quantiles(X, q_min=q_min, q_max=q_max, column_budget_bytes=8 * X.shape[0])
    var adjust = Float64(1)
    if with_scaling and unit_variance:
        adjust = _norm_ppf(q_max / 100.0) - _norm_ppf(q_min / 100.0)
    for j in range(shift.size):
        shift.unsafe_set(j, Float64(Scalar[d](quantiles.median.unsafe_get(j))) if with_centering else 0)
        var s = Float64(1)
        if with_scaling:
            s = quantiles.upper.unsafe_get(j) - quantiles.lower.unsafe_get(j)
            if s < 10 * _F64_EPS:
                s = 1
            if unit_variance:
                s = s / adjust
        divisor.unsafe_set(j, s)


# ---------------------------------------------------------------------------
# StandardScaler


@fieldwise_init
struct SampleCount(Copyable, Movable):
    """sklearn `n_samples_seen_`: a scalar when all features agree, else a vector.

    `values` always holds one Float64 count per feature. `is_integer` marks the
    int64 representation sklearn uses when neither centering nor scaling is done.
    """
    var values: NDArray[DType.float64]
    var is_scalar: Bool
    var is_integer: Bool

    @staticmethod
    def empty() raises -> Self:
        return Self(vector[DType.float64](0), True, True)

    def value(self) raises -> Float64:
        """The scalar count; raises when counts differ by feature."""
        if not self.is_scalar or self.values.shape[0] == 0:
            raise Error("ValueError: n_samples_seen_ is a per-feature vector")
        return self.values.unsafe_get(0)


@fieldwise_init
struct StandardScalerParams(Copyable, Movable):
    var copy_input: Bool
    var with_mean: Bool
    var with_std: Bool


struct StandardScaler[dtype: DType = DType.float64](InvertiblePreprocessor, OnlinePreprocessor):
    """Standardize features by removing the mean and scaling to unit variance.

    `fit`, `partial_fit` and weighted variants keep O(features) Float64 moments
    (Youngs–Cramer updates with a two-pass batch correction, as in sklearn).
    """
    comptime Params = StandardScalerParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var copy: Bool
    var with_mean: Bool
    var with_std: Bool
    var mean_: Optional[NDArray[DType.float64]]
    var var_: Optional[NDArray[DType.float64]]
    var scale_: Optional[NDArray[DType.float64]]
    var n_samples_seen_: SampleCount
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    var _state: Optional[MomentsStateV1]
    var _count_only: Bool
    var _weighted: Bool
    var _shift: NDArray[Self.dtype]
    var _divisor: NDArray[Self.dtype]

    def __init__(out self, *, copy: Bool = True, with_mean: Bool = True, with_std: Bool = True) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 only"
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_samples_seen_ = SampleCount.empty()
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self._state = None
        self._count_only = False
        self._weighted = False
        self._shift = vector[Self.dtype](0)
        self._divisor = vector[Self.dtype](0)

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.copy, self.with_mean, self.with_std)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.copy = params.copy_input
        self.with_mean = params.with_mean
        self.with_std = params.with_std
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        copy: Optional[Bool] = None,
        with_mean: Optional[Bool] = None,
        with_std: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        if copy:
            self.copy = copy.value()
        if with_mean:
            self.with_mean = with_mean.value()
        if with_std:
            self.with_std = with_std.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._state)

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    # -- fitting ------------------------------------------------------------

    def _start(self, X: Self.Input, reset: Bool) raises -> MomentsStateV1:
        check_matrix(X, allow_nan=True)
        if reset or not self._state:
            return MomentsStateV1(X.shape[1], track_variance=self.with_std)
        check_n_features("StandardScaler", self.n_features_in_, X.shape[1])
        var count_only = not self.with_mean and not self.with_std
        if count_only != self._count_only or (self.with_std and not self._state.value().track_variance):
            raise Error("ValueError: with_mean/with_std changed between partial_fit calls; call fit to restart")
        return self._state.value().copy()

    def _commit(mut self, var state: MomentsStateV1, n_features: Int, weighted: Bool, reset: Bool) raises:
        """Publish learned attributes from a successfully updated state."""
        self._count_only = not self.with_mean and not self.with_std
        self._weighted = weighted or (self._weighted and not reset)
        var p = n_features
        var counts = state.count.copy()
        var scalar = state.scalar_count()
        self.n_samples_seen_ = SampleCount(counts^, scalar, self._count_only and not self._weighted)
        self._shift = vector[Self.dtype](p)
        self._divisor = vector[Self.dtype](p)
        for j in range(p):
            self._divisor.unsafe_set(j, 1)
        if self._count_only:
            self.mean_ = None
            self.var_ = None
            self.scale_ = None
        else:
            self.mean_ = state.mean.copy()
            if self.with_mean:
                for j in range(p):
                    self._shift.unsafe_set(j, Scalar[Self.dtype](state.mean.unsafe_get(j)))
            if self.with_std:
                self.var_ = state.variance.copy()
                var scale = vector[DType.float64](p)
                for j in range(p):
                    var s = Float64(1) if state.constant_at(j) else sqrt(state.variance.unsafe_get(j))
                    scale.unsafe_set(j, s)
                    self._divisor.unsafe_set(j, Scalar[Self.dtype](s))
                self.scale_ = scale^
            else:
                self.var_ = None
                self.scale_ = None
        self._state = state^
        if reset:
            self.feature_names_in_ = None
        self.n_features_in_ = p

    def _count_update(self, X: Self.Input, mut state: MomentsStateV1) raises:
        # Neither statistic requested: sklearn only counts non-missing values.
        for j in range(X.shape[1]):
            var present = 0
            for i in range(X.shape[0]):
                if not isnan(at(X, i, j)):
                    present += 1
            state.count.unsafe_set(j, state.count.unsafe_get(j) + Float64(present))

    def _partial(mut self, X: Self.Input, reset: Bool) raises:
        var state = self._start(X, reset)
        if not self.with_mean and not self.with_std:
            self._count_update(X, state)
        else:
            update_moments(X, state)
        self._commit(state^, X.shape[1], False, reset)

    def _partial_weighted(mut self, X: Self.Input, sample_weight: NDArray[Self.dtype], reset: Bool) raises:
        var state = self._start(X, reset)
        if sample_weight.ndim != 1 or sample_weight.shape[0] != X.shape[0]:
            raise Error("ValueError: sample_weight.shape == (" + String(sample_weight.shape[0]) + ",), expected (" + String(X.shape[0]) + ",)!")
        if not self.with_mean and not self.with_std:
            # sklearn ignores weights in this branch but switches to a float count.
            self._count_update(X, state)
        else:
            update_moments_weighted(X, sample_weight, state)
        self._commit(state^, X.shape[1], True, reset)

    def _partial_scalar(mut self, X: Self.Input, sample_weight: Float64, reset: Bool) raises:
        var state = self._start(X, reset)
        if not self.with_mean and not self.with_std:
            self._count_update(X, state)
        else:
            update_moments_scalar(X, sample_weight, state)
        self._commit(state^, X.shape[1], True, reset)

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self._partial(X, True)
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
        self._partial_weighted(X, sample_weight, True)
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
        self._partial_weighted(X, sample_weight, True)
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
        """Scalar weights are broadcast logically, never expanded."""
        self._partial_scalar(X, sample_weight, True)
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

    def partial_fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self._partial(X, not self._state)
        return self

    def partial_fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target],
        ref[sample_weight_origin] sample_weight: NDArray[Self.dtype],
    ) raises -> ref[self] Self:
        self._partial_weighted(X, sample_weight, not self._state)
        return self

    def partial_fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        *,
        ref[sample_weight_origin] sample_weight: NDArray[Self.dtype],
    ) raises -> ref[self] Self:
        self._partial_weighted(X, sample_weight, not self._state)
        return self

    def partial_fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        *,
        ref[sample_weight_origin] sample_weight: Float64,
    ) raises -> ref[self] Self:
        self._partial_scalar(X, sample_weight, not self._state)
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def fit_transform(mut self, X: Self.Input, *, sample_weight: NDArray[Self.dtype]) raises -> Self.Transformed:
        _ = self.fit(X, sample_weight=sample_weight)
        return self.transform(X)

    # -- transformation -------------------------------------------------------

    def _check(self, X: Self.Input) raises:
        check_fitted(self.__sklearn_is_fitted__(), "StandardScaler")
        check_matrix(X, allow_nan=True, min_samples=0)
        check_n_features("StandardScaler", self.n_features_in_, X.shape[1])

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check(X)
        return engine.transform(X, self._shift, self._divisor, operation=0)

    def transform(self, X: Self.Input, copy: Optional[Bool]) raises -> Self.Transformed:
        """`copy` is accepted for API parity; borrowed input is never modified."""
        return self.transform(X)

    def transform_inplace(self, mut X: Self.Input) raises:
        self._check(X)
        check_writable(X)
        engine.transform_inplace(X, self._shift, self._divisor, operation=0)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        self._check(X)
        return engine.transform(X, self._shift, self._divisor, operation=1)

    def inverse_transform(self, X: Self.Transformed, copy: Optional[Bool]) raises -> Self.Input:
        return self.inverse_transform(X)

    def inverse_transform_inplace(self, mut X: Self.Transformed) raises:
        self._check(X)
        check_writable(X)
        engine.transform_inplace(X, self._shift, self._divisor, operation=1)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        check_fitted(self.__sklearn_is_fitted__(), "StandardScaler")
        return feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)


# ---------------------------------------------------------------------------
# MinMaxScaler


@fieldwise_init
struct MinMaxScalerParams(Copyable, Movable):
    var feature_range: Tuple[Float64, Float64]
    var copy_input: Bool
    var clip: Bool


struct MinMaxScaler[dtype: DType = DType.float64](InvertiblePreprocessor, OnlinePreprocessor):
    """Scale each feature to `feature_range`; learned values keep X's dtype."""
    comptime Params = MinMaxScalerParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var feature_range: Tuple[Float64, Float64]
    var copy: Bool
    var clip: Bool
    var min_: NDArray[Self.dtype]
    var scale_: NDArray[Self.dtype]
    var data_min_: NDArray[Self.dtype]
    var data_max_: NDArray[Self.dtype]
    var data_range_: NDArray[Self.dtype]
    var n_samples_seen_: Int
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    var _state: Optional[ExtremaStateV1]

    def __init__(out self, feature_range: Tuple[Float64, Float64] = (0, 1), *, copy: Bool = True, clip: Bool = False) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 only"
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        self.min_ = vector[Self.dtype](0)
        self.scale_ = vector[Self.dtype](0)
        self.data_min_ = vector[Self.dtype](0)
        self.data_max_ = vector[Self.dtype](0)
        self.data_range_ = vector[Self.dtype](0)
        self.n_samples_seen_ = 0
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self._state = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.feature_range, self.copy, self.clip)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.feature_range = params.feature_range
        self.copy = params.copy_input
        self.clip = params.clip
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        feature_range: Optional[Tuple[Float64, Float64]] = None,
        copy: Optional[Bool] = None,
        clip: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        if feature_range:
            self.feature_range = feature_range.value()
        if copy:
            self.copy = copy.value()
        if clip:
            self.clip = clip.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.feature_range, copy=self.copy, clip=self.clip)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._state)

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    def _range(self) raises -> Tuple[Scalar[Self.dtype], Scalar[Self.dtype]]:
        var lo = self.feature_range[0]
        var hi = self.feature_range[1]
        if lo >= hi:
            raise Error("ValueError: Minimum of desired feature range must be smaller than maximum. Got (" + String(lo) + ", " + String(hi) + ").")
        return (Scalar[Self.dtype](lo), Scalar[Self.dtype](hi))

    def _partial(mut self, X: Self.Input, reset: Bool) raises:
        var bounds = self._range()
        check_matrix(X, allow_nan=True)
        var first = reset or not self._state
        if not first:
            check_n_features("MinMaxScaler", self.n_features_in_, X.shape[1])
        var state = ExtremaStateV1(X.shape[1]) if first else self._state.value().copy()
        update_extrema(X, state)
        var p = X.shape[1]
        self.data_min_ = vector[Self.dtype](p)
        self.data_max_ = vector[Self.dtype](p)
        self.data_range_ = vector[Self.dtype](p)
        self.scale_ = vector[Self.dtype](p)
        self.min_ = vector[Self.dtype](p)
        for j in range(p):
            var lo = Scalar[Self.dtype](state.minimum.unsafe_get(j))
            var hi = Scalar[Self.dtype](state.maximum.unsafe_get(j))
            self.data_min_.unsafe_set(j, lo)
            self.data_max_.unsafe_set(j, hi)
            self.data_range_.unsafe_set(j, hi - lo)
        _minmax_affine(state, bounds[0], bounds[1], self.min_, self.scale_)
        self.n_samples_seen_ = X.shape[0] if first else self.n_samples_seen_ + X.shape[0]
        self._state = state^
        if first:
            self.feature_names_in_ = None
        self.n_features_in_ = p

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self._partial(X, True)
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

    def partial_fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self._partial(X, False)
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def _check(self, X: Self.Input, features: Bool) raises:
        check_fitted(self.__sklearn_is_fitted__(), "MinMaxScaler")
        check_matrix(X, allow_nan=True, min_samples=0)
        if features:
            check_n_features("MinMaxScaler", self.n_features_in_, X.shape[1])

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check(X, True)
        var bounds = (Scalar[Self.dtype](self.feature_range[0]), Scalar[Self.dtype](self.feature_range[1]))
        return engine.transform(X, self.min_, self.scale_, operation=2 if self.clip else 1, lower=bounds[0], upper=bounds[1])

    def transform_inplace(self, mut X: Self.Input) raises:
        self._check(X, True)
        check_writable(X)
        var bounds = (Scalar[Self.dtype](self.feature_range[0]), Scalar[Self.dtype](self.feature_range[1]))
        engine.transform_inplace(X, self.min_, self.scale_, operation=2 if self.clip else 1, lower=bounds[0], upper=bounds[1])

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        self._check(X, True)
        return engine.transform(X, self.min_, self.scale_, operation=3)

    def inverse_transform_inplace(self, mut X: Self.Transformed) raises:
        self._check(X, True)
        check_writable(X)
        engine.transform_inplace(X, self.min_, self.scale_, operation=3)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        check_fitted(self.__sklearn_is_fitted__(), "MinMaxScaler")
        return feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)


# ---------------------------------------------------------------------------
# MaxAbsScaler


@fieldwise_init
struct MaxAbsScalerParams(Copyable, Movable):
    var copy_input: Bool
    var clip: Bool


struct MaxAbsScaler[dtype: DType = DType.float64](InvertiblePreprocessor, OnlinePreprocessor):
    """Scale each feature by its maximum absolute value; learned values keep X's dtype."""
    comptime Params = MaxAbsScalerParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var copy: Bool
    var clip: Bool
    var scale_: NDArray[Self.dtype]
    var max_abs_: NDArray[Self.dtype]
    var n_samples_seen_: Int
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    var _state: Optional[ExtremaStateV1]
    var _zero: NDArray[Self.dtype]

    def __init__(out self, *, copy: Bool = True, clip: Bool = False) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 only"
        self.copy = copy
        self.clip = clip
        self.scale_ = vector[Self.dtype](0)
        self.max_abs_ = vector[Self.dtype](0)
        self.n_samples_seen_ = 0
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self._state = None
        self._zero = vector[Self.dtype](0)

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.copy, self.clip)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.copy = params.copy_input
        self.clip = params.clip
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        copy: Optional[Bool] = None,
        clip: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        if copy:
            self.copy = copy.value()
        if clip:
            self.clip = clip.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(copy=self.copy, clip=self.clip)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._state)

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    def _partial(mut self, X: Self.Input, reset: Bool) raises:
        check_matrix(X, allow_nan=True)
        var first = reset or not self._state
        if not first:
            check_n_features("MaxAbsScaler", self.n_features_in_, X.shape[1])
        var state = ExtremaStateV1(X.shape[1]) if first else self._state.value().copy()
        update_extrema(X, state)
        var p = X.shape[1]
        self.max_abs_ = vector[Self.dtype](p)
        self.scale_ = vector[Self.dtype](p)
        self._zero = vector[Self.dtype](p)
        for j in range(p):
            self.max_abs_.unsafe_set(j, Scalar[Self.dtype](state.max_abs.unsafe_get(j)))
        _maxabs_divisor(state, self.scale_)
        self.n_samples_seen_ = X.shape[0] if first else self.n_samples_seen_ + X.shape[0]
        self._state = state^
        if first:
            self.feature_names_in_ = None
        self.n_features_in_ = p

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self._partial(X, True)
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

    def partial_fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self._partial(X, False)
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def _check(self, X: Self.Input) raises:
        check_fitted(self.__sklearn_is_fitted__(), "MaxAbsScaler")
        check_matrix(X, allow_nan=True, min_samples=0)
        check_n_features("MaxAbsScaler", self.n_features_in_, X.shape[1])

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check(X)
        var result = engine.transform(X, self._zero, self.scale_, operation=0)
        if self.clip:
            engine.transform_inplace(result, self._zero, self.scale_, operation=4, lower=-1, upper=1)
        return result^

    def transform_inplace(self, mut X: Self.Input) raises:
        self._check(X)
        check_writable(X)
        engine.transform_inplace(X, self._zero, self.scale_, operation=0)
        if self.clip:
            engine.transform_inplace(X, self._zero, self.scale_, operation=4, lower=-1, upper=1)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        self._check(X)
        return engine.transform(X, self._zero, self.scale_, operation=1)

    def inverse_transform_inplace(self, mut X: Self.Transformed) raises:
        self._check(X)
        check_writable(X)
        engine.transform_inplace(X, self._zero, self.scale_, operation=1)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        check_fitted(self.__sklearn_is_fitted__(), "MaxAbsScaler")
        return feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)


# ---------------------------------------------------------------------------
# RobustScaler


@fieldwise_init
struct RobustScalerParams(Copyable, Movable):
    var with_centering: Bool
    var with_scaling: Bool
    var quantile_range: Tuple[Float64, Float64]
    var copy_input: Bool
    var unit_variance: Bool


struct RobustScaler[dtype: DType = DType.float64](InvertiblePreprocessor):
    """Center on the median and scale by a quantile range (exact quantiles).

    `center_` keeps X's dtype; `scale_` is Float64, matching sklearn.
    Fitting gathers one Float64 column workspace at a time (8 * n_samples bytes).
    """
    comptime Params = RobustScalerParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var with_centering: Bool
    var with_scaling: Bool
    var quantile_range: Tuple[Float64, Float64]
    var copy: Bool
    var unit_variance: Bool
    var center_: Optional[NDArray[Self.dtype]]
    var scale_: Optional[NDArray[DType.float64]]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    var _fitted: Bool
    var _shift: NDArray[DType.float64]
    var _divisor: NDArray[DType.float64]

    def __init__(out self, *, with_centering: Bool = True, with_scaling: Bool = True, quantile_range: Tuple[Float64, Float64] = (25.0, 75.0), copy: Bool = True, unit_variance: Bool = False) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 only"
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        self.unit_variance = unit_variance
        self.center_ = None
        self.scale_ = None
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self._fitted = False
        self._shift = vector[DType.float64](0)
        self._divisor = vector[DType.float64](0)

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.with_centering, self.with_scaling, self.quantile_range, self.copy, self.unit_variance)

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.with_centering = params.with_centering
        self.with_scaling = params.with_scaling
        self.quantile_range = params.quantile_range
        self.copy = params.copy_input
        self.unit_variance = params.unit_variance
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        with_centering: Optional[Bool] = None,
        with_scaling: Optional[Bool] = None,
        quantile_range: Optional[Tuple[Float64, Float64]] = None,
        copy: Optional[Bool] = None,
        unit_variance: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        if with_centering:
            self.with_centering = with_centering.value()
        if with_scaling:
            self.with_scaling = with_scaling.value()
        if quantile_range:
            self.quantile_range = quantile_range.value()
        if copy:
            self.copy = copy.value()
        if unit_variance:
            self.unit_variance = unit_variance.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(with_centering=self.with_centering, with_scaling=self.with_scaling, quantile_range=self.quantile_range, copy=self.copy, unit_variance=self.unit_variance)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        check_matrix(X, allow_nan=True)
        _check_quantile_range(self.quantile_range[0], self.quantile_range[1])
        var p = X.shape[1]
        var shift = vector[DType.float64](p)
        var divisor = vector[DType.float64](p)
        _robust_affine(X, self.with_centering, self.with_scaling, self.quantile_range[0], self.quantile_range[1], self.unit_variance, shift, divisor)
        var center = Optional[NDArray[Self.dtype]](None)
        var scale = Optional[NDArray[DType.float64]](None)
        if self.with_centering:
            # The shift holds the dtype-rounded median, so this cast is exact.
            var values = vector[Self.dtype](p)
            for j in range(p):
                values.unsafe_set(j, Scalar[Self.dtype](shift.unsafe_get(j)))
            center = values^
        if self.with_scaling:
            scale = divisor.copy()
        self.center_ = center^
        self.scale_ = scale^
        self._shift = shift^
        self._divisor = divisor^
        self.n_features_in_ = p
        self.feature_names_in_ = None
        self._fitted = True
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

    def _check(self, X: Self.Input) raises:
        check_fitted(self._fitted, "RobustScaler")
        check_matrix(X, allow_nan=True, min_samples=0)
        check_n_features("RobustScaler", self.n_features_in_, X.shape[1])

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._check(X)
        return engine.transform(X, self._shift, self._divisor, operation=0)

    def transform_inplace(self, mut X: Self.Input) raises:
        self._check(X)
        check_writable(X)
        engine.transform_inplace(X, self._shift, self._divisor, operation=0)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        self._check(X)
        return engine.transform(X, self._shift, self._divisor, operation=1)

    def inverse_transform_inplace(self, mut X: Self.Transformed) raises:
        self._check(X)
        check_writable(X)
        engine.transform_inplace(X, self._shift, self._divisor, operation=1)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        check_fitted(self._fitted, "RobustScaler")
        return feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)


# ---------------------------------------------------------------------------
# Functions. Each accepts a vector or matrix; axis=1 fits per row through a
# borrowed transposed view (no copy) and applies per-row parameters.


def _check_function_input[dtype: DType](X: NDArray[dtype], axis: Int, owner: String) raises -> NDArray[dtype]:
    comptime assert dtype == DType.float32 or dtype == DType.float64, "Float32/Float64 only"
    _check_axis(axis, owner)
    var view = _matrix_view(X)
    check_matrix(view, allow_nan=True)
    return view^


def scale[dtype: DType](X: NDArray[dtype], *, axis: Int = 0, with_mean: Bool = True, with_std: Bool = True, copy: Bool = True) raises -> NDArray[dtype]:
    """Standardize along an axis with sklearn's input-precision nanmean/nanstd.

    Unlike StandardScaler, statistics stay in X's dtype and two correction
    passes remove residual means, exactly as sklearn's `scale`. The result is a
    new array (`copy` has no effect on the borrowed input).
    """
    comptime assert dtype == DType.float32 or dtype == DType.float64, "Float32/Float64 only"
    _check_axis(axis, "scale")
    if X.ndim == 1 and axis == 1:
        raise Error("ValueError: axis 1 is out of bounds for array of dimension 1")
    var view = _matrix_view(X)
    check_matrix(view, allow_nan=True, min_samples=0, min_features=0)
    var rows = view.shape[0]
    var cols = view.shape[1]
    var k = cols if axis == 0 else rows
    # Three per-axis vectors (k = features for axis 0, rows for axis 1): the
    # statistics, then `shift` doubles as the residual correction and `unit` as
    # the ones/zeros operand, so no table-sized or extra k-sized storage is live.
    var shift = vector[dtype](k)
    var std = vector[dtype](k)
    var unit = vector[dtype](k)
    var tiny = Scalar[dtype](10 * _eps[dtype]())
    for t in range(k):
        var stats = nanmean_std_at(view, axis, t)
        shift.unsafe_set(t, stats.mean if with_mean else 0)
        std.unsafe_set(t, Scalar[dtype](1) if stats.stddev < tiny else stats.stddev)
    _fill(unit, 1)
    var result = matrix[dtype](rows, cols) if X.ndim == 2 else vector[dtype](rows)
    var out = _writable_view(result, rows, cols)
    # x - mean (or the exact copy x - 0) directly into the result.
    engine.transform_into(view, out, shift, unit, operation=0, axis=axis)
    if with_mean:
        for t in range(k):
            shift.unsafe_set(t, nanmean_at(out, axis, t))
        if not _all_close_zero(shift):
            engine.transform_inplace(out, shift, unit, operation=0, axis=axis)
    if with_std:
        _fill(unit, 0)
        engine.transform_inplace(out, unit, std, operation=0, axis=axis)
        if with_mean:
            for t in range(k):
                shift.unsafe_set(t, nanmean_at(out, axis, t))
            if not _all_close_zero(shift):
                _fill(unit, 1)
                engine.transform_inplace(out, shift, unit, operation=0, axis=axis)
    _ = out^
    return result^


def minmax_scale[dtype: DType](X: NDArray[dtype], feature_range: Tuple[Float64, Float64] = (0, 1), *, axis: Int = 0, copy: Bool = True) raises -> NDArray[dtype]:
    """MinMaxScaler(feature_range).fit_transform along `axis`; vectors stay vectors."""
    var view = _check_function_input(X, axis, "minmax_scale")
    var lower = feature_range[0]
    var upper = feature_range[1]
    if lower >= upper:
        raise Error("ValueError: Minimum of desired feature range must be smaller than maximum. Got (" + String(lower) + ", " + String(upper) + ").")
    # A fresh borrowed view either way: copying an NDArray view may copy its buffer.
    var fit_view = _matrix_view(X) if axis == 0 else _transposed(view)
    check_matrix(fit_view, allow_nan=True)
    var shift = vector[dtype](fit_view.shape[1])
    var scale = vector[dtype](fit_view.shape[1])
    var state = ExtremaStateV1(fit_view.shape[1])
    update_extrema(fit_view, state)
    _minmax_affine(state, Scalar[dtype](lower), Scalar[dtype](upper), shift, scale)
    _ = state^  # owner state is released before the result is allocated
    return _scaled_output(X, shift, scale, 1, axis)


def maxabs_scale[dtype: DType](X: NDArray[dtype], *, axis: Int = 0, copy: Bool = True) raises -> NDArray[dtype]:
    """MaxAbsScaler().fit_transform along `axis`; vectors stay vectors."""
    var view = _check_function_input(X, axis, "maxabs_scale")
    # A fresh borrowed view either way: copying an NDArray view may copy its buffer.
    var fit_view = _matrix_view(X) if axis == 0 else _transposed(view)
    check_matrix(fit_view, allow_nan=True)
    var zero = vector[dtype](fit_view.shape[1])
    var scale = vector[dtype](fit_view.shape[1])
    var state = ExtremaStateV1(fit_view.shape[1])
    update_extrema(fit_view, state)
    _maxabs_divisor(state, scale)
    _ = state^  # owner state is released before the result is allocated
    return _scaled_output(X, zero, scale, 0, axis)


def robust_scale[dtype: DType](X: NDArray[dtype], *, axis: Int = 0, with_centering: Bool = True, with_scaling: Bool = True, quantile_range: Tuple[Float64, Float64] = (25.0, 75.0), copy: Bool = True, unit_variance: Bool = False) raises -> NDArray[dtype]:
    """RobustScaler(...).fit_transform along `axis`; vectors stay vectors."""
    var view = _check_function_input(X, axis, "robust_scale")
    _check_quantile_range(quantile_range[0], quantile_range[1])
    # A fresh borrowed view either way: copying an NDArray view may copy its buffer.
    var fit_view = _matrix_view(X) if axis == 0 else _transposed(view)
    check_matrix(fit_view, allow_nan=True)
    var shift = vector[DType.float64](fit_view.shape[1])
    var divisor = vector[DType.float64](fit_view.shape[1])
    # Quantile state and its column workspace end inside _robust_affine.
    _robust_affine(fit_view, with_centering, with_scaling, quantile_range[0], quantile_range[1], unit_variance, shift, divisor)
    return _scaled_output(X, shift, divisor, 0, axis)
