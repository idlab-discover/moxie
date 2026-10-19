# Adapted from scikit-learn; SPDX-License-Identifier: BSD-3-Clause
"""Dense population-variance selection, allowing missing values during fit."""
import numojo as nm
from std.math import isnan, isfinite, nan, inf
from moxie.base import FitTransformer
from ._base import UnsupervisedSelector
from ._selection import SelectionState, SupportResult

@fieldwise_init
struct VarianceThresholdParams(Copyable, Movable):
    var threshold: Float64

def _validate[dtype: DType](X: nm.NDArray[dtype], allow_nan: Bool = True) raises:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise Error("ValueError: expected nonempty rank-2 X")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            var value = X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))
            if not isfinite(value) and not (allow_nan and isnan(value)):
                raise Error("ValueError: X contains infinity or disallowed NaN")

struct VarianceThreshold[dtype: DType = DType.float64](UnsupervisedSelector, FitTransformer):
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Transformed = Self.Input
    comptime Support = SupportResult
    comptime Params = VarianceThresholdParams
    var threshold: Float64
    var _selection: SelectionState
    var _variances: Optional[nm.NDArray[DType.float64]]
    var _names: List[String]

    def __init__(out self, threshold: Float64 = 0.0):
        self.threshold = threshold
        self._selection = SelectionState()
        self._variances = None
        self._names = List[String]()

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self.fit_named(X, y, List[String]())
        return self

    def fit_named(mut self, X: Self.Input, y: Optional[Self.Target], names: List[String]) raises:
        if not isfinite(self.threshold) or self.threshold < 0:
            raise Error("ValueError: threshold must be finite and nonnegative")
        _validate(X)
        var variances = nm.zeros[DType.float64](nm.Shape(X.shape[1]))
        var mask = List[Bool]()
        var any_finite_selected = False
        for j in range(X.shape[1]):
            var count = 0
            for i in range(X.shape[0]):
                count += Int(not isnan(X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))))
            var variance = nan[DType.float64]()
            if count > 0:
                var column = nm.zeros[DType.float64](nm.Shape(count, 1))
                var low = inf[DType.float64]()
                var high = -inf[DType.float64]()
                var k = 0
                for i in range(X.shape[0]):
                    var value = Float64(X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1])))
                    if not isnan(value):
                        column.unsafe_set(k, value)
                        low = min(low, value)
                        high = max(high, value)
                        k += 1
                var mean = nm.sum(column) / Float64(count)
                var centered = column - mean
                var squared = centered * centered
                variance = nm.sum(squared) / Float64(count)
                # np.ptp propagates NaN, so the guard applies only to complete columns.
                if self.threshold == 0 and count == X.shape[0]:
                    var spread = high - low
                    if isnan(variance) or spread < variance:
                        variance = spread
            variances.unsafe_set(j, variance)
            mask.append(variance > self.threshold)
            any_finite_selected = any_finite_selected or (isfinite(variance) and variance > self.threshold)
        if not any_finite_selected:
            raise Error("ValueError: No feature in X meets the variance threshold")
        # Failed fits preserve the previously committed native state.
        self._selection.commit(mask, names)
        self._variances = variances^
        self._names = names.copy()

    def _selected(self) raises -> SelectionState:
        self._selection.check()
        var mask = List[Bool]()
        ref variances = self._variances.value()
        for j in range(variances.shape[0]):
            mask.append(variances.unsafe_get(j) > self.threshold)
        var result = self._selection.copy()
        result.commit(mask, self._names)
        return result^

    def get_support(self, indices: Bool = False) raises -> Self.Support:
        return self._selected().get_support(indices)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        _validate(X)
        return self._selected().transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        _validate(X, False)
        return self._selected().inverse_transform(X)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        return self._selected().get_feature_names_out(input_features)

    def variances_(self) raises -> nm.NDArray[DType.float64]:
        self._selection.check()
        return self._variances.value().copy()

    def n_features_in_(self) raises -> Int:
        return self._selection.n_features()

    def feature_names_in_(self) raises -> List[String]:
        self._selection.check()
        return self._names.copy()

    def get_params(self, deep: Bool = True) -> Self.Params:
        return Self.Params(self.threshold)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.threshold = params.threshold
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, threshold: Float64) raises -> ref[self] Self:
        self.threshold = threshold
        return self

    def __sklearn_clone__(self) -> Self:
        return Self(self.threshold)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._selection.is_fitted()
