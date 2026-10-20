"""Univariate dense imputation. See docs/api.md for native adaptations."""
import numojo as nm
from std.math import isfinite
from std.utils import Variant
from moxie._feature_metadata import FeatureNamesOut
from moxie.impute._base import Imputer
from moxie.impute._plumbing import MissingValue, missing_mask, FeatureLayout, select_features, append_indicator, validate_matrix
from moxie.impute._indicator import MissingIndicator, _indicator_mask


trait ColumnStrategy(Copyable, Movable, Deinitable):
    def __init__(out self):
        ...

    def __call__(self, values: nm.NDArray[DType.float64]) raises -> Float64:
        ...


struct NoCallable(ColumnStrategy):
    def __init__(out self):
        pass

    def __call__(self, values: nm.NDArray[DType.float64]) raises -> Float64:
        raise Error("ValueError: callable strategy requires a native ColumnStrategy")


struct SimpleParams[dtype: DType, Callback: ColumnStrategy = NoCallable](Copyable, Movable):
    var missing_values: MissingValue[Self.dtype]
    var strategy: String
    var fill_value: Optional[SIMD[Self.dtype, 1]]
    var copy_input: Bool
    var add_indicator: Bool
    var keep_empty_features: Bool
    var callback: Self.Callback

    def __init__(out self, *, missing_values: MissingValue[Self.dtype] = MissingValue[Self.dtype](), strategy: String = "mean", fill_value: Optional[SIMD[Self.dtype, 1]] = None, copy: Bool = True, add_indicator: Bool = False, keep_empty_features: Bool = False, callback: Self.Callback = Self.Callback()):
        self.missing_values = missing_values.copy()
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy_input = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.callback = callback.copy()


struct SimpleImputer[dtype: DType = DType.float64, Callback: ColumnStrategy = NoCallable, Result: Copyable = nm.NDArray[dtype]](Imputer, FeatureNamesOut, Copyable, Movable):
    comptime Params = SimpleParams[Self.dtype, Self.Callback]
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime _floating = Self.dtype == DType.float32 or Self.dtype == DType.float64
    comptime Transformed = Self.Result
    var _params: Self.Params
    var _layout: FeatureLayout
    var statistics_: nm.NDArray[DType.float64]
    var indicator_: Optional[MissingIndicator[Self.dtype]]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]

    def __init__(out self, *, missing_values: MissingValue[Self.dtype] = MissingValue[Self.dtype](), strategy: String = "mean", fill_value: Optional[SIMD[Self.dtype, 1]] = None, copy: Bool = True, add_indicator: Bool = False, keep_empty_features: Bool = False, callback: Self.Callback = Self.Callback()) raises:
        comptime assert Self._floating or Self.dtype == DType.int8 or Self.dtype == DType.int16 or Self.dtype == DType.int32 or Self.dtype == DType.int64 or Self.dtype == DType.uint8 or Self.dtype == DType.uint16 or Self.dtype == DType.uint32 or Self.dtype == DType.uint64
        comptime assert Self.Result == (nm.NDArray[Self.dtype] if Self._floating else Variant[nm.NDArray[Self.dtype], nm.NDArray[DType.float64]]), "use IntegerSimpleImputer for integer inputs"
        self._params = Self.Params(missing_values=missing_values, strategy=strategy, fill_value=fill_value, copy=copy, add_indicator=add_indicator, keep_empty_features=keep_empty_features, callback=callback)
        self._layout = FeatureLayout()
        self.statistics_ = nm.zeros[DType.float64](nm.Shape(0))
        self.indicator_ = None
        self.n_features_in_ = 0
        self.feature_names_in_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self = Self(missing_values=params.missing_values, strategy=params.strategy, fill_value=params.fill_value, copy=params.copy_input, add_indicator=params.add_indicator, keep_empty_features=params.keep_empty_features, callback=params.callback)
        return self

    def _updated_params(self, *, missing_values: Optional[MissingValue[Self.dtype]] = None, strategy: Optional[String] = None, copy: Optional[Bool] = None, add_indicator: Optional[Bool] = None, keep_empty_features: Optional[Bool] = None, callback: Optional[Self.Callback] = None) raises -> Self.Params:
        var params = self.get_params()
        if missing_values:
            params.missing_values = missing_values.value().copy()
        if strategy:
            params.strategy = strategy.value()
        if copy:
            params.copy_input = copy.value()
        if add_indicator:
            params.add_indicator = add_indicator.value()
        if keep_empty_features:
            params.keep_empty_features = keep_empty_features.value()
        if callback:
            params.callback = callback.value().copy()
        return params^

    def set_params[origin: MutOrigin](ref[origin] self, *, missing_values: Optional[MissingValue[Self.dtype]] = None, strategy: Optional[String] = None, copy: Optional[Bool] = None, add_indicator: Optional[Bool] = None, keep_empty_features: Optional[Bool] = None, callback: Optional[Self.Callback] = None) raises -> ref[self] Self:
        return self.set_params(self._updated_params(missing_values=missing_values, strategy=strategy, copy=copy, add_indicator=add_indicator, keep_empty_features=keep_empty_features, callback=callback))

    def set_params[origin: MutOrigin](ref[origin] self, *, fill_value: Optional[SIMD[Self.dtype, 1]], missing_values: Optional[MissingValue[Self.dtype]] = None, strategy: Optional[String] = None, copy: Optional[Bool] = None, add_indicator: Optional[Bool] = None, keep_empty_features: Optional[Bool] = None, callback: Optional[Self.Callback] = None) raises -> ref[self] Self:
        # A required keyword distinguishes clearing the fill from leaving it alone.
        var params = self._updated_params(missing_values=missing_values, strategy=strategy, copy=copy, add_indicator=add_indicator, keep_empty_features=keep_empty_features, callback=callback)
        params.fill_value = fill_value
        return self.set_params(params)

    def __sklearn_clone__(self) raises -> Self:
        var result = Self(callback=self._params.callback)
        _ = result.set_params(self._params)
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._layout.state.is_fitted()

    def _promote_integer_input(self, X: Self.Input) raises -> nm.NDArray[DType.float64]:
        validate_matrix(X)
        # Pinned astype traverses physical storage, not arbitrary strides.
        # Materialize logical row order with NuMojo before casting a view.
        if X.is_c_contiguous():
            return X.astype[DType.float64]()
        return X.contiguous().astype[DType.float64]()

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self._fit(X, y)
        return self

    def _fit(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises:
        var strategy = self._params.strategy
        if strategy != "mean" and strategy != "median" and strategy != "most_frequent" and strategy != "constant" and strategy != "callable":
            raise Error("ValueError: unknown imputation strategy")
        comptime if not Self._floating:
            if strategy == "mean" or strategy == "median" or strategy == "callable":
                # sklearn validates these strategies as Float64 BEFORE masking.
                # Reuse the floating estimator, including rounded-marker semantics.
                var marker = MissingValue[DType.float64]()
                if not self._params.missing_values.is_nan:
                    marker = MissingValue[DType.float64](Float64(self._params.missing_values.value))
                var promoted = self._promote_integer_input(X)
                var model = SimpleImputer[DType.float64, Self.Callback](missing_values=marker, strategy=strategy, copy=self._params.copy_input, add_indicator=self._params.add_indicator, keep_empty_features=self._params.keep_empty_features, callback=self._params.callback)
                _ = model.fit(promoted)
                var indicator: Optional[MissingIndicator[Self.dtype]] = None
                if self._params.add_indicator:
                    var next_indicator = MissingIndicator[Self.dtype](missing_values=self._params.missing_values, error_on_new=False)
                    next_indicator.fit_mask(missing_mask(promoted, marker))
                    indicator = next_indicator^
                self.statistics_ = model.statistics_.copy()
                self._layout = model._layout.copy()
                self.indicator_ = indicator^
                self.n_features_in_ = X.shape[1]
                self.feature_names_in_ = None
                return
        var mask = _indicator_mask(X, self._params.missing_values)
        var layout = FeatureLayout()
        layout.fit(mask, keep_empty_features=self._params.keep_empty_features, error_on_new=False)
        var statistics = nm.zeros[DType.float64](nm.Shape(X.shape[1]))
        var retained = List[Int]()
        for j in range(X.shape[1]):
            var count = 0
            for i in range(X.shape[0]):
                count += Int(not mask.unsafe_get(i * X.shape[1] + j))
            var value = Float64("nan")
            if strategy == "callable":
                var values = nm.zeros[DType.float64](nm.Shape(count))
                var k = 0
                for i in range(X.shape[0]):
                    if not mask.unsafe_get(i * X.shape[1] + j):
                        values.unsafe_set(k, Float64(X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))))
                        k += 1
                value = self._params.callback(values)
            elif count == 0:
                if self._params.keep_empty_features:
                    value = 0
                    if strategy == "constant":
                        value = Float64((self._params.fill_value.value() if self._params.fill_value else SIMD[Self.dtype, 1](0)))
            elif strategy == "constant":
                value = Float64((self._params.fill_value.value() if self._params.fill_value else SIMD[Self.dtype, 1](0)))
            else:
                var values = nm.zeros[Self.dtype](nm.Shape(count))
                var k = 0
                for i in range(X.shape[0]):
                    if not mask.unsafe_get(i * X.shape[1] + j):
                        values.unsafe_set(k, X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1])))
                        k += 1
                if strategy == "mean":
                    value = Float64(nm.sum(values)) / Float64(count)
                    # np.ma.mean masks a nonfinite quotient (including overflow).
                    if not isfinite(value):
                        value = Float64(0) if self._params.keep_empty_features else Float64("nan")
                elif strategy == "median":
                    value = Float64(nm.median[returned_dtype=Self.dtype](values))
                else:
                    var ordered = nm.sort(values)
                    var best = 0
                    var run = 0
                    var previous = ordered.unsafe_get(0)
                    for k in range(count):
                        var current = ordered.unsafe_get(k)
                        if current == previous:
                            run += 1
                        else:
                            run = 1
                        if run > best:
                            best = run
                            value = Float64(current)
                        previous = current
            statistics.unsafe_set(j, value)
            if self._params.keep_empty_features or value == value:
                retained.append(j)
        layout.set_retained(retained)
        var indicator: Optional[MissingIndicator[Self.dtype]] = None
        if self._params.add_indicator:
            var next_indicator = MissingIndicator[Self.dtype](missing_values=self._params.missing_values, error_on_new=False)
            next_indicator.fit_mask(mask)
            indicator = next_indicator^
        self.statistics_ = statistics^
        self.indicator_ = indicator^
        self.n_features_in_ = X.shape[1]
        self._layout = layout^
        self.feature_names_in_ = None

    def fit_named(mut self, X: Self.Input, feature_names: List[String], y: Optional[Self.Target] = None) raises:
        if X.ndim != 2 or len(feature_names) != X.shape[1]:
            raise Error("ValueError: feature_names length mismatch")
        _ = self.fit(X, y)
        self.feature_names_in_ = feature_names.copy()

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        comptime if Self._floating:
            return rebind_var[Self.Transformed](self._transform(X))
        else:
            if self._params.strategy == "mean" or self._params.strategy == "median" or self._params.strategy == "callable":
                return rebind_var[Self.Transformed](Variant[nm.NDArray[Self.dtype], nm.NDArray[DType.float64]](self._transform(self._promote_integer_input(X))))
            return rebind_var[Self.Transformed](Variant[nm.NDArray[Self.dtype], nm.NDArray[DType.float64]](self._transform(X)))

    def transform[output_dtype: DType](self, X: nm.NDArray[output_dtype]) raises -> nm.NDArray[output_dtype]:
        comptime assert Self._floating, "cross-dtype transform is supported only for floating estimators"
        comptime assert output_dtype == DType.float32 or output_dtype == DType.float64
        return self._transform(X)

    def _transform[output_dtype: DType](self, X: nm.NDArray[output_dtype]) raises -> nm.NDArray[output_dtype]:
        self._layout.state.check_features(X.shape[1] if X.ndim == 2 else 0)
        var marker = MissingValue[output_dtype]()
        if not self._params.missing_values.is_nan:
            marker = MissingValue[output_dtype](self._params.missing_values.value.cast[output_dtype]())
        var mask = _indicator_mask(X, marker)
        var result = select_features(X, self._layout.retained_features)
        for i in range(X.shape[0]):
            for k in range(len(self._layout.retained_features)):
                var j = self._layout.retained_features[k]
                if mask.unsafe_get(i * X.shape[1] + j):
                    comptime if not Self._floating and output_dtype == Self.dtype:
                        if self._params.strategy == "constant":
                            # Do not round a 64-bit constant through Float64 statistics.
                            result.unsafe_set(i * result.shape[1] + k, (self._params.fill_value.value() if self._params.fill_value else SIMD[Self.dtype, 1](0)).cast[output_dtype]())
                            continue
                        # Float64 mode statistics can round beyond the integer range.
                        # Reject undefined native float-to-int casts explicitly.
                        var value = self.statistics_.unsafe_get(j)
                        comptime if output_dtype == DType.int64:
                            if value >= Float64(9223372036854775808):
                                raise Error("ValueError: mode statistic is outside Int64 range after Float64 rounding")
                        comptime if output_dtype == DType.uint64:
                            if value >= Float64(18446744073709551616):
                                raise Error("ValueError: mode statistic is outside UInt64 range after Float64 rounding")
                    result.unsafe_set(i * result.shape[1] + k, self.statistics_.unsafe_get(j).cast[output_dtype]())
        if self._params.add_indicator:
            return append_indicator(result, mask, self._layout)
        return result^

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> nm.NDArray[DType.float64]:
        comptime if Self._floating:
            return self._inverse_transform(rebind[nm.NDArray[Self.dtype]](X))
        else:
            ref arrays = rebind[Variant[nm.NDArray[Self.dtype], nm.NDArray[DType.float64]]](X)
            if arrays.isa[nm.NDArray[Self.dtype]]():
                return self._inverse_transform(arrays[nm.NDArray[Self.dtype]])
            return self._inverse_transform(arrays[nm.NDArray[DType.float64]])

    def _inverse_transform[input_dtype: DType](self, X: nm.NDArray[input_dtype]) raises -> nm.NDArray[DType.float64]:
        self._layout.state.check_features(self.n_features_in_)
        if not self._params.add_indicator:
            raise Error("ValueError: inverse_transform requires add_indicator=True")
        var n_retained = len(self._layout.retained_features)
        if X.ndim != 2:
            raise Error("ValueError: inverse input must be rank 2")
        if X.shape[1] != n_retained + len(self._layout.indicator_features):
            raise Error("ValueError: inverse input width does not match fitted output")
        var result = nm.zeros[DType.float64](nm.Shape(X.shape[0], self.n_features_in_))
        var original_mask = nm.zeros[DType.bool](nm.Shape(X.shape[0], self.n_features_in_))
        for i in range(X.shape[0]):
            for k in range(len(self._layout.indicator_features)):
                var absent = X.unsafe_get(i * Int(X.strides[0]) + (n_retained + k) * Int(X.strides[1])) != 0
                original_mask.unsafe_set(i * self.n_features_in_ + self._layout.indicator_features[k], absent)
        # Follow the pinned inverse's all-indicator-column traversal. A fully
        # flagged retained column can exhaust the original width; fail safely.
        var original = 0
        for k in range(n_retained):
            while True:
                if original >= self.n_features_in_:
                    raise Error("ValueError: inverse indicator layout exhausts original features")
                var all_missing = True
                for i in range(X.shape[0]):
                    all_missing = all_missing and original_mask.unsafe_get(i * self.n_features_in_ + original)
                if not all_missing:
                    break
                original += 1
            for i in range(X.shape[0]):
                result.unsafe_set(i * self.n_features_in_ + original, Float64(X.unsafe_get(i * Int(X.strides[0]) + k * Int(X.strides[1]))))
            original += 1
        var missing = Float64(self._params.missing_values.value)
        if self._params.missing_values.is_nan:
            missing = Float64("nan")
        for i in range(X.shape[0] * self.n_features_in_):
            if original_mask.unsafe_get(i):
                result.unsafe_set(i, missing)
        return result^

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        self._layout.state.check_features(self.n_features_in_)
        var names = List[String]()
        if input_features:
            if len(input_features.value()) != self.n_features_in_:
                raise Error("ValueError: input_features length mismatch")
            names = input_features.value().copy()
            if self.feature_names_in_:
                if names != self.feature_names_in_.value():
                    raise Error("ValueError: input_features differ from fitted names")
        elif self.feature_names_in_:
            names = self.feature_names_in_.value().copy()
        else:
            for j in range(self.n_features_in_):
                names.append("x" + String(j))
        var result = List[String]()
        for j in range(self.n_features_in_):
            if self.statistics_.unsafe_get(j) == self.statistics_.unsafe_get(j):
                result.append(names[j])
        if self._params.add_indicator:
            for j in self._layout.indicator_features:
                result.append("missingindicator_" + names[j])
        return result^


comptime IntegerSimpleImputer[dtype: DType, Callback: ColumnStrategy = NoCallable] = SimpleImputer[dtype, Callback, Variant[nm.NDArray[dtype], nm.NDArray[DType.float64]]]
