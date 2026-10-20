"""Owned nullable text and string-only simple imputation."""
import numojo as nm
from std.utils import Variant
from moxie.base import UnsupervisedEstimator, TransformerMixin, FitTransformer
from moxie.impute._base import Imputer
from moxie.impute._plumbing import FeatureLayout
from moxie.impute._indicator import MissingIndicator


struct StringMatrix(Copyable, Movable):
    """Rank-two owned UTF-8 cells; None is an explicit missing tag."""
    var _values: List[Optional[String]]
    var _rows: Int
    var _columns: Int

    def __init__(out self, values: List[Optional[String]], rows: Int, columns: Int) raises:
        if rows < 0 or columns < 0:
            raise Error("ValueError: negative string matrix dimension")
        if columns != 0 and rows > len(values) // columns:
            raise Error("ValueError: string matrix shape does not match cells")
        if rows * columns != len(values):
            raise Error("ValueError: string matrix shape does not match cells")
        self._values = values.copy()
        self._rows = rows
        self._columns = columns

    @staticmethod
    def from_strings(values: List[String], rows: Int, columns: Int) raises -> Self:
        var cells = List[Optional[String]]()
        for value in values:
            cells.append(Optional[String](value.copy()))
        return Self(cells, rows, columns)

    def rows(self) -> Int:
        return self._rows

    def columns(self) -> Int:
        return self._columns

    def values(self) -> List[Optional[String]]:
        return self._values.copy()

    def _check_index(self, row: Int, column: Int) raises:
        if row < 0 or row >= self._rows or column < 0 or column >= self._columns:
            raise Error("ValueError: string matrix index out of bounds")

    def get(self, row: Int, column: Int) raises -> Optional[String]:
        self._check_index(row, column)
        return self._values[row * self._columns + column].copy()

    def set(mut self, row: Int, column: Int, value: Optional[String]) raises:
        self._check_index(row, column)
        self._values[row * self._columns + column] = value.copy()

    def missing_mask(self, missing_values: Optional[String] = None) raises -> nm.NDArray[DType.bool]:
        if self._rows == 0 or self._columns == 0:
            raise Error("ValueError: input must have samples and features")
        var mask = nm.zeros[DType.bool](nm.Shape(self._rows, self._columns))
        for i in range(len(self._values)):
            var absent = not self._values[i]
            if missing_values:
                if absent:
                    raise Error("ValueError: a missing tag is invalid with a string missing_values sentinel")
                absent = self._values[i].value() == missing_values.value()
            mask.unsafe_set(i, absent)
        return mask^


struct StringImputationResult(Copyable, Movable):
    """Retained text columns followed logically by Boolean indicator columns."""
    var values: StringMatrix
    var indicators: nm.NDArray[DType.bool]
    var shape: nm.Shape

    def __init__(out self, var values: StringMatrix, var indicators: nm.NDArray[DType.bool]) raises:
        if indicators.ndim != 2 or indicators.shape[0] != values.rows():
            raise Error("ValueError: string result and indicators must have matching rows")
        self.shape = nm.Shape(values.rows(), values.columns() + indicators.shape[1])
        self.values = values^
        self.indicators = indicators^


struct StringSimpleParams(Copyable, Movable):
    var missing_values: Optional[String]
    var strategy: String
    var fill_value: Optional[String]
    var copy_input: Bool
    var add_indicator: Bool
    var keep_empty_features: Bool

    def __init__(out self, *, missing_values: Optional[String] = None, strategy: String = "mean", fill_value: Optional[String] = None, copy: Bool = True, add_indicator: Bool = False, keep_empty_features: Bool = False):
        self.missing_values = missing_values.copy()
        self.strategy = strategy.copy()
        self.fill_value = fill_value.copy()
        self.copy_input = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features


struct StringSimpleImputer(Imputer, Copyable, Movable):
    comptime Params = StringSimpleParams
    comptime Input = StringMatrix
    comptime Target = StringMatrix
    comptime Transformed = StringImputationResult
    var _params: Self.Params
    var _layout: FeatureLayout
    var statistics_: List[Optional[String]]
    var indicator_: Optional[MissingIndicator[DType.float64]]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]

    def __init__(out self, *, missing_values: Optional[String] = None, strategy: String = "mean", fill_value: Optional[String] = None, copy: Bool = True, add_indicator: Bool = False, keep_empty_features: Bool = False):
        self._params = Self.Params(missing_values=missing_values, strategy=strategy, fill_value=fill_value, copy=copy, add_indicator=add_indicator, keep_empty_features=keep_empty_features)
        self._layout = FeatureLayout()
        self.statistics_ = List[Optional[String]]()
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
        self = Self(missing_values=params.missing_values, strategy=params.strategy, fill_value=params.fill_value, copy=params.copy_input, add_indicator=params.add_indicator, keep_empty_features=params.keep_empty_features)
        return self

    def _updated_params(self, *, strategy: Optional[String] = None, copy: Optional[Bool] = None, add_indicator: Optional[Bool] = None, keep_empty_features: Optional[Bool] = None) raises -> Self.Params:
        var params = self.get_params()
        if strategy:
            params.strategy = strategy.value().copy()
        if copy:
            params.copy_input = copy.value()
        if add_indicator:
            params.add_indicator = add_indicator.value()
        if keep_empty_features:
            params.keep_empty_features = keep_empty_features.value()
        return params^

    def set_params[origin: MutOrigin](ref[origin] self, *, strategy: Optional[String] = None, copy: Optional[Bool] = None, add_indicator: Optional[Bool] = None, keep_empty_features: Optional[Bool] = None) raises -> ref[self] Self:
        return self.set_params(self._updated_params(strategy=strategy, copy=copy, add_indicator=add_indicator, keep_empty_features=keep_empty_features))

    def set_params[origin: MutOrigin](ref[origin] self, *, fill_value: Optional[String], strategy: Optional[String] = None, copy: Optional[Bool] = None, add_indicator: Optional[Bool] = None, keep_empty_features: Optional[Bool] = None) raises -> ref[self] Self:
        var params = self._updated_params(strategy=strategy, copy=copy, add_indicator=add_indicator, keep_empty_features=keep_empty_features)
        params.fill_value = fill_value.copy()
        return self.set_params(params)

    def set_params[origin: MutOrigin](ref[origin] self, *, missing_values: Optional[String], strategy: Optional[String] = None, copy: Optional[Bool] = None, add_indicator: Optional[Bool] = None, keep_empty_features: Optional[Bool] = None) raises -> ref[self] Self:
        var params = self._updated_params(strategy=strategy, copy=copy, add_indicator=add_indicator, keep_empty_features=keep_empty_features)
        params.missing_values = missing_values.copy()
        return self.set_params(params)

    def set_params[origin: MutOrigin](ref[origin] self, *, fill_value: Optional[String], missing_values: Optional[String], strategy: Optional[String] = None, copy: Optional[Bool] = None, add_indicator: Optional[Bool] = None, keep_empty_features: Optional[Bool] = None) raises -> ref[self] Self:
        var params = self._updated_params(strategy=strategy, copy=copy, add_indicator=add_indicator, keep_empty_features=keep_empty_features)
        params.fill_value = fill_value.copy()
        params.missing_values = missing_values.copy()
        return self.set_params(params)

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        _ = result.set_params(self._params)
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._layout.state.is_fitted()

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
        if self._params.strategy != "most_frequent" and self._params.strategy != "constant":
            raise Error("ValueError: StringSimpleImputer supports only most_frequent and constant; select strategy explicitly")
        var mask = X.missing_mask(self._params.missing_values)
        var layout = FeatureLayout()
        layout.fit(mask, keep_empty_features=self._params.keep_empty_features, error_on_new=False)
        if self._params.strategy == "most_frequent" and self._params.keep_empty_features and len(layout.empty_features) != 0:
            raise Error("ValueError: an empty most_frequent feature requires numeric zero; use constant or keep_empty_features=False for strings")
        var statistics = List[Optional[String]]()
        for j in range(X.columns()):
            var counts = Dict[String, Int]()
            var observed = False
            for i in range(X.rows()):
                if not mask.unsafe_get(i * X.columns() + j):
                    observed = True
                    if self._params.strategy == "most_frequent":
                        var value = X.get(i, j).value().copy()
                        if value in counts:
                            counts[value] += 1
                        else:
                            counts[value] = 1
            var statistic: Optional[String] = None
            if observed or self._params.keep_empty_features:
                if self._params.strategy == "constant":
                    statistic = self._params.fill_value.value().copy() if self._params.fill_value else String("missing_value")
                else:
                    var best_count = 0
                    for entry in counts.items():
                        if entry.value > best_count or (entry.value == best_count and entry.key < statistic.value()):
                            statistic = entry.key.copy()
                            best_count = entry.value
            statistics.append(statistic^)
        var indicator: Optional[MissingIndicator[DType.float64]] = None
        if self._params.add_indicator:
            var next_indicator = MissingIndicator[DType.float64](error_on_new=False)
            next_indicator.fit_mask(mask)
            indicator = next_indicator^
        self.statistics_ = statistics^
        self.indicator_ = indicator^
        self._layout = layout^
        self.n_features_in_ = X.columns()
        self.feature_names_in_ = None

    def fit_named(mut self, X: Self.Input, feature_names: List[String], y: Optional[Self.Target] = None) raises:
        if len(feature_names) != X.columns():
            raise Error("ValueError: feature_names length mismatch")
        _ = self.fit(X, y)
        self.feature_names_in_ = feature_names.copy()

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._layout.state.check_features(X.columns())
        var mask = X.missing_mask(self._params.missing_values)
        var cells = List[Optional[String]]()
        for i in range(X.rows()):
            for j in self._layout.retained_features:
                if mask.unsafe_get(i * X.columns() + j):
                    cells.append(self.statistics_[j].copy())
                else:
                    cells.append(X.get(i, j))
        var values = StringMatrix(cells, X.rows(), len(self._layout.retained_features))
        var indicators = nm.zeros[DType.bool](nm.Shape(X.rows(), 0))
        if self._params.add_indicator:
            indicators = self._layout.indicator(mask)
        return StringImputationResult(values^, indicators^)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> StringMatrix:
        self._layout.state.check_features(self.n_features_in_)
        if not self._params.add_indicator:
            raise Error("ValueError: inverse_transform requires add_indicator=True")
        raise Error("ValueError: inverse_transform is unsupported for strings; the reference inverse requires numeric values")

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
        for j in self._layout.retained_features:
            result.append(names[j])
        if self._params.add_indicator:
            for j in self._layout.indicator_features:
                result.append("missingindicator_" + names[j])
        return result^


struct StringIndicatorParams(Copyable, Movable):
    var missing_values: Optional[String]
    var features: String
    var sparse: Variant[Bool, String]
    var error_on_new: Bool

    def __init__(out self, *, missing_values: Optional[String] = None, features: String = "missing-only", sparse: Variant[Bool, String] = "auto", error_on_new: Bool = True):
        self.missing_values = missing_values.copy()
        self.features = features
        self.sparse = sparse.copy()
        self.error_on_new = error_on_new


struct StringMissingIndicator(UnsupervisedEstimator, TransformerMixin, FitTransformer, Copyable, Movable):
    comptime Params = StringIndicatorParams
    comptime Input = StringMatrix
    comptime Target = StringMatrix
    comptime Transformed = nm.NDArray[DType.bool]
    var _params: Self.Params
    var _layout: FeatureLayout
    var features_: List[Int]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]

    def __init__(out self, *, missing_values: Optional[String] = None, features: String = "missing-only", sparse: Variant[Bool, String] = "auto", error_on_new: Bool = True):
        self._params = Self.Params(missing_values=missing_values, features=features, sparse=sparse, error_on_new=error_on_new)
        self._layout = FeatureLayout()
        self.features_ = List[Int]()
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
        self = Self(missing_values=params.missing_values, features=params.features, sparse=params.sparse, error_on_new=params.error_on_new)
        return self

    def _updated_params(self, features: Optional[String], sparse: Optional[Bool], error_on_new: Optional[Bool]) raises -> Self.Params:
        var params = self.get_params()
        if features:
            params.features = features.value().copy()
        if sparse:
            params.sparse = sparse.value()
        if error_on_new:
            params.error_on_new = error_on_new.value()
        return params^

    def set_params[origin: MutOrigin](ref[origin] self, *, features: Optional[String] = None, sparse: Optional[Bool] = None, error_on_new: Optional[Bool] = None) raises -> ref[self] Self:
        var params = self._updated_params(features, sparse, error_on_new)
        return self.set_params(params)

    def set_params[origin: MutOrigin](ref[origin] self, *, sparse: String, features: Optional[String] = None, error_on_new: Optional[Bool] = None) raises -> ref[self] Self:
        var params = self._updated_params(features, None, error_on_new)
        params.sparse = sparse
        return self.set_params(params)

    def set_params[origin: MutOrigin](ref[origin] self, *, missing_values: Optional[String], features: Optional[String] = None, sparse: Optional[Bool] = None, error_on_new: Optional[Bool] = None) raises -> ref[self] Self:
        var params = self._updated_params(features, sparse, error_on_new)
        params.missing_values = missing_values.copy()
        return self.set_params(params)

    def set_params[origin: MutOrigin](ref[origin] self, *, missing_values: Optional[String], sparse: String, features: Optional[String] = None, error_on_new: Optional[Bool] = None) raises -> ref[self] Self:
        var params = self._updated_params(features, None, error_on_new)
        params.sparse = sparse
        params.missing_values = missing_values.copy()
        return self.set_params(params)

    def __sklearn_clone__(self) raises -> Self:
        return Self(missing_values=self._params.missing_values, features=self._params.features, sparse=self._params.sparse, error_on_new=self._params.error_on_new)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._layout.state.is_fitted()

    def fit_mask(mut self, mask: nm.NDArray[DType.bool]) raises:
        if self._params.sparse.isa[String]():
            if self._params.sparse[String] != "auto":
                raise Error("ValueError: sparse must be a Bool or 'auto'")
        elif self._params.sparse[Bool]:
            raise Error("ValueError: sparse output is deferred; select sparse=False or 'auto'")
        self._layout.fit(mask, features=self._params.features, error_on_new=self._params.error_on_new)
        self.features_ = self._layout.indicator_features.copy()
        self.n_features_in_ = mask.shape[1]
        self.feature_names_in_ = None

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
        var mask = X.missing_mask(self._params.missing_values)
        self.fit_mask(mask)

    def fit_named(mut self, X: Self.Input, feature_names: List[String], y: Optional[Self.Target] = None) raises:
        if len(feature_names) != X.columns():
            raise Error("ValueError: feature_names length mismatch")
        _ = self.fit(X, y)
        self.feature_names_in_ = feature_names.copy()

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._layout.state.check_features(X.columns())
        var mask = X.missing_mask(self._params.missing_values)
        return self._layout.indicator(mask)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

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
        for j in self.features_:
            result.append("missingindicator_" + names[j])
        return result^
