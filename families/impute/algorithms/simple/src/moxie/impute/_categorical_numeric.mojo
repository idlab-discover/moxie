"""Simple imputation for ordered numeric category dictionaries."""
import numojo as nm
from std.math import isfinite
from moxie._categorical import CategoricalColumn, NumericCategories
from moxie._feature_metadata import EstimatorClassName, FeatureNamesOut
from moxie.impute._base import Imputer
from moxie.utils.validation import FittedState


@fieldwise_init
struct CategoricalNumericParams[dtype: DType](Copyable, Movable):
    var strategy: String
    var fill_value: Optional[Scalar[Self.dtype]]
    var keep_empty_features: Bool


struct CategoricalNumericImputer[dtype: DType = DType.int64](
    Imputer, FeatureNamesOut, EstimatorClassName, Copyable, Movable,
):
    """Impute tagged missing values without converting category scalars."""
    comptime Domain = NumericCategories[Self.dtype]
    comptime Column = CategoricalColumn[Self.Domain]
    comptime Input = List[Self.Column]
    comptime Target = nm.NDArray[DType.float64]
    comptime Transformed = Self.Input
    comptime Params = CategoricalNumericParams[Self.dtype]
    comptime Value = Scalar[Self.dtype]

    var _params: Self.Params
    var _state: FittedState
    var statistics_: List[Optional[Self.Value]]
    var n_features_in_: Int

    def __init__(
        out self, *, strategy: String = "most_frequent",
        fill_value: Optional[Self.Value] = None, keep_empty_features: Bool = False,
    ):
        comptime assert Self.dtype in (DType.bool, DType.int64, DType.float32, DType.float64)
        self._params = Self.Params(strategy, fill_value, keep_empty_features)
        self._state = FittedState()
        self.statistics_ = []
        self.n_features_in_ = 0

    @staticmethod
    def name() -> String:
        return "SimpleImputer"

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self = Self(
            strategy=params.strategy, fill_value=params.fill_value,
            keep_empty_features=params.keep_empty_features,
        )
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, strategy: Optional[String] = None,
        fill_value: Optional[Self.Value] = None,
        keep_empty_features: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        var params = self.get_params()
        params.strategy = strategy.or_else(params.strategy)
        if fill_value:
            params.fill_value = fill_value
        params.keep_empty_features = keep_empty_features.or_else(params.keep_empty_features)
        return self.set_params(params)

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            strategy=self._params.strategy, fill_value=self._params.fill_value,
            keep_empty_features=self._params.keep_empty_features,
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.is_fitted()

    @staticmethod
    def _rows(X: Self.Input) raises -> Int:
        if len(X) == 0 or X[0].rows() == 0:
            raise Error("ValueError: input must have samples and features")
        var rows = X[0].rows()
        for column in X:
            if column.rows() != rows:
                raise Error("ValueError: categorical columns must have matching rows")
        return rows

    def _statistic(self, column: Self.Column) raises -> Optional[Self.Value]:
        var codes = column.codes()
        if self._params.strategy == "constant":
            var fill_value = self._params.fill_value.or_else(Self.Value(0))
            if self._params.keep_empty_features:
                return fill_value
            for i in range(codes.size):
                if codes.unsafe_get(i) >= 0:
                    return fill_value
            return None

        var domain = column.domain()
        var counts = List[Int](length=domain.size(), fill=0)
        for i in range(codes.size):
            var code = Int(codes.unsafe_get(i))
            if code >= 0:
                counts[code] += 1
        var best = -1
        var best_count = 0
        # Numeric domain order breaks frequency ties by smallest category value.
        for i in range(len(counts)):
            if counts[i] > best_count:
                best = i
                best_count = counts[i]
        if best >= 0:
            return domain.get(best)
        if self._params.keep_empty_features:
            return Self.Value(0)
        return None

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        if self._params.strategy != "most_frequent" and self._params.strategy != "constant":
            raise Error("ValueError: categorical numeric imputation requires most_frequent or constant")
        comptime if Self.dtype == DType.float32 or Self.dtype == DType.float64:
            var fill_value = self._params.fill_value.or_else(Self.Value(0))
            if self._params.strategy == "constant" and not isfinite(fill_value):
                raise Error("ValueError: categorical fill_value must be finite")
        var rows = Self._rows(X)
        var statistics = List[Optional[Self.Value]](capacity=len(X))
        for column in X:
            statistics.append(self._statistic(column))
        self._state.record_fit(rows, len(X))
        self.statistics_ = statistics^
        self.n_features_in_ = len(X)
        return self

    @staticmethod
    def _fill(column: Self.Column, value: Self.Value) raises -> Self.Column:
        var codes = column.codes()
        var first_missing = 0
        while first_missing < codes.size and codes.unsafe_get(first_missing) >= 0:
            first_missing += 1
        if first_missing == codes.size:
            return column.copy()

        var domain = column.domain()
        var fill_code = domain.find(value)
        if fill_code < 0:
            var old = domain.values()
            var categories = nm.zeros[Self.dtype](nm.Shape(old.size + 1))
            fill_code = 0
            while fill_code < old.size and old.unsafe_get(fill_code) < value:
                fill_code += 1
            categories.unsafe_set(fill_code, value)
            for i in range(old.size):
                categories.unsafe_set(i if i < fill_code else i + 1, old.unsafe_get(i))
            domain = Self.Domain(categories)
            for i in range(codes.size):
                var code = codes.unsafe_get(i)
                if code >= Int64(fill_code):
                    codes.unsafe_set(i, code + 1)
        for i in range(first_missing, codes.size):
            if codes.unsafe_get(i) < 0:
                codes.unsafe_set(i, Int64(fill_code))
        return Self.Column(domain, codes)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._state.check_features(len(X))
        _ = Self._rows(X)
        var result = Self.Transformed(capacity=len(X))
        for j in range(len(X)):
            if self.statistics_[j]:
                result.append(Self._fill(X[j], self.statistics_[j].value()))
        return result^

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        self._state.check_features(self.n_features_in_)
        if input_features and len(input_features.value()) != self.n_features_in_:
            raise Error("ValueError: input_features length mismatch")
        var result = List[String]()
        for j in range(self.n_features_in_):
            if self.statistics_[j]:
                result.append(input_features.value()[j] if input_features else "x" + String(j))
        return result^
