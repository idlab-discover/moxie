"""Dense missing indicators using the family-owned missingness layout."""
import numojo as nm
from std.utils import Variant
from moxie.base import UnsupervisedEstimator, TransformerMixin, FitTransformer
from moxie.impute._plumbing import MissingValue, missing_mask, FeatureLayout, validate_matrix


def _indicator_mask[dtype: DType](X: nm.NDArray[dtype], marker: MissingValue[dtype]) raises -> nm.NDArray[DType.bool]:
    comptime if dtype == DType.float32 or dtype == DType.float64:
        return missing_mask(X, marker)
    else:
        comptime assert dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64
        validate_matrix(X)
        var mask = nm.zeros[DType.bool](nm.Shape(X.shape[0], X.shape[1]))
        # Integer data cannot contain NaN. Preserve exact integer sentinel equality.
        if not marker.is_nan:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    mask.unsafe_set(i * X.shape[1] + j, X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1])) == marker.value)
        return mask^


struct IndicatorParams[dtype: DType](Copyable, Movable):
    var missing_values: MissingValue[Self.dtype]
    var features: String
    var sparse: Variant[Bool, String]
    var error_on_new: Bool

    def __init__(out self, *, missing_values: MissingValue[Self.dtype] = MissingValue[Self.dtype](), features: String = "missing-only", sparse: Variant[Bool, String] = "auto", error_on_new: Bool = True):
        self.missing_values = missing_values.copy()
        self.features = features
        self.sparse = sparse.copy()
        self.error_on_new = error_on_new


struct MissingIndicator[dtype: DType = DType.float64](UnsupervisedEstimator, TransformerMixin, FitTransformer, Copyable, Movable):
    comptime Params = IndicatorParams[Self.dtype]
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime Transformed = nm.NDArray[DType.bool]
    var _params: Self.Params
    var _layout: FeatureLayout
    var features_: List[Int]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]

    def __init__(out self, *, missing_values: MissingValue[Self.dtype] = MissingValue[Self.dtype](), features: String = "missing-only", sparse: Variant[Bool, String] = "auto", error_on_new: Bool = True):
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

    def set_params[origin: MutOrigin](ref[origin] self, *, missing_values: Optional[MissingValue[Self.dtype]] = None, features: Optional[String] = None, sparse: Optional[Bool] = None, error_on_new: Optional[Bool] = None) raises -> ref[self] Self:
        var params = self.get_params()
        if missing_values:
            params.missing_values = missing_values.value().copy()
        if features:
            params.features = features.value()
        if sparse:
            params.sparse = sparse.value()
        if error_on_new:
            params.error_on_new = error_on_new.value()
        return self.set_params(params)

    def set_params[origin: MutOrigin](ref[origin] self, *, sparse: String, missing_values: Optional[MissingValue[Self.dtype]] = None, features: Optional[String] = None, error_on_new: Optional[Bool] = None) raises -> ref[self] Self:
        var params = self.get_params()
        if missing_values:
            params.missing_values = missing_values.value().copy()
        params.sparse = sparse
        if features:
            params.features = features.value()
        if error_on_new:
            params.error_on_new = error_on_new.value()
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
        var mask = _indicator_mask(X, self._params.missing_values)
        self.fit_mask(mask)

    def fit_named(mut self, X: Self.Input, feature_names: List[String], y: Optional[Self.Target] = None) raises:
        if X.ndim != 2 or len(feature_names) != X.shape[1]:
            raise Error("ValueError: feature_names length mismatch")
        _ = self.fit(X, y)
        self.feature_names_in_ = feature_names.copy()

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._layout.state.check_features(X.shape[1] if X.ndim == 2 else 0)
        var mask = _indicator_mask(X, self._params.missing_values)
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
