"""Test transformers with observable dispatch and state; not public estimators."""
from numojo import NDArray, Shape, zeros
from moxie.base import UnsupervisedEstimator, TransformerMixin, FitTransformer
from moxie.pipeline import UnionFeatureNames, UnionInputFeatures

@fieldwise_init
struct TestParams(Copyable, Movable):
    var factor: Float64
    var kind: Int

struct TestTransformer[dtype: DType, fused: Bool = False](
    UnsupervisedEstimator, TransformerMixin, UnionFeatureNames, UnionInputFeatures,
    FitTransformer where fused,
):
    comptime Input = NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Transformed = Self.Input
    comptime Params = TestParams
    var factor: Float64
    var kind: Int
    var fitted: Bool
    var fit_calls: Int
    var fused_calls: Int
    var n_features: Int
    var target_sum: Float64

    def __init__(out self, factor: Float64 = 1, kind: Int = 0):
        self.factor = factor
        self.kind = kind
        self.fitted = False
        self.fit_calls = 0
        self.fused_calls = 0
        self.n_features = 0
        self.target_sum = 0
    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.factor, self.kind)
    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.factor = params.factor
        self.kind = params.kind
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self(self.factor, self.kind)
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted
    def _learn(mut self, X: Self.Input, y: Optional[Self.Target]) raises:
        self.fitted = False
        if self.kind == 4:
            raise Error("ValueError: requested fixture fit failure")
        self.n_features = X.shape[1]
        self.target_sum = 0
        if y:
            for i in range(y.value().size):
                self.target_sum += Float64(y.value().item(i))
        self.fitted = True
    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self.fit_calls += 1
        self._learn(X, y)
        return self
    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.fused_calls += 1
        self._learn(X, y)
        return self._output(X, 10)
    def _output(self, X: Self.Input, extra: Float64) raises -> Self.Transformed:
        if not self.fitted:
            raise Error("NotFittedError: test transformer")
        if self.kind == 2:
            return zeros[Self.dtype](Shape(X.shape[0]))
        var rows = X.shape[0] + (1 if self.kind == 3 else 0)
        var cols = 0 if self.kind == 5 else (min(1, X.shape[1]) if self.kind == 1 else X.shape[1])
        var result = zeros[Self.dtype](Shape(rows, cols))
        for i in range(X.shape[0]):
            for j in range(cols):
                result.itemset(i * cols + j, Scalar[Self.dtype](Float64(X.item(i, j)) * self.factor + extra))
        return result^
    def transform(self, X: Self.Input) raises -> Self.Transformed:
        return self._output(X, 0)
    def n_features_in_(self) raises -> Int:
        if not self.fitted:
            raise Error("AttributeError: fixture has no feature count before fit")
        return self.n_features
    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if not self.fitted:
            raise Error("NotFittedError: feature names")
        if input_features and len(input_features.value()) != self.n_features:
            raise Error("ValueError: feature name count mismatch")
        var cols = 0 if self.kind == 5 else (min(1, self.n_features) if self.kind == 1 else self.n_features)
        var result = List[String]()
        for j in range(cols):
            result.append(input_features.value()[j] if input_features else "x" + String(j))
        return result^

from moxie.base import SupervisedEstimator

struct SupervisedOnly[dtype: DType](SupervisedEstimator, TransformerMixin):
    comptime Input = NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Transformed = Self.Input
    comptime Params = Int
    var fitted: Bool
    def __init__(out self): self.fitted = False
    def get_params(self, deep: Bool = True) raises -> Int: return 0
    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        return self
    def __sklearn_clone__(self) raises -> Self: return Self()
    def __sklearn_is_fitted__(self) -> Bool: return self.fitted
    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        self.fitted = True
        return self
    def transform(self, X: Self.Input) raises -> Self.Transformed:
        if not self.fitted: raise Error("NotFittedError: fixture")
        return X.copy()

struct FusedOnly[dtype: DType](FitTransformer, TransformerMixin):
    comptime Input = NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Transformed = Self.Input
    comptime Params = Int
    var fitted: Bool
    def __init__(out self): self.fitted = False
    def get_params(self, deep: Bool = True) raises -> Int: return 0
    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        return self
    def __sklearn_clone__(self) raises -> Self: return Self()
    def __sklearn_is_fitted__(self) -> Bool: return self.fitted
    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.fitted = True
        return X.copy()
    def transform(self, X: Self.Input) raises -> Self.Transformed:
        if not self.fitted: raise Error("NotFittedError: fixture")
        return X.copy()
