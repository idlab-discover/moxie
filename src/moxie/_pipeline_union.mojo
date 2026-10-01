# Adapted from sklearn.pipeline; BSD-3-Clause, see LICENSES/scikit-learn.txt.
"""Dense FeatureUnion with typed, recursively composed branch ownership."""
from numojo import NDArray, Shape, zeros
from numojo.routines.manipulation import _concatenate_list
from std.time import perf_counter_ns
from moxie.base import (
    TransformerMixin, FitTransformer, SupervisedEstimator, UnsupervisedEstimator, clone,
)
from moxie._feature_metadata import FeatureNamesOut as UnionFeatureNames, InputFeatureCount
from moxie._pipeline import CompositionStep, PairParams, StepMode, validate_names, check_union_rows

comptime UnionWeights = Optional[Dict[String, Float64]]


trait UnionInputFeatures:
    """Optional native adapter exposing the constituent feature count."""
    def n_features_in_(self) raises -> Int: ...


trait UnionBranches(CompositionStep):
    comptime Params: Copyable & Deinitable
    comptime dtype: DType
    def names(self) -> List[String]: ...
    def active_count(self) -> Int: ...
    def input_feature_count(self) raises -> Int: ...
    @staticmethod
    def param_names(params: Self.Params) -> List[String]: ...
    def fit_branches(mut self, X: NDArray[Self.dtype], y: Optional[NDArray[Self.dtype]],
                     transform: Bool, weights: UnionWeights, verbose: Bool,
                     mut outputs: List[NDArray[Self.dtype]]) raises: ...
    def transform_branches(self, X: NDArray[Self.dtype], weights: UnionWeights,
                           mut outputs: List[NDArray[Self.dtype]]) raises: ...
    def feature_names(self, input_features: Optional[List[String]], prefix: Bool) raises -> List[String]: ...


def _append_names(mut result: List[String], names: List[String]):
    for name in names:
        result.append(name)


def _named_features(name: String, var features: List[String], prefix: Bool) -> List[String]:
    if prefix:
        for i in range(len(features)):
            features[i] = name + "__" + features[i]
    return features^


def _collect[dtype: DType](name: String, var output: NDArray[dtype],
                          weights: UnionWeights, mut outputs: List[NDArray[dtype]]) raises:
    if weights:
        if name in weights.value():
            output = output * Scalar[dtype](weights.value()[name])
    outputs.append(output^)


def _fit_branch[dtype: DType, E: TransformerMixin](mut estimator: E, X: NDArray[dtype],
                                                y: Optional[NDArray[dtype]]) raises:
    comptime assert E.Input == NDArray[dtype], "union input must be NDArray[dtype]"
    comptime if conforms_to(E, UnsupervisedEstimator):
        comptime assert E.Target == NDArray[dtype], "union target must be NDArray[dtype]"
        estimator.fit(rebind[E.Input](X), rebind[Optional[E.Target]](y))
    elif conforms_to(E, SupervisedEstimator):
        comptime assert E.Target == NDArray[dtype], "union target must be NDArray[dtype]"
        if not y:
            raise Error("ValueError: supervised union branch requires y")
        estimator.fit(rebind[E.Input](X), rebind[E.Target](y.value()))
    else:
        raise Error("TypeError: branch has no fit; use fit_transform")


def _fit_transform_branch[dtype: DType, E: TransformerMixin](mut estimator: E, X: NDArray[dtype],
                                                           y: Optional[NDArray[dtype]]) raises -> NDArray[dtype]:
    comptime assert E.Input == NDArray[dtype], "union input must be NDArray[dtype]"
    comptime assert E.Transformed == NDArray[dtype], "union output must be NDArray[dtype]"
    comptime if conforms_to(E, FitTransformer):
        comptime assert E.Target == NDArray[dtype], "union target must be NDArray[dtype]"
        return rebind_var[NDArray[dtype]](estimator.fit_transform(rebind[E.Input](X), rebind[Optional[E.Target]](y)))
    else:
        _fit_branch(estimator, X, y)
        return rebind_var[NDArray[dtype]](estimator.transform(rebind[E.Input](X)))


@fieldwise_init
struct UnionStepParams[P: Copyable & Deinitable](Copyable, Movable):
    var name: String
    var estimator: Self.P


struct UnionStep[dtype_: DType, E: TransformerMixin & Deinitable](UnionBranches) where conforms_to(E.Params, Deinitable):
    comptime dtype = Self.dtype_
    comptime Params = UnionStepParams[Self.E.Params]
    var name: String
    var estimator: Self.E

    def __init__(out self, name: String, var estimator: Self.E):
        comptime assert Self.E.Input == NDArray[Self.dtype], "union input must be NDArray[dtype]"
        comptime assert Self.E.Transformed == NDArray[Self.dtype], "union output must be NDArray[dtype]"
        comptime assert conforms_to(Self.E, FitTransformer) or conforms_to(Self.E, SupervisedEstimator) or conforms_to(Self.E, UnsupervisedEstimator), "union branch needs fit or fit_transform"
        self.name = name
        self.estimator = estimator^

    def active_count(self) -> Int:
        return 1
    def input_feature_count(self) raises -> Int:
        comptime if conforms_to(Self.E, UnionInputFeatures):
            return self.estimator.n_features_in_()
        elif conforms_to(Self.E, InputFeatureCount):
            return self.estimator.get_n_features_in()
        else:
            raise Error("AttributeError: first union transformer does not expose n_features_in_")
    def names(self) -> List[String]:
        return [self.name]
    @staticmethod
    def param_names(params: Self.Params) -> List[String]:
        return [params.name]
    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.name, self.estimator.get_params(deep))
    def set_params(mut self, params: Self.Params) raises:
        self.name = params.name
        self.estimator.set_params(params.estimator)
    def __sklearn_clone__(self) raises -> Self:
        return Self(self.name, clone(self.estimator))
    def __sklearn_is_fitted__(self) -> Bool:
        return self.estimator.__sklearn_is_fitted__()
    def fit_branches(mut self, X: NDArray[Self.dtype], y: Optional[NDArray[Self.dtype]],
                     transform: Bool, weights: UnionWeights, verbose: Bool,
                     mut outputs: List[NDArray[Self.dtype]]) raises:
        var start = perf_counter_ns()
        if transform:
            var output = _fit_transform_branch(self.estimator, X, y)
            _collect(self.name, output^, weights, outputs)
        else:
            _fit_branch(self.estimator, X, y)
        if verbose:
            print("[FeatureUnion] Processing", self.name, "total seconds:", Float64(perf_counter_ns() - start) / 1e9)
    def transform_branches(self, X: NDArray[Self.dtype], weights: UnionWeights,
                           mut outputs: List[NDArray[Self.dtype]]) raises:
        var output = rebind_var[NDArray[Self.dtype]](self.estimator.transform(rebind[Self.E.Input](X)))
        _collect(self.name, output^, weights, outputs)
    def feature_names(self, input_features: Optional[List[String]], prefix: Bool) raises -> List[String]:
        comptime if conforms_to(Self.E, UnionFeatureNames):
            return _named_features(self.name, self.estimator.get_feature_names_out(input_features), prefix)
        else:
            raise Error("AttributeError: union transformer does not provide get_feature_names_out")


def union_step[E: TransformerMixin & Deinitable, //, dtype: DType = DType.float64](
    name: String, var estimator: E,
) -> UnionStep[dtype, E] where conforms_to(E.Params, Deinitable):
    return UnionStep[dtype, E](name, estimator^)


@fieldwise_init
struct UnionSentinelParams(Copyable, Movable):
    var name: String
    var mode: Int


struct UnionSentinel[dtype_: DType = DType.float64](UnionBranches):
    """Passthrough/drop without an estimator or learned array buffers."""
    comptime dtype = Self.dtype_
    comptime Params = UnionSentinelParams
    var name: String
    var mode: Int
    var _n_features: Optional[Int]

    def __init__(out self, name: String, mode: Int = StepMode.passthrough) raises:
        if mode != StepMode.passthrough and mode != StepMode.drop:
            raise Error("ValueError: union sentinel must be passthrough or drop")
        self.name = name
        self.mode = mode
        self._n_features = None
    def active_count(self) -> Int:
        return 0 if self.mode == StepMode.drop else 1
    def input_feature_count(self) raises -> Int:
        if self.mode != StepMode.passthrough or not self._n_features:
            raise Error("AttributeError: first union branch has no n_features_in_")
        return self._n_features.value()
    def names(self) -> List[String]:
        return [self.name]
    @staticmethod
    def param_names(params: Self.Params) -> List[String]:
        return [params.name]
    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.name, self.mode)
    def set_params(mut self, params: Self.Params) raises:
        if params.mode != StepMode.passthrough and params.mode != StepMode.drop:
            raise Error("ValueError: union sentinel must be passthrough or drop")
        if self.mode != params.mode:
            self._n_features = None
        self.name = params.name
        self.mode = params.mode
    def __sklearn_clone__(self) raises -> Self:
        return Self(self.name, self.mode)
    def __sklearn_is_fitted__(self) -> Bool:
        return True
    def fit_branches(mut self, X: NDArray[Self.dtype], y: Optional[NDArray[Self.dtype]],
                     transform: Bool, weights: UnionWeights, verbose: Bool,
                     mut outputs: List[NDArray[Self.dtype]]) raises:
        if self.mode == StepMode.passthrough:
            self._n_features = X.shape[1]
            if transform:
                self.transform_branches(X, weights, outputs)
            if verbose:
                print("[FeatureUnion] Processing", self.name, "(passthrough)")
        elif self.mode != StepMode.drop:
            raise Error("ValueError: invalid union sentinel mode")
    def transform_branches(self, X: NDArray[Self.dtype], weights: UnionWeights,
                           mut outputs: List[NDArray[Self.dtype]]) raises:
        if self.mode == StepMode.passthrough:
            _collect(self.name, X.copy(), weights, outputs)
        elif self.mode != StepMode.drop:
            raise Error("ValueError: invalid union sentinel mode")
    def feature_names(self, input_features: Optional[List[String]], prefix: Bool) raises -> List[String]:
        if self.mode == StepMode.drop:
            return List[String]()
        if self.mode != StepMode.passthrough:
            raise Error("ValueError: invalid union sentinel mode")
        var result = List[String]()
        if input_features:
            if self._n_features and len(input_features.value()) != self._n_features.value():
                raise Error("ValueError: input_features length differs from n_features_in_")
            result = input_features.value().copy()
        else:
            if not self._n_features:
                raise Error("ValueError: supply input_features or fit the passthrough first")
            for i in range(self._n_features.value()):
                result.append("x" + String(i))
        return _named_features(self.name, result^, prefix)


struct UnionPair[A: UnionBranches, B: UnionBranches](UnionBranches):
    """Recursive heterogeneous branch list; nesting here is flattened."""
    comptime dtype = Self.A.dtype
    comptime Params = PairParams[Self.A.Params, Self.B.Params]
    var first: Self.A
    var second: Self.B
    def __init__(out self, var first: Self.A, var second: Self.B):
        comptime assert Self.A.dtype == Self.B.dtype, "union branch dtypes must match"
        self.first = first^
        self.second = second^
    def active_count(self) -> Int:
        return self.first.active_count() + self.second.active_count()
    def input_feature_count(self) raises -> Int:
        return self.first.input_feature_count()
    def names(self) -> List[String]:
        var names = self.first.names()
        _append_names(names, self.second.names())
        return names^
    @staticmethod
    def param_names(params: Self.Params) -> List[String]:
        var names = Self.A.param_names(params.first)
        _append_names(names, Self.B.param_names(params.second))
        return names^
    def get_params(self, deep: Bool = True) raises -> Self.Params:
        var first: Self.A.Params = self.first.get_params(deep)
        var second: Self.B.Params = self.second.get_params(deep)
        return Self.Params(first^, second^)
    def set_params(mut self, params: Self.Params) raises:
        self.first.set_params(params.first)
        self.second.set_params(params.second)
    def __sklearn_clone__(self) raises -> Self:
        var first: Self.A = clone(self.first)
        var second: Self.B = clone(self.second)
        return Self(first^, second^)
    def __sklearn_is_fitted__(self) -> Bool:
        return self.first.__sklearn_is_fitted__() and self.second.__sklearn_is_fitted__()
    def fit_branches(mut self, X: NDArray[Self.dtype], y: Optional[NDArray[Self.dtype]],
                     transform: Bool, weights: UnionWeights, verbose: Bool,
                     mut outputs: List[NDArray[Self.dtype]]) raises:
        self.first.fit_branches(X, y, transform, weights, verbose, outputs)
        self.second.fit_branches(rebind[NDArray[Self.B.dtype]](X), rebind[Optional[NDArray[Self.B.dtype]]](y), transform, weights, verbose,
                                 rebind[List[NDArray[Self.B.dtype]]](outputs))
    def transform_branches(self, X: NDArray[Self.dtype], weights: UnionWeights,
                           mut outputs: List[NDArray[Self.dtype]]) raises:
        self.first.transform_branches(X, weights, outputs)
        self.second.transform_branches(rebind[NDArray[Self.B.dtype]](X), weights, rebind[List[NDArray[Self.B.dtype]]](outputs))
    def feature_names(self, input_features: Optional[List[String]], prefix: Bool) raises -> List[String]:
        var result = self.first.feature_names(input_features, prefix)
        _append_names(result, self.second.feature_names(input_features, prefix))
        return result^


@fieldwise_init
struct FeatureUnionParams[P: Copyable & Deinitable](Copyable, Movable):
    var transformer_list: Self.P
    var n_jobs: Optional[Int]
    var transformer_weights: UnionWeights
    var verbose: Bool
    var verbose_feature_names_out: Bool


def _validate_union(names: List[String], n_jobs: Optional[Int], weights: UnionWeights) raises:
    if len(names) == 0:
        raise Error("ValueError: FeatureUnion needs at least one declared branch")
    validate_names(names, ["transformer_list", "n_jobs", "transformer_weights", "verbose", "verbose_feature_names_out"])
    if n_jobs and n_jobs.value() != 1:
        raise Error("NotImplementedError: native FeatureUnion supports n_jobs=None or 1")
    if weights:
        for name in weights.value():
            var found = False
            for candidate in names:
                found = found or name == candidate
            if not found:
                raise Error("ValueError: weight names must belong to transformer_list")


struct FeatureUnion[B: UnionBranches](TransformerMixin, FitTransformer, UnsupervisedEstimator, UnionFeatureNames, UnionInputFeatures):
    comptime Input = NDArray[Self.B.dtype]
    comptime Target = Self.Input
    comptime Transformed = Self.Input
    comptime Params = FeatureUnionParams[Self.B.Params]
    var transformer_list: Self.B
    var n_jobs: Optional[Int]
    var transformer_weights: UnionWeights
    var verbose: Bool
    var verbose_feature_names_out: Bool

    def __init__(out self, var transformer_list: Self.B, *, n_jobs: Optional[Int] = None,
                 transformer_weights: UnionWeights = None, verbose: Bool = False,
                 verbose_feature_names_out: Bool = True) raises:
        comptime assert Self.B.dtype == DType.float32 or Self.B.dtype == DType.float64, "FeatureUnion requires Float32 or Float64"
        _validate_union(transformer_list.names(), n_jobs, transformer_weights)
        self.transformer_list = transformer_list^
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights.copy()
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out
    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.transformer_list.get_params(deep), self.n_jobs, self.transformer_weights.copy(), self.verbose, self.verbose_feature_names_out)
    def set_params(mut self, params: Self.Params) raises:
        _validate_union(Self.B.param_names(params.transformer_list), params.n_jobs, params.transformer_weights)
        self.transformer_list.set_params(params.transformer_list)
        self.n_jobs = params.n_jobs
        self.transformer_weights = params.transformer_weights.copy()
        self.verbose = params.verbose
        self.verbose_feature_names_out = params.verbose_feature_names_out
    def __sklearn_clone__(self) raises -> Self:
        return Self(clone(self.transformer_list), n_jobs=self.n_jobs, transformer_weights=self.transformer_weights,
                    verbose=self.verbose, verbose_feature_names_out=self.verbose_feature_names_out)
    def __sklearn_is_fitted__(self) -> Bool:
        return self.transformer_list.__sklearn_is_fitted__()
    def n_features_in_(self) raises -> Int:
        return self.transformer_list.input_feature_count()
    def _validate(self, X: Self.Input) raises:
        _validate_union(self.transformer_list.names(), self.n_jobs, self.transformer_weights)
        if X.ndim != 2:
            raise Error("ValueError: dense FeatureUnion input must be rank 2")
    def _hstack(self, outputs: List[Self.Input], rows: Int) raises -> Self.Transformed:
        for output in outputs:
            if output.ndim != 2:
                raise Error("ValueError: union transformer must return a rank-2 array")
            check_union_rows(rows, output.shape[0])
        if len(outputs) == 0:
            return zeros[Self.B.dtype](Shape(rows, 0))
        if len(outputs) == 1:
            return outputs[0].copy()
        return _concatenate_list(outputs, axis=1)
    def fit(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises:
        self._validate(X)
        var outputs = List[Self.Input]()
        self.transformer_list.fit_branches(X, y, False, self.transformer_weights, self.verbose, outputs)
    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self._validate(X)
        var outputs = List[Self.Input]()
        self.transformer_list.fit_branches(X, y, True, self.transformer_weights, self.verbose, outputs)
        return self._hstack(outputs, X.shape[0])
    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._validate(X)
        var outputs = List[Self.Input]()
        self.transformer_list.transform_branches(X, self.transformer_weights, outputs)
        return self._hstack(outputs, X.shape[0])
    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        var result = self.transformer_list.feature_names(input_features, self.verbose_feature_names_out)
        if not self.verbose_feature_names_out:
            if self.transformer_list.active_count() == 0:
                raise Error("ValueError: no feature names to concatenate")
            for i in range(len(result)):
                for j in range(i):
                    if result[i] == result[j]:
                        raise Error("ValueError: output feature names are not unique; enable verbose_feature_names_out")
        return result^
