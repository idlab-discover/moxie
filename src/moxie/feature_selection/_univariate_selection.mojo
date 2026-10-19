"""Typed dense univariate selectors; sklearn names and threshold semantics."""
import numojo as nm
from std.utils import Variant
from moxie._labels import StringTarget
from moxie.base import FitTransformer
from moxie.feature_selection._base import UnsupervisedSelector
from moxie.feature_selection._selection import SelectionState, SupportResult
from moxie.feature_selection._contracts import ScoreResult, UnivariateScorer
from moxie.feature_selection._univariate_policy import UnivariatePolicy
from moxie.feature_selection._univariate_scores import chi2, f_classif, f_oneway, f_regression, r_regression, _validate_x

trait ConfiguredScorer(UnivariateScorer, Copyable, Movable):
    """Native callable configuration; clone_config must discard learned state."""
    def __init__(out self):
        ...
    def clone_config(self) -> Self:
        ...

struct BuiltinScorer[dt: DType = DType.float64, target_dt: DType = dt, target_type: Copyable = nm.NDArray[target_dt]](ConfiguredScorer):
    comptime dtype = Self.dt
    comptime score_dtype = DType.float64
    comptime Target = Self.target_type
    var name: String
    var center: Bool
    var force_finite: Bool

    def __init__(out self):
        self.name = "f_classif"
        self.center = True
        self.force_finite = True

    def __init__(out self, name: String, *, center: Bool = True, force_finite: Bool = True):
        self.name = name
        self.center = center
        self.force_finite = force_finite

    def clone_config(self) -> Self:
        return Self(self.name, center=self.center, force_finite=self.force_finite)

    def score(self, x: nm.NDArray[Self.dtype], y: Optional[Self.Target]) raises -> ScoreResult[Self.score_dtype]:
        comptime assert Self.Target == StringTarget or Self.Target == nm.NDArray[Self.target_dt]
        if not y:
            raise Error("ValueError: selected scorer requires y")
        comptime if Self.Target == StringTarget:
            ref target = rebind[StringTarget](y.value())
            if self.name == "f_classif":
                return f_classif(x, target)
            if self.name == "chi2":
                return chi2(x, target)
            raise Error("ValueError: string targets require a classification scorer")
        else:
            ref target = rebind[nm.NDArray[Self.target_dt]](y.value())
            if self.name == "f_classif":
                return f_classif(x, target)
            if self.name == "chi2":
                return chi2(x, target)
            if self.name == "r_regression":
                var values = r_regression(x, target, center=self.center, force_finite=self.force_finite)
                return ScoreResult[Self.score_dtype](values^)
            if self.name == "f_regression":
                return f_regression(x, target, center=self.center, force_finite=self.force_finite)
            raise Error("ValueError: unknown built-in scorer")

comptime SelectionParameter = Variant[Int, Float64, String]

def _policy[dt: DType](mode: String, parameter: SelectionParameter) raises -> UnivariatePolicy[dt]:
    if parameter.isa[String]():
        if parameter[String] != "all":
            raise Error("ValueError: only the string all is supported")
        return UnivariatePolicy[dt](mode, 0, True)
    if parameter.isa[Int]():
        return UnivariatePolicy[dt](mode, Float64(parameter[Int]))
    return UnivariatePolicy[dt](mode, parameter[Float64])

@fieldwise_init
struct SelectorParams[S: ConfiguredScorer & Deinitable](Copyable, Movable):
    var score_func: Self.S
    var mode: String
    var param: SelectionParameter

struct _FilterState[dt: DType](Movable):
    var selection: SelectionState
    var result: Optional[ScoreResult[Self.dt]]
    var names: List[String]

    def __init__(out self):
        self.selection = SelectionState()
        self.result = None
        self.names = List[String]()

    def commit(mut self, var result: ScoreResult[Self.dt], features: Int, names: List[String]) raises:
        if result.scores.shape[0] != features:
            raise Error("ValueError: scorer returned wrong feature count")
        var mask = List[Bool]()
        for _ in range(features):
            mask.append(True)
        self.selection.commit(mask, names)
        self.result = result^
        self.names = names.copy()

    def selected(self, policy: UnivariatePolicy[Self.dt]) raises -> SelectionState:
        self.selection.check()
        var p: Optional[nm.NDArray[Self.dt]] = None
        if self.result.value().pvalues:
            p = self.result.value().pvalues.value().copy()
        var result = ScoreResult[Self.dt](self.result.value().scores.copy(), p^)
        var selected = self.selection.copy()
        selected.commit(policy.select(result), self.names)
        return selected^

struct SelectKBest[S: ConfiguredScorer & Deinitable = BuiltinScorer[DType.float64]](UnsupervisedSelector, FitTransformer):
    comptime Input = nm.NDArray[Self.S.dtype]
    comptime Target = Self.S.Target
    comptime Transformed = Self.Input
    comptime Support = SupportResult
    comptime Params = SelectorParams[Self.S]
    var score_func: Self.S
    var k: SelectionParameter
    var _state: _FilterState[Self.S.score_dtype]

    def __init__(out self, var score_func: Self.S = Self.S(), *, k: SelectionParameter = SelectionParameter(10)):
        self.score_func = score_func^
        self.k = k.copy()
        self._state = _FilterState[Self.S.score_dtype]()

    def _policy(self) raises -> UnivariatePolicy[Self.S.score_dtype]:
        return _policy[Self.S.score_dtype]("k_best", self.k.copy())

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self.fit_named(X, y, List[String]())
        return self

    def fit_named(mut self, X: Self.Input, y: Optional[Self.Target], names: List[String]) raises:
        _validate_x(X)
        self._policy().validate()
        var result = self.score_func.score(X, y)
        self._state.commit(result^, X.shape[1], names)

    def get_support(self, indices: Bool = False) raises -> Self.Support:
        return self._state.selected(self._policy()).get_support(indices)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        _validate_x(X)
        return self._state.selected(self._policy()).transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        _validate_x(X)
        return self._state.selected(self._policy()).inverse_transform(X)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        return self._state.selected(self._policy()).get_feature_names_out(input_features)

    def scores_(self) raises -> nm.NDArray[Self.S.score_dtype]:
        self._state.selection.check()
        return self._state.result.value().scores.copy()

    def pvalues_(self) raises -> Optional[nm.NDArray[Self.S.score_dtype]]:
        self._state.selection.check()
        if self._state.result.value().pvalues:
            return self._state.result.value().pvalues.value().copy()
        return None

    def n_features_in_(self) raises -> Int:
        return self._state.selection.n_features()

    def feature_names_in_(self) raises -> List[String]:
        self._state.selection.check()
        return self._state.names.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.score_func.clone_config(), "k_best", self.k.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        var policy = _policy[Self.S.score_dtype](params.mode, params.param)
        policy.validate()
        if params.mode != "k_best":
            raise Error("ValueError: selector mode is fixed")
        self.score_func = params.score_func.clone_config()
        self.k = params.param.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, k: SelectionParameter) raises -> ref[self] Self:
        var params = self.get_params()
        params.param = k.copy()
        _ = self.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.score_func.clone_config(), k=self.k.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.selection.is_fitted()

struct SelectPercentile[S: ConfiguredScorer & Deinitable = BuiltinScorer[DType.float64]](UnsupervisedSelector, FitTransformer):
    comptime Input = nm.NDArray[Self.S.dtype]
    comptime Target = Self.S.Target
    comptime Transformed = Self.Input
    comptime Support = SupportResult
    comptime Params = SelectorParams[Self.S]
    var score_func: Self.S
    var percentile: Float64
    var _state: _FilterState[Self.S.score_dtype]

    def __init__(out self, var score_func: Self.S = Self.S(), *, percentile: Float64 = 10):
        self.score_func = score_func^
        self.percentile = percentile
        self._state = _FilterState[Self.S.score_dtype]()

    def _policy(self) raises -> UnivariatePolicy[Self.S.score_dtype]:
        return _policy[Self.S.score_dtype]("percentile", SelectionParameter(self.percentile))

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self.fit_named(X, y, List[String]())
        return self

    def fit_named(mut self, X: Self.Input, y: Optional[Self.Target], names: List[String]) raises:
        _validate_x(X)
        self._policy().validate()
        var result = self.score_func.score(X, y)
        self._state.commit(result^, X.shape[1], names)

    def get_support(self, indices: Bool = False) raises -> Self.Support:
        return self._state.selected(self._policy()).get_support(indices)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        _validate_x(X)
        return self._state.selected(self._policy()).transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        _validate_x(X)
        return self._state.selected(self._policy()).inverse_transform(X)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        return self._state.selected(self._policy()).get_feature_names_out(input_features)

    def scores_(self) raises -> nm.NDArray[Self.S.score_dtype]:
        self._state.selection.check()
        return self._state.result.value().scores.copy()

    def pvalues_(self) raises -> Optional[nm.NDArray[Self.S.score_dtype]]:
        self._state.selection.check()
        if self._state.result.value().pvalues:
            return self._state.result.value().pvalues.value().copy()
        return None

    def n_features_in_(self) raises -> Int:
        return self._state.selection.n_features()

    def feature_names_in_(self) raises -> List[String]:
        self._state.selection.check()
        return self._state.names.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.score_func.clone_config(), "percentile", SelectionParameter(self.percentile))

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        var policy = _policy[Self.S.score_dtype](params.mode, params.param)
        policy.validate()
        if params.mode != "percentile":
            raise Error("ValueError: selector mode is fixed")
        self.score_func = params.score_func.clone_config()
        self.percentile = policy.param
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, percentile: Float64) raises -> ref[self] Self:
        var params = self.get_params()
        params.param = SelectionParameter(percentile)
        _ = self.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.score_func.clone_config(), percentile=self.percentile)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.selection.is_fitted()

struct SelectFpr[S: ConfiguredScorer & Deinitable = BuiltinScorer[DType.float64]](UnsupervisedSelector, FitTransformer):
    comptime Input = nm.NDArray[Self.S.dtype]
    comptime Target = Self.S.Target
    comptime Transformed = Self.Input
    comptime Support = SupportResult
    comptime Params = SelectorParams[Self.S]
    var score_func: Self.S
    var alpha: Float64
    var _state: _FilterState[Self.S.score_dtype]

    def __init__(out self, var score_func: Self.S = Self.S(), *, alpha: Float64 = 0.05):
        self.score_func = score_func^
        self.alpha = alpha
        self._state = _FilterState[Self.S.score_dtype]()

    def _policy(self) raises -> UnivariatePolicy[Self.S.score_dtype]:
        return _policy[Self.S.score_dtype]("fpr", SelectionParameter(self.alpha))

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self.fit_named(X, y, List[String]())
        return self

    def fit_named(mut self, X: Self.Input, y: Optional[Self.Target], names: List[String]) raises:
        _validate_x(X)
        self._policy().validate()
        var result = self.score_func.score(X, y)
        self._state.commit(result^, X.shape[1], names)

    def get_support(self, indices: Bool = False) raises -> Self.Support:
        return self._state.selected(self._policy()).get_support(indices)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        _validate_x(X)
        return self._state.selected(self._policy()).transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        _validate_x(X)
        return self._state.selected(self._policy()).inverse_transform(X)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        return self._state.selected(self._policy()).get_feature_names_out(input_features)

    def scores_(self) raises -> nm.NDArray[Self.S.score_dtype]:
        self._state.selection.check()
        return self._state.result.value().scores.copy()

    def pvalues_(self) raises -> Optional[nm.NDArray[Self.S.score_dtype]]:
        self._state.selection.check()
        if self._state.result.value().pvalues:
            return self._state.result.value().pvalues.value().copy()
        return None

    def n_features_in_(self) raises -> Int:
        return self._state.selection.n_features()

    def feature_names_in_(self) raises -> List[String]:
        self._state.selection.check()
        return self._state.names.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.score_func.clone_config(), "fpr", SelectionParameter(self.alpha))

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        var policy = _policy[Self.S.score_dtype](params.mode, params.param)
        policy.validate()
        if params.mode != "fpr":
            raise Error("ValueError: selector mode is fixed")
        self.score_func = params.score_func.clone_config()
        self.alpha = policy.param
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, alpha: Float64) raises -> ref[self] Self:
        var params = self.get_params()
        params.param = SelectionParameter(alpha)
        _ = self.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.score_func.clone_config(), alpha=self.alpha)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.selection.is_fitted()

struct SelectFdr[S: ConfiguredScorer & Deinitable = BuiltinScorer[DType.float64]](UnsupervisedSelector, FitTransformer):
    comptime Input = nm.NDArray[Self.S.dtype]
    comptime Target = Self.S.Target
    comptime Transformed = Self.Input
    comptime Support = SupportResult
    comptime Params = SelectorParams[Self.S]
    var score_func: Self.S
    var alpha: Float64
    var _state: _FilterState[Self.S.score_dtype]

    def __init__(out self, var score_func: Self.S = Self.S(), *, alpha: Float64 = 0.05):
        self.score_func = score_func^
        self.alpha = alpha
        self._state = _FilterState[Self.S.score_dtype]()

    def _policy(self) raises -> UnivariatePolicy[Self.S.score_dtype]:
        return _policy[Self.S.score_dtype]("fdr", SelectionParameter(self.alpha))

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self.fit_named(X, y, List[String]())
        return self

    def fit_named(mut self, X: Self.Input, y: Optional[Self.Target], names: List[String]) raises:
        _validate_x(X)
        self._policy().validate()
        var result = self.score_func.score(X, y)
        self._state.commit(result^, X.shape[1], names)

    def get_support(self, indices: Bool = False) raises -> Self.Support:
        return self._state.selected(self._policy()).get_support(indices)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        _validate_x(X)
        return self._state.selected(self._policy()).transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        _validate_x(X)
        return self._state.selected(self._policy()).inverse_transform(X)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        return self._state.selected(self._policy()).get_feature_names_out(input_features)

    def scores_(self) raises -> nm.NDArray[Self.S.score_dtype]:
        self._state.selection.check()
        return self._state.result.value().scores.copy()

    def pvalues_(self) raises -> Optional[nm.NDArray[Self.S.score_dtype]]:
        self._state.selection.check()
        if self._state.result.value().pvalues:
            return self._state.result.value().pvalues.value().copy()
        return None

    def n_features_in_(self) raises -> Int:
        return self._state.selection.n_features()

    def feature_names_in_(self) raises -> List[String]:
        self._state.selection.check()
        return self._state.names.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.score_func.clone_config(), "fdr", SelectionParameter(self.alpha))

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        var policy = _policy[Self.S.score_dtype](params.mode, params.param)
        policy.validate()
        if params.mode != "fdr":
            raise Error("ValueError: selector mode is fixed")
        self.score_func = params.score_func.clone_config()
        self.alpha = policy.param
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, alpha: Float64) raises -> ref[self] Self:
        var params = self.get_params()
        params.param = SelectionParameter(alpha)
        _ = self.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.score_func.clone_config(), alpha=self.alpha)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.selection.is_fitted()

struct SelectFwe[S: ConfiguredScorer & Deinitable = BuiltinScorer[DType.float64]](UnsupervisedSelector, FitTransformer):
    comptime Input = nm.NDArray[Self.S.dtype]
    comptime Target = Self.S.Target
    comptime Transformed = Self.Input
    comptime Support = SupportResult
    comptime Params = SelectorParams[Self.S]
    var score_func: Self.S
    var alpha: Float64
    var _state: _FilterState[Self.S.score_dtype]

    def __init__(out self, var score_func: Self.S = Self.S(), *, alpha: Float64 = 0.05):
        self.score_func = score_func^
        self.alpha = alpha
        self._state = _FilterState[Self.S.score_dtype]()

    def _policy(self) raises -> UnivariatePolicy[Self.S.score_dtype]:
        return _policy[Self.S.score_dtype]("fwe", SelectionParameter(self.alpha))

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self.fit_named(X, y, List[String]())
        return self

    def fit_named(mut self, X: Self.Input, y: Optional[Self.Target], names: List[String]) raises:
        _validate_x(X)
        self._policy().validate()
        var result = self.score_func.score(X, y)
        self._state.commit(result^, X.shape[1], names)

    def get_support(self, indices: Bool = False) raises -> Self.Support:
        return self._state.selected(self._policy()).get_support(indices)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        _validate_x(X)
        return self._state.selected(self._policy()).transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        _validate_x(X)
        return self._state.selected(self._policy()).inverse_transform(X)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        return self._state.selected(self._policy()).get_feature_names_out(input_features)

    def scores_(self) raises -> nm.NDArray[Self.S.score_dtype]:
        self._state.selection.check()
        return self._state.result.value().scores.copy()

    def pvalues_(self) raises -> Optional[nm.NDArray[Self.S.score_dtype]]:
        self._state.selection.check()
        if self._state.result.value().pvalues:
            return self._state.result.value().pvalues.value().copy()
        return None

    def n_features_in_(self) raises -> Int:
        return self._state.selection.n_features()

    def feature_names_in_(self) raises -> List[String]:
        self._state.selection.check()
        return self._state.names.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.score_func.clone_config(), "fwe", SelectionParameter(self.alpha))

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        var policy = _policy[Self.S.score_dtype](params.mode, params.param)
        policy.validate()
        if params.mode != "fwe":
            raise Error("ValueError: selector mode is fixed")
        self.score_func = params.score_func.clone_config()
        self.alpha = policy.param
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, alpha: Float64) raises -> ref[self] Self:
        var params = self.get_params()
        params.param = SelectionParameter(alpha)
        _ = self.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.score_func.clone_config(), alpha=self.alpha)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.selection.is_fitted()

struct GenericUnivariateSelect[S: ConfiguredScorer & Deinitable = BuiltinScorer[DType.float64]](UnsupervisedSelector, FitTransformer):
    comptime Input = nm.NDArray[Self.S.dtype]
    comptime Target = Self.S.Target
    comptime Transformed = Self.Input
    comptime Support = SupportResult
    comptime Params = SelectorParams[Self.S]
    var score_func: Self.S
    var param: SelectionParameter
    var mode: String
    var _state: _FilterState[Self.S.score_dtype]

    def __init__(out self, var score_func: Self.S = Self.S(), *, param: SelectionParameter = SelectionParameter(Float64(1e-5)), mode: String = "percentile"):
        self.score_func = score_func^
        self.param = param.copy()
        self.mode = mode
        self._state = _FilterState[Self.S.score_dtype]()

    def _policy(self) raises -> UnivariatePolicy[Self.S.score_dtype]:
        return _policy[Self.S.score_dtype](self.mode, self.param.copy())

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self.fit_named(X, y, List[String]())
        return self

    def fit_named(mut self, X: Self.Input, y: Optional[Self.Target], names: List[String]) raises:
        _validate_x(X)
        self._policy().validate()
        var result = self.score_func.score(X, y)
        self._state.commit(result^, X.shape[1], names)

    def get_support(self, indices: Bool = False) raises -> Self.Support:
        return self._state.selected(self._policy()).get_support(indices)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        _validate_x(X)
        return self._state.selected(self._policy()).transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        _validate_x(X)
        return self._state.selected(self._policy()).inverse_transform(X)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: List[String] = List[String]()) raises -> List[String]:
        return self._state.selected(self._policy()).get_feature_names_out(input_features)

    def scores_(self) raises -> nm.NDArray[Self.S.score_dtype]:
        self._state.selection.check()
        return self._state.result.value().scores.copy()

    def pvalues_(self) raises -> Optional[nm.NDArray[Self.S.score_dtype]]:
        self._state.selection.check()
        if self._state.result.value().pvalues:
            return self._state.result.value().pvalues.value().copy()
        return None

    def n_features_in_(self) raises -> Int:
        return self._state.selection.n_features()

    def feature_names_in_(self) raises -> List[String]:
        self._state.selection.check()
        return self._state.names.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.score_func.clone_config(), self.mode, self.param.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        var policy = _policy[Self.S.score_dtype](params.mode, params.param)
        policy.validate()
        self.mode = params.mode
        self.score_func = params.score_func.clone_config()
        self.param = params.param.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, param: SelectionParameter) raises -> ref[self] Self:
        var params = self.get_params()
        params.param = param.copy()
        _ = self.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.score_func.clone_config(), param=self.param.copy(), mode=self.mode)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.selection.is_fitted()
