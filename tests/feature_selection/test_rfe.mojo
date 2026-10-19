import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from std.math import nan, inf
from moxie.base import clone, Predictor, SupervisedScorer, RegressorMixin, ProbabilityPredictor, LogProbabilityPredictor, DecisionFunction
from moxie.linear_model import Lasso, ElasticNet
from moxie.feature_selection import RFE, FeatureCount, LassoImportance, ElasticNetImportance, SelectorMixin, SelectionState, SelectionEstimator, SelectionClassLabels

@fieldwise_init
struct MaskSelector[dt: DType](SelectorMixin, Copyable, Movable):
    comptime dtype = Self.dt
    comptime Target = nm.NDArray[Self.dtype]
    var state: SelectionState
    var allow_nan: Bool
    def _selection_state(self) raises -> SelectionState:
        self.state.check()
        return self.state.copy()
    def _allows_nan(self) -> Bool:
        return self.allow_nan


def mixin[dtype: DType]() raises:
    var selector = MaskSelector[dtype](SelectionState(), False)
    var X = nm.zeros[dtype](nm.Shape(2, 3))
    for i in range(6):
        X.unsafe_set(i, SIMD[dtype, 1](i))
    with assert_raises():
        _ = selector.get_support()
    with assert_raises():
        _ = selector.transform(X)
    selector.state.commit([True, False, True], [String("a"), String("b"), String("c")])
    assert_equal(selector.get_feature_names_out()[1], "c")
    with assert_raises():
        _ = selector.get_feature_names_out([String("a"), String("bad"), String("c")])
    var indices = selector.get_support(True)[List[Int]].copy()
    indices[0] = 2
    assert_equal(selector.get_support(True)[List[Int]][0], 0)
    var view = X.view_with_layout(nm.Shape(2, 3), nm.NDArrayStrides(-3, -1), 5)
    var reduced = selector.transform(view)
    assert_equal(reduced.unsafe_get(0), 5)
    assert_equal(reduced.unsafe_get(3), 0)
    var expanded = selector.inverse_transform(reduced)
    assert_equal(expanded.unsafe_get(1), 0)
    assert_equal(expanded.unsafe_get(2), 3)
    reduced.unsafe_set(0, 20)
    assert_equal(X.unsafe_get(5), 5)
    with assert_raises():
        _ = selector.transform(nm.zeros[dtype](nm.Shape(2, 2)))
    with assert_raises():
        _ = selector.inverse_transform(X)
    X.unsafe_set(0, nan[dtype]())
    with assert_raises():
        _ = selector.transform(X)
    selector.allow_nan = True
    _ = selector.transform(X)
    X.unsafe_set(0, inf[dtype]())
    _ = selector.transform(X)
    with assert_raises():
        _ = selector.inverse_transform(selector.transform(X))
    selector.state.commit([False, False, False])
    assert_equal(selector.transform(X).shape[1], 0)
    with assert_raises():
        _ = selector.inverse_transform(nm.zeros[dtype](nm.Shape(2, 0)))


def exercise[dtype: DType]() raises:
    comptime assert conforms_to(RFE[LassoImportance[dtype]], RegressorMixin)
    comptime assert conforms_to(RFE[LassoImportance[dtype]], Predictor)
    comptime assert conforms_to(RFE[LassoImportance[dtype]], SupervisedScorer)
    comptime assert not conforms_to(RFE[LassoImportance[dtype]], ProbabilityPredictor)
    var X = nm.zeros[dtype](nm.Shape(8, 4))
    var y = nm.zeros[dtype](nm.Shape(8))
    var beta: List[Float64] = [6, -3, 0.5, 0]
    for i in range(8):
        X.unsafe_set(i * 4 + i % 4, 2)
        y.unsafe_set(i, SIMD[dtype, 1](2 * beta[i % 4]))
    var selector = RFE(LassoImportance(Lasso[dtype](alpha=0.1, fit_intercept=False, tol=1e-7)))
    with assert_raises():
        _ = selector.ranking_()
    with assert_raises():
        _ = selector.predict(X)
    selector.fit_named(X, y, [String("a"), String("b"), String("c"), String("d")])
    assert_equal(selector.n_features_(), 2)
    assert_equal(selector.n_features_in_(), 4)
    assert_equal(selector.get_feature_names_out()[1], "b")
    assert_false(selector.estimator.__sklearn_is_fitted__())
    var ranks = selector.ranking_()
    assert_equal(ranks[0], 1)
    assert_equal(ranks[1], 1)
    assert_equal(ranks[2], 2)
    assert_equal(ranks[3], 3)
    ranks[0] = 99
    assert_equal(selector.ranking_()[0], 1)
    var fitted = selector.estimator_()
    assert_equal(fitted.n_features(), 2)
    assert_almost_equal(Float64(fitted.model.coef_.unsafe_get(0)), 5.9, atol=1e-5)
    var expected = fitted.predict(selector.transform(X))
    var predicted = selector.predict(X)
    for i in range(8):
        assert_equal(predicted.unsafe_get(i), expected.unsafe_get(i))
    var weights = nm.ones[dtype](nm.Shape(8))
    weights.unsafe_set(0, 3)
    assert_equal(selector.score(X, y, weights.copy()), fitted.score(selector.transform(X), y, weights.copy()))
    fitted.model.coef_.unsafe_set(0, 99)
    assert_almost_equal(Float64(selector.estimator_().model.coef_.unsafe_get(0)), 5.9, atol=1e-5)
    var fresh = clone(selector)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_false(fresh.estimator.__sklearn_is_fitted__())
    var params = selector.get_params()
    params.n_features_to_select = FeatureCount(1)
    params.step = FeatureCount(0.5)
    params.estimator.model.alpha = 0.2
    assert_equal(selector.estimator.model.alpha, 0.1)
    _ = selector.set_params(params).fit(X, y, sample_weight=weights)
    assert_equal(selector.n_features_(), 1)
    assert_equal(selector.ranking_()[3], 3)
    assert_equal(selector.ranking_()[2], 3)
    assert_equal(selector.ranking_()[1], 2)
    var direct = Lasso[dtype](alpha=0.2, fit_intercept=False, tol=1e-7)
    _ = direct.fit(selector.transform(X), y, sample_weight=weights)
    assert_equal(selector.estimator_().model.coef_.unsafe_get(0), direct.coef_.unsafe_get(0))
    with assert_raises():
        _ = selector.fit(X, nm.zeros[dtype](nm.Shape(3)))
    assert_equal(selector.n_features_(), 1)
    with assert_raises():
        selector.fit_named(X, y, [String("bad")])
    assert_equal(selector.n_features_(), 1)
    _ = selector.set_params(n_features_to_select=FeatureCount(0.5))
    _ = selector.set_params(step=FeatureCount(1))
    _ = selector.fit(X, nm.zeros[dtype](nm.Shape(8)))
    # All zero importances: stable ties remove the first columns first.
    assert_equal(selector.get_support(True)[List[Int]][0], 2)
    assert_equal(selector.get_support(True)[List[Int]][1], 3)
    _ = selector.set_params(n_features_to_select=FeatureCount(99))
    _ = selector.fit(X, y)
    assert_equal(selector.n_features_(), 4)
    assert_equal(selector.ranking_()[0], 1)
    _ = selector.set_params(n_features_to_select=FeatureCount(1.0))
    assert_equal(selector.fit_transform(X, y.copy()).shape[1], 4)
    with assert_raises():
        _ = selector.fit_transform(X)
    _ = selector.set_params(n_features_to_select=FeatureCount(0.01))
    with assert_raises():
        _ = selector.fit(X, y)
    assert_equal(selector.n_features_(), 4)
    _ = selector.set_params(n_features_to_select=None)
    for bad in [0, -1]:
        _ = selector.set_params(step=FeatureCount(bad))
        with assert_raises():
            _ = selector.fit(X, y)
    for bad in [Float64(0), Float64(1), Float64(1.5), nan[DType.float64](), inf[DType.float64]()]:
        _ = selector.set_params(step=FeatureCount(bad))
        with assert_raises():
            _ = selector.fit(X, y)
    _ = selector.set_params(step=FeatureCount(1))
    for bad in [0, -1]:
        _ = selector.set_params(n_features_to_select=FeatureCount(bad))
        with assert_raises():
            _ = selector.fit(X, y)
    for bad in [Float64(0), Float64(-0.1), Float64(1.1), nan[DType.float64](), inf[DType.float64]()]:
        _ = selector.set_params(n_features_to_select=FeatureCount(bad))
        with assert_raises():
            _ = selector.fit(X, y)
    _ = selector.set_params(n_features_to_select=None)
    with assert_raises():
        _ = selector.fit(nm.zeros[dtype](nm.Shape(8, 1)), y)
    with assert_raises():
        _ = selector.fit(nm.zeros[dtype](nm.Shape(8)), y)
    selector.verbose = -1
    with assert_raises():
        _ = selector.fit(X, y)
    selector.verbose = 0
    selector.importance_getter = "missing.coef_"
    with assert_raises():
        _ = selector.fit(X, y)
    selector.importance_getter = "coef_"
    with assert_raises():
        _ = selector.fit(nm.zeros[dtype](nm.Shape(0, 4)), nm.zeros[dtype](nm.Shape(0)))
    var reversed = X.view_with_layout(nm.Shape(8, 4), nm.NDArrayStrides(4, -1), 3)
    _ = selector.fit(reversed, y)
    assert_equal(selector.get_support(True)[List[Int]][0], 2)
    assert_equal(selector.get_support(True)[List[Int]][1], 3)
    var multi = nm.zeros[dtype](nm.Shape(8, 2))
    for i in range(8):
        multi.unsafe_set(i * 2, y.unsafe_get(i))
        multi.unsafe_set(i * 2 + 1, -2 * y.unsafe_get(i))
    var elastic = RFE(ElasticNetImportance(ElasticNet[dtype](alpha=0.1, fit_intercept=False)))
    _ = elastic.fit(X, multi)
    assert_equal(elastic.estimator_().model.coef_.ndim, 2)
    assert_equal(elastic.predict(X).shape[1], 2)
    assert_equal(elastic.get_support(True)[List[Int]][0], 0)
    X.unsafe_set(0, nan[dtype]())
    with assert_raises():
        _ = selector.fit(X, y)
    with assert_raises():
        _ = selector.transform(X)



struct RoutingProbe[capabilities: Bool](
    SelectionEstimator,
    SelectionClassLabels where capabilities,
    LogProbabilityPredictor where capabilities,
    DecisionFunction where capabilities,
):
    """Test-only routing spy; never used as an implemented statistical estimator."""
    comptime dtype = DType.float64
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Prediction = Self.Input
    comptime Probabilities = Self.Input
    comptime Decision = Self.Input
    comptime Classes = List[Int]
    comptime Params = Int
    var width: Int
    var invalid_importance: Int
    def __init__(out self, invalid_importance: Int = 0):
        self.width = 0
        self.invalid_importance = invalid_importance
    def get_params(self, deep: Bool = True) -> Int:
        return self.invalid_importance
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Int) raises -> ref[self] Self:
        self.invalid_importance = params
        return self

    def __sklearn_clone__(self) -> Self:
        return Self(self.invalid_importance)
    def __sklearn_is_fitted__(self) -> Bool:
        return self.width != 0
    def fit_selection(mut self, X: Self.Input, y: Optional[Self.Target], sample_weight: Optional[Self.Input]) raises:
        assert_equal(self.width, 0)  # Every iteration must use a fresh clone.
        self.width = X.shape[1]
    def n_features(self) -> Int:
        return self.width
    def uses_l1_threshold(self) -> Bool:
        return False
    def feature_importances(self) raises -> Self.Input:
        var result = nm.ones[Self.dtype](nm.Shape(self.width + Int(self.invalid_importance == 1)))
        if self.invalid_importance == 3:
            for j in range(self.width):
                result.unsafe_set(j, Float64(j if self.width == 4 else self.width - j))
        if self.invalid_importance == 2:
            result.unsafe_set(0, nan[Self.dtype]())
        return result^
    def predict(self, X: Self.Input) raises -> Self.Prediction:
        assert_equal(X.shape[1], self.width)
        return X.copy()
    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return self.predict(X)
    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return self.predict(X)
    def decision_function(self, X: Self.Input) raises -> Self.Decision:
        return self.predict(X)
    def classes_(self) -> Self.Classes:
        return [42, 7]


def routing() raises:
    comptime assert not conforms_to(RFE[RoutingProbe[False]], Predictor)
    comptime assert conforms_to(RFE[RoutingProbe[True]], LogProbabilityPredictor)
    comptime assert conforms_to(RFE[RoutingProbe[True]], DecisionFunction)
    var X = nm.ones[DType.float64](nm.Shape(3, 4))
    var y = nm.ones[DType.float64](nm.Shape(3))
    var selector = RFE(RoutingProbe[True]())
    with assert_raises():
        _ = selector.classes_()
    with assert_raises():
        _ = selector.predict_proba(X)
    with assert_raises():
        _ = selector.predict_log_proba(X)
    with assert_raises():
        _ = selector.decision_function(X)
    _ = selector.fit(X, y)
    assert_equal(selector.classes_()[0], 42)
    assert_equal(selector.classes_()[1], 7)
    assert_equal(selector.predict_proba(X).shape[1], 2)
    assert_equal(selector.predict_log_proba(X).shape[1], 2)
    assert_equal(selector.decision_function(X).shape[1], 2)
    var transform_only = RFE(RoutingProbe[False]())
    _ = transform_only.fit(X, y)
    assert_equal(transform_only.transform(X).shape[1], 2)
    var adaptive = RFE(RoutingProbe[False](3))
    _ = adaptive.fit(X, y)
    assert_equal(adaptive.get_support(True)[List[Int]][0], 1)
    assert_equal(adaptive.get_support(True)[List[Int]][1], 2)
    assert_equal(adaptive.ranking_()[0], 3)
    assert_equal(adaptive.ranking_()[3], 2)
    var odd = RFE(RoutingProbe[False]())
    _ = odd.fit(nm.ones[DType.float64](nm.Shape(3, 5)), y)
    assert_equal(odd.n_features_(), 2)
    assert_equal(odd.get_support(True)[List[Int]][0], 3)
    for bad in [1, 2]:
        var invalid = RFE(RoutingProbe[False](bad))
        with assert_raises():
            _ = invalid.fit(X, y)
        assert_false(invalid.__sklearn_is_fitted__())

def main() raises:
    routing()
    mixin[DType.float32]()
    mixin[DType.float64]()
    exercise[DType.float32]()
    exercise[DType.float64]()
    print("SelectorMixin and RFE contracts passed")
