"""Elimination-path contracts; PathProbe is a lifecycle spy, not an estimator."""
import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from moxie.linear_model import Lasso
from moxie.feature_selection import RFE, FeatureCount, LassoImportance, SelectionState, SelectionEstimator
from moxie.feature_selection._rfe import _RFEStepObserver
from rfe_path import HeldOutPath, PathStep


struct PathProbe(SelectionEstimator):
    comptime dtype = DType.float64
    comptime Target = nm.NDArray[Self.dtype]
    comptime Params = Int
    var width: Int
    var fail_width: Int
    def __init__(out self, fail_width: Int = 0):
        self.width = 0
        self.fail_width = fail_width
    def get_params(self, deep: Bool = True) -> Int:
        return self.fail_width
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Int) raises -> ref[self] Self:
        self.fail_width = params
        return self

    def __sklearn_clone__(self) -> Self:
        return Self(self.fail_width)
    def __sklearn_is_fitted__(self) -> Bool:
        return self.width > 0
    def fit_selection(mut self, X: Self.Target, y: Optional[Self.Target], sample_weight: Optional[Self.Target]) raises:
        assert_equal(self.width, 0)
        if X.shape[1] == self.fail_width:
            raise Error("intentional fit failure")
        if sample_weight:
            assert_equal(sample_weight.value().shape[0], X.shape[0])
            assert_equal(sample_weight.value().unsafe_get(0), 3.0)
        self.width = X.shape[1]
    def n_features(self) -> Int:
        return self.width
    def uses_l1_threshold(self) -> Bool:
        return False
    def feature_importances(self) raises -> Self.Target:
        return nm.ones[Self.dtype](nm.Shape(self.width + Int(self.fail_width == -1)))


struct Recorder(_RFEStepObserver):
    comptime Estimator = PathProbe
    var steps: List[PathStep]
    var fail_at: Int
    var calls: Int
    def __init__(out self, fail_at: Int = -1):
        self.steps = List[PathStep]()
        self.fail_at = fail_at
        self.calls = 0
    def observe(mut self, estimator: PathProbe, selection: SelectionState, ranking: List[Int]) raises:
        assert_true(estimator.__sklearn_is_fitted__())
        assert_equal(estimator.n_features(), selection.selected_count())
        var support = selection.get_support()[List[Bool]].copy()
        for j in range(len(support)):
            assert_equal(ranking[j] == 1, support[j])
        self.calls += 1
        if len(self.steps) == self.fail_at:
            raise Error("intentional observer failure")
        self.steps.append(PathStep(selection.selected_count(), 0, support^, ranking.copy()))


def lifecycle() raises:
    var X = nm.ones[DType.float64](nm.Shape(3, 5))
    var y = nm.ones[DType.float64](nm.Shape(3))
    var weights = nm.ones[DType.float64](nm.Shape(3))
    weights.unsafe_set(0, 3)
    var selector = RFE(PathProbe(), n_features_to_select=FeatureCount(1), step=FeatureCount(0.4))
    var observer = Recorder()
    selector._fit_with_observer(X, y, observer, [String("a"), String("b"), String("c"), String("d"), String("e")], weights.copy())
    assert_equal(observer.calls, 3)
    assert_equal(len(observer.steps), 3)
    for i in range(3):
        assert_equal(observer.steps[i].n_features, 5 - 2 * i)
        for j in range(5):
            assert_equal(observer.steps[i].support[j], j >= 2 * i)
    assert_equal(observer.steps[0].ranking[0], 1)
    assert_equal(observer.steps[1].ranking[0], 2)
    assert_equal(observer.steps[2].ranking[0], 3)
    assert_equal(observer.steps[2].ranking[2], 2)
    assert_equal(observer.steps[2].ranking[4], 1)
    assert_equal(selector.get_feature_names_out()[0], "e")
    assert_false(selector.estimator.__sklearn_is_fitted__())
    # Caller snapshots are independent of other steps and learned state.
    observer.steps[0].support[0] = False
    observer.steps[2].ranking[4] = 99
    assert_true(observer.steps[2].support[4])
    assert_equal(selector.ranking_()[4], 1)
    var failed = Recorder(1)
    with assert_raises():
        selector._fit_with_observer(X, y, failed)
    assert_equal(failed.calls, 2)
    assert_equal(len(failed.steps), 1)
    assert_equal(selector.n_features_(), 1)
    assert_equal(selector.get_feature_names_out()[0], "e")
    # A failing observer on the final fit also leaves old learned state intact.
    var final_failure = Recorder(2)
    with assert_raises():
        selector._fit_with_observer(X, y, final_failure)
    assert_equal(final_failure.calls, 3)
    assert_equal(len(final_failure.steps), 2)
    assert_equal(selector.get_feature_names_out()[0], "e")
    for fail_width in [5, 3, 1]:
        selector.estimator.fail_width = fail_width
        var fit_failure = Recorder()
        with assert_raises():
            selector._fit_with_observer(X, y, fit_failure)
        assert_equal(fit_failure.calls, (5 - fail_width) // 2)
        assert_equal(selector.n_features_(), 1)
    # Observation precedes importance extraction and its validation failure.
    selector.estimator.fail_width = -1
    var bad_importance = Recorder()
    with assert_raises():
        selector._fit_with_observer(X, y, bad_importance)
    assert_equal(bad_importance.calls, 1)
    assert_equal(bad_importance.steps[0].n_features, 5)
    selector.estimator.fail_width = 0
    var invalid = Recorder()
    with assert_raises():
        selector._fit_with_observer(X, y, invalid, [String("wrong length")])
    assert_equal(invalid.calls, 0)
    # The observer owns/reset its own history, including across repeated fits.
    var repeated = Recorder()
    selector._fit_with_observer(X, y, repeated)
    selector._fit_with_observer(X, y, repeated)
    assert_equal(repeated.calls, 6)
    assert_equal(repeated.steps[3].n_features, 5)
    _ = selector.set_params(n_features_to_select=FeatureCount(99))
    var no_elimination = Recorder()
    selector._fit_with_observer(X, y, no_elimination)
    assert_equal(no_elimination.calls, 1)
    assert_equal(no_elimination.steps[0].n_features, 5)
    _ = selector.set_params(n_features_to_select=FeatureCount(2))
    _ = selector.set_params(step=FeatureCount(2))
    var clipped_step = Recorder()
    selector._fit_with_observer(X, y, clipped_step)
    assert_equal(clipped_step.steps[2].n_features, 2)
    # Ordinary fitting uses the same path without requiring scoring/prediction.
    _ = selector.fit(X, y)
    assert_equal(selector.n_features_(), 2)
    assert_equal(selector.ranking_()[0], clipped_step.steps[2].ranking[0])
    var unfitted = RFE(PathProbe())
    var failing = Recorder(0)
    with assert_raises():
        unfitted._fit_with_observer(X, y, failing)
    assert_false(unfitted.__sklearn_is_fitted__())


def real_score[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(8, 4))
    var y = nm.zeros[dtype](nm.Shape(8))
    var beta: List[Float64] = [6, -3, 0.5, 0]
    for i in range(8):
        X.unsafe_set(i * 4 + i % 4, 2)
        y.unsafe_set(i, SIMD[dtype, 1](2 * beta[i % 4]))
    var weights = nm.ones[dtype](nm.Shape(8))
    weights.unsafe_set(0, 3)
    var score_weights = nm.ones[dtype](nm.Shape(8))
    score_weights.unsafe_set(3, 4)
    var y_test = y.copy()
    for i in range(8):
        y_test.unsafe_set(i, y_test.unsafe_get(i) + 0.25)
    var selector = RFE(LassoImportance(Lasso[dtype](alpha=0.1, fit_intercept=False)), n_features_to_select=FeatureCount(1), step=FeatureCount(2))
    var observer = HeldOutPath[LassoImportance[dtype]](X, y_test, score_weights.copy())
    selector._fit_with_observer(X, y, observer, sample_weight=weights.copy())
    assert_equal(len(observer.steps), 3)
    for i in range(3):
        var direct = RFE(LassoImportance(Lasso[dtype](alpha=0.1, fit_intercept=False)), n_features_to_select=FeatureCount(observer.steps[i].n_features), step=FeatureCount(2))
        _ = direct.fit(X, y, sample_weight=weights)
        assert_almost_equal(observer.steps[i].score, direct.score(X, y_test, score_weights.copy()), atol=1e-6)
        for j in range(4):
            assert_equal(observer.steps[i].support[j], direct.support_()[j])
            assert_equal(observer.steps[i].ranking[j], direct.ranking_()[j])
    var plain = RFE(LassoImportance(Lasso[dtype](alpha=0.1, fit_intercept=False)), n_features_to_select=FeatureCount(1), step=FeatureCount(2))
    _ = plain.fit(X, y, sample_weight=weights)
    var expected = plain.predict(X)
    var actual = selector.predict(X)
    for i in range(8):
        assert_equal(actual.unsafe_get(i), expected.unsafe_get(i))


def main() raises:
    lifecycle()
    real_score[DType.float32]()
    real_score[DType.float64]()
    print("PASS RFE path timing, snapshots, failure atomicity, ordinary fit and real held-out scoring")
