"""RFECV lifecycle, score policies, owned paths and real-model contracts."""
import numojo as nm
from std.math import isnan, nan
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from moxie.base import clone, Predictor, SupervisedScorer, ProbabilityPredictor
from moxie.linear_model import Lasso
from moxie.feature_selection import RFECV, DenseSupervisedCV, SequentialCV, LassoImportance, FeatureCount, RFECVResults
from rfecv_support import TestCV, TestScore


def exercise[dtype: DType]() raises:
    comptime E = LassoImportance[dtype]
    comptime CV = TestCV[E]
    comptime Model = RFECV[CV]
    comptime assert conforms_to(Model, Predictor)
    comptime assert conforms_to(Model, SupervisedScorer)
    comptime assert not conforms_to(Model, ProbabilityPredictor)
    var X = nm.zeros[dtype](nm.Shape(40, 5))
    var y = nm.zeros[dtype](nm.Shape(40))
    for i in range(40):
        X.unsafe_set(i * 5 + i % 5, 2)
        y.unsafe_set(i, SIMD[dtype, 1](6 if i % 5 == 4 else 0))
    var model = E(Lasso[dtype](alpha=0.1, fit_intercept=False))
    _ = model.fit(X, y)
    var s = Model(model.copy(), scoring=TestScore[E](1), step=FeatureCount(2))
    with assert_raises():
        _ = s.support_()
    with assert_raises():
        _ = s.cv_results_()
    with assert_raises():
        _ = s.predict(X)
    with assert_raises():
        _ = s.score(X, y)
    with assert_raises():
        _ = s.estimator_()
    with assert_raises():
        _ = s.fit_transform(X)
    var names: List[String] = ["a", "b", "c", "d", "e"]
    s.fit_named(X, y, names)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.n_features(), 5)
    assert_equal(s.n_features_(), 1)
    assert_equal(s.n_features_in_(), 5)
    assert_equal(s.feature_names_in_(), names)
    assert_equal(s.get_feature_names_out()[0], "e")
    assert_equal(s.score(X, y), Float64(7))
    assert_equal(s.estimator_().n_features(), 1)
    var results = s.cv_results_()
    var counts = results["n_features"][List[Int]].copy()
    assert_equal(counts, [1, 3, 5])
    for j in range(3):
        assert_equal(results["mean_test_score"][nm.NDArray[DType.float64]].unsafe_get(j), Float64(7))
        assert_equal(results["std_test_score"][nm.NDArray[DType.float64]].unsafe_get(j), Float64(0))
    for fold in range(5):
        ref masks = results["split" + String(fold) + "_support"][List[List[Bool]]]
        ref ranks = results["split" + String(fold) + "_ranking"][List[List[Int]]]
        assert_equal(masks[0], [False, False, False, False, True])
        assert_equal(ranks[0], [3, 3, 2, 2, 1])
        assert_equal(masks[1], [False, False, True, True, True])
        assert_equal(ranks[2], [1, 1, 1, 1, 1])
    results["split0_support"][List[List[Bool]]][0][4] = False
    results["split0_ranking"][List[List[Int]]][0][4] = 99
    results["mean_test_score"][nm.NDArray[DType.float64]].unsafe_set(0, 99)
    results["n_features"][List[Int]][0] = 99
    var again = s.cv_results_()
    assert_true(again["split0_support"][List[List[Bool]]][0][4])
    assert_equal(again["split0_ranking"][List[List[Int]]][0][4], 1)
    assert_equal(again["mean_test_score"][nm.NDArray[DType.float64]].unsafe_get(0), Float64(7))
    assert_equal(again["n_features"][List[Int]][0], 1)
    var transformed = s.transform(X)
    var inverse = s.inverse_transform(transformed)
    assert_equal(inverse.shape[1], 5)
    transformed.unsafe_set(0, 99)
    assert_equal(X.unsafe_get(4), SIMD[dtype, 1](0))
    var parameters = s.get_params()
    parameters.min_features_to_select = 3
    parameters.scoring = TestScore[E](2)
    assert_equal(s.min_features_to_select, 1)
    _ = s.set_params(parameters)
    assert_equal(s.n_features_(), 1)
    assert_equal(s.score(X, y), Float64(1))
    var fresh = clone(s)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_false(fresh.estimator.__sklearn_is_fitted__())
    _ = fresh.fit(X, y)
    assert_equal(fresh.n_features_(), 5)
    s.scoring = TestScore[E](3)
    with assert_raises():
        _ = s.fit(X, y)
    assert_equal(s.n_features_(), 1)
    assert_equal(s.feature_names_in_(), names)
    assert_equal(s.cv_results_()["n_features"][List[Int]][0], 1)
    s.scoring = TestScore[E](1)
    s.importance_getter = "bad"
    with assert_raises():
        _ = s.fit(X, y)
    assert_equal(s.n_features_(), 1)
    s.importance_getter = "auto"
    # Both bad folds and bad metadata fail before replacing fitted state.
    s.cv = SequentialCV(99)
    with assert_raises():
        _ = s.fit(X, y)
    s.cv = SequentialCV()
    with assert_raises():
        s.fit_named(X, y, names, sample_weight=nm.ones[dtype](nm.Shape(39)))
    with assert_raises():
        s.fit_named(X, y, [String("wrong")])
    with assert_raises():
        _ = s.fit(X, nm.zeros[dtype](nm.Shape(39)))
    with assert_raises():
        _ = s.fit(nm.zeros[dtype](nm.Shape(40, 1)), y)
    with assert_raises():
        _ = s.transform(nm.zeros[dtype](nm.Shape(40, 4)))
    s.min_features_to_select = 0
    with assert_raises():
        _ = s.fit(X, y)
    s.min_features_to_select = 1
    s.n_jobs = 0
    with assert_raises():
        _ = s.fit(X, y)
    s.n_jobs = -1
    s.verbose = -1
    with assert_raises():
        _ = s.fit(X, y)
    s.verbose = 0
    s.step = FeatureCount(0)
    with assert_raises():
        _ = s.fit(X, y)
    s.step = FeatureCount(Float64(0.4))
    s.scoring = TestScore[E](4)
    _ = s.fit(X, y)
    assert_equal(s.n_features_(), 3)
    assert_true(isnan(s.cv_results_()["mean_test_score"][nm.NDArray[DType.float64]].unsafe_get(1)))
    assert_equal(len(s.feature_names_in_()), 0)
    for mode in [5, 6]:
        s.scoring = TestScore[E](mode)
        _ = s.fit(X, y)
        assert_equal(s.n_features_(), 1)
        assert_true(isnan(s.cv_results_()["std_test_score"][nm.NDArray[DType.float64]].unsafe_get(0)))
    s.scoring = TestScore[E](1)
    s.min_features_to_select = 99
    _ = s.fit(X, y)
    assert_equal(s.n_features_(), 5)
    assert_equal(s.cv_results_()["n_features"][List[Int]], [5])
    var copy = s.copy()
    copy.min_features_to_select = 1
    _ = copy.fit(X, y)
    assert_equal(s.n_features_(), 5)
    assert_equal(copy.n_features_(), 1)
    var bad = X.copy()
    bad.unsafe_set(0, nan[dtype]())
    with assert_raises():
        _ = s.fit(bad, y)
    assert_equal(s.n_features_(), 5)
    var real = RFECV[DenseSupervisedCV[E, dtype]](E(Lasso[dtype](alpha=0.1, fit_intercept=False)))
    var weights = nm.ones[dtype](nm.Shape(40))
    _ = real.fit(X, y, sample_weight=weights)
    assert_equal(real.n_features_(), 1)
    assert_almost_equal(real.score(X, y), real.estimator_().score(real.transform(X), y), atol=1e-7)
    assert_equal(real.fit_transform(X, y.copy()).shape[1], 1)
    var fitted = real.estimator_()
    fitted.model.coef_.unsafe_set(0, 99)
    assert_true(real.estimator_().model.coef_.unsafe_get(0) < 4)


def routing() raises:
    from rfecv_support import RoutingSpy
    from moxie.base import ClassifierMixin, LogProbabilityPredictor, DecisionFunction
    from moxie.model_selection._indices import IndexSplit
    from moxie.model_selection._split import SplitLabels
    comptime CV = TestCV[RoutingSpy]
    comptime assert conforms_to(RFECV[CV], ClassifierMixin)
    comptime assert conforms_to(RFECV[CV], LogProbabilityPredictor)
    comptime assert conforms_to(RFECV[CV], DecisionFunction)
    from std.memory import ArcPointer
    var trace = ArcPointer(nm.zeros[DType.float64](nm.Shape(100)))
    var X = nm.zeros[DType.float64](nm.Shape(20, 5))
    var y = nm.zeros[DType.float64](nm.Shape(20))
    var weights = nm.zeros[DType.float64](nm.Shape(20))
    var score_weights = nm.zeros[DType.float64](nm.Shape(20))
    for i in range(20):
        for j in range(5):
            X.unsafe_set(i * 5 + j, Float64(i * 10 + j))
        y.unsafe_set(i, Float64(42 if i < 10 else 7))
        weights.unsafe_set(i, Float64(i + 1))
        score_weights.unsafe_set(i, Float64(i + 101))
    var s = RFECV[CV](RoutingSpy(trace.copy()), step=FeatureCount(2))
    with assert_raises():
        _ = s.classes_()
    with assert_raises():
        _ = s.predict_proba(X)
    s.fit_named(X, y, List[String](), sample_weight=weights.copy(), score_metadata=score_weights.copy())
    assert_equal(Int(trace[].unsafe_get(0)), 19)  # 5*3 fold fits + 3 full RFE + final refit.
    for call in range(1, 20):
        assert_equal(Int(trace[].unsafe_get(call * 2 - 1)), 16 if call <= 15 else 20)
        var expected = 5 - 2 * ((call - 1) % 3) if call <= 18 else 1
        assert_equal(Int(trace[].unsafe_get(call * 2)), expected)
    assert_equal(s.n_features_(), 1)
    assert_equal(s.classes_(), [42, 7])
    assert_equal(s.predict_proba(X).shape[1], 2)
    assert_equal(s.predict_log_proba(X).shape[1], 2)
    assert_equal(s.decision_function(X).shape[0], 20)
    assert_equal(s.score(X, y, score_weights.copy()), Float64(-1))
    for fail in [2, 17, 19]:
        trace[].unsafe_set(0, 0)
        s.estimator.fail_call = fail
        with assert_raises():
            _ = s.fit(X, y)
        assert_equal(Int(trace[].unsafe_get(0)), fail)
        assert_equal(s.n_features_(), 1)
        assert_equal(s.cv_results_()["n_features"][List[Int]], [1, 3, 5])
        assert_equal(s.predict(X).shape[0], 20)
    # Materialized folds own their indices and reject out-of-range indices.
    var pairs = SequentialCV(2).splits(20, SplitLabels(y.copy()), True)
    var cv = SequentialCV(pairs)
    pairs[0].train.unsafe_set(0, 999)
    assert_true(Int(cv.splits(20, None, False)[0].train.unsafe_get(0)) < 20)
    var clean_pairs = cv.splits(20, None, False)
    var one_pair = List[IndexSplit]()
    one_pair.append(clean_pairs[0].copy())
    s.cv = SequentialCV(one_pair)
    trace[].unsafe_set(0, 0)
    s.estimator.fail_call = 0
    _ = s.fit(X, y)
    assert_equal(s.n_features_(), 1)
    assert_equal(len(s.cv_results_()), 6)
    assert_equal(s.cv_results_()["std_test_score"][nm.NDArray[DType.float64]].unsafe_get(0), Float64(0))
    var bad_cv = SequentialCV(pairs)
    s.cv = bad_cv.copy()
    with assert_raises():
        _ = s.fit(X, y)
    assert_equal(s.n_features_(), 1)
    var empty = List[IndexSplit]()
    s.cv = SequentialCV(empty)
    with assert_raises():
        _ = s.fit(X, y)


def main() raises:
    routing()
    exercise[DType.float32]()
    exercise[DType.float64]()
    print("PASS RFECV real fits, lifecycle, path ownership, scoring, ties, NaNs and validation")
