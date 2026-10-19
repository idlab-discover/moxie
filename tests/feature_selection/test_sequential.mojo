import numojo as nm
from moxie.linear_model import Lasso
from moxie.feature_selection._sequential import SequentialFeatureSelector, SequentialCV, SequentialCount
from moxie.feature_selection._sequential_adapter import DenseSupervisedCV
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from std.math import nan, inf
from sequential_support import ProbeEstimator, ProbeAdapter, ProbeScore, Dense
from moxie.model_selection._split import KFold
from moxie.model_selection._indices import IndexSplit

def real_model() raises:
    var X = nm.zeros[DType.float64](nm.Shape(20, 3))
    var y = nm.zeros[DType.float64](nm.Shape(20))
    for i in range(20):
        X.unsafe_set(i * 3, Float64(i))
        X.unsafe_set(i * 3 + 1, Float64(i % 3))
        X.unsafe_set(i * 3 + 2, Float64(i % 2))
        y.unsafe_set(i, Float64(i * 2))
    var model = Lasso(alpha=0.01)
    var selector = SequentialFeatureSelector[DenseSupervisedCV[Lasso[]]](model.copy(), n_features_to_select=SequentialCount(1))
    _ = selector.fit(X, y)
    assert_equal(selector.get_support(True)[List[Int]][0], 0)
    assert_false(model.__sklearn_is_fitted__())
    assert_false(selector.estimator.__sklearn_is_fitted__())
    print("PASS sequential selection")

def controls() raises:
    var X = nm.zeros[DType.float64](nm.Shape(20, 5))
    var y = nm.zeros[DType.float64](nm.Shape(20))
    var weights = nm.zeros[DType.float64](nm.Shape(20))
    for i in range(20):
        for j in range(5):
            X.unsafe_set(i * 5 + j, Float64(j * 100 + i))
        y.unsafe_set(i, Float64(i // 10))
        weights.unsafe_set(i, Float64(i + 1))
    var original = ProbeEstimator()
    original.fitted = True
    var s = SequentialFeatureSelector[ProbeAdapter](original.copy())
    with assert_raises():
        _ = s.support_()
    with assert_raises():
        _ = s.transform(X)
    # No target is necessary for an unsupervised adapter; ties select leftmost.
    _ = s.fit(X)
    assert_true(original.fitted)
    assert_equal(s.n_features_to_select_(), 2)
    var indices = s.get_support(True)[List[Int]].copy()
    assert_equal(indices[0], 0)
    assert_equal(indices[1], 1)
    var params = s.get_params()
    assert_true(params.estimator.fitted)
    params.direction = "backward"
    _ = s.set_params(params)
    assert_equal(s.n_features_to_select_(), 2)
    _ = s.fit(X)
    assert_equal(s.n_features_to_select_(), 3)
    assert_equal(s.get_support(True)[List[Int]][0], 2)
    var clone = s.__sklearn_clone__()
    assert_false(clone.__sklearn_is_fitted__())
    var clone_params = clone.get_params()
    assert_false(clone_params.estimator.fitted)
    _ = clone.fit(X)
    assert_equal(clone.support_(), s.support_())
    s.direction = "forward"
    s.n_features_to_select = SequentialCount(3)
    var names: List[String] = ["a", "b", "c", "d", "e"]
    s.fit_named(X, y.copy(), names, fit_metadata=Optional[Dense](weights.copy()), score_metadata=Optional[Dense](weights.copy()))
    assert_equal(s.feature_names_in_(), names)
    assert_equal(s.get_feature_names_out()[2], "c")
    var transformed = s.transform(X)
    transformed.unsafe_set(0, -99)
    assert_equal(X.unsafe_get(0), Float64(0))
    var inverse = s.inverse_transform(s.transform(X))
    assert_equal(inverse.unsafe_get(4), Float64(0))
    var saved = s.support_()
    s.n_features_to_select = SequentialCount(5)
    with assert_raises():
        _ = s.fit(X)
    assert_equal(s.support_(), saved)
    s.n_features_to_select = SequentialCount(Float64(1))
    _ = s.fit(X)
    assert_equal(s.n_features_to_select_(), 5)
    s.n_features_to_select = SequentialCount(Float64(.01))
    _ = s.fit(X)
    assert_equal(s.n_features_to_select_(), 0)
    assert_equal(s.transform(X).shape[1], 0)
    s.direction = "backward"
    with assert_raises():
        _ = s.fit(X)
    s.direction = "forward"
    s.n_features_to_select = SequentialCount(String("auto"))
    s.scoring = ProbeScore(1)
    s.tol = 0.0
    _ = s.fit(X)
    assert_equal(s.n_features_to_select_(), 1)
    s.scoring = ProbeScore(0)
    _ = s.fit(X)
    assert_equal(s.n_features_to_select_(), 4)
    s.tol = .01
    _ = s.fit(X)
    assert_equal(s.n_features_to_select_(), 1)
    s.direction = "backward"
    s.scoring = ProbeScore(2)
    s.tol = -.5
    _ = s.fit(X)
    assert_equal(s.n_features_to_select_(), 4)
    s.tol = -2.0
    _ = s.fit(X)
    assert_equal(s.n_features_to_select_(), 1)
    s.direction = "forward"
    s.tol = None
    s.scoring = ProbeScore(4)
    _ = s.fit(X)
    assert_equal(s.get_support(True)[List[Int]][0], 0)
    s.scoring = ProbeScore(3)
    s.n_features_to_select = SequentialCount(1)
    _ = s.fit(X)
    assert_equal(s.get_support(True)[List[Int]][0], 0)
    s.scoring = None
    s.estimator.failure = 1
    _ = s.fit(X)
    assert_equal(s.n_features_to_select_(), 1)
    s.estimator.failure = 2
    with assert_raises():
        _ = s.fit(X)
    s.estimator.failure = 0
    # Materialized split configuration owns the indices supplied by its caller.
    var cursor = KFold(2).split(20)
    var pairs = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        pairs.append(pair.value().copy())
    var explicit_cv = SequentialCV(pairs)
    pairs[0].test.unsafe_set(0, 999)
    s.cv = explicit_cv.copy()
    _ = s.fit(X)
    var independent_cv = explicit_cv.copy()
    explicit_cv.pairs.value()[0][1].unsafe_set(0, 999)
    _ = independent_cv.splits(20, None, False)
    with assert_raises():
        _ = explicit_cv.splits(20, None, False)
    var classifier = ProbeAdapter()
    classifier.classification = True
    var default_cv = SequentialCV()
    var splits = default_cv.splits(20, classifier.cv_labels(y.copy()), True)
    assert_equal(len(splits), 5)
    for i in range(5):
        var ones = 0
        for j in range(splits[i].test.shape[0]):
            ones += Int(y.unsafe_get(Int(splits[i].test.unsafe_get(j))))
        assert_equal(ones, 2)
    var classified = SequentialFeatureSelector[ProbeAdapter](ProbeEstimator(), classifier^)
    var classification_target = Optional(y.copy())
    _ = classified.fit(X, classification_target)
    assert_equal(classified.n_features_to_select_(), 2)
    s.cv = SequentialCV()
    s.n_jobs = -1
    _ = s.fit(X)
    s.n_jobs = 0
    with assert_raises():
        _ = s.fit(X)
    s.n_jobs = None
    s.direction = "sideways"
    with assert_raises():
        _ = s.fit(X)
    s.direction = "forward"
    s.tol = -.01
    with assert_raises():
        _ = s.fit(X)
    s.tol = nan[DType.float64]()
    with assert_raises():
        _ = s.fit(X)
    s.tol = None
    s.n_features_to_select = SequentialCount(String("wrong"))
    with assert_raises():
        _ = s.fit(X)
    s.n_features_to_select = SequentialCount(0)
    with assert_raises():
        _ = s.fit(X)
    s.n_features_to_select = SequentialCount(Float64(1.1))
    with assert_raises():
        _ = s.fit(X)
    s.n_features_to_select = SequentialCount(1)
    s.cv = SequentialCV(1)
    with assert_raises():
        _ = s.fit(X)
    s.cv = SequentialCV(21)
    with assert_raises():
        _ = s.fit(X)
    var no_pairs = List[IndexSplit]()
    s.cv = SequentialCV(no_pairs)
    with assert_raises():
        _ = s.fit(X)
    s.cv = SequentialCV()
    with assert_raises():
        var wrong = nm.zeros[DType.float64](nm.Shape(19))
        var wrong_target = Optional(wrong^)
        _ = s.fit(X, wrong_target)
    with assert_raises():
        var single = nm.zeros[DType.float64](nm.Shape(20, 1))
        _ = s.fit(single)
    with assert_raises():
        var wrong_names: List[String] = ["a"]
        s.fit_named(X, None, wrong_names)
    X.unsafe_set(0, inf[DType.float64]())
    with assert_raises():
        _ = s.fit(X)
    X.unsafe_set(0, 0)
    _ = s.fit_transform(X)
    # Zero-step requests never execute default folds or construct a job pool.
    var tiny = nm.ones[DType.float64](nm.Shape(2, 5))
    s.n_jobs = 0
    s.n_features_to_select = SequentialCount(Float64(.01))
    _ = s.fit(tiny)
    assert_equal(s.n_features_to_select_(), 0)
    s.direction = "backward"
    s.n_features_to_select = SequentialCount(Float64(1.0))
    _ = s.fit(tiny)
    assert_equal(s.n_features_to_select_(), 5)
    s.cv = SequentialCV(1)
    with assert_raises():
        _ = s.fit(tiny)
    print("PASS sequential contracts")


def main() raises:
    real_model()
    controls()
