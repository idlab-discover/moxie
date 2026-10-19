import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.base import clone
from moxie.tree import DecisionTreeClassifier, DecisionTreeRegressor

def main() raises:
    var X = nm.zeros[DType.float32](nm.Shape(6, 2))
    var y = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        X.unsafe_set(i * 2, Float32(i))
        X.unsafe_set(i * 2 + 1, Float32(i % 2))
        y.unsafe_set(i, Float64(10 if i < 3 else 20))
    var classifier = DecisionTreeClassifier(random_state=0)
    with assert_raises():
        _ = classifier.predict(X)
    _ = classifier.fit(X, y)
    assert_equal(classifier.get_n_leaves(), 2)
    assert_equal(classifier.score(X, y), 1.0)
    assert_equal(classifier.classes_[0].unsafe_get(0), 10.0)
    assert_equal(classifier.tree_.nodes[0].threshold, 2.5)
    var proba = classifier.predict_proba(X)
    assert_equal(proba[0].unsafe_get(0), 1.0)
    var logs = classifier.predict_log_proba(X)
    assert_equal(logs[0].unsafe_get(0), 0.0)
    var copied = clone(classifier)
    assert_true(not copied.__sklearn_is_fitted__())
    var params = classifier.get_params()
    params.max_depth = 1
    _ = copied.set_params(params)
    _ = copied.fit(X, y)
    var regressor = DecisionTreeRegressor(random_state=0)
    _ = regressor.fit(X, y)
    assert_equal(regressor.score(X, y), 1.0)
    assert_equal(regressor.predict(X).unsafe_get(0), 10.0)
    var path = classifier.cost_complexity_pruning_path(X, y)
    assert_equal(path.ccp_alphas.unsafe_get(1), 0.5)
    print("PASS public classifier/regressor lifecycle, predictions, probabilities, score, clone, pruning path")
