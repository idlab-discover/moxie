from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.dummy import DummyRegressor, DummyClassifier, Dense, Label
from moxie.base import clone
from moxie._random import RandomState, check_random_state, seed_global_random_state
from numojo import Shape, Item
import numojo as nm


def test_default_random_state() raises:
    var X = nm.zeros[DType.float64](Shape(12, 1))
    var labels: List[Label] = [Label(0), Label(1), Label(2), Label(0), Label(1), Label(2), Label(0), Label(1), Label(2), Label(0), Label(1), Label(2)]
    var a = DummyClassifier(strategy="uniform")
    var b = DummyClassifier(strategy="stratified")
    _ = a.fit(X, labels)
    _ = b.fit(X, labels)
    seed_global_random_state(42)
    var expected = RandomState(42)
    for _ in range(3):
        var actual = a.predict(X)
        var reference = a.predict(X, expected)
        for i in range(12):
            assert_true(actual[i] == reference[i])
        var actual_log = b.predict_log_proba(X)
        var reference_log = b.predict_log_proba(X, expected)
        for i in range(12):
            for j in range(3):
                assert_equal(actual_log.item(i, j), reference_log.item(i, j))
        # Uniform probabilities use no random draws.
        _ = a.predict_proba(X)
        with check_random_state() as state:
            assert_equal(state[].state_position(), expected.state_position())
            for i in range(624):
                assert_equal(state[].state_word(i), expected.state_word(i))
    # A failing operation releases its default-stream lease.
    var scalar = nm.zeros[DType.float64](Shape())
    with assert_raises():
        _ = a.predict(scalar)
    seed_global_random_state(42)
    expected = RandomState(42)
    var after_error = a.predict(X)
    var reference = a.predict(X, expected)
    for i in range(12):
        assert_true(after_error[i] == reference[i])
    # An integer seed must not consume the shared default stream.
    _ = a.set_params(random_state=Optional[Optional[Int]](7))
    _ = a.predict(X)
    with check_random_state() as state:
        assert_equal(state[].state_position(), expected.state_position())
        for i in range(624):
            assert_equal(state[].state_word(i), expected.state_word(i))
    print("PASS shared default RNG, interleaved estimators and exception release")

def test_single_output_api() raises:
    var X = nm.zeros[DType.float64](Shape(4, 2))
    var labels: List[Label] = [Label(3), Label(1), Label(1), Label(2)]
    var rows: List[List[Label]] = [[Label(3)], [Label(1)], [Label(1)], [Label(2)]]
    var weights = nm.ones[DType.float64](Shape(4))
    weights[Item(1)] = 3
    for strategy in ["prior", "most_frequent", "uniform", "stratified", "constant"]:
        var model = DummyClassifier(strategy=strategy, random_state=42, constant=1)
        var nested = DummyClassifier[multioutput=True](strategy=strategy, random_state=42, constant=1)
        _ = model.fit(X, labels, sample_weight=weights)
        _ = nested.fit(X, rows, sample_weight=weights)
        assert_equal(model.n_classes_, 3)
        assert_equal(model.classes_[0].integer, 1)
        assert_equal(model.class_prior_.ndim, 1)
        var predictions = model.predict(X)
        var expected = nested.predict(X)
        assert_equal(len(predictions), 4)
        for i in range(4):
            assert_true(predictions[i] == expected[i][0])
        var probabilities = model.predict_proba(X)
        var logs = model.predict_log_proba(X)
        assert_equal(probabilities.shape, Shape(4, 3))
        var nested_probabilities = nested.predict_proba(X)
        var nested_logs = nested.predict_log_proba(X)
        for i in range(4):
            for j in range(3):
                assert_equal(probabilities.item(i, j), nested_probabilities[0].item(i, j))
                assert_equal(logs.item(i, j), nested_logs[0].item(i, j))
        assert_equal(model.score(X, labels, weights.copy()), nested.score(X, rows, weights.copy()))
        assert_equal(model.score(Optional[Dense](), labels), model.score(X, rows))
        var rng = RandomState(7)
        var other_rng = RandomState(7)
        assert_equal(model.score(X, labels, rng), nested.score(X, rows, other_rng))
        var fresh = clone(model)
        assert_false(fresh.__sklearn_is_fitted__())
        # One-column targets preserve the default flat API.
        _ = model.fit(X, rows)
        assert_equal(model.n_outputs_.value(), 1)
        var multiple: List[List[Label]] = [[Label(0), Label(1)], [Label(0), Label(1)], [Label(1), Label(0)], [Label(1), Label(1)]]
        with assert_raises():
            _ = model.fit(X, multiple)
        # Failed fit does not replace valid learned state.
        assert_equal(model.n_outputs_.value(), 1)
    var text_labels: List[Label] = [Label("cat"), Label("dog"), Label("cat"), Label("cat")]
    var text = DummyClassifier(strategy="constant", constant="cat")
    _ = text.fit(X, text_labels)
    assert_equal(text.predict(X)[0].text, "cat")
    _ = text.set_params(constant="dog")
    assert_equal(text.predict(X)[0].text, "dog")
    var integer_constant = DummyClassifier(strategy="constant", constant=3)
    _ = integer_constant.fit(X, labels)
    assert_equal(integer_constant.predict(X)[0].integer, 3)
    _ = integer_constant.set_params(constant=1)
    assert_equal(integer_constant.predict(X)[0].integer, 1)
    var y = nm.ones[DType.float64](Shape(4))
    var regressor = DummyRegressor(strategy="constant", constant=43.0)
    _ = regressor.fit(X, y)
    assert_equal(regressor.predict(X).item(0), 43.0)
    _ = regressor.set_params(constant=7.0)
    _ = regressor.fit(X, y)
    assert_equal(regressor.predict(X).item(0), 7.0)
    print("PASS flat classifier results, learned attributes, scoring and scalar constants")


def main() raises:
    test_single_output_api()
    test_default_random_state()
    var X = nm.zeros[DType.float64](Shape(4, 2))
    var y = nm.zeros[DType.float64](Shape(4))
    for i in range(4):
        y[Item(i)] = Float64(i)
    var reg = DummyRegressor()
    with assert_raises():
        _ = reg.predict(X)
    _ = reg.fit(X, y)
    assert_equal(reg.constant_.value().item(0, 0), 1.5)
    assert_equal(reg.score(X, y), 0.0)
    assert_false(Bool(reg.predict(X, return_std=False).std))
    var fresh = clone(reg)
    assert_false(fresh.__sklearn_is_fitted__())
    _ = reg.set_params(strategy="median")
    _ = reg.fit(X, y)
    assert_equal(reg.constant_.value().item(0, 0), 1.5)
    var weights = nm.ones[DType.float64](Shape(4))
    _ = reg.fit(X, y, sample_weight=weights)
    assert_equal(reg.constant_.value().item(0, 0), 1.0)
    var constant = nm.ones[DType.float64](Shape(1))
    _ = reg.set_params(strategy="constant", constant=Optional[Optional[Dense]](constant.copy()))
    _ = reg.fit(X, y)
    constant[Item(0)] = 99
    assert_equal(reg.predict(X).item(0), 1.0)
    var params = reg.get_params()
    params.constant.value()[Item(0)] = 8
    assert_equal(reg.constant.value().item(0), 1.0)
    with assert_raises():
        var bad = nm.zeros[DType.float64](Shape(0))
        _ = reg.fit(X, bad)
    var labels: List[Label] = [Label("東京"), Label("Paris"), Label("東京"), Label("Paris")]
    var clf = DummyClassifier[multioutput=True]()
    with assert_raises():
        _ = clf.predict(X)
    _ = clf.fit(X, labels)
    assert_equal(clf.classes_[0][0].text, "Paris")
    assert_equal(clf.predict(X)[0][0].text, "Paris")
    assert_equal(clf.predict_proba(X)[0].item(0, 0), 0.5)
    var copied = clone(clf)
    assert_false(copied.__sklearn_is_fitted__())
    _ = clf.set_params(strategy="uniform")
    # Fitted strategy is frozen until refit, matching sklearn.
    assert_equal(clf.predict(X)[0][0].text, "Paris")
    _ = clf.fit(X, labels)
    _ = clf.predict(X)
    var rng = RandomState(42)
    _ = clf.predict(X, rng)
    _ = clf.set_params(random_state=Optional[Optional[Int]](42))
    var a = clf.predict(X)
    var b = clf.predict(X)
    for i in range(4):
        assert_true(a[i][0] == b[i][0])
    _ = clf.set_params(strategy="bad")
    with assert_raises():
        _ = clf.fit(X, labels)
    print("PASS native lifecycle, shapes, ownership, strings, parameters and errors")
    # Weighted subset accuracy and multiclass-multioutput errors.
    var multi: List[List[Label]] = [[Label(0), Label(1)], [Label(0), Label(1)], [Label(1), Label(0)], [Label(1), Label(1)]]
    var prior = DummyClassifier[multioutput=True]()
    _ = prior.fit(X, multi)
    assert_equal(prior.score(X, multi), 0.5)
    assert_equal(prior.score(Optional[Dense](), multi), 0.5)
    weights[Item(0)] = 3
    assert_equal(prior.score(X, multi, weights.copy()), 4.0 / 6.0)
    multi[0][1] = Label(5)
    with assert_raises():
        _ = prior.score(X, multi)
    var continuous: List[List[Label]] = [[Label(Float64(0.1))], [Label(Float64(0.2))], [Label(Float64(0.1))], [Label(Float64(0.2))]]
    _ = prior.fit(X, continuous)
    with assert_raises():
        _ = prior.score(X, continuous)
    var uniform = DummyClassifier[multioutput=True](strategy="uniform")
    _ = uniform.fit(X, labels)
    assert_equal(uniform.predict_proba(X)[0].item(0, 0), 0.5)
    var one: List[Label] = [Label("a"), Label("a"), Label("a"), Label("a")]
    _ = uniform.fit(X, one)
    assert_equal(uniform.n_classes_[0], 1)
    assert_equal(uniform.classes_[0][0].text, "a")
    var too_short = nm.ones[DType.float64](Shape(3))
    with assert_raises():
        _ = prior.fit(X, labels, sample_weight=too_short.copy())
    var invalid_constant: List[Label] = [Label("absent")]
    var constant_model = DummyClassifier[multioutput=True](strategy="constant", constant=invalid_constant.copy())
    with assert_raises():
        _ = constant_model.fit(X, labels)
    var missing = DummyClassifier[multioutput=True](strategy="constant")
    with assert_raises():
        _ = missing.fit(X, labels)
    var bad_seed = DummyClassifier[multioutput=True](random_state=-1)
    with assert_raises():
        _ = bad_seed.fit(X, labels)
    var bad_quantile = DummyRegressor(strategy="quantile", quantile=2.0)
    with assert_raises():
        _ = bad_quantile.fit(X, y)
    var missing_quantile = DummyRegressor(strategy="quantile")
    with assert_raises():
        _ = missing_quantile.fit(X, y)
    var no_constant = DummyRegressor(strategy="constant")
    with assert_raises():
        _ = no_constant.fit(X, y)
    var nonfinite = y.copy()
    from std.math import nan
    nonfinite[Item(0)] = nan[DType.float64]()
    with assert_raises():
        _ = reg.fit(X, nonfinite)
    var singleton = nm.ones[DType.float64](Shape(1))
    var basic = DummyRegressor()
    _ = basic.fit(singleton, singleton)
    assert_true(basic.score(singleton, singleton) != basic.score(singleton, singleton))
    assert_equal(basic.predict(nm.zeros[DType.float64](Shape(0, 5))).shape[0], 0)
    assert_false(Bool(basic.n_features_in_))
    print("PASS weighted subset accuracy, score errors, repeat-fit and degenerate inputs")
    var zeros = nm.zeros[DType.float64](Shape(4))
    with assert_raises():
        _ = reg.fit(X, y, sample_weight=zeros)
    with assert_raises():
        _ = prior.fit(X, labels, sample_weight=zeros.copy())
    # Explicit Float32 conversion and independent learned state.
    var narrow = nm.ones[DType.float32](Shape(4))
    var converted = narrow.astype[DType.float64]()
    _ = basic.fit(X, converted)
    converted[Item(0)] = 99
    assert_equal(basic.predict(X).item(0), 1.0)
    # Logical indexing over a reversed target view.
    var backing = nm.zeros[DType.float64](Shape(8))
    for i in range(8):
        backing[Item(i)] = Float64(i)
    var strided = backing.view_with_layout(Shape(4), nm.NDArrayStrides(-2), 7)
    _ = basic.fit(X, strided)
    assert_equal(basic.constant_.value().item(0, 0), 4.0)
    var frozen: List[Label] = [Label("Paris")]
    var fixed = DummyClassifier[multioutput=True](strategy="constant", constant=frozen.copy())
    _ = fixed.fit(X, labels)
    var config = fixed.get_params()
    config.constant.value()[0] = Label("Tokyo")
    assert_equal(fixed.predict(X)[0][0].text, "Paris")
    var nan_labels: List[Label] = [Label(nan[DType.float64]()), Label(nan[DType.float64]()), Label(nan[DType.float64]()), Label(nan[DType.float64]())]
    var nan_constant: List[Label] = [Label(nan[DType.float64]())]
    var reject_nan_constant = DummyClassifier[multioutput=True](strategy="constant", constant=nan_constant.copy())
    with assert_raises():
        _ = reject_nan_constant.fit(X, nan_labels)
    var overflowing = nm.ones[DType.float64](Shape(4))
    for i in range(4):
        overflowing[Item(i)] = 1e308
    var overflow_model = DummyRegressor()
    _ = overflow_model.fit(X, overflowing)
    with assert_raises():
        _ = overflow_model.score(X, overflowing)
