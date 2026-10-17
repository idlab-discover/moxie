from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import BaseEstimator, RegressorMixin, clone
from moxie.utils.validation import FittedState, check_is_fitted
from kernel_contract_fixture import DemoRegressor, DemoParams


def exercise_generic[
    E: RegressorMixin
](mut estimator: E, X: E.Input, y: E.Target) raises -> E.Prediction:
    estimator.fit(X, y)
    return estimator.predict(X)


def main() raises:
    var state = FittedState()
    assert_false(state.is_fitted())
    with assert_raises():
        state.check_features(2)
    with assert_raises():
        state.record_fit(0, 2)
    with assert_raises():
        state.record_fit(2, 0)
    with assert_raises():
        state.record_fit(2, 3, n_targets=1)
    assert_false(state.is_fitted())
    state.record_fit(2, 3, n_targets=2)
    assert_true(state.is_fitted())
    state.check_features(3)
    with assert_raises():
        state.check_features(2)
    with assert_raises():
        state.record_fit(2, 1, n_targets=0)
    state.check_features(3)
    state.record_fit(4, 5)
    state.check_features(5)
    state.reset()
    assert_false(state.is_fitted())

    var model = DemoRegressor(offset=2.0)
    var X: List[Float64] = [1.0, 2.0]
    var y: List[Float64] = [3.0, 4.0]
    with assert_raises():
        _ = model.predict(X)
    var result = exercise_generic(model, X, y)
    assert_equal(result[0], 3.0)
    assert_equal(result[1], 4.0)
    check_is_fitted(model)
    var params = model.get_params()
    params.offset = 7.0
    assert_equal(model.offset, 2.0)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.offset, 2.0)
    fresh.set_params(params)
    assert_equal(fresh.offset, 7.0)
    assert_equal(model.offset, 2.0)
    fresh.fit(X, y)
    fresh.fit(X, y)
    assert_equal(fresh.predict(X)[0], 8.0)
    with assert_raises():
        _ = fresh.score(X, y)
    print("Core estimator contract checks passed")
