import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.svm import LinearSVC, LinearSVR
from moxie.base import BaseEstimator, SupervisedEstimator, clone
from moxie._random import RandomState


def generic_fit[T: SupervisedEstimator](mut model: T, X: T.Input, y: T.Target) raises:
    assert_true(model.fit(X, y).__sklearn_is_fitted__())


def generic_params[T: BaseEstimator](mut model: T, params: T.Params) raises:
    assert_true(model.set_params(params).__sklearn_is_fitted__())


def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(4, 1))
    var target = nm.zeros[DType.float64](nm.Shape(4))
    var labels: List[Float64] = [0, 0, 1, 1]
    for i in range(4):
        x.unsafe_set(i, Float64(i) - 1.5)
        target.unsafe_set(i, labels[i])
    var weights = nm.ones[DType.float64](nm.Shape(4))
    var single = x.astype[DType.float32]()
    var single_target = target.astype[DType.float32]()
    var svc = LinearSVC(random_state=42)
    assert_equal(svc.set_params(C=2.0).fit(x, labels, weights.copy()).score(x, labels), 1.0)
    # Updating through the returned borrow must update the original, not a copy.
    _ = svc.fit(single, labels, weights.copy()).set_params(C=3.0)
    assert_equal(svc.C, 3.0)
    assert_equal(svc.score(x, labels), 1.0)
    assert_equal(svc.fit(x, target, weights.copy()).score(x, labels), 1.0)
    assert_equal(svc.fit(x, single_target).score(x, labels), 1.0)
    var rng = RandomState(42)
    assert_equal(svc.fit_with_random_state(x, labels, rng, weights.copy()).score(x, labels), 1.0)
    var expected_rng = RandomState(42)
    _ = expected_rng.randint(2147483647)
    assert_equal(rng.next_uint32(), expected_rng.next_uint32())
    var params = svc.get_params()
    generic_params(svc, params)
    generic_fit(svc, x, labels)
    var fresh = clone(svc)
    assert_true(not fresh.__sklearn_is_fitted__())
    var strings: List[String] = ["a", "a", "b", "b"]
    var named = LinearSVC[String](random_state=42)
    assert_equal(named.set_params(C=2.0).fit(single, strings).predict(single)[0], "a")

    var svr = LinearSVR(random_state=42, loss="squared_epsilon_insensitive", dual=False)
    assert_equal(svr.set_params(C=2.0).fit(x, target, weights.copy()).predict(x).shape[0], 4)
    _ = svr.fit(single, target, weights.copy()).set_params(C=3.0)
    assert_equal(svr.C, 3.0)
    assert_equal(svr.fit(x, single_target, weights.copy()).predict(x).shape[0], 4)
    assert_equal(svr.fit(single, single_target, weights.copy()).predict(single).shape[0], 4)
    assert_equal(svr.fit_with_random_state(x, target, rng).predict(x).shape[0], 4)
    var regparams = svr.get_params()
    generic_params(svr, regparams)
    generic_fit(svr, x, target)
    var zero = nm.zeros[DType.float64](nm.Shape(4))
    with assert_raises():
        _ = svr.fit(x, target, zero.copy()).set_params(C=99.0)
    assert_equal(svr.C, 3.0)
    assert_true(svr.__sklearn_is_fitted__())
    print("PASS public linear fluent overloads, generic dispatch, borrow identity and failure")
