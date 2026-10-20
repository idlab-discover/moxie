import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from std.math import abs, isnan, inf
from moxie.svm import LinearSVC, LinearSVR, ClassWeight, Dual
from moxie.base import clone
from moxie._random import RandomState
from moxie.svm._contracts import Dense

def close(a: Dense, b: Dense) raises:
    assert_equal(a.ndim, b.ndim)
    assert_equal(a.size, b.size)
    for i in range(a.size):
        assert_true(abs(a.unsafe_get(i)-b.unsafe_get(i)) < 1e-9)

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(6, 2))
    var y: List[Float64] = [-7,-7,-7,4,4,4]
    var target = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        x.unsafe_set(i*2, Float64(i)-2.5)
        x.unsafe_set(i*2+1, Float64(i % 2))
        target.unsafe_set(i, Float64(i)*0.5-1.0)
    var svc = LinearSVC(random_state=42, tol=1e-8, max_iter=5000)
    var svr = LinearSVR(random_state=42, tol=1e-8, max_iter=5000)
    with assert_raises():
        _ = svc.predict(x)
    with assert_raises():
        _ = svr.predict(x)
    _ = svc.fit(x, y)
    _ = svr.fit(x, target)
    assert_true(svc.__sklearn_is_fitted__())
    assert_equal(svc.classes_[0], -7.0)
    assert_equal(svc.classes_[1], 4.0)
    assert_equal(svc.coef_.ndim, 2)
    assert_equal(svr.coef_.ndim, 1)
    assert_equal(svc.decision_function(x).ndim, 1)
    assert_equal(svc.score(x, y), 1.0)
    assert_true(svr.score(x, target) > 0.9)
    var saved = svc.coef_.copy()
    _ = svc.fit(x, y)
    close(saved, svc.coef_)
    var copied = clone(svc)
    assert_true(not copied.__sklearn_is_fitted__())
    assert_equal(copied.get_params().random_state, 42)
    with assert_raises():
        _ = copied.decision_function(x)
    copied.set_params(C=0.1)
    assert_equal(svc.C, 1.0)
    _ = copied.fit(x, y)
    copied.coef_.unsafe_set(0, 777)
    close(saved, svc.coef_)
    var regclone = clone(svr)
    assert_true(not regclone.__sklearn_is_fitted__())
    var params = svr.get_params()
    params.epsilon = 0.2
    _ = regclone.set_params(params)
    assert_equal(svr.epsilon, 0.0)
    # Float32 conversion and array target adapter.
    var single = x.astype[DType.float32]()
    svc.fit(single, y)
    close(saved, svc.coef_)
    close(svc.decision_function(single), svc.decision_function(x))
    var numeric = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        numeric.unsafe_set(i, y[i])
    svc.fit(x, numeric)
    close(saved, svc.coef_)
    svr.fit(single, target.astype[DType.float32]())
    # Negative feature strides and strided weights/targets.
    var backing = nm.zeros[DType.float64](nm.Shape(6,4))
    var wb = nm.ones[DType.float64](nm.Shape(12))
    for i in range(6):
        backing.unsafe_set(i*4+3, x.unsafe_get(i*2))
        backing.unsafe_set(i*4+1, x.unsafe_get(i*2+1))
    var strided = backing.view_with_layout(nm.Shape(6,2), nm.NDArrayStrides(4,-2), 3)
    var weights = wb.view_with_layout(nm.Shape(6), nm.NDArrayStrides(2), 0)
    svc.fit(strided, y, weights.copy())
    close(saved, svc.coef_)
    close(svc.decision_function(strided), svc.decision_function(x))
    # Seeded fits reset; explicitly borrowed RNG streams advance once per fit.
    var rng = RandomState(42)
    svc.fit_with_random_state(x, y, rng)
    var expected_rng = RandomState(42)
    _ = expected_rng.randint(2147483647)
    assert_equal(rng.state_position(), expected_rng.state_position())
    close(saved, svc.coef_)
    svc.fit_with_random_state(x, y, rng)
    _ = expected_rng.randint(2147483647)
    assert_equal(rng.state_position(), expected_rng.state_position())
    # Sorted string labels, multiclass decisions and clone-owned configuration.
    var labels: List[String] = ["z","z","a","a","m","m"]
    var names: List[String] = ["z"]
    var values: List[Float64] = [2.0]
    var text = LinearSVC[String](class_weight=ClassWeight[String](names, values), random_state=9)
    _ = text.fit(x, labels)
    assert_equal(text.classes_[0], "a")
    assert_equal(text.classes_[1], "m")
    assert_equal(text.classes_[2], "z")
    assert_equal(text.decision_function(x).shape[1], 3)
    var text_clone = clone(text)
    text_clone.class_weight.values[0] = 8
    assert_equal(text.class_weight.values[0], 2.0)
    _ = text.predict(x)
    var wide: List[Int] = [9007199254740993,9007199254740993,9007199254740993,9007199254740995,9007199254740995,9007199254740995]
    var integers = LinearSVC[Int](random_state=42)
    _ = integers.fit(x, wide)
    assert_equal(integers.classes_[0], 9007199254740993)
    assert_equal(integers.predict(x)[5], 9007199254740995)
    # Refit removes old classes and feature counts; invalid fit keeps old model.
    var two: List[String] = ["b","b","b","a","a","a"]
    with assert_raises():
        _ = text.fit(x, two)
    assert_equal(len(text.classes_), 3)
    _ = text.set_params(class_weight=ClassWeight[String]("none")).fit(x, two)
    assert_equal(len(text.classes_), 2)
    var short: List[Float64] = [0]
    with assert_raises():
        _ = svc.fit(x, short)
    close(saved, svc.coef_)
    var wrong = nm.ones[DType.float64](nm.Shape(2,3))
    with assert_raises():
        _ = svc.predict(wrong)
    var continuous = y.copy()
    continuous[0] = 0.5
    with assert_raises():
        _ = svc.fit(x, continuous)
    var only: List[Float64] = [0,0,0,0,0,0]
    with assert_raises():
        _ = svc.fit(x, only)
    var bad = x.copy()
    bad.unsafe_set(0, inf[DType.float64]())
    with assert_raises():
        _ = svc.fit(bad, y)
    var zero_weights = nm.zeros[DType.float64](nm.Shape(6))
    with assert_raises():
        svc.fit(x, y, zero_weights.copy())
    with assert_raises():
        svr.fit(x, target, zero_weights.copy())
    for code in range(10):
        var invalid = clone(svc)
        if code == 0:
            invalid.C = 0
        elif code == 1:
            invalid.tol = 0
        elif code == 2:
            invalid.intercept_scaling = 0
        elif code == 3:
            invalid.max_iter = -1
        elif code == 4:
            invalid.random_state = -2
        elif code == 5:
            invalid.penalty = "l1"
            invalid.loss = "hinge"
        elif code == 6:
            invalid.dual = Dual(False)
            invalid.loss = "hinge"
        elif code == 7:
            invalid.class_weight = ClassWeight[Float64]("bad")
        elif code == 8:
            invalid.multi_class = "bad"
        else:
            invalid.dual = Dual("bad")
        with assert_raises():
            _ = invalid.fit(x,y)
    var invalid_reg = LinearSVR(dual=False)
    with assert_raises():
        _ = invalid_reg.fit(x,target)
    invalid_reg.dual = Dual(True)
    invalid_reg.epsilon = -1
    with assert_raises():
        _ = invalid_reg.fit(x,target)
    var limited = LinearSVC(max_iter=0)
    _ = limited.fit(x,y)
    assert_equal(limited.n_iter_,0)
    assert_equal(limited.fit_status_,1)
    var limited_reg = LinearSVR(max_iter=0)
    _ = limited_reg.fit(x,target)
    assert_equal(limited_reg.fit_status_,1)
    var one = nm.ones[DType.float64](nm.Shape(1,2))
    var one_y = nm.ones[DType.float64](nm.Shape(1))
    _ = limited_reg.fit(one,one_y)
    assert_true(isnan(limited_reg.score(one,one_y)))
    print("PASS linear lifecycle, labels, dtype, strides, RNG, weights, invalid modes and convergence")
