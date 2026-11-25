from std.math import abs, inf, nan
from std.testing import assert_equal, assert_raises, assert_true
from moxie.base import clone
from moxie.linear_model import LogisticRegression
from moxie.linear_model._linear_loss import LogisticLoss
from numojo import NDArray, Shape, Strides, zeros


def data[dtype: DType]() raises -> Tuple[NDArray[dtype], NDArray[DType.int64]]:
    var X = zeros[dtype](Shape(6, 2))
    var y = zeros[DType.int64](Shape(6))
    for i in range(6):
        X.unsafe_set(i * 2, Scalar[dtype](i - 3))
        X.unsafe_set(i * 2 + 1, Scalar[dtype](i % 2))
        y.unsafe_set(i, Int64(-3 if i < 3 else 8))
    return (X^, y^)


def contract[dtype: DType]() raises:
    var inputs = data[dtype]()
    ref X = inputs[0]
    ref y = inputs[1]
    var model = LogisticRegression[dtype]()
    var defaults = model.get_params()
    assert_equal(defaults.penalty.value(), "deprecated")
    assert_equal(defaults.solver, "lbfgs")
    assert_equal(defaults.max_iter, 100)
    assert_equal(defaults.l1_ratio.value(), 0)
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.decision_function(X)
    with assert_raises():
        _ = model.predict_log_proba(X)

    var labels = model.set_params(C=2.0).fit(X, y).predict(X)
    assert_equal(labels.unsafe_get(0), -3)
    assert_equal(labels.unsafe_get(5), 8)
    assert_equal(model.coef_.ndim, 2)
    assert_equal(model.coef_.shape[0], 1)
    assert_equal(model.decision_function(X).ndim, 1)
    assert_equal(model.n_iter_.shape[0], 1)
    assert_true(model.n_iter_.unsafe_get(0) > 0)
    assert_equal(model.score(X, y), 1.0)

    var original = model.coef_.unsafe_get(0)
    var copied = model.copy()
    var fresh = clone(model)
    assert_equal(fresh.get_params().C, 2.0)
    with assert_raises():
        _ = fresh.predict(X)
    _ = model.fit(X, y)
    assert_equal(model.coef_.unsafe_get(0), original)

    var wrong_rank = zeros[DType.int64](Shape(6, 2))
    var one_class = zeros[DType.int64](Shape(6))
    var bad_X = X.copy()
    bad_X.unsafe_set(0, nan[dtype]())
    var zero_weights = zeros[dtype](Shape(6))
    with assert_raises():
        _ = model.fit(X, wrong_rank)
    with assert_raises():
        _ = model.fit(X, one_class)
    with assert_raises():
        _ = model.fit(bad_X, y)
    with assert_raises():
        _ = model.fit(X, y, sample_weight=zero_weights)
    with assert_raises():
        _ = model.predict(zeros[dtype](Shape(6, 3)))
    assert_equal(model.coef_.unsafe_get(0), original)

    _ = model.set_params(C=0.0)
    with assert_raises():
        _ = model.fit(X, y)
    _ = model.set_params(C=2.0, l1_ratio=0.5)
    with assert_raises():
        _ = model.fit(X, y)
    _ = model.set_params(l1_ratio=0.0, solver=String("saga"))
    with assert_raises():
        _ = model.fit(X, y)
    _ = model.set_params(solver=String("lbfgs"), dual=True)
    with assert_raises():
        _ = model.fit(X, y)
    var updated = model.get_params()
    updated.penalty = None
    updated.dual = False
    updated.verbose = 1
    updated.n_jobs = 1
    _ = model.set_params(updated).fit(X, y, sample_weight=2.0)
    var unpenalized = LogisticRegression[dtype](C=inf[DType.float64]())
    _ = unpenalized.fit(X, y, sample_weight=2.0)
    assert_equal(model.coef_.unsafe_get(0), unpenalized.coef_.unsafe_get(0))

    var Xs = X.view_with_layout(Shape(6, 2), Strides(-2, 1), 10)
    var ys = y.view_with_layout(Shape(6), Strides(-1), 5)
    var ws = zero_weights.view_with_layout(Shape(6), Strides(-1), 5)
    for i in range(6):
        zero_weights.unsafe_set(i, Scalar[dtype](i + 1))
    var solvers: List[String] = ["lbfgs", "newton-cholesky", "newton-cg"]
    for solver in solvers:
        var strided = LogisticRegression[dtype](solver=solver.copy())
        _ = strided.fit(Xs, ys, sample_weight=ws)
        var contiguous = LogisticRegression[dtype](solver=solver.copy())
        _ = contiguous.fit(X, y, sample_weight=zero_weights)
        assert_true(abs(Float64(strided.coef_.unsafe_get(0) - contiguous.coef_.unsafe_get(0))) < 1e-6)
        assert_equal(strided.predict(Xs).unsafe_get(0), 8)
        assert_equal(zero_weights.unsafe_get(0), 1)

    var multiclass = y.copy()
    multiclass.unsafe_set(1, 23)
    multiclass.unsafe_set(4, 23)
    _ = model.set_params(warm_start=True)
    with assert_raises():
        _ = model.fit(X, multiclass)
    _ = model.set_params(warm_start=False, fit_intercept=False).fit(X, multiclass)
    assert_equal(model.classes_.unsafe_get(2), 23)
    assert_equal(model.coef_.shape[0], 3)
    assert_equal(model.intercept_.unsafe_get(2), 0)
    assert_equal(model.decision_function(X).shape[1], 3)
    var p = model.predict_proba(X)
    var log_p = model.predict_log_proba(X)
    for i in range(6):
        var total = Float64(0)
        for k in range(3):
            total += Float64(p.unsafe_get(i * 3 + k))
            assert_true(log_p.unsafe_get(i * 3 + k) <= 0)
        assert_true(abs(total - 1) < 1e-6)
    assert_equal(copied.coef_.unsafe_get(0), original)
    assert_equal(copied.classes_.shape[0], 2)
    var bad_warm_start = copied.copy()
    bad_warm_start.coef_ = zeros[dtype](Shape(1))
    _ = bad_warm_start.set_params(warm_start=True)
    with assert_raises(contains="warm-start coefficients"):
        _ = bad_warm_start.fit(X, y)

    # Large inference margins must not overflow the sigmoid exponential.
    var extremes = zeros[dtype](Shape(2, 2))
    extremes.unsafe_set(0, -100000)
    extremes.unsafe_set(2, 100000)
    var extreme_prob = copied.predict_proba(extremes)
    assert_equal(extreme_prob.unsafe_get(0), 1)
    assert_equal(extreme_prob.unsafe_get(3), 1)
    assert_equal(copied.predict_log_proba(extremes).unsafe_get(1), -inf[dtype]())


def hessian_product_contract[dtype: DType](outputs: Int, fit_intercept: Bool) raises:
    var X = zeros[dtype](Shape(4, 2))
    for i in range(X.size):
        X.unsafe_set(i, Scalar[dtype](i - 4) / 4)
    var weights = zeros[DType.float64](Shape(4))
    weights.unsafe_set(0, 0.5)
    weights.unsafe_set(1, -0.25)
    weights.unsafe_set(2, 0.75)
    var targets: List[Int] = [0, 1, max(1, outputs - 1), 0]
    var objective = LogisticLoss(X^, targets^, weights^, outputs, 0.1, fit_intercept, False)
    var shape = Shape((2 + Int(fit_intercept)) * outputs)
    var current = zeros[DType.float64](shape)
    var direction = zeros[DType.float64](shape)
    var plus = zeros[DType.float64](shape)
    var minus = zeros[DType.float64](shape)
    comptime delta = 1e-5
    for i in range(current.size):
        var value = Float64(i + 1) / 8
        var perturbation = Float64((i % 3) - 1)
        current.unsafe_set(i, value)
        direction.unsafe_set(i, perturbation)
        plus.unsafe_set(i, value + delta * perturbation)
        minus.unsafe_set(i, value - delta * perturbation)
    var product = zeros[DType.float64](shape)
    var scratch = List[Float64](length=outputs, fill=0)
    objective.hessian_product(current, direction, product, scratch)
    var gradient_plus = zeros[DType.float64](shape)
    var gradient_minus = zeros[DType.float64](shape)
    _ = objective.evaluate(plus, gradient_plus)
    _ = objective.evaluate(minus, gradient_minus)
    for i in range(product.size):
        var difference = (gradient_plus.unsafe_get(i) - gradient_minus.unsafe_get(i)) / (2 * delta)
        assert_true(abs(product.unsafe_get(i) - difference) < 1e-9)
    assert_equal(objective.hessian.size, 0)


def numeric_label_contract() raises:
    var X = zeros[DType.float64](Shape(3, 1))
    var y = zeros[DType.float64](Shape(3, 1))
    y.unsafe_set(1, -3.0)
    y.unsafe_set(2, -3.0)
    var model = LogisticRegression[DType.float64, DType.float64](String("l2"))
    _ = model.fit(X, y)
    assert_equal(model.classes_.unsafe_get(0), -3.0)
    assert_equal(model.predict(X).unsafe_get(1), -3.0)
    y.unsafe_set(0, 9223372036854775808.0)
    with assert_raises(contains="signed integer range"):
        _ = model.fit(X, y)
    y.unsafe_set(0, 0.5)
    with assert_raises():
        _ = model.fit(X, y)


def main() raises:
    contract[DType.float64]()
    contract[DType.float32]()
    hessian_product_contract[DType.float32](1, True)
    hessian_product_contract[DType.float64](3, False)
    hessian_product_contract[DType.float64](3, True)
    numeric_label_contract()
    print("PASS: dense LogisticRegression contracts in Float32 and Float64")
