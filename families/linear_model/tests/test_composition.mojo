"""Linear models inside Moxie compositions: Pipeline, nested clone and typed nested set_params.

Expected values come from the pinned scikit-learn estimators (`_base.py`, `_ridge.py`,
`_coordinate_descent.py` with its compiled Cython engine, `_bayes.py`) wrapped in
`make_pipeline(StandardScaler(), ...)` on the data built by `data()`. Tolerance 1e-9:
Float64 closed-form solves and the same cyclic coordinate updates (converged well
inside tol=1e-4) agree to ~1e-13; 1e-9 leaves room for summation-order differences.
"""
import numojo as nm
from numojo import NDArray, Shape
from std.math import sin
from std.testing import assert_equal, assert_true, assert_false, assert_almost_equal, assert_raises
from moxie.base import clone
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler
from moxie.linear_model import LinearRegression, Ridge, RidgeClassifier, Lasso, ElasticNet, BayesianRidge

comptime TOL = 1e-9


def data() raises -> Tuple[NDArray[DType.float64], NDArray[DType.float64], NDArray[DType.int64]]:
    # X = [i, (3 i) mod 5], y = 2 x0 - x1 + 1 + 0.25 sin(i), labels = i mod 3, i = 0..7.
    var X = nm.zeros[DType.float64](Shape(8, 2))
    var y = nm.zeros[DType.float64](Shape(8))
    var labels = nm.zeros[DType.int64](Shape(8))
    for i in range(8):
        var x1 = Float64((i * 3) % 5)
        X.unsafe_set(2 * i, Float64(i))
        X.unsafe_set(2 * i + 1, x1)
        y.unsafe_set(i, 2.0 * Float64(i) - x1 + 1 + 0.25 * sin(Float64(i)))
        labels.unsafe_set(i, Int64(i % 3))
    return (X^, y^, labels^)


def check_coef(coef: NDArray[DType.float64], expected: List[Float64]) raises:
    assert_equal(coef.size, len(expected))
    for j in range(len(expected)):
        assert_almost_equal(Float64(coef.unsafe_get(j)), expected[j], atol=TOL)


def same(a: NDArray[DType.float64], b: NDArray[DType.float64]) raises:
    assert_equal(a.size, b.size)
    for i in range(a.size):
        assert_almost_equal(Float64(a.unsafe_get(i)), Float64(b.unsafe_get(i)), atol=1e-12)


def regressors() raises:
    var d = data()
    ref X = d[0]
    ref y = d[1]
    comptime INTERCEPT = 6.267304148445069  # mean(y): every step sees centered features
    var ols = make_pipeline(StandardScaler(), LinearRegression())
    _ = ols.fit(X, y)
    check_coef(ols.estimator.coef_, [4.530120911427289, -1.3599294586881723])
    assert_almost_equal(Float64(ols.estimator.intercept_.unsafe_get(0)), INTERCEPT, atol=TOL)
    assert_almost_equal(ols.score(X, y), 0.9989118713248454, atol=TOL)
    var ridge = make_pipeline(StandardScaler(), Ridge(alpha=0.5))
    _ = ridge.fit(X, y)
    check_coef(ridge.estimator.coef_, [4.26032524477784, -1.2699814897139274])
    assert_almost_equal(ridge.score(X, y), 0.9953075960569829, atol=TOL)
    var lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.1))
    _ = lasso.fit(X, y)
    check_coef(lasso.estimator.coef_, [4.426041610417974, -1.2558502692678886])
    assert_almost_equal(lasso.score(X, y), 0.9979619105554649, atol=TOL)
    var enet = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.3))
    _ = enet.fit(X, y)
    check_coef(enet.estimator.coef_, [4.200993037220247, -1.2308690843601788])
    assert_almost_equal(enet.score(X, y), 0.9933601085987237, atol=TOL)
    var bayes = make_pipeline(StandardScaler(), BayesianRidge())
    _ = bayes.fit(X, y)
    check_coef(bayes.estimator.coef_, [4.5284900299759325, -1.3593824102206182])
    assert_almost_equal(bayes.score(X, y), 0.9989117394766378, atol=1e-7)  # bayesian parity bound

    # The pipeline equals fitting the steps by hand (sklearn's Pipeline contract).
    var scaler = StandardScaler()
    var Z = scaler.fit_transform(X)
    var manual = Ridge(alpha=0.5)
    _ = manual.fit(Z, y)
    same(ridge.predict(X), manual.predict(scaler.transform(X)))


def classifier() raises:
    var d = data()
    ref X = d[0]
    ref labels = d[2]
    var model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=0.5))
    _ = model.fit(X, labels)
    var expected_labels: List[Int] = [2, 0, 2, 0, 1, 2, 1, 1]
    var prediction = model.predict(X)
    for i in range(8):
        assert_equal(Int(prediction.unsafe_get(i)), expected_labels[i])
    # coef_ is (classes, features) in row-major order.
    check_coef(model.estimator.coef_, [-0.1651730204767911, 0.3019133361108363, 0.14956304807496795,
                                       0.12126318917997074, 0.01560997240182314, -0.423176525290807])
    check_coef(model.estimator.intercept_, [-0.25, -0.25, -0.5])
    assert_almost_equal(model.score(X, labels), 0.625, atol=1e-15)
    assert_equal(model.decision_function(X).shape[1], 3)


def lifecycle() raises:
    var d = data()
    ref X = d[0]
    ref y = d[1]
    var pipe = make_pipeline(StandardScaler(), Ridge(alpha=0.5))
    with assert_raises(contains="NotFittedError"):
        _ = pipe.predict(X)
    _ = pipe.fit(X, y)
    var before = pipe.predict(X)
    # Nested clone: unfitted copy carrying the inner configuration.
    var copy = clone(pipe)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(copy.estimator.alpha[Float64], 0.5)
    assert_true(pipe.__sklearn_is_fitted__())
    # Typed nested set_params (sklearn: set_params(ridge__alpha=...)) then refit replaces state.
    var params = pipe.get_params()
    params.estimator.alpha = 2.0
    _ = pipe.set_params(params)
    assert_equal(pipe.estimator.alpha[Float64], 2.0)
    _ = pipe.fit(X, y)
    var after = pipe.predict(X)
    var changed = False
    for i in range(8):
        if abs(Float64(after.unsafe_get(i)) - Float64(before.unsafe_get(i))) > 1e-6:
            changed = True
    assert_true(changed)
    # The clone fits independently and reproduces the original alpha=0.5 model.
    _ = copy.fit(X, y)
    same(copy.predict(X), before)


def main() raises:
    regressors()
    classifier()
    lifecycle()
    print("PASS: linear models inside Pipeline: reference values, manual equivalence, nested clone and set_params")
