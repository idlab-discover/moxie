"""Native checks for GP algebra adapters, L-BFGS-B adapter and restarts."""
from numojo import NDArray, Shape, zeros
from std.math import abs, log
from std.testing import assert_almost_equal, assert_equal, assert_raises, assert_true, TestSuite

from moxie._random import RandomState
from moxie.gaussian_process._algebra import CholeskyAlgebra, identity
from moxie.gaussian_process._optimization import (
    LBFGSBOptimizer,
    RestartingOptimizer,
    restart_initial_theta,
)
from moxie.gaussian_process._plumbing import Dense64, LikelihoodObjective, ObjectiveEvaluation


def matrix(rows: Int, cols: Int, values: List[Float64]) raises -> Dense64:
    var out = zeros[DType.float64](Shape(rows, cols))
    for i in range(len(values)):
        out.unsafe_set(i, values[i])
    return out^


def vector(values: List[Float64]) raises -> Dense64:
    var out = zeros[DType.float64](Shape(len(values)))
    for i in range(len(values)):
        out.unsafe_set(i, values[i])
    return out^


def test_scalar_cholesky_contract() raises:
    var algebra = CholeskyAlgebra()
    var a = matrix(1, 1, [4.0])
    var l = algebra.cholesky(a)
    assert_equal(l.item(0, 0), 2.0)
    assert_equal(algebra.solve_triangular(l, matrix(1, 1, [6.0])).item(0, 0), 3.0)
    assert_equal(algebra.cho_solve(l, matrix(1, 1, [6.0])).item(0, 0), 1.5)
    assert_almost_equal(algebra.log_determinant(l), log(4.0), atol=1e-15)
    assert_equal(a.item(0, 0), 4.0)  # input unchanged


def test_cho_solve_reconstructs() raises:
    var algebra = CholeskyAlgebra()
    var a = matrix(3, 3, [4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0])
    var l = algebra.cholesky(a)
    var inv = algebra.cho_solve(l, identity(3))
    for i in range(3):
        for j in range(3):
            var acc: Float64 = 0.0
            for k in range(3):
                acc += a.item(i, k) * inv.item(k, j)
            assert_almost_equal(acc, 1.0 if i == j else 0.0, atol=1e-14)
    var y = algebra.solve_triangular(l, identity(3), transpose=True)
    # L.T y = I  =>  y = L^-T, so L.T @ y is the identity.
    for i in range(3):
        for j in range(3):
            var acc: Float64 = 0.0
            for k in range(3):
                acc += l.item(k, i) * y.item(k, j)
            assert_almost_equal(acc, 1.0 if i == j else 0.0, atol=1e-14)


def test_not_positive_definite_is_explicit() raises:
    var algebra = CholeskyAlgebra()
    with assert_raises(contains="not positive definite"):
        _ = algebra.cholesky(matrix(2, 2, [1.0, 2.0, 2.0, 1.0]))
    with assert_raises(contains="ValueError"):
        _ = algebra.cholesky(matrix(2, 3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


@fieldwise_init
struct Quadratic(LikelihoodObjective):
    """(theta - center)^2 summed; counts evaluations."""

    var center: Float64
    var calls: Int

    def evaluate(mut self, theta: Dense64, eval_gradient: Bool = True) raises -> ObjectiveEvaluation:
        self.calls += 1
        var value: Float64 = 0.0
        var g = zeros[DType.float64](Shape(theta.size))
        for i in range(theta.size):
            var d = theta.unsafe_get(i) - self.center
            value += d * d
            g.unsafe_set(i, 2.0 * d)
        return ObjectiveEvaluation(value=value, gradient=g^)


def test_lbfgsb_reaches_interior_and_bound() raises:
    var optimizer = LBFGSBOptimizer[Quadratic]()
    var objective = Quadratic(1.0, 0)
    var result = optimizer.minimize(objective, vector([0.0]), matrix(1, 2, [-2.0, 2.0]))
    assert_true(result.converged)
    assert_almost_equal(result.theta.item(0), 1.0, atol=1e-6)
    assert_true(objective.calls > 1)
    var far = Quadratic(5.0, 0)
    var bounded = optimizer.minimize(far, vector([0.0, -1.0]), matrix(2, 2, [-2.0, 2.0, -2.0, 2.0]))
    assert_equal(bounded.theta.item(0), 2.0)
    assert_equal(bounded.theta.item(1), 2.0)


def test_restarts_draw_uniformly_and_pick_minimum() raises:
    var bounds = matrix(2, 2, [-1.0, 3.0, 0.0, 2.0])
    var rng = RandomState(7)
    var expected = RandomState(7)
    var start = restart_initial_theta(bounds, rng)
    assert_equal(start.item(0), expected.uniform(-1.0, 3.0))
    assert_equal(start.item(1), expected.uniform(0.0, 2.0))
    var restarts = RestartingOptimizer(LBFGSBOptimizer[Quadratic]())
    var objective = Quadratic(0.5, 0)
    var stream = RandomState(0)
    var outcome = restarts.run_all(objective, vector([2.5, 1.5]), bounds, 3, stream)
    assert_almost_equal(outcome.best.theta.item(0), 0.5, atol=1e-5)
    assert_almost_equal(outcome.best.theta.item(1), 0.5, atol=1e-5)
    assert_equal(len(outcome.warnings), 0)
    var infinite = zeros[DType.float64](Shape(1, 2))
    infinite.unsafe_set(0, -1.0)
    infinite.unsafe_set(1, Float64.MAX * 2.0)
    with assert_raises(contains="requires that all bounds are finite"):
        _ = restarts.run_all(objective, vector([0.0]), infinite, 1, stream)
    with assert_raises(contains="bounds must have shape (n_dims, 2)"):
        _ = restarts.run_all(objective, vector([0.0, 1.0, 2.0]), bounds, 0, stream)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
