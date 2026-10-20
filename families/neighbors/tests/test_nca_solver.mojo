"""Deterministic stopping-order and rollback regressions for NCA's optimizer."""
import numojo as nm
from std.testing import assert_equal
from evaluate_nca_solver import Objective
from moxie.neighbors._nca_lbfgs import minimize, Stop
from moxie.neighbors import NCACallback
from moxie.neighbors._nca_math import Matrix


struct Count(NCACallback):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def __call__(mut self, x: Matrix, iteration: Int) raises:
        self.calls += 1
        assert_equal(self.calls, iteration)
        assert_equal(x.ndim, 1)


def main() raises:
    for kind in [0, 2, 3, 5]:
        var x = nm.zeros[DType.float64](nm.Shape(1, 2))
        x.unsafe_set(0, Float64(2 if kind == 5 else 1))
        var count = Count()
        var result = minimize(Objective(kind), x^, count, 2, 0)
        assert_equal(count.calls, result.iterations)
        if kind == 0:
            assert_equal(result.status, Stop.gradient)
            assert_equal(result.iterations, 1)
            assert_equal(result.x.unsafe_get(0), 0)
        elif kind == 2:
            assert_equal(result.status, Stop.iterations)
            assert_equal(result.iterations, 2)
        else:
            assert_equal(result.status, Stop.search)
            assert_equal(result.x.unsafe_get(0), 1)
            assert_equal(result.restarts, 1 if kind == 5 else 0)
            assert_equal(result.iterations, 1 if kind == 5 else 0)
    for maxfun in [1, 2]:
        for maxiter in [1, 2]:
            var x = nm.zeros[DType.float64](nm.Shape(1, 2))
            x.unsafe_set(0, 1)
            var count = Count()
            var result = minimize(Objective(0), x^, count, maxiter, 1e-5,
                                  maxfun=maxfun)
            var expected = Stop.gradient
            if maxiter == 1:
                expected = Stop.iterations
            elif maxfun == 1:
                expected = Stop.evaluations
            assert_equal(result.status, expected)
            assert_equal(result.evaluations, 2)
            assert_equal(count.calls, 1)
    var x = nm.zeros[DType.float64](nm.Shape(1, 2))
    var count = Count()
    var result = minimize(Objective(0), x^, count, 1, 0)
    assert_equal(result.status, Stop.gradient)
    assert_equal(result.iterations, 0)
    assert_equal(result.evaluations, 1)
    assert_equal(count.calls, 0)
    print("PASS: NCA solver budgets, convergence, warnings, rollback and retry")
