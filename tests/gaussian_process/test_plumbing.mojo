"""Exact validation/ownership contracts; numerical engines are not implemented."""
import numojo as nm
from numojo import Shape, Item
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import inf, nan
from moxie.gaussian_process._plumbing import (
    Dense64, HyperparameterSpec, KernelEvaluation, ObjectiveEvaluation,
    PredictionResult, OptimizationResult, LatentMoments,
    KernelContract, PositiveDefiniteAlgebra, LikelihoodObjective,
    HyperparameterOptimizer, OptimizerRestarts,
    validate_uncertainty, validate_gradient, validate_likelihood,
    validate_theta, retain_training,
)
from moxie._random import RandomState


@fieldwise_init
struct KernelProbe(KernelContract):
    comptime Input = Dense64
    comptime Params = Int
    def get_params(self, deep: Bool = True) raises -> Int: return 0
    def set_params(mut self, params: Int) raises: raise Error("Probe: no kernel")
    def hyperparameters(self) raises -> List[HyperparameterSpec]: raise Error("Probe: no kernel")
    def theta(self) raises -> Dense64: raise Error("Probe: no kernel")
    def bounds(self) raises -> Dense64: raise Error("Probe: no kernel")
    def set_theta(mut self, theta: Dense64) raises: raise Error("Probe: no kernel")
    def clone_with_theta(self, theta: Dense64) raises -> Self: raise Error("Probe: no kernel")
    def evaluate(self, X: Dense64, Y: Optional[Dense64] = None, eval_gradient: Bool = False) raises -> KernelEvaluation:
        raise Error("Probe: no kernel")
    def diag(self, X: Dense64) raises -> Dense64: raise Error("Probe: no kernel")
    def is_stationary(self) -> Bool: return False
    def requires_vector_input(self) -> Bool: return True


@fieldwise_init
struct AlgebraProbe(PositiveDefiniteAlgebra):
    def cholesky(self, matrix: Dense64) raises -> Dense64: raise Error("Probe: no algebra")
    def solve_triangular(self, lower: Dense64, rhs: Dense64, transpose: Bool = False) raises -> Dense64: raise Error("Probe: no algebra")
    def cho_solve(self, lower: Dense64, rhs: Dense64) raises -> Dense64: raise Error("Probe: no algebra")
    def log_determinant(self, lower: Dense64) raises -> Float64: raise Error("Probe: no algebra")


@fieldwise_init
struct ObjectiveProbe(LikelihoodObjective):
    def evaluate(mut self, theta: Dense64, eval_gradient: Bool = True) raises -> ObjectiveEvaluation:
        raise Error("Probe: no posterior")


@fieldwise_init
struct OptimizerProbe(HyperparameterOptimizer, OptimizerRestarts):
    comptime Objective = ObjectiveProbe
    def minimize(self, mut objective: ObjectiveProbe, initial_theta: Dense64, bounds: Dense64) raises -> OptimizationResult:
        raise Error("Probe: no optimizer")
    def optimize(self, mut objective: ObjectiveProbe, initial_theta: Dense64, bounds: Dense64, n_restarts: Int, mut rng: RandomState) raises -> OptimizationResult:
        raise Error("Probe: no optimizer")


def ownership[dtype: DType]() raises:
    var X = nm.arange[dtype](6).reshape(Shape(2, 3))
    var snapshot = retain_training(X)
    var aliasing = retain_training(X, False)
    X[Item(0, 0)] = 42
    assert_equal(snapshot.item(0, 0), 0)
    assert_equal(aliasing.item(0, 0), 42)
    aliasing[Item(1, 2)] = 17
    assert_equal(X.item(1, 2), 17)
    var strided = X.view_with_layout(Shape(2, 2), nm.NDArrayStrides(3, -1), 2)
    var kept = retain_training(strided)
    assert_equal(kept.item(0, 0), 2)
    assert_equal(kept.item(0, 1), 1)
    var replacement = nm.ones[dtype](Shape(1, 1))
    snapshot = retain_training(replacement)
    assert_equal(snapshot.shape[0], 1)
    assert_equal(X.item(0, 0), 42)
    var empty = nm.zeros[dtype](Shape(0, 2))
    var vector = nm.zeros[dtype](Shape(2))
    with assert_raises():
        _ = retain_training(empty)
    with assert_raises():
        _ = retain_training(vector)


def main() raises:
    ownership[DType.float32]()
    ownership[DType.float64]()
    for std_flag in range(2):
        for cov_flag in range(2):
            if std_flag == 1 and cov_flag == 1:
                with assert_raises(): validate_uncertainty(True, True)
            else:
                validate_uncertainty(Bool(std_flag), Bool(cov_flag))
    validate_gradient(False, True)
    validate_gradient(True, False)
    with assert_raises(): validate_gradient(True, True)
    validate_likelihood(False, False)
    validate_likelihood(True, True)
    with assert_raises(): validate_likelihood(False, True)
    var theta = nm.zeros[DType.float64](Shape(2))
    var bounds = nm.zeros[DType.float64](Shape(2, 2))
    bounds[Item(0, 1)] = 1
    bounds[Item(1, 1)] = 2
    validate_theta(theta, bounds, 2, 1)
    var spec = HyperparameterSpec("length_scale", 2, False, bounds)
    var cloned = spec.copy()
    bounds[Item(0, 1)] = 3
    cloned.bounds[Item(0, 1)] = 4
    assert_equal(spec.bounds.item(0, 1), 1)
    assert_equal(spec.n_dims(), 2)
    var empty_bounds = nm.zeros[DType.float64](Shape(0, 2))
    var empty_theta = nm.zeros[DType.float64](Shape(0))
    var fixed = HyperparameterSpec("noise_level", 1, True, empty_bounds)
    assert_equal(fixed.n_dims(), 0)
    validate_theta(empty_theta, empty_bounds, 0)
    with assert_raises(): _ = HyperparameterSpec("x", 0, False, bounds)
    with assert_raises(): _ = HyperparameterSpec("x", 1, False, bounds)
    with assert_raises(): validate_theta(theta, bounds, 1)
    with assert_raises(): validate_theta(theta, bounds, 2, -1)
    with assert_raises(): validate_theta(theta, empty_theta, 2)
    bounds[Item(0, 0)] = -inf[DType.float64]()
    validate_theta(theta, bounds, 2)
    with assert_raises(): validate_theta(theta, bounds, 2, 1)
    bounds[Item(0, 0)] = 10
    with assert_raises(): validate_theta(theta, bounds, 2)
    bounds[Item(0, 0)] = nan[DType.float64]()
    with assert_raises(): validate_theta(theta, bounds, 2)
    bounds[Item(0, 0)] = 0
    theta[Item(0)] = nan[DType.float64]()
    with assert_raises(): validate_theta(theta, bounds, 2)
    var kernel = KernelProbe()
    var algebra = AlgebraProbe()
    var objective = ObjectiveProbe()
    var optimizer = OptimizerProbe()
    var rng = RandomState(42)
    with assert_raises(): _ = kernel.evaluate(bounds)
    with assert_raises(): _ = algebra.cholesky(bounds)
    with assert_raises(): _ = algebra.solve_triangular(bounds, bounds)
    with assert_raises(): _ = algebra.cho_solve(bounds, bounds)
    with assert_raises(): _ = algebra.log_determinant(bounds)
    with assert_raises(): _ = objective.evaluate(theta)
    with assert_raises(): _ = optimizer.minimize(objective, theta, bounds)
    with assert_raises(): _ = optimizer.optimize(objective, theta, bounds, 0, rng)
    var result = PredictionResult(bounds.copy(), None, None)
    assert_false(result.std)
    assert_false(result.covariance)
    var latent = LatentMoments(theta.copy(), theta.copy())
    assert_equal(latent.mean.shape[0], 2)
    print("PASS: GP plumbing validation, ownership and raising extension probes")
