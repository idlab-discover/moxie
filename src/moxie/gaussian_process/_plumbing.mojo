"""Private family contracts and validation only; no GP numerical engine."""
import numojo as nm
from std.math import isfinite, isnan

comptime Dense64 = nm.NDArray[DType.float64]


@fieldwise_init
struct KernelEvaluation(Movable):
    var covariance: Dense64
    var gradient: Optional[Dense64]


@fieldwise_init
struct ObjectiveEvaluation(Movable):
    var value: Float64
    var gradient: Optional[Dense64]


@fieldwise_init
struct OptimizationResult(Movable):
    var theta: Dense64
    var value: Float64
    var converged: Bool
    var message: String


@fieldwise_init
struct PredictionResult(Movable):
    var mean: Dense64
    var std: Optional[Dense64]
    var covariance: Optional[Dense64]


@fieldwise_init
struct LatentMoments(Movable):
    var mean: Dense64
    var variance: Dense64


struct HyperparameterSpec(Copyable, Movable):
    """Internal normalized metadata; bounds are in original parameter units."""
    var name: String
    var n_elements: Int
    var fixed: Bool
    var bounds: Dense64

    def __init__(out self, name: String, n_elements: Int, fixed: Bool, bounds: Dense64) raises:
        if n_elements < 1:
            raise Error("ValueError: n_elements must be positive")
        # The kernel owner expands broadcast bounds before constructing this record.
        if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] != (0 if fixed else n_elements):
            raise Error("ValueError: expected normalized hyperparameter bounds")
        self.name = name
        self.n_elements = n_elements
        self.fixed = fixed
        self.bounds = bounds.copy()

    def n_dims(self) -> Int:
        return 0 if self.fixed else self.n_elements


trait KernelContract(Copyable, Movable):
    comptime Input: Movable
    comptime Params: Movable

    def get_params(self, deep: Bool = True) raises -> Self.Params: ...
    def set_params(mut self, params: Self.Params) raises: ...
    def hyperparameters(self) raises -> List[HyperparameterSpec]: ...
    def theta(self) raises -> Dense64: ...
    def bounds(self) raises -> Dense64: ...
    def set_theta(mut self, theta: Dense64) raises: ...
    def clone_with_theta(self, theta: Dense64) raises -> Self: ...
    def evaluate(self, X: Self.Input, Y: Optional[Self.Input] = None, eval_gradient: Bool = False) raises -> KernelEvaluation: ...
    def diag(self, X: Self.Input) raises -> Dense64: ...
    def is_stationary(self) -> Bool: ...
    def requires_vector_input(self) -> Bool: ...


trait PositiveDefiniteAlgebra:
    """Engine extension point: lower factors, rank-2 RHS, owned results."""
    def cholesky(self, matrix: Dense64) raises -> Dense64: ...
    def solve_triangular(self, lower: Dense64, rhs: Dense64, transpose: Bool = False) raises -> Dense64: ...
    def cho_solve(self, lower: Dense64, rhs: Dense64) raises -> Dense64: ...
    def log_determinant(self, lower: Dense64) raises -> Float64: ...


trait LikelihoodObjective(Movable):
    """Minimization objective: negative LML and its log-theta gradient."""
    def evaluate(mut self, theta: Dense64, eval_gradient: Bool = True) raises -> ObjectiveEvaluation: ...


trait HyperparameterOptimizer:
    comptime Objective: LikelihoodObjective
    def minimize(self, mut objective: Self.Objective, initial_theta: Dense64, bounds: Dense64) raises -> OptimizationResult: ...


trait OptimizerRestarts:
    comptime Objective: LikelihoodObjective
    # The supplied RNG is advanced in place, never reseeded by this engine.
    def optimize(self, mut objective: Self.Objective, initial_theta: Dense64, bounds: Dense64, n_restarts: Int, mut rng: RandomState) raises -> OptimizationResult: ...


from moxie._random import RandomState


def validate_uncertainty(return_std: Bool, return_cov: Bool) raises:
    if return_std and return_cov:
        raise Error("ValueError: at most one of return_std or return_cov can be requested")


def validate_gradient(has_y: Bool, eval_gradient: Bool) raises:
    if has_y and eval_gradient:
        raise Error("ValueError: kernel gradients require Y=None")


def validate_likelihood(has_theta: Bool, eval_gradient: Bool) raises:
    if eval_gradient and not has_theta:
        raise Error("ValueError: gradient requires explicit theta")


def validate_theta(theta: Dense64, bounds: Dense64, n_dims: Int, n_restarts: Int = 0) raises:
    """Validate normalized log-space optimizer inputs, without clipping theta."""
    if n_dims < 0 or n_restarts < 0:
        raise Error("ValueError: dimensions and restart count must be nonnegative")
    if theta.ndim != 1 or theta.shape[0] != n_dims:
        raise Error("ValueError: theta has incorrect shape")
    if bounds.ndim != 2 or bounds.shape[0] != n_dims or bounds.shape[1] != 2:
        raise Error("ValueError: bounds must have shape (n_dims, 2)")
    for i in range(n_dims):
        if not isfinite(theta.item(i)):
            raise Error("ValueError: theta must be finite")
        var low = bounds.item(i, 0)
        var high = bounds.item(i, 1)
        if isnan(low) or isnan(high) or low > high:
            raise Error("ValueError: invalid bounds interval")
        if n_restarts > 0 and (not isfinite(low) or not isfinite(high)):
            raise Error("ValueError: optimizer restarts require finite bounds")


def retain_training[dtype: DType](mut X: nm.NDArray[dtype], copy_X_train: Bool = True) raises -> nm.NDArray[dtype]:
    """Dense feature retention; false deliberately preserves caller aliasing."""
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise Error("ValueError: expected nonempty dense feature matrix")
    if copy_X_train:
        return X.copy()
    return X.view()
