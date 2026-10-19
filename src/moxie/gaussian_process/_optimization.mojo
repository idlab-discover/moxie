"""Hyperparameter optimization shared by GP posteriors.

`LBFGSBOptimizer` adapts a family `LikelihoodObjective` (negative log marginal
likelihood and its log-theta gradient) to the pinned mojo-optimize L-BFGS-B
port, called exactly as sklearn calls `scipy.optimize.minimize(obj, theta0,
method="L-BFGS-B", jac=True, bounds=bounds)` with SciPy's default options.
`optimize_with_restarts` implements sklearn's restart policy.
"""
from numojo import NDArray, Shape, zeros
from std.math import inf, isfinite

from moxie._optimize import LBFGSB, LBFGSBConfig, Objective, Status
from moxie._random import RandomState
from moxie.gaussian_process._plumbing import (
    Dense64,
    HyperparameterOptimizer,
    LikelihoodObjective,
    ObjectiveEvaluation,
    OptimizationResult,
    OptimizerRestarts,
    validate_theta,
)


struct _ObjectiveAdapter[O: LikelihoodObjective, origin: MutOrigin](Objective):
    """mojo-optimize callback view of a borrowed family objective."""

    comptime dtype = DType.float64
    var target: Pointer[Self.O, Self.origin]

    def __init__(out self, ref[Self.origin] target: Self.O):
        self.target = Pointer(to=target)

    def evaluate(
        mut self, x: NDArray[DType.float64], mut gradient: NDArray[DType.float64]
    ) raises -> Float64:
        var result = self.target[].evaluate(x.copy(), True)
        if not result.gradient:
            raise Error("ValueError: likelihood objective returned no gradient")
        ref g = result.gradient.value()
        if g.size != gradient.size:
            raise Error(
                "ValueError: objective gradient has "
                + String(g.size)
                + " entries, expected "
                + String(gradient.size)
            )
        var stride = Int(g.strides[0])
        for i in range(gradient.size):
            gradient.unsafe_set(i, g.unsafe_get(i * stride))
        return result.value


struct LBFGSBOptimizer[O: LikelihoodObjective](HyperparameterOptimizer, ImplicitlyCopyable, Movable):
    """sklearn `optimizer="fmin_l_bfgs_b"`: bounded L-BFGS-B with SciPy defaults.

    The returned value is SciPy's `res.fun` (last evaluated value), which sklearn
    uses for restart selection; `converged` is False and `message` carries
    SciPy's text when SciPy would report status != 0 (sklearn then warns).
    """

    comptime Objective = Self.O
    var history_size: Int
    var max_iterations: Int
    var max_evaluations: Int
    var function_tolerance: Float64
    var projected_gradient_tolerance: Float64
    var max_line_search_steps: Int

    def __init__(
        out self,
        history_size: Int = 10,
        max_iterations: Int = 15000,
        max_evaluations: Int = 15000,
        function_tolerance: Float64 = 2.2204460492503131e-09,
        projected_gradient_tolerance: Float64 = 1e-5,
        max_line_search_steps: Int = 20,
    ):
        self.history_size = history_size
        self.max_iterations = max_iterations
        self.max_evaluations = max_evaluations
        self.function_tolerance = function_tolerance
        self.projected_gradient_tolerance = projected_gradient_tolerance
        self.max_line_search_steps = max_line_search_steps

    def minimize(
        self, mut objective: Self.O, initial_theta: Dense64, bounds: Dense64
    ) raises -> OptimizationResult:
        var n = initial_theta.size
        validate_theta(initial_theta, bounds, n)
        if n == 0:
            var value = objective.evaluate(initial_theta.copy(), False).value
            return OptimizationResult(
                theta=initial_theta.copy(), value=value, converged=True, message="no free hyperparameters"
            )
        var lower = zeros[DType.float64](Shape(n))
        var upper = zeros[DType.float64](Shape(n))
        for i in range(n):
            lower.unsafe_set(i, bounds.item(i, 0))
            upper.unsafe_set(i, bounds.item(i, 1))
        # sklearn's likelihood returns (+inf, 0) on Cholesky failure; SciPy
        # propagates such values into its line search, so do the same.
        var config = LBFGSBConfig(
            history_size=self.history_size,
            max_iterations=self.max_iterations,
            max_evaluations=self.max_evaluations,
            function_tolerance=self.function_tolerance,
            projected_gradient_tolerance=self.projected_gradient_tolerance,
            max_line_search_steps=self.max_line_search_steps,
            propagate_nonfinite=True,
        )
        var solver = LBFGSB[DType.float64](n, config^)
        var adapter = _ObjectiveAdapter(objective)
        var status = solver.minimize(adapter, initial_theta, lower, upper)
        if status == Status.callback_error:
            raise Error("ValueError: likelihood evaluation failed during optimization: " + solver.state.error_message)
        if not solver.state.has_accepted:
            raise Error("ValueError: L-BFGS-B could not evaluate the initial theta: " + solver.message())
        var message = solver.message()
        var converged = solver.scipy_warnflag() == 0
        if not converged:
            message = (
                "lbfgs failed to converge after "
                + String(solver.state.iterations)
                + " iteration(s) (status="
                + String(solver.scipy_warnflag())
                + "): "
                + message
            )
        return OptimizationResult(
            theta=solver.state.x.copy(),
            value=Float64(solver.scipy_result_value),
            converged=converged,
            message=message,
        )


def restart_initial_theta(bounds: Dense64, mut rng: RandomState) raises -> Dense64:
    """numpy `rng.uniform(bounds[:, 0], bounds[:, 1])`, coordinates in order."""
    var n = bounds.shape[0]
    var theta = zeros[DType.float64](Shape(n))
    for i in range(n):
        theta.unsafe_set(i, rng.uniform(bounds.item(i, 0), bounds.item(i, 1)))
    return theta^


def check_restart_bounds(bounds: Dense64) raises:
    for i in range(bounds.size):
        if not isfinite(bounds.unsafe_get(i)):
            raise Error(
                "ValueError: Multiple optimizer restarts (n_restarts_optimizer>0)"
                " requires that all bounds are finite."
            )


def select_best(var results: List[OptimizationResult]) raises -> OptimizationResult:
    """`np.argmin` of the minimized values: first minimum wins ties."""
    if len(results) == 0:
        raise Error("ValueError: no optimizer results to select from")
    var best = 0
    for i in range(1, len(results)):
        if results[i].value < results[best].value:
            best = i
    return results.pop(best)


struct RestartingOptimizer[Opt: HyperparameterOptimizer & Copyable & Movable & Deinitable](Movable, OptimizerRestarts):
    """sklearn restart policy around any native optimizer.

    Run 0 starts at the supplied theta; each restart draws theta uniformly in
    the (finite) log bounds from the borrowed RandomState, in coordinate order.
    The lowest minimized value wins (first on ties).
    """

    comptime Objective = Self.Opt.Objective
    var optimizer: Self.Opt

    def __init__(out self, var optimizer: Self.Opt):
        self.optimizer = optimizer^

    def optimize(
        self,
        mut objective: Self.Opt.Objective,
        initial_theta: Dense64,
        bounds: Dense64,
        n_restarts: Int,
        mut rng: RandomState,
    ) raises -> OptimizationResult:
        return self.run_all(objective, initial_theta, bounds, n_restarts, rng).take_best()

    def run_all(
        self,
        mut objective: Self.Opt.Objective,
        initial_theta: Dense64,
        bounds: Dense64,
        n_restarts: Int,
        mut rng: RandomState,
    ) raises -> RestartOutcome:
        """All runs' warnings plus the selected optimum."""
        if n_restarts < 0:
            raise Error("ValueError: n_restarts_optimizer must be nonnegative")
        var results = List[OptimizationResult]()
        results.append(self.optimizer.minimize(objective, initial_theta, bounds))
        if n_restarts > 0:
            check_restart_bounds(bounds)
            for _ in range(n_restarts):
                var start = restart_initial_theta(bounds, rng)
                results.append(self.optimizer.minimize(objective, start, bounds))
        var warnings = List[String]()
        for i in range(len(results)):
            if not results[i].converged:
                warnings.append(results[i].message)
        return RestartOutcome(best=select_best(results^), warnings=warnings^)


@fieldwise_init
struct RestartOutcome(Movable):
    """Selected optimum and the convergence warnings of all runs."""

    var best: OptimizationResult
    var warnings: List[String]

    def take_best(deinit self) -> OptimizationResult:
        return self.best^
