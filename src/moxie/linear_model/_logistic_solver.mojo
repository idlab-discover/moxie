# Adapted from scikit-learn sklearn/linear_model/_glm/_newton_solver.py
# and sklearn/utils/optimize.py.
# Authors: The scikit-learn developers
# Original SciPy authors: Travis Oliphant, Eric Jones
# SPDX-License-Identifier: BSD-3-Clause
"""Native dense solvers for the smooth logistic objective."""

from std.math import abs, inf, isfinite, max, sqrt
from mojo_optimize import Config, EvaluationState, LBFGSB, LBFGSBConfig, LineSearch, LineSearchConfig
from mojo_optimize._vectors import dot
from numojo import NDArray, Shape, full, zeros
from moxie._linear_linalg import IndefiniteFactor, symmetric_indefinite
from moxie.linear_model._linear_loss import LogisticLoss


@fieldwise_init
struct LogisticSolution(Movable):
    var coefficients: NDArray[DType.float64]
    var iterations: Int


def logistic_lbfgs[dtype: DType](
    mut objective: LogisticLoss[dtype], initial: NDArray[DType.float64],
    max_iter: Int, tol: Float64,
) raises -> LogisticSolution:
    var size = initial.size
    var optimizer = LBFGSB[DType.float64](size, LBFGSBConfig(
        max_iterations=max_iter, function_tolerance=1.4210854715202004e-14,
        projected_gradient_tolerance=tol, max_line_search_steps=50,
    ))
    var lower = full[DType.float64](Shape(size), -inf[DType.float64]())
    var upper = full[DType.float64](Shape(size), inf[DType.float64]())
    _ = optimizer.minimize(objective, initial, lower, upper)
    if not optimizer.state.has_accepted:
        raise Error(String("RuntimeError: LogisticRegression lbfgs failed: ", optimizer.message()))
    return LogisticSolution(optimizer.state.x.copy(), min(optimizer.state.iterations, max_iter))


def _gradient_norm(gradient: NDArray[DType.float64], infinity: Bool) -> Float64:
    var norm = Float64(0)
    for j in range(gradient.size):
        var value = abs(gradient.unsafe_get(j))
        if not isfinite(value):
            return inf[DType.float64]()
        norm = max(norm, value) if infinity else norm + value
    return norm


def _newton_line_search[dtype: DType](
    mut objective: LogisticLoss[dtype], current: NDArray[DType.float64],
    gradient: NDArray[DType.float64], direction: NDArray[DType.float64],
    active: List[Int], slope: Float64, loss: Float64, tol: Float64,
    mut trial: NDArray[DType.float64], mut trial_gradient: NDArray[DType.float64],
) raises -> Bool:
    comptime armijo = 0.00048828125
    comptime epsilon = 3.552713678800501e-15  # 16 * Float64 epsilon.
    comptime min_step = 1e-12
    var step = Float64(1)
    var previous_step = Float64(1)
    var previous_loss = loss
    var old_gradient_norm = _gradient_norm(gradient, False)
    for i in range(21):
        for j in range(len(active)):
            var index = active[j]
            trial.unsafe_set(index, current.unsafe_get(index) + step * direction.unsafe_get(j))
        var trial_loss = objective.evaluate(trial, trial_gradient)
        var change = trial_loss - loss
        if isfinite(trial_loss):
            if change <= step * armijo * slope:
                return True
            if i == 0 and abs(armijo * slope) <= tol and _gradient_norm(trial_gradient, True) <= tol:
                return True
            if abs(change) <= abs(loss * epsilon) and _gradient_norm(trial_gradient, False) < old_gradient_norm:
                return True

        # Interpolate the loss at the trial step and, from the second
        # rejection onward, the previous trial step.
        var residual = change - slope * step
        var next_step = -slope * step * step / (2 * residual)
        if i > 0:
            var old_residual = previous_loss - loss - slope * previous_step
            var product = previous_step * step
            var denominator = product * product * (step - previous_step)
            var a = (previous_step * previous_step * residual - step * step * old_residual) / denominator
            var b = (step * step * step * old_residual - previous_step * previous_step * previous_step * residual) / denominator
            var discriminant = b * b - 3 * a * slope
            if a != 0 and discriminant >= 0:
                next_step = (-b + sqrt(discriminant)) / (3 * a)
        previous_step = step
        previous_loss = trial_loss
        if not isfinite(next_step) or next_step >= step or next_step <= min_step:
            next_step = 0.5 * step
        step = max(next_step, max(0.01 * step, min_step))
    return False


def _has_tiny_pivot(factor: IndefiniteFactor[DType.float64]) -> Bool:
    if factor.is_singular():
        return True
    var diagonal = factor.diagonal_factor()
    var blocks = factor.blocks()
    var order = len(blocks)
    var largest = Float64(0)
    var smallest = inf[DType.float64]()
    for i in range(order):
        if blocks[i] == 0:
            continue
        var pivot = abs(diagonal.unsafe_get(i * order + i))
        if blocks[i] == 2:
            # Complete pivoting bounds both diagonal entries relative to this
            # off-diagonal, so it sets the magnitude of the 2-by-2 block.
            pivot = abs(diagonal.unsafe_get((i + 1) * order + i))
        smallest = min(smallest, pivot)
        largest = max(largest, pivot)
    return smallest <= 2.220446049250313e-16 * largest


def logistic_newton[dtype: DType](
    mut objective: LogisticLoss[dtype], var current: NDArray[DType.float64],
    outputs: Int, features: Int, fit_intercept: Bool, l2: Float64,
    max_iter: Int, tol: Float64,
) raises -> LogisticSolution:
    if outputs > 1:
        # Move warm starts to the Newton system's reference-class coordinates.
        var first = 0 if l2 == 0 else features
        for j in range(first, features + Int(fit_intercept)):
            var last = current.unsafe_get(j * outputs + outputs - 1)
            for k in range(outputs):
                var index = j * outputs + k
                current.unsafe_set(index, current.unsafe_get(index) - last)

    if max_iter == 0:
        return LogisticSolution(current^, 0)
    var active = objective.active_parameters()
    var trial = current.copy()
    var gradient = zeros[DType.float64](current.shape)
    var trial_gradient = zeros[DType.float64](current.shape)
    var rhs = zeros[DType.float64](Shape(len(active)))
    var accepted = 0
    while accepted < max_iter:
        var loss = objective.gradient_hessian(current, gradient)
        if objective.hessian_warning or not isfinite(loss):
            break
        for j in range(len(active)):
            rhs.unsafe_set(j, -gradient.unsafe_get(active[j]))
        var direction: NDArray[DType.float64]
        try:
            var factor = symmetric_indefinite(objective.hessian)
            if _has_tiny_pivot(factor):
                break
            direction = factor.solve(rhs)
        except:
            break
        var slope = Float64(0)
        for j in range(len(active)):
            slope -= rhs.unsafe_get(j) * direction.unsafe_get(j)
        if not isfinite(slope) or slope > 0:
            break
        if not _newton_line_search(objective, current, gradient, direction, active, slope, loss, tol, trial, trial_gradient):
            break
        swap(current, trial)
        accepted += 1
        # For H d = -g, the squared Newton decrement d' H d equals -g' d.
        if _gradient_norm(trial_gradient, True) <= tol and -0.5 * slope <= tol:
            return LogisticSolution(current^, accepted)

    if accepted < max_iter:
        # The failed Newton iteration consumes the first slot of the fallback budget.
        var solution = logistic_lbfgs(objective, current, max_iter - accepted - 1, tol)
        solution.iterations += accepted
        return solution^
    return LogisticSolution(current^, accepted)

def _cg_direction[dtype: DType](
    mut objective: LogisticLoss[dtype], current: NDArray[DType.float64],
    gradient: NDArray[DType.float64], mut direction: NDArray[DType.float64],
    mut residual: NDArray[DType.float64], mut search: NDArray[DType.float64],
    mut product: NDArray[DType.float64], mut projected: List[Float64],
) raises:
    var size = gradient.size
    var magnitude = _gradient_norm(gradient, False)
    var target = min(0.5, sqrt(magnitude)) * magnitude
    for j in range(size):
        var value = gradient.unsafe_get(j)
        direction.unsafe_set(j, 0)
        residual.unsafe_set(j, value)
        search.unsafe_set(j, -value)
    var residual_squared = dot(residual, residual)
    comptime curvature_eps = 3.552713678800501e-15  # 16 * Float64 epsilon.
    for iteration in range(201):
        if _gradient_norm(residual, False) <= target:
            break
        objective.hessian_product(current, search, product, projected)
        var curvature = dot(search, product)
        if not isfinite(curvature):
            break
        if curvature <= 0:
            var norm_squared = dot(search, search)
            if curvature < -curvature_eps * norm_squared:
                # Orient the normalized negative-curvature direction downhill.
                var scale = abs(curvature / norm_squared) / sqrt(norm_squared)
                if dot(search, gradient) > 0:
                    scale = -scale
                for j in range(size):
                    direction.unsafe_set(j, scale * search.unsafe_get(j))
            elif iteration == 0:
                for j in range(size):
                    direction.unsafe_set(j, search.unsafe_get(j))
            break
        var step = residual_squared / curvature
        if not isfinite(step):
            break
        for j in range(size):
            direction.unsafe_set(j, direction.unsafe_get(j) + step * search.unsafe_get(j))
            residual.unsafe_set(j, residual.unsafe_get(j) + step * product.unsafe_get(j))
        var next_squared = dot(residual, residual)
        var beta = next_squared / residual_squared
        for j in range(size):
            search.unsafe_set(j, -residual.unsafe_get(j) + beta * search.unsafe_get(j))
        residual_squared = next_squared


def logistic_newton_cg[dtype: DType](
    mut objective: LogisticLoss[dtype], var initial: NDArray[DType.float64],
    outputs: Int, max_iter: Int, tol: Float64,
) raises -> LogisticSolution:
    if max_iter == 0:
        return LogisticSolution(initial^, 0)
    var size = initial.size
    var state = EvaluationState[DType.float64](
        size, Config(max_iterations=max_iter, max_evaluations=Int.MAX),
    )
    if not state.initialize(objective, initial):
        raise Error("RuntimeError: LogisticRegression newton-cg could not evaluate its initial state")
    var line_search = LineSearch[DType.float64](
        size, LineSearchConfig(min_step=1e-8, max_step=50),
    )
    var direction = zeros[DType.float64](initial.shape)
    var residual = zeros[DType.float64](initial.shape)
    var search = zeros[DType.float64](initial.shape)
    var product = zeros[DType.float64](initial.shape)
    var projected = List[Float64](length=outputs, fill=0)
    var previous_loss = state.value
    while state.iterations < max_iter:
        if _gradient_norm(state.gradient, True) <= tol:
            break
        _cg_direction(objective, state.x, state.gradient, direction, residual, search, product, projected)
        var slope = dot(state.gradient, direction)
        if not isfinite(slope) or slope >= 0:
            break
        var initial_step = Float64(1)
        if state.iterations > 0:
            var estimate = 2.02 * (state.value - previous_loss) / slope
            if estimate > 0:
                initial_step = min(1.0, estimate)
        previous_loss = state.value
        if not line_search.search(state, objective, direction, initial_step):
            break
        if not state.accept_trial():
            break
    return LogisticSolution(state.x.copy(), state.iterations)
