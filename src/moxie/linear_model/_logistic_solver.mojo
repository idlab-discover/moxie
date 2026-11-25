# Adapted from scikit-learn sklearn/linear_model/_glm/_newton_solver.py and
# sklearn/utils/optimize.py.
# Authors: The scikit-learn developers
# Original SciPy authors: Travis Oliphant, Eric Jones
# SPDX-License-Identifier: BSD-3-Clause
"""Native solvers for dense logistic objectives."""

from std.math import abs, copysign, inf, isfinite, max, sqrt
from mojo_optimize import Config, EvaluationState, LBFGSB, LBFGSBConfig, LineSearch, LineSearchConfig
from mojo_optimize._vectors import dot
from numojo import NDArray, Shape, full, zeros
from numojo.routines.math.extrema import max as array_max
from moxie._dense import row_squared_norms
from moxie._linear_linalg import IndefiniteFactor, symmetric_indefinite, symmetric_pinv
from moxie.linear_model._sag import SAGSolution, dense_sag
from moxie.linear_model._cd_fast import coordinate_descent, coordinate_descent_gram
from moxie.linear_model._linear_base import preprocess_data
from moxie.linear_model._linear_loss import LogisticLoss, logistic_probability


comptime LogisticSolution = SAGSolution


def logistic_sag[dtype: DType](
    X: NDArray[dtype], targets: List[Int], sample_weights: NDArray[DType.float64],
    var coefficients: NDArray[DType.float64], outputs: Int, fit_intercept: Bool,
    alpha: Float64, beta: Float64, max_iter: Int, tol: Float64, seed: UInt32,
    saga: Bool,
) raises -> LogisticSolution:
    var features = X.shape[1]

    def row_gradient(
        sample: Int, weights: NDArray[DType.float64], mut scores: List[Float64],
    ) {imm X, imm features, imm fit_intercept, imm targets, imm outputs}:
        LogisticLoss[dtype]._project_row(X, sample * features, features, fit_intercept, weights, scores)
        var target = targets[sample]
        if outputs == 1:
            scores[0] = -logistic_probability(-scores[0]) if target == 1 else logistic_probability(scores[0])
        else:
            var _, _, tail = LogisticLoss[dtype]._exp_shifted_scores(scores)
            for k in range(outputs):
                scores[k] = scores[k] / (1 + tail) - Float64(Int(k == target))

    return dense_sag(
        X, sample_weights, coefficients^, outputs, fit_intercept, alpha, beta,
        max_iter, tol, seed, saga, Float64(array_max(row_squared_norms(X))),
        0.25, row_gradient,
    )


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


def _l1_penalty(coef: NDArray[DType.float64], features: Int, strength: Float64) -> Float64:
    var norm = Float64(0)
    for j in range(features):
        norm += abs(coef.unsafe_get(j))
    return strength * norm


def _subgradient_norm(
    gradient: NDArray[DType.float64], coef: NDArray[DType.float64],
    features: Int, l1: Float64, infinity: Bool,
) -> Float64:
    if l1 == 0:
        return _gradient_norm(gradient, infinity)
    var norm = Float64(0)
    for j in range(gradient.size):
        var value = gradient.unsafe_get(j)
        if not isfinite(value):
            return inf[DType.float64]()
        if j < features:
            var coefficient = coef.unsafe_get(j)
            if coefficient == 0:
                value = max(0.0, abs(value) - l1)
            else:
                value += copysign(l1, coefficient)
        norm = max(norm, abs(value)) if infinity else norm + abs(value)
    return norm


def _newton_line_search[dtype: DType](
    mut objective: LogisticLoss[dtype], current: NDArray[DType.float64],
    gradient: NDArray[DType.float64], direction: NDArray[DType.float64],
    active: List[Int], slope: Float64, loss: Float64, tol: Float64,
    mut trial: NDArray[DType.float64], mut trial_gradient: NDArray[DType.float64],
    *, l1: Float64 = 0, penalized_features: Int = 0,
) raises -> Bool:
    comptime armijo = 0.00048828125
    comptime epsilon = 3.552713678800501e-15  # 16 * Float64 epsilon.
    comptime min_step = 1e-12
    var step = Float64(1)
    var previous_step = Float64(1)
    var previous_change = Float64(0)
    var penalty = Float64(0)
    var armijo_slope = slope
    if l1 > 0:
        penalty = _l1_penalty(current, penalized_features, l1)
        var new_norm = Float64(0)
        # L1 callers use the full, ordered coefficient vector.
        for j in range(penalized_features):
            new_norm += abs(current.unsafe_get(j) + direction.unsafe_get(j))
        armijo_slope += l1 * new_norm - penalty
    var old_gradient_norm = _subgradient_norm(gradient, current, penalized_features, l1, False)
    for i in range(21):
        for j in range(len(active)):
            var index = active[j]
            trial.unsafe_set(index, current.unsafe_get(index) + step * direction.unsafe_get(j))
        var trial_loss = objective.evaluate(trial, trial_gradient)
        var change = trial_loss - loss
        if l1 > 0:
            change += _l1_penalty(trial, penalized_features, l1) - penalty
        if isfinite(trial_loss):
            if change <= step * armijo * armijo_slope:
                return True
            if i == 0 and abs(armijo * armijo_slope) <= tol and _subgradient_norm(trial_gradient, trial, penalized_features, l1, True) <= tol:
                return True
            if abs(change) <= abs((loss + penalty) * epsilon) and _subgradient_norm(trial_gradient, trial, penalized_features, l1, False) < old_gradient_norm:
                return True

        # Interpolate the loss at the trial step and, from the second
        # rejection onward, the previous trial step.
        var residual = change - slope * step
        var next_step = -slope * step * step / (2 * residual)
        if i > 0:
            var old_residual = previous_change - slope * previous_step
            var product = previous_step * step
            var denominator = product * product * (step - previous_step)
            var a = (previous_step * previous_step * residual - step * step * old_residual) / denominator
            var b = (step * step * step * old_residual - previous_step * previous_step * previous_step * residual) / denominator
            var discriminant = b * b - 3 * a * slope
            if a != 0 and discriminant >= 0:
                next_step = (-b + sqrt(discriminant)) / (3 * a)
        previous_step = step
        previous_change = change
        if not isfinite(next_step) or next_step >= step or next_step <= min_step:
            next_step = 0.5 * step
        step = max(next_step, max(0.01 * step, min_step))
    return False


def logistic_newton_cd[dtype: DType](
    mut objective: LogisticLoss[dtype], var current: NDArray[DType.float64],
    l1: Float64, l2: Float64, max_iter: Int, tol: Float64,
) raises -> LogisticSolution:
    if max_iter == 0:
        return LogisticSolution(current^, 0)
    var X = objective._X.astype[DType.float64]()
    var rows = X.shape[0]
    var features = X.shape[1]
    var fit_intercept = objective._fit_intercept
    var response = zeros[DType.float64](Shape(rows))
    var curvature = zeros[DType.float64](Shape(rows))
    var trial = current.copy()
    var gradient = zeros[DType.float64](current.shape)
    var trial_gradient = zeros[DType.float64](current.shape)
    var direction = zeros[DType.float64](current.shape)
    var start = zeros[DType.float64](Shape(features))
    var active = List[Int](capacity=current.size)
    for j in range(current.size):
        active.append(j)
    var projected = List[Float64](length=1, fill=0)
    var inner_tol = tol
    for iteration in range(max_iter):
        var loss = objective.evaluate(current, gradient)
        if not isfinite(loss):
            raise Error("ValueError: LogisticRegression newton-cd encountered a nonfinite objective")
        objective.binary_working_response(current, response, curvature)
        # Fit the quadratic model with the existing weighted least-squares
        # preprocessing and residual coordinate solver.
        var data = preprocess_data(
            X, response, Optional(curvature.view()), fit_intercept, owner="LogisticRegression newton-cd",
        )
        for j in range(features):
            start.unsafe_set(j, current.unsafe_get(j))
        var result = coordinate_descent(
            data.X, data.y.reshape(Shape(rows)), start, l1, l2, 1000, inner_tol,
            early_stopping=False,
        )
        var intercept = data.y_offset.unsafe_get(0)
        for j in range(features):
            var coefficient = result.coef.unsafe_get(j)
            direction.unsafe_set(j, coefficient - current.unsafe_get(j))
            intercept -= data.X_offset.unsafe_get(j) * coefficient
        if fit_intercept:
            direction.unsafe_set(features, intercept - current.unsafe_get(features))
        if result.iterations <= 4:
            inner_tol *= 0.25
        var slope = dot(gradient, direction)
        if not _newton_line_search(
            objective, current, gradient, direction, active, slope, loss, tol, trial, trial_gradient,
            l1=l1, penalized_features=features,
        ):
            if l1 > 0:
                raise Error("RuntimeError: LogisticRegression newton-cd line search failed with a nonzero L1 penalty")
            var solution = logistic_lbfgs(objective, current, max_iter - iteration - 1, tol)
            solution.iterations += iteration
            return solution^
        swap(current, trial)
        if _subgradient_norm(trial_gradient, current, features, l1, True) <= tol:
            var decrement_squared = Float64(0)
            for j in range(features):
                var delta = direction.unsafe_get(j)
                decrement_squared += l2 * delta * delta
            for i in range(rows):
                LogisticLoss[dtype]._project_row(
                    objective._X, i * features, features, fit_intercept, direction, projected,
                )
                decrement_squared += curvature.unsafe_get(i) * projected[0] * projected[0]
            if 0.5 * decrement_squared <= tol:
                return LogisticSolution(current^, iteration + 1)
    return LogisticSolution(current^, max_iter)


def _reference_class_start(
    mut current: NDArray[DType.float64], outputs: Int, features: Int,
    fit_intercept: Bool, shift_features: Bool,
):
    if outputs == 1:
        return
    var first = 0 if shift_features else features
    for j in range(first, features + Int(fit_intercept)):
        var last = current.unsafe_get(j * outputs + outputs - 1)
        for k in range(outputs):
            var index = j * outputs + k
            current.unsafe_set(index, current.unsafe_get(index) - last)


def _hessian_entry(hessian: NDArray[DType.float64], order: Int, row: Int, col: Int) -> Float64:
    return hessian.unsafe_get(max(row, col) * order + min(row, col))


def _gram_newton_direction(
    hessian: NDArray[DType.float64], gradient: NDArray[DType.float64],
    current: NDArray[DType.float64], active: List[Int], penalized: Int,
    l1: Float64, l2: Float64, target_norm2: Float64, inner_tol: Float64,
    mut direction: NDArray[DType.float64],
) raises -> Int:
    var order = len(active)
    var intercepts = order - penalized
    var gram = zeros[DType.float64](Shape(penalized, penalized))
    var cross = zeros[DType.float64](Shape(penalized))
    var start = zeros[DType.float64](Shape(penalized))
    var intercept_rhs = zeros[DType.float64](Shape(intercepts, penalized + 1))
    var intercept_block = zeros[DType.float64](Shape(intercepts, intercepts))
    for i in range(order):
        var value = -gradient.unsafe_get(active[i])
        for j in range(order):
            var entry = _hessian_entry(hessian, order, i, j)
            value += entry * current.unsafe_get(active[j])
            if i < penalized and j < penalized:
                gram.unsafe_set(i * penalized + j, entry)
            elif i >= penalized:
                var row = i - penalized
                if j < penalized:
                    intercept_rhs.unsafe_set(row * (penalized + 1) + j, entry)
                else:
                    intercept_block.unsafe_set(row * intercepts + j - penalized, entry)
        if i < penalized:
            cross.unsafe_set(i, value)
            start.unsafe_set(i, current.unsafe_get(active[i]))
        else:
            intercept_rhs.unsafe_set((i - penalized) * (penalized + 1) + penalized, value)

    # Eliminate the unpenalized intercepts with a Schur complement. The
    # intercept solution is affine in the solution of the reduced system.
    var intercept_map = zeros[DType.float64](Shape(0, 0))
    if intercepts > 0:
        var inverse = symmetric_pinv(intercept_block, rtol=1e-15, atol=0)
        intercept_map = inverse.matrix @ intercept_rhs
        for i in range(penalized):
            var cross_value = cross.unsafe_get(i)
            for k in range(intercepts):
                cross_value -= _hessian_entry(hessian, order, penalized + k, i) * intercept_map.unsafe_get(
                    k * (penalized + 1) + penalized
                )
            cross.unsafe_set(i, cross_value)
            for j in range(i + 1):
                var value = gram.unsafe_get(i * penalized + j)
                for k in range(intercepts):
                    value -= _hessian_entry(hessian, order, penalized + k, i) * intercept_map.unsafe_get(k * (penalized + 1) + j)
                gram.unsafe_set(i * penalized + j, value)
                gram.unsafe_set(j * penalized + i, value)

    var result = coordinate_descent_gram(gram, cross, start^, target_norm2, l1, l2, 1000, inner_tol)
    for j in range(penalized):
        direction.unsafe_set(active[j], result.coef.unsafe_get(j) - current.unsafe_get(active[j]))
    for k in range(intercepts):
        var row = k * (penalized + 1)
        var value = intercept_map.unsafe_get(row + penalized)
        for j in range(penalized):
            value -= intercept_map.unsafe_get(row + j) * result.coef.unsafe_get(j)
        var index = active[penalized + k]
        direction.unsafe_set(index, value - current.unsafe_get(index))
    return result.iterations


def _binary_gram_target_norm[dtype: DType](
    mut objective: LogisticLoss[dtype], current: NDArray[DType.float64],
    mut response: NDArray[DType.float64], mut curvature: NDArray[DType.float64],
) raises -> Float64:
    objective.binary_working_response(current, response, curvature)
    var mean = Float64(0)
    if objective._fit_intercept:
        var mass = Float64(0)
        for i in range(response.size):
            var weight = curvature.unsafe_get(i)
            mass += weight
            mean += weight * response.unsafe_get(i)
        if mass > 0:
            mean /= mass
    var norm2 = Float64(0)
    for i in range(response.size):
        var value = sqrt(curvature.unsafe_get(i)) * (response.unsafe_get(i) - mean)
        norm2 += value * value
    return norm2


def logistic_newton_cd_gram[dtype: DType](
    mut objective: LogisticLoss[dtype], var current: NDArray[DType.float64],
    outputs: Int, features: Int, fit_intercept: Bool, l1: Float64, l2: Float64,
    max_iter: Int, tol: Float64,
) raises -> LogisticSolution:
    _reference_class_start(current, outputs, features, fit_intercept, l2 == 0)
    if max_iter == 0:
        return LogisticSolution(current^, 0)
    var active = objective.active_parameters()
    var intercepts = (1 if outputs == 1 else outputs - 1) if fit_intercept else 0
    var penalized = len(active) - intercepts
    var all_indices = List[Int](capacity=current.size)
    for j in range(current.size):
        all_indices.append(j)

    var trial = current.copy()
    var gradient = zeros[DType.float64](current.shape)
    var trial_gradient = zeros[DType.float64](current.shape)
    var direction = zeros[DType.float64](current.shape)
    var rows = objective._X.shape[0]
    var response = zeros[DType.float64](Shape(rows if outputs == 1 else 0))
    var curvature = zeros[DType.float64](response.shape)
    var inner_tol = tol
    var accepted = 0
    while accepted < max_iter:
        var loss = objective.gradient_hessian[False](current, gradient)
        if objective.hessian_warning or not isfinite(loss):
            break
        # The pinned multinomial solver uses an all-ones dual-gap target.
        var target_norm2 = Float64(rows)
        if outputs == 1:
            target_norm2 = _binary_gram_target_norm(objective, current, response, curvature)
        if not isfinite(target_norm2):
            break
        var iterations: Int
        try:
            iterations = _gram_newton_direction(
                objective.hessian, gradient, current, active, penalized,
                l1, l2, target_norm2, inner_tol, direction,
            )
        except:
            break
        if iterations <= 4:
            inner_tol *= 0.25
        for j in range(features * outputs):
            var value = current.unsafe_get(j)
            var regularizer = l2 * value
            gradient.unsafe_set(j, gradient.unsafe_get(j) + regularizer)
            loss += 0.5 * regularizer * value
        var slope = dot(gradient, direction)
        if not _newton_line_search(
            objective, current, gradient, direction, all_indices, slope, loss, tol, trial, trial_gradient,
            l1=l1, penalized_features=features * outputs,
        ):
            break
        swap(current, trial)
        accepted += 1
        if _subgradient_norm(trial_gradient, current, features * outputs, l1, True) <= tol:
            var decrement_squared = Float64(0)
            for i in range(len(active)):
                var di = direction.unsafe_get(active[i])
                var diagonal = _hessian_entry(objective.hessian, len(active), i, i)
                decrement_squared += di * di * (diagonal + (l2 if i < penalized else 0))
                for j in range(i):
                    decrement_squared += 2 * di * direction.unsafe_get(active[j]) * _hessian_entry(objective.hessian, len(active), i, j)
            if 0.5 * decrement_squared <= tol:
                return LogisticSolution(current^, accepted)
    if accepted < max_iter:
        if l1 > 0:
            raise Error("RuntimeError: LogisticRegression newton-cd-gram cannot fall back with a nonzero L1 penalty")
        var solution = logistic_lbfgs(objective, current, max_iter - accepted - 1, tol)
        solution.iterations += accepted
        return solution^
    return LogisticSolution(current^, accepted)


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
    _reference_class_start(current, outputs, features, fit_intercept, l2 == 0)
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
