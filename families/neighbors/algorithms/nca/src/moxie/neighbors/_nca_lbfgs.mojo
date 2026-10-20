"""Family-local unbounded L-BFGS for NCA, with SciPy L-BFGS-B search rules.

No bound-constrained API is provided. Numeric workspaces remain NuMojo arrays.
"""
import numojo as nm
from std.math import abs, sqrt
from moxie.neighbors._nca_math import Matrix, inner
from moxie.neighbors._nca_linesearch import MoreThuente, SearchTask


trait NCACallback(Copyable, Defaultable, Deinitable, Movable):
    def __call__(mut self, transformation: Matrix, n_iter: Int) raises:
        ...


trait NCAObjective:
    def __call__(self, x: Matrix) raises -> Tuple[Float64, Matrix]:
        ...


struct Stop:
    comptime gradient = 401
    comptime function = 402
    comptime iterations = 504
    comptime evaluations = 502
    comptime search = 8


@fieldwise_init
struct OptimizationResult(Movable):
    var x: Matrix
    var value: Float64
    var iterations: Int
    var evaluations: Int
    var status: Int
    var restarts: Int


def gradient_norm(gradient: Matrix) -> Float64:
    var norm = Float64(0)
    for j in range(gradient.size):
        norm = max(norm, abs(gradient.unsafe_get(j)))
    return norm


def minimize[Objective: NCAObjective, Callback: NCACallback](
    objective: Objective,
    var A: Matrix,
    mut callback: Callback,
    max_iter: Int,
    tol: Float64,
    verbose: Int = 0,
    maxfun: Int = 15000,
    maxls: Int = 20,
) raises -> OptimizationResult:
    # Internal options expose the fixed SciPy defaults to regression tests.
    var evaluation = objective(A)
    var value = evaluation[0]
    var gradient = evaluation[1].copy()
    var evaluations = 1
    var iterations = 0
    var restarts = 0
    var status = Stop.gradient
    var length = A.size
    var steps = nm.zeros[DType.float64](nm.Shape(10, length))
    var changes = nm.zeros[DType.float64](nm.Shape(10, length))
    var rho = nm.zeros[DType.float64](nm.Shape(10))
    var alpha = nm.zeros[DType.float64](nm.Shape(10))
    var stored = 0
    var head = 0
    # Match ScalarFunction's one-point cache, including repeated search trials.
    var cached_x = A.copy()
    var cached_gradient = gradient.copy()
    var cached_value = value
    comptime epsilon: Float64 = 2.220446049250313e-16
    var ftol = (tol / epsilon) * epsilon
    if verbose:
        print("[NeighborhoodComponentsAnalysis] Iteration Objective Value")
    while gradient_norm(gradient) > tol:
        var direction = gradient.copy()
        for age in range(stored):
            var slot = (head - 1 - age + 10) % 10
            var product = Float64(0)
            for j in range(length):
                product += steps.unsafe_get(
                    slot * length + j
                ) * direction.unsafe_get(j)
            alpha.unsafe_set(slot, rho.unsafe_get(slot) * product)
            for j in range(length):
                direction.unsafe_set(
                    j,
                    direction.unsafe_get(j)
                    - alpha.unsafe_get(slot)
                    * changes.unsafe_get(slot * length + j),
                )
        var scale = Float64(1)
        if stored:
            var slot = (head + 9) % 10
            var yy = Float64(0)
            for j in range(length):
                yy += changes.unsafe_get(slot * length + j) ** 2
            scale = 1 / (rho.unsafe_get(slot) * yy)
        for j in range(length):
            direction.unsafe_set(j, scale * direction.unsafe_get(j))
        for age in range(stored - 1, -1, -1):
            var slot = (head - 1 - age + 10) % 10
            var product = Float64(0)
            for j in range(length):
                product += changes.unsafe_get(
                    slot * length + j
                ) * direction.unsafe_get(j)
            var beta = rho.unsafe_get(slot) * product
            for j in range(length):
                direction.unsafe_set(
                    j,
                    direction.unsafe_get(j)
                    + (alpha.unsafe_get(slot) - beta)
                    * steps.unsafe_get(slot * length + j),
                )
        for j in range(length):
            direction.unsafe_set(j, -direction.unsafe_get(j))
        var slope = inner(gradient, direction)
        var step = Float64(1)
        if iterations == 0:
            step = min(1 / sqrt(inner(direction, direction)), Float64(1e10))
        var trial = A.copy()
        var trial_gradient = gradient.copy()
        var trial_value = value
        var trial_slope = slope
        var search = MoreThuente()
        var trials = 0
        var accepted = False
        if slope < 0:
            search.search(value, slope, step, 1e-3, 0.9, 0.1, 0, 1e10)
            while search.task == SearchTask.fg and trials < maxls:
                var cached = True
                for j in range(length):
                    var xj = A.unsafe_get(j) + step * direction.unsafe_get(j)
                    trial.unsafe_set(j, xj)
                    cached = cached and xj == cached_x.unsafe_get(j)
                if not cached:
                    var next_eval = objective(trial)
                    cached_value = next_eval[0]
                    cached_gradient = next_eval[1].copy()
                    cached_x = trial.copy()
                    evaluations += 1
                trial_value = cached_value
                trial_gradient = cached_gradient.copy()
                trial_slope = inner(trial_gradient, direction)
                trials += 1
                search.search(
                    trial_value, trial_slope, step, 1e-3, 0.9, 0.1, 0, 1e10
                )
            accepted = (
                search.task == SearchTask.convergence
                or search.task == SearchTask.warning
            )
        if not accepted:
            if stored:
                # Retry from the last accepted point with the identity Hessian.
                stored = 0
                head = 0
                restarts += 1
                continue
            status = Stop.search
            break
        var previous = value
        var old_gradient = gradient^
        A = trial^
        gradient = trial_gradient^
        value = trial_value
        iterations += 1
        var flat = A.reshape(nm.Shape(length))
        callback(flat, iterations)
        if verbose:
            print("[NeighborhoodComponentsAnalysis]", iterations, -value)
        # Python's NEW_X budget checks precede Fortran/C convergence checks.
        if iterations >= max_iter:
            status = Stop.iterations
            break
        if evaluations > maxfun:
            status = Stop.evaluations
            break
        if gradient_norm(gradient) <= tol:
            status = Stop.gradient
            break
        if previous - value <= ftol * max(
            Float64(1), max(abs(previous), abs(value))
        ):
            status = Stop.function
            break
        var sy = (trial_slope - slope) * step
        if sy > epsilon * (-slope * step):
            for j in range(length):
                steps.unsafe_set(
                    head * length + j, step * direction.unsafe_get(j)
                )
                changes.unsafe_set(
                    head * length + j,
                    gradient.unsafe_get(j) - old_gradient.unsafe_get(j),
                )
            rho.unsafe_set(head, 1 / sy)
            head = (head + 1) % 10
            stored = min(stored + 1, 10)
    return OptimizationResult(A^, value, iterations, evaluations, status, restarts)
