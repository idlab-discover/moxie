"""The single EM lifecycle shared by GaussianMixture and BayesianGaussianMixture.

Native translation of sklearn.mixture._base.BaseMixture.fit_predict (pinned
4358542a): restarts, initialization, alternating E/M steps, strict absolute
objective-change convergence (|change| < tol, decreases not clamped), best
restart selection (first strictly larger objective, or the first restart),
warm starts (skip initialization, resume the stored objective, one restart),
max_iter = 0 semantics and the final E-step that makes fit_predict equal
fit(X).predict(X). Algorithm-specific numerics enter only through EMHooks.
"""
from numojo import NDArray, Shape, zeros
from std.math import abs
from std.sys import stderr
from std.time import perf_counter_ns
from std.utils.numerics import inf, isnan
from moxie._random import RandomState
from moxie.mixture._contracts import EMHooks
from moxie.mixture._initialization import initial_responsibilities
from moxie.mixture._responsibilities import log_responsibilities, mean_vector, argmax_rows

comptime CONVERGENCE_WARNING = (
    "ConvergenceWarning: Best performing initialization did not converge. Try"
    " different init parameters, or increase max_iter, tol, or check for"
    " degenerate data."
)


trait MixtureEMModel(EMHooks):
    """EMHooks plus the one decision BaseMixture delegates to subclasses.

    GaussianMixture skips responsibility initialization (and its RNG draws)
    when weights_init, means_init and precisions_init are all supplied; the
    model then receives an empty (0, k) responsibility matrix.
    """

    def needs_responsibilities(self) -> Bool:
        ...


@fieldwise_init
struct EMSettings(Copyable, Movable):
    """Validated controls for one fit (see moxie.mixture._contracts.EMControl)."""

    var n_components: Int
    var tol: Float64
    var max_iter: Int
    var n_init: Int
    var init_params: String
    var verbose: Int
    var verbose_interval: Int
    var do_init: Bool
    var initial_lower_bound: Float64


struct EMResult(Movable):
    """Outcome of a fit: learned-state diagnostics and final labels."""

    var converged: Bool
    var n_iter: Int
    var lower_bound: Float64
    var lower_bounds: List[Float64]
    var labels: NDArray[DType.int64]
    var warning: String

    def __init__(out self, converged: Bool, n_iter: Int, lower_bound: Float64, var lower_bounds: List[Float64], var labels: NDArray[DType.int64], var warning: String):
        self.converged = converged
        self.n_iter = n_iter
        self.lower_bound = lower_bound
        self.lower_bounds = lower_bounds^
        self.labels = labels^
        self.warning = warning^


def weighted_log_prob[M: EMHooks](model: M, X: NDArray[M.dtype]) raises -> NDArray[M.dtype]:
    """Return estimate_log_prob(X) + estimate_log_weights() (row broadcast, dtype sum)."""
    var log_prob = model.estimate_log_prob(X)
    var log_weights = model.estimate_log_weights()
    var n = log_prob.shape[0]
    var k = log_prob.shape[1]
    if log_weights.ndim != 1 or log_weights.shape[0] != k:
        raise Error("ValueError: log weights do not match the component count")
    for i in range(n):
        for j in range(k):
            log_prob.unsafe_set(i * k + j, log_prob.unsafe_get(i * k + j) + log_weights.unsafe_get(j))
    return log_prob^


def e_step[M: EMHooks](model: M, X: NDArray[M.dtype]) raises -> Tuple[Float64, NDArray[M.dtype]]:
    """(mean log_prob_norm, log_resp) as in BaseMixture._e_step."""
    var normalized = log_responsibilities(weighted_log_prob(model, X))
    var mean = mean_vector(normalized[0])
    return (mean, normalized[1].copy())


def _elapsed(start: Int) -> Float64:
    return Float64(perf_counter_ns() - start) * 1e-9


def run_em[M: MixtureEMModel](mut model: M, X: NDArray[M.dtype], settings: EMSettings, mut rng: RandomState) raises -> EMResult:
    """Run all restarts, restore the best state, and return diagnostics + labels.

    Verbose output follows the reference messages on stdout. A nonconverged
    best restart (with max_iter > 0) prints CONVERGENCE_WARNING to stderr and
    returns it in `EMResult.warning` (native stand-in for ConvergenceWarning).
    """
    var negative_infinity = -inf[DType.float64]()
    var max_lower_bound = negative_infinity
    var best_lower_bounds = List[Float64]()
    var converged_best = False
    var best_n_iter = 0
    var best_state = Optional[M.State](None)
    var k = settings.n_components
    var init_start = perf_counter_ns()
    var iter_start = perf_counter_ns()
    for init in range(settings.n_init):
        if settings.verbose >= 1:
            print("Initialization", init)
            init_start = perf_counter_ns()
            iter_start = init_start
        if settings.do_init:
            if model.needs_responsibilities():
                var resp = initial_responsibilities(X, k, settings.init_params, rng)
                model.initialize(X, resp)
            else:
                model.initialize(X, zeros[M.dtype](Shape(0, k)))
        var lower_bound = negative_infinity if settings.do_init else settings.initial_lower_bound
        var current_lower_bounds = List[Float64]()
        if settings.max_iter == 0:
            best_state = model.get_state()
            best_n_iter = 0
        else:
            var converged = False
            var n_iter = 0
            for iteration in range(1, settings.max_iter + 1):
                n_iter = iteration
                var previous = lower_bound
                var step = e_step(model, X)
                model.m_step(X, step[1])
                lower_bound = model.lower_bound(step[1], step[0])
                current_lower_bounds.append(lower_bound)
                var change = lower_bound - previous
                if iteration % settings.verbose_interval == 0:
                    if settings.verbose == 1:
                        print("  Iteration", iteration)
                    elif settings.verbose >= 2:
                        print("  Iteration", iteration, "\t time lapse", _elapsed(iter_start), "s\t ll change", change)
                        iter_start = perf_counter_ns()
                if abs(change) < settings.tol:
                    converged = True
                    break
            if settings.verbose >= 1:
                var message = "converged" if converged else "did not converge"
                if settings.verbose == 1:
                    print("Initialization " + message + ".")
                else:
                    print("Initialization " + message + ". time lapse", _elapsed(init_start), "s\t lower bound", lower_bound, ".")
            if lower_bound > max_lower_bound or max_lower_bound == negative_infinity:
                max_lower_bound = lower_bound
                best_state = model.get_state()
                best_n_iter = n_iter
                best_lower_bounds = current_lower_bounds.copy()
                converged_best = converged
    var warning = String("")
    if not converged_best and settings.max_iter > 0:
        warning = String(CONVERGENCE_WARNING)
        print(warning, file=stderr)
    if not best_state:
        raise Error("RuntimeError: EM finished without a selected state")
    model.set_state(best_state.take())
    var final_step = e_step(model, X)
    var labels = argmax_rows(final_step[1])
    return EMResult(converged_best, best_n_iter, max_lower_bound, best_lower_bounds^, labels^, warning^)
