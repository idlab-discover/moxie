"""Dense MLP adapter for the shared L-BFGS-B solver.

Adapted from scikit-learn (BSD-3-Clause; see LICENSES).
"""
from mojo_optimize import LBFGSB, LBFGSBConfig, Objective, Status
from numojo import NDArray, Shape, full
from std.math import inf
from moxie.neural_network._mlp_engine import _loss_gradient, _pack_into, allocate, pack, unpack
from moxie.neural_network._plumbing import LayerLayout, LayerTensors


struct LBFGSOptions(ImplicitlyCopyable, Movable):
    var max_iter: Int
    var max_fun: Int
    var tol: Float64

    def __init__(out self, max_iter: Int = 200, max_fun: Int = 15000, tol: Float64 = 1e-4):
        self.max_iter = max_iter
        self.max_fun = max_fun
        self.tol = tol


@fieldwise_init
struct LBFGSResult(Movable):
    var params: LayerTensors[DType.float64]
    var loss: Float64
    var n_iter: Int
    var evaluations: Int
    var warning: String


@fieldwise_init
struct _MLPObjective[
    data_dtype: DType, x_origin: ImmOrigin, y_origin: ImmOrigin, weight_origin: ImmOrigin,
](Objective):
    comptime dtype = DType.float64
    var X: ImmPointer[NDArray[Self.data_dtype], Self.x_origin]
    var y: ImmPointer[NDArray[Self.data_dtype], Self.y_origin]
    var weights: Optional[ImmPointer[NDArray[Self.data_dtype], Self.weight_origin]]
    var params: LayerTensors[Self.data_dtype]
    var hidden_activation: String
    var output_activation: String
    var loss: String
    var alpha: Float64

    def evaluate(
        mut self, x: NDArray[Self.dtype], mut gradient: NDArray[Self.dtype],
    ) raises -> Scalar[Self.dtype]:
        # SciPy evaluates using the initial dtype and retains the accepted
        # coordinates in double precision.
        unpack(x, self.params)
        var result = _loss_gradient(
            self.X[], self.y[], self.params, self.hidden_activation,
            self.output_activation, self.loss, self.alpha, self.weights,
        )
        _pack_into(result[1], gradient)
        return result[0]


def fit_lbfgs[dtype: DType, x_origin: ImmOrigin, y_origin: ImmOrigin](
    ref[x_origin] X: NDArray[dtype], ref[y_origin] y: NDArray[dtype],
    initial: LayerTensors[dtype], hidden_activation: String,
    output_activation: String, loss: String, alpha: Float64 = 0.0001,
    options: LBFGSOptions = LBFGSOptions(),
) raises -> LBFGSResult:
    var weights: Optional[ImmPointer[NDArray[dtype], x_origin]] = None
    return _fit_lbfgs(X, y, initial, hidden_activation, output_activation, loss, alpha, options, weights)


def fit_lbfgs[dtype: DType, x_origin: ImmOrigin, y_origin: ImmOrigin, weight_origin: ImmOrigin](
    ref[x_origin] X: NDArray[dtype], ref[y_origin] y: NDArray[dtype],
    initial: LayerTensors[dtype], hidden_activation: String,
    output_activation: String, loss: String, alpha: Float64,
    ref[weight_origin] sample_weight: NDArray[dtype],
    options: LBFGSOptions = LBFGSOptions(),
) raises -> LBFGSResult:
    var weights: Optional[ImmPointer[NDArray[dtype], weight_origin]] = Pointer(to=sample_weight)
    return _fit_lbfgs(X, y, initial, hidden_activation, output_activation, loss, alpha, options, weights)


def _fit_lbfgs[dtype: DType, x_origin: ImmOrigin, y_origin: ImmOrigin, weight_origin: ImmOrigin](
    ref[x_origin] X: NDArray[dtype], ref[y_origin] y: NDArray[dtype],
    initial: LayerTensors[dtype], hidden_activation: String,
    output_activation: String, loss: String, alpha: Float64,
    options: LBFGSOptions, weights: Optional[ImmPointer[NDArray[dtype], weight_origin]],
) raises -> LBFGSResult:
    if options.max_iter <= 0 or options.max_fun <= 0:
        raise Error("ValueError: max_iter and max_fun must be positive")
    var start = pack(initial)
    var config = LBFGSBConfig(
        max_iterations=options.max_iter,
        max_evaluations=options.max_fun,
        projected_gradient_tolerance=options.tol,
        propagate_nonfinite=True,
    )
    var solver = LBFGSB[DType.float64](start.size, config^)
    var lower = full[DType.float64](Shape(start.size), -inf[DType.float64]())
    var upper = full[DType.float64](Shape(start.size), inf[DType.float64]())
    var objective = _MLPObjective[dtype, x_origin, y_origin, weight_origin](
        X=Pointer(to=X), y=Pointer(to=y), weights=weights, params=initial.copy(),
        hidden_activation=hidden_activation, output_activation=output_activation,
        loss=loss, alpha=alpha,
    )
    var status = solver.minimize(objective, start, lower, upper)
    if status == Status.callback_error:
        raise Error(solver.state.error_message)
    if not solver.state.has_accepted:
        raise Error("ValueError: L-BFGS-B could not evaluate the initial parameters: " + solver.message())

    var units = List[Int]()
    units.append(initial.coefs[0].shape[0])
    for coefs in initial.coefs:
        units.append(coefs.shape[1])
    var params = allocate[DType.float64](LayerLayout(units))
    unpack(solver.state.x, params)

    var n_iter = min(solver.state.iterations, options.max_iter)
    var warning = String()
    if not solver.converged():
        warning = (
            "lbfgs failed to converge after " + String(n_iter) + " iteration(s) (status="
            + String(solver.scipy_warnflag()) + "):\n" + solver.message() + "\n"
        )
        if n_iter == options.max_iter:
            warning += (
                "\nIncrease the number of iterations to improve the convergence (max_iter="
                + String(options.max_iter) + ")."
            )
        warning += (
            "\nYou might also want to scale the data as shown in:\n"
            "    https://scikit-learn.org/stable/modules/preprocessing.html"
        )
    return LBFGSResult(
        params=params^, loss=solver.scipy_result_value, n_iter=n_iter,
        evaluations=solver.state.evaluations, warning=warning^,
    )
