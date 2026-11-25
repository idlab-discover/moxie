# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Private estimator validation, solver dispatch and dense inference."""

from numojo import NDArray, Shape, zeros
from std.math import isfinite, nan
from moxie._random import RandomState
from moxie.warnings import FitWarning
from ._mlp_engine import _validate_hidden_activation, _weight_sum, allocate, forward, initialize
from ._mlp_lbfgs import LBFGSOptions, LBFGSResult, _fit_lbfgs
from ._mlp_stochastic import OptimizerOptions, StochasticOptions, StochasticTrainer
from ._plumbing import LayerLayout, LayerTensors, validate_matrix


@fieldwise_init
struct MLPParams(Copyable, Movable):
    """Constructor configuration shared by the dense MLP estimators."""

    var hidden_layer_sizes: List[Int]
    var activation: String
    var solver: String
    var alpha: Float64
    var batch_size: Optional[Int]
    var learning_rate: String
    var learning_rate_init: Float64
    var power_t: Float64
    var max_iter: Int
    var shuffle: Bool
    var random_state: Optional[Int]
    var tol: Float64
    var verbose: Bool
    var warm_start: Bool
    var momentum: Float64
    var nesterovs_momentum: Bool
    var early_stopping: Bool
    var validation_fraction: Float64
    var beta_1: Float64
    var beta_2: Float64
    var epsilon: Float64
    var n_iter_no_change: Optional[Int]
    var max_fun: Int
    var loss: String

    def _optimizer(self) -> OptimizerOptions:
        return OptimizerOptions(
            solver="adam" if self.solver == "lbfgs" else self.solver,
            learning_rate_init=self.learning_rate_init,
            learning_rate=self.learning_rate,
            power_t=self.power_t,
            momentum=self.momentum,
            nesterovs_momentum=self.nesterovs_momentum,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
        )

    def _validate(self) raises:
        _validate_hidden_activation(self.activation)
        if self.solver not in ("lbfgs", "sgd", "adam"):
            raise Error("ValueError: solver must be lbfgs, sgd or adam")
        self._optimizer().validate()
        if not isfinite(self.alpha) or self.alpha < 0:
            raise Error("ValueError: alpha must be finite and nonnegative")
        if not isfinite(self.tol) or self.tol < 0:
            raise Error("ValueError: tol must be finite and nonnegative")
        if self.max_iter <= 0 or self.max_fun <= 0:
            raise Error("ValueError: max_iter and max_fun must be positive")
        if self.batch_size.or_else(1) <= 0 or self.n_iter_no_change.or_else(1) <= 0:
            raise Error("ValueError: batch_size and n_iter_no_change must be positive")
        if not (self.validation_fraction >= 0 and self.validation_fraction < 1):
            raise Error("ValueError: validation_fraction must be in [0, 1)")
        if self.early_stopping and self.solver != "lbfgs":
            raise Error("NotImplementedError: MLP validation-set early stopping is unfinished")

    def _seed(self) raises -> Int:
        if not self.random_state:
            raise Error("ValueError: random_state=None requires an explicit mutable RNG context")
        return self.random_state.value()


struct _MLPState[dtype: DType](Copyable, Movable):
    var _stochastic: Optional[StochasticTrainer[Self.dtype]]
    var _lbfgs: Optional[LBFGSResult]
    var out_activation: String
    var n_features: Int
    var n_outputs: Int
    var n_layers: Int
    var n_iter: Int
    var t: Int
    var loss: Float64
    var warnings: List[FitWarning]

    def __init__[weight_origin: ImmOrigin](
        out self, X: NDArray[Self.dtype], y: NDArray[Self.dtype], params: MLPParams,
        output_activation: String, loss: String, mut rng: RandomState,
        weights: Optional[ImmPointer[NDArray[Self.dtype], weight_origin]],
        previous: Optional[Self], incremental: Bool,
    ) raises:
        params._validate()
        validate_matrix(X)
        validate_matrix(y)
        if X.shape[0] != y.shape[0]:
            raise Error("ValueError: X and y must have the same number of rows")
        _ = _weight_sum(weights, X.shape[0])

        var units = List[Int](capacity=len(params.hidden_layer_sizes) + 2)
        units.append(X.shape[1])
        for width in params.hidden_layer_sizes:
            units.append(width)
        units.append(y.shape[1])
        var layout = LayerLayout(units)
        self.out_activation = output_activation
        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]
        self.n_layers = len(units)
        self.warnings = []
        var trainer: StochasticTrainer[Self.dtype]
        if incremental and previous:
            ref fitted = previous.value()
            if not fitted._stochastic:
                raise Error("ValueError: partial_fit requires stochastic fitted state")
            if X.shape[1] != fitted.n_features:
                raise Error("ValueError: input width differs from the fitted network")
            if y.shape[1] != fitted.n_outputs or output_activation != fitted.out_activation:
                raise Error("ValueError: output shape or activation differs from the fitted network")
            ref retained = fitted._stochastic.value()
            if retained._layout.units != units:
                raise Error("ValueError: hidden layer sizes differ from the fitted network")
            trainer = retained.copy()
        else:
            var initial = allocate[Self.dtype](layout)
            initialize(initial, rng, params.activation)
            if params.solver == "lbfgs":
                var result = _fit_lbfgs(
                    X, y, initial, params.activation, output_activation, loss,
                    params.alpha, LBFGSOptions(params.max_iter, params.max_fun, params.tol),
                    weights,
                )
                self._stochastic = None
                self.n_iter = result.n_iter
                self.t = 0
                self.loss = result.loss
                if result.warning:
                    self.warnings.append(FitWarning("ConvergenceWarning", result.warning.copy()))
                self._lbfgs = result^
                return
            trainer = StochasticTrainer(initial, params._optimizer())

        self._lbfgs = None
        var options = StochasticOptions(
            batch_size=params.batch_size, max_iter=params.max_iter, shuffle=params.shuffle,
            tol=params.tol, n_iter_no_change=params.n_iter_no_change, verbose=params.verbose,
            incremental=incremental,
        )
        var report = trainer._train(
            X, y, rng, params.activation, output_activation, loss, params.alpha,
            options, weights,
        )
        self.n_iter = trainer.n_iter
        self.t = trainer.t
        self.loss = trainer.loss
        self._stochastic = trainer^
        if report.batch_size_clipped:
            self.warnings.append(FitWarning(
                "UserWarning",
                "Got `batch_size` less than 1 or larger than sample size. It is going to be clipped",
            ))
        if report.iteration_limit:
            self.warnings.append(FitWarning(
                "ConvergenceWarning",
                String("Stochastic Optimizer: Maximum iterations (", params.max_iter,
                       ") reached and the optimization hasn't converged yet."),
            ))

    def best_loss(self) -> Optional[Float64]:
        if self._stochastic:
            return self._stochastic.value().best_loss
        return None

    def loss_curve(self) -> Optional[List[Float64]]:
        if self._stochastic:
            return self._stochastic.value().loss_curve.copy()
        return None

    def parameter_dtype(self) -> DType:
        return DType.float64 if self._lbfgs else Self.dtype

    def coefs[parameter_dtype: DType](self) raises -> List[NDArray[parameter_dtype]]:
        self._check_parameter_dtype[parameter_dtype]()
        if self._lbfgs:
            return _copy_layers[parameter_dtype](self._lbfgs.value().params.coefs)
        return _copy_layers[parameter_dtype](self._stochastic.value().parameters().coefs)

    def intercepts[parameter_dtype: DType](self) raises -> List[NDArray[parameter_dtype]]:
        self._check_parameter_dtype[parameter_dtype]()
        if self._lbfgs:
            return _copy_layers[parameter_dtype](self._lbfgs.value().params.intercepts)
        return _copy_layers[parameter_dtype](self._stochastic.value().parameters().intercepts)

    def _check_parameter_dtype[parameter_dtype: DType](self) raises:
        if parameter_dtype != self.parameter_dtype():
            raise Error("ValueError: requested dtype differs from fitted MLP parameters")

    def _predict_lbfgs(
        self, X: NDArray[Self.dtype], activation: String,
    ) raises -> NDArray[DType.float64]:
        validate_matrix(X)
        # NuMojo astype walks contiguous storage. Compact a strided view first.
        var inputs = X.astype[DType.float64]() if X.is_c_contiguous() else (
            X.contiguous().astype[DType.float64]()
        )
        return forward(inputs, self._lbfgs.value().params, activation, self.out_activation)

    def predict_matrix(
        self, X: NDArray[Self.dtype], activation: String,
    ) raises -> NDArray[DType.float64]:
        if self._lbfgs:
            return self._predict_lbfgs(X, activation)
        var prediction = forward(X, self._stochastic.value().parameters(), activation, self.out_activation)
        return prediction.astype[DType.float64]()

    def predict_feature_matrix(
        self, X: NDArray[Self.dtype], activation: String,
    ) raises -> NDArray[Self.dtype]:
        if self._lbfgs:
            return self._predict_lbfgs(X, activation).astype[Self.dtype]()
        return forward(X, self._stochastic.value().parameters(), activation, self.out_activation)


def _copy_layers[target: DType, dtype: DType](layers: List[NDArray[dtype]]) raises -> List[NDArray[target]]:
    var result = List[NDArray[target]](capacity=len(layers))
    for layer in layers:
        result.append(layer.astype[target]())
    return result^


def _target_matrix[dtype: DType](y: NDArray[dtype], rows: Int) raises -> NDArray[dtype]:
    if y.ndim not in (1, 2) or y.shape[0] != rows:
        raise Error("ValueError: y must be a vector or matrix with one entry per sample")
    if y.ndim == 2:
        validate_matrix(y)
        return y.copy()
    var result = zeros[dtype](Shape(rows, 1))
    for row in range(rows):
        var value = y.unsafe_get(row * Int(y.strides[0]))
        if not isfinite(value):
            raise Error("ValueError: expected finite target values")
        result.unsafe_set(row, value)
    return result^


def _regression_output[dtype: DType](var prediction: NDArray[dtype]) raises -> NDArray[dtype]:
    if prediction.shape[1] == 1:
        return prediction.reshape(Shape(prediction.shape[0]))
    return prediction^


def _sample_weight[dtype: DType](weights: Optional[NDArray[dtype]], row: Int) raises -> Float64:
    if weights:
        ref array = weights.value()
        return Float64(array.unsafe_get(row * Int(array.strides[0])))
    return 1.0


def _regression_score[dtype: DType, weight_origin: ImmOrigin](
    y: NDArray[dtype], prediction: NDArray[DType.float64],
    weights: Optional[NDArray[dtype]],
    weight_ptr: Optional[ImmPointer[NDArray[dtype], weight_origin]],
) raises -> Float64:
    var rows = prediction.shape[0]
    var outputs = prediction.shape[1]
    var target = _target_matrix(y, rows)
    if target.shape[1] != outputs:
        raise Error("ValueError: y and predictions must have the same output width")
    var mass = _weight_sum(weight_ptr, rows)
    if rows < 2:
        return nan[DType.float64]()
    # A fitted Poisson model can overflow at inference even for finite inputs.
    for i in range(prediction.size):
        if not isfinite(prediction.unsafe_get(i)):
            return nan[DType.float64]()
    var score: Float64 = 0
    for output in range(outputs):
        var mean: Float64 = 0
        for row in range(rows):
            var offset = row * Int(target.strides[0]) + output * Int(target.strides[1])
            mean += _sample_weight(weights, row) * Float64(target.unsafe_get(offset))
        mean /= mass
        var residual: Float64 = 0
        var total: Float64 = 0
        for row in range(rows):
            var offset = row * Int(target.strides[0]) + output * Int(target.strides[1])
            var value = Float64(target.unsafe_get(offset))
            var error = value - prediction.unsafe_get(row * outputs + output)
            var deviation = value - mean
            var weight = _sample_weight(weights, row)
            residual += weight * error * error
            total += weight * deviation * deviation
        if total != 0:
            score += 1 - residual / total
        elif residual == 0:
            score += 1
    return score / Float64(outputs)
