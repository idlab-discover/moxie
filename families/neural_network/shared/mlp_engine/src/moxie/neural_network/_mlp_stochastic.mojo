"""Dense stochastic MLP training with SGD and Adam.

Adapted from scikit-learn (BSD-3-Clause; see LICENSES).
"""

from std.algorithm import vectorize
from std.math import inf, isfinite, sqrt
from std.sys import llvm_intrinsic, simd_width_of

from numojo import NDArray, Shape, arange

from moxie._random import RandomState
from moxie.neural_network._mlp_engine import (
    allocate,
    loss_gradient,
    validate_tensors,
)
from moxie.neural_network._plumbing import LayerLayout, LayerTensors, validate_matrix


struct OptimizerOptions(Copyable, Movable):
    var solver: String
    var learning_rate_init: Float64
    var learning_rate: String
    var power_t: Float64
    var momentum: Float64
    var nesterovs_momentum: Bool
    var beta_1: Float64
    var beta_2: Float64
    var epsilon: Float64

    def __init__(
        out self, *,
        solver: String = "adam",
        learning_rate_init: Float64 = 0.001,
        learning_rate: String = "constant",
        power_t: Float64 = 0.5,
        momentum: Float64 = 0.9,
        nesterovs_momentum: Bool = True,
        beta_1: Float64 = 0.9,
        beta_2: Float64 = 0.999,
        epsilon: Float64 = 1e-8,
    ):
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def validate(self) raises:
        if self.solver not in ("sgd", "adam"):
            raise Error("ValueError: stochastic training requires sgd or adam")
        if self.learning_rate not in ("constant", "invscaling", "adaptive"):
            raise Error("ValueError: unsupported learning rate schedule")
        if not isfinite(self.learning_rate_init) or self.learning_rate_init <= 0:
            raise Error("ValueError: learning_rate_init must be positive and finite")
        if not isfinite(self.power_t) or self.power_t < 0:
            raise Error("ValueError: power_t must be nonnegative and finite")
        if not (
            self.momentum >= 0 and self.momentum <= 1
            and self.beta_1 >= 0 and self.beta_1 < 1
            and self.beta_2 >= 0 and self.beta_2 < 1
        ):
            raise Error("ValueError: momentum must be in [0, 1] and beta values in [0, 1)")
        if not isfinite(self.epsilon) or self.epsilon <= 0:
            raise Error("ValueError: epsilon must be positive and finite")


struct StochasticOptions(ImplicitlyCopyable, Movable):
    var batch_size: Optional[Int]
    var max_iter: Int
    var shuffle: Bool
    var tol: Float64
    var n_iter_no_change: Optional[Int]
    var incremental: Bool
    var verbose: Bool

    def __init__(
        out self, *,
        batch_size: Optional[Int] = None,
        max_iter: Int = 200,
        shuffle: Bool = True,
        tol: Float64 = 1e-4,
        n_iter_no_change: Optional[Int] = 10,
        incremental: Bool = False,
        verbose: Bool = False,
    ):
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.incremental = incremental
        self.verbose = verbose


@fieldwise_init
struct StochasticReport(ImplicitlyCopyable, Movable):
    var batch_size_clipped: Bool
    var iteration_limit: Bool


def _pow(base: Float64, exponent: Float64) -> Float64:
    # The locked std.math.pow loses precision for fractional exponents.
    return llvm_intrinsic["llvm.pow", Float64, has_side_effect=False](
        base, exponent,
    )


# These buffers are contiguous and their lengths agree by construction.
def _sgd_update[dtype: DType](
    mut params: NDArray[dtype], gradient: NDArray[dtype],
    mut velocity: NDArray[dtype], options: OptimizerOptions, rate: Float64,
):
    var momentum = options.momentum.cast[dtype]()
    var step = rate.cast[dtype]()
    var nesterov = options.nesterovs_momentum

    def update[width: Int](offset: Int) {
        mut params, gradient, mut velocity, momentum, step, nesterov
    }:
        var descent = step * gradient.unsafe_load[width=width](offset)
        var speed = momentum * velocity.unsafe_load[width=width](offset) - descent
        velocity.unsafe_store[width=width](offset, speed)
        var delta = momentum * speed - descent if nesterov else speed
        params.unsafe_store[width=width](offset, params.unsafe_load[width=width](offset) + delta)

    vectorize[simd_width_of[dtype]()](params.size, update)


def _adam_update[dtype: DType](
    mut params: NDArray[dtype], gradient: NDArray[dtype],
    mut first: NDArray[dtype], mut second: NDArray[dtype],
    options: OptimizerOptions, rate: Float64,
):
    var beta_1 = options.beta_1.cast[dtype]()
    var beta_2 = options.beta_2.cast[dtype]()
    var grad_weight = (1 - options.beta_1).cast[dtype]()
    var square_weight = (1 - options.beta_2).cast[dtype]()
    var epsilon = options.epsilon.cast[dtype]()

    def update[width: Int](offset: Int) {
        mut params, gradient, mut first, mut second,
        beta_1, beta_2, grad_weight, square_weight, epsilon, rate
    }:
        var grad = gradient.unsafe_load[width=width](offset)
        var m = beta_1 * first.unsafe_load[width=width](offset) + grad_weight * grad
        var v = beta_2 * second.unsafe_load[width=width](offset) + square_weight * (grad * grad)
        first.unsafe_store[width=width](offset, m)
        second.unsafe_store[width=width](offset, v)
        # NumPy's Float64 Adam step promotes the update but the moments
        # and the square root plus epsilon retain the parameter dtype.
        var delta = -rate * m.cast[DType.float64]() / (sqrt(v) + epsilon).cast[DType.float64]()
        var updated = params.unsafe_load[width=width](offset).cast[DType.float64]() + delta
        params.unsafe_store[width=width](offset, updated.cast[dtype]())

    vectorize[simd_width_of[dtype]()](params.size, update)


def _batch_rows[dtype: DType](
    source: NDArray[dtype], order: NDArray[DType.int], start: Int, count: Int,
) raises -> NDArray[dtype]:
    """Gather only the current batch from a validated matrix or weight vector."""
    var matrix = source.ndim == 2
    var width = source.shape[1] if matrix else 1
    var batch = NDArray[dtype](Shape(count, width) if matrix else Shape(count))
    var row_stride = Int(source.strides[0])
    var column_stride = Int(source.strides[1]) if matrix else 0
    for row in range(count):
        var offset = Int(order.unsafe_get(start + row)) * row_stride
        for j in range(width):
            batch.unsafe_set(row * width + j, source.unsafe_get(offset + j * column_stride))
    return batch^


struct StochasticTrainer[dtype: DType](Copyable, Movable):
    var _layout: LayerLayout
    var _params: LayerTensors[Self.dtype]
    var _first: LayerTensors[Self.dtype]
    var _second: Optional[LayerTensors[Self.dtype]]
    var _options: OptimizerOptions

    var learning_rate: Float64
    var updates: Int
    var n_iter: Int
    var t: Int
    var loss: Float64
    var best_loss: Float64
    var loss_curve: List[Float64]
    var no_improvement_count: Int

    def __init__(
        out self, initial: LayerTensors[Self.dtype], options: OptimizerOptions = OptimizerOptions(),
    ) raises:
        options.validate()
        validate_tensors(initial)
        var units: List[Int] = [initial.coefs[0].shape[0]]
        self._params = LayerTensors(contiguous=initial)
        for layer in range(len(initial.coefs)):
            units.append(initial.coefs[layer].shape[1])
        self._layout = LayerLayout(units)
        self._first = allocate[Self.dtype](self._layout)
        self._second = allocate[Self.dtype](self._layout) if options.solver == "adam" else None
        self._options = options.copy()
        self.learning_rate = options.learning_rate_init
        self.updates = 0
        self.n_iter = 0
        self.t = 0
        self.loss = inf[DType.float64]()
        self.best_loss = inf[DType.float64]()
        self.loss_curve = []
        self.no_improvement_count = 0
        self._check_finite_parameters()

    def parameters(self) -> ref[self._params] LayerTensors[Self.dtype]:
        return self._params

    def reset_optimizer(mut self, options: OptimizerOptions) raises:
        """Restart optimizer buffers and rate, preserving learned parameters and history."""
        options.validate()
        var first = allocate[Self.dtype](self._layout)
        var second: Optional[LayerTensors[Self.dtype]] = (
            allocate[Self.dtype](self._layout) if options.solver == "adam" else None
        )
        self._first = first^
        self._second = second^
        self._options = options.copy()
        self.learning_rate = options.learning_rate_init
        self.updates = 0

    def _check_finite_parameters(self) raises:
        for layer in range(len(self._params.coefs)):
            ref coefs = self._params.coefs[layer]
            ref biases = self._params.intercepts[layer]
            for i in range(coefs.size):
                if not isfinite(coefs.unsafe_get(i)):
                    raise Error("ValueError: stochastic training produced nonfinite weights")
            for i in range(biases.size):
                if not isfinite(biases.unsafe_get(i)):
                    raise Error("ValueError: stochastic training produced nonfinite weights")

    def _update(mut self, gradient: LayerTensors[Self.dtype]) raises:
        self.updates += 1
        var rate = self.learning_rate
        if self._options.solver == "adam":
            rate = (
                self._options.learning_rate_init
                * sqrt(1 - _pow(self._options.beta_2, Float64(self.updates)))
                / (1 - _pow(self._options.beta_1, Float64(self.updates)))
            )
            self.learning_rate = rate
            ref second = self._second.value()
            for layer in range(len(self._params.coefs)):
                _adam_update(
                    self._params.coefs[layer], gradient.coefs[layer],
                    self._first.coefs[layer], second.coefs[layer], self._options, rate,
                )
                _adam_update(
                    self._params.intercepts[layer], gradient.intercepts[layer],
                    self._first.intercepts[layer], second.intercepts[layer], self._options, rate,
                )
        else:
            for layer in range(len(self._params.coefs)):
                _sgd_update(
                    self._params.coefs[layer], gradient.coefs[layer],
                    self._first.coefs[layer], self._options, rate,
                )
                _sgd_update(
                    self._params.intercepts[layer], gradient.intercepts[layer],
                    self._first.intercepts[layer], self._options, rate,
                )

    def train(
        mut self, X: NDArray[Self.dtype], y: NDArray[Self.dtype], mut rng: RandomState,
        hidden_activation: String, output_activation: String, loss: String,
        alpha: Float64 = 0.0001, options: StochasticOptions = StochasticOptions(),
    ) raises -> StochasticReport:
        var weights: Optional[ImmPointer[NDArray[Self.dtype], ImmStaticOrigin]] = None
        return self._train(X, y, rng, hidden_activation, output_activation, loss, alpha, options, weights)

    def train[weight_origin: ImmOrigin](
        mut self, X: NDArray[Self.dtype], y: NDArray[Self.dtype], mut rng: RandomState,
        hidden_activation: String, output_activation: String, loss: String,
        alpha: Float64, ref[weight_origin] sample_weight: NDArray[Self.dtype],
        options: StochasticOptions = StochasticOptions(),
    ) raises -> StochasticReport:
        var weights: Optional[ImmPointer[NDArray[Self.dtype], weight_origin]] = Pointer(to=sample_weight)
        return self._train(X, y, rng, hidden_activation, output_activation, loss, alpha, options, weights)

    def _train[weight_origin: ImmOrigin](
        mut self, X: NDArray[Self.dtype], y: NDArray[Self.dtype], mut rng: RandomState,
        hidden_activation: String, output_activation: String, loss: String,
        alpha: Float64, options: StochasticOptions,
        weights: Optional[ImmPointer[NDArray[Self.dtype], weight_origin]],
    ) raises -> StochasticReport:
        validate_matrix(X)
        validate_matrix(y)
        var n_samples = X.shape[0]
        if X.shape[1] != self._layout.units[0]:
            raise Error("ValueError: input width differs from the first layer")
        var output_width = self._layout.units[len(self._layout.units) - 1]
        if y.shape[0] != n_samples or y.shape[1] != output_width:
            raise Error("ValueError: target shape differs from the network output shape")
        if weights:
            ref supplied = weights.value()[]
            if supplied.ndim != 1 or supplied.shape[0] != n_samples:
                raise Error("ValueError: sample weights must be a vector with one entry per row")
        if options.max_iter <= 0 or options.n_iter_no_change.or_else(1) <= 0:
            raise Error("ValueError: max_iter and n_iter_no_change must be positive")
        if options.tol < 0 or not isfinite(options.tol):
            raise Error("ValueError: tol must be finite and nonnegative")
        var batch_size = options.batch_size.or_else(min(200, n_samples))
        if batch_size <= 0:
            raise Error("ValueError: batch_size must be positive")
        var clipped = batch_size > n_samples
        batch_size = min(batch_size, n_samples)

        var sample_order = arange[DType.int](n_samples)
        self.n_iter = 0
        var limit = 1 if options.incremental else options.max_iter
        var stopped = False
        for epoch in range(limit):
            if options.shuffle:
                rng.shuffle(sample_order)
            var accumulated_loss: Float64 = 0
            for start in range(0, n_samples, batch_size):
                var stop = min(start + batch_size, n_samples)
                var size = stop - start
                var X_batch = _batch_rows(X, sample_order, start, size)
                var y_batch = _batch_rows(y, sample_order, start, size)
                var objective: Tuple[Float64, LayerTensors[Self.dtype]]
                if weights:
                    ref values = weights.value()[]
                    var batch_weights = _batch_rows(values, sample_order, start, size)
                    objective = loss_gradient(
                        X_batch, y_batch, self._params, hidden_activation, output_activation,
                        loss, alpha, batch_weights,
                    )
                else:
                    objective = loss_gradient(
                        X_batch, y_batch, self._params, hidden_activation, output_activation,
                        loss, alpha,
                    )
                accumulated_loss += objective[0] * Float64(size)
                self._update(objective[1])

            self.n_iter = epoch + 1
            self.t += n_samples
            self.loss = accumulated_loss / Float64(n_samples)
            self.loss_curve.append(self.loss)
            if options.verbose:
                print("Iteration ", self.n_iter, ", loss = ", self.loss, sep="")
            stopped = self._epoch_ends(options)
            if stopped:
                break

        self._check_finite_parameters()
        return StochasticReport(
            batch_size_clipped=clipped,
            iteration_limit=not stopped and not options.incremental,
        )

    def _epoch_ends(mut self, options: StochasticOptions) -> Bool:
        if self.loss > self.best_loss - options.tol:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        if self.loss < self.best_loss:
            self.best_loss = self.loss

        if self._options.solver == "sgd" and self._options.learning_rate == "invscaling":
            self.learning_rate = self._options.learning_rate_init / _pow(
                Float64(self.t + 1), self._options.power_t,
            )
        if options.n_iter_no_change:
            if self.no_improvement_count > options.n_iter_no_change.value():
                if (
                    self._options.solver != "sgd"
                    or self._options.learning_rate != "adaptive"
                    or self.learning_rate <= 1e-6
                ):
                    return True
                self.learning_rate /= 5
                self.no_improvement_count = 0
        return False
