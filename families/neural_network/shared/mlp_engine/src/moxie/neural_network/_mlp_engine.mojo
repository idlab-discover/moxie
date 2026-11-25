"""Dense MLP parameters, forward propagation and batch gradients.

Adapted from scikit-learn (BSD-3-Clause; see LICENSES).
"""
from numojo import NDArray, Shape, transpose, zeros
from std.math import clamp, isfinite, isnan, sqrt, tanh
from std.sys import llvm_intrinsic
from moxie._dense import sum_axis2d
from moxie._random import RandomState
from moxie.neural_network._plumbing import LayerLayout, LayerTensors, validate_matrix


def allocate[dtype: DType](layout: LayerLayout) raises -> LayerTensors[dtype]:
    var coefs = List[NDArray[dtype]]()
    var intercepts = List[NDArray[dtype]]()
    for l in range(len(layout.units) - 1):
        coefs.append(zeros[dtype](Shape(layout.units[l], layout.units[l + 1])))
        intercepts.append(zeros[dtype](Shape(layout.units[l + 1])))
    return LayerTensors[dtype](layout, coefs, intercepts)


def validate_tensors[dtype: DType](params: LayerTensors[dtype]) raises:
    """Validate public mutable metadata before unsafe element access."""
    if len(params.coefs) == 0 or len(params.coefs) != len(params.intercepts):
        raise Error("ValueError: inconsistent tensor counts")
    for l in range(len(params.coefs)):
        if params.coefs[l].ndim != 2 or params.intercepts[l].ndim != 1:
            raise Error("ValueError: coefficient matrix and bias vector required")
        if params.coefs[l].shape[0] <= 0 or params.coefs[l].shape[1] <= 0:
            raise Error("ValueError: empty layer")
        if params.intercepts[l].shape[0] != params.coefs[l].shape[1]:
            raise Error("ValueError: bias width mismatch")
        if l > 0:
            if params.coefs[l - 1].shape[1] != params.coefs[l].shape[0]:
                raise Error("ValueError: disconnected layers")


def initialize[dtype: DType](mut params: LayerTensors[dtype], mut rng: RandomState, activation: String) raises:
    validate_tensors(params)
    _validate_hidden_activation(activation)
    var factor = Float64(6)
    if activation == "logistic":
        factor = 2
    for l in range(len(params.coefs)):
        var bound = sqrt(factor / Float64(params.coefs[l].shape[0] + params.coefs[l].shape[1]))
        for i in range(params.coefs[l].shape[0]):
            for j in range(params.coefs[l].shape[1]):
                var index = i * Int(params.coefs[l].strides[0]) + j * Int(params.coefs[l].strides[1])
                params.coefs[l].unsafe_set(index, rng.uniform(-bound, bound).cast[dtype]())
        for j in range(params.intercepts[l].shape[0]):
            params.intercepts[l].unsafe_set(j * Int(params.intercepts[l].strides[0]), rng.uniform(-bound, bound).cast[dtype]())


def parameter_count[dtype: DType](params: LayerTensors[dtype]) raises -> Int:
    validate_tensors(params)
    var count = 0
    for l in range(len(params.coefs)):
        count += params.coefs[l].shape[0] * params.coefs[l].shape[1] + params.intercepts[l].shape[0]
    return count


def pack[dtype: DType](params: LayerTensors[dtype]) raises -> NDArray[DType.float64]:
    """Return an independent Float64 vector: all coefficients, then biases."""
    var result = zeros[DType.float64](Shape(parameter_count(params)))
    var k = 0
    for l in range(len(params.coefs)):
        for i in range(params.coefs[l].shape[0]):
            for j in range(params.coefs[l].shape[1]):
                result.unsafe_set(k, Float64(params.coefs[l].unsafe_get(i * Int(params.coefs[l].strides[0]) + j * Int(params.coefs[l].strides[1]))))
                k += 1
    for l in range(len(params.intercepts)):
        for j in range(params.intercepts[l].shape[0]):
            result.unsafe_set(k, Float64(params.intercepts[l].unsafe_get(j * Int(params.intercepts[l].strides[0]))))
            k += 1
    return result^


def unpack[dtype: DType](packed: NDArray[DType.float64], mut params: LayerTensors[dtype]) raises:
    """Validate the whole vector before mutating parameters; preserve layouts."""
    var count = parameter_count(params)
    if packed.ndim != 1:
        raise Error("ValueError: packed parameters must be rank one")
    if packed.shape[0] != count:
        raise Error("ValueError: packed parameter count mismatch")
    for k in range(count):
        var value = packed.unsafe_get(k * Int(packed.strides[0]))
        if not isfinite(value) or not isfinite(value.cast[dtype]()):
            raise Error("ValueError: nonfinite packed parameter or dtype overflow")
    var k = 0
    for l in range(len(params.coefs)):
        for i in range(params.coefs[l].shape[0]):
            for j in range(params.coefs[l].shape[1]):
                params.coefs[l].unsafe_set(i * Int(params.coefs[l].strides[0]) + j * Int(params.coefs[l].strides[1]), packed.unsafe_get(k * Int(packed.strides[0])).cast[dtype]())
                k += 1
    for l in range(len(params.intercepts)):
        for j in range(params.intercepts[l].shape[0]):
            params.intercepts[l].unsafe_set(j * Int(params.intercepts[l].strides[0]), packed.unsafe_get(k * Int(packed.strides[0])).cast[dtype]())
            k += 1


# The locked std.math exp/log lose Float64 precision on these objectives.
def _exp[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    return llvm_intrinsic["llvm.exp", Scalar[dtype], has_side_effect=False](x)


def _log[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    return llvm_intrinsic["llvm.log", Scalar[dtype], has_side_effect=False](x)


def _validate_hidden_activation(activation: String) raises:
    if activation not in ("identity", "logistic", "tanh", "relu"):
        raise Error("ValueError: unsupported hidden activation")


def _validate_forward[dtype: DType](
    X: NDArray[dtype], params: LayerTensors[dtype], hidden_activation: String,
) raises:
    validate_tensors(params)
    validate_matrix(X)
    _validate_hidden_activation(hidden_activation)
    if X.shape[1] != params.coefs[0].shape[0]:
        raise Error("ValueError: input width differs from the first layer")


def _activate[dtype: DType](
    mut values: NDArray[dtype], activation: String,
) raises:
    comptime assert dtype.is_floating_point()
    if activation == "identity":
        return
    var width = values.shape[1]
    if activation == "softmax":
        for row in range(values.shape[0]):
            var start = row * width
            var largest = values.unsafe_get(start)
            for j in range(1, width):
                largest = max(largest, values.unsafe_get(start + j))
            var total = SIMD[dtype, 1](0)
            for j in range(width):
                var value = _exp(values.unsafe_get(start + j) - largest)
                values.unsafe_set(start + j, value)
                total += value
            for j in range(width):
                values.unsafe_set(start + j, values.unsafe_get(start + j) / total)
        return

    for i in range(values.size):
        var x = values.unsafe_get(i)
        if activation == "logistic":
            var e = _exp(-abs(x))
            values.unsafe_set(i, (1 if x >= 0 else e) / (1 + e))
        elif activation == "tanh":
            values.unsafe_set(i, tanh(x))
        elif activation == "relu":
            values.unsafe_set(i, max(x, 0))
        else:
            values.unsafe_set(i, _exp(x))


def _layer[dtype: DType](
    X: NDArray[dtype], coefs: NDArray[dtype], intercepts: NDArray[dtype],
    activation: String,
) raises -> NDArray[dtype]:
    var result = X @ coefs
    var width = result.shape[1]
    var bias_stride = Int(intercepts.strides[0])
    for row in range(result.shape[0]):
        for j in range(width):
            var index = row * width + j
            result.unsafe_set(index, result.unsafe_get(index) + intercepts.unsafe_get(j * bias_stride))
    _activate(result, activation)
    return result^


def forward[dtype: DType](
    X: NDArray[dtype], params: LayerTensors[dtype], hidden_activation: String,
    output_activation: String,
) raises -> NDArray[dtype]:
    """Return the final activation without retaining earlier layers."""
    _validate_forward(X, params, hidden_activation)
    if output_activation not in ("identity", "logistic", "softmax", "exp"):
        raise Error("ValueError: unsupported output activation")
    var last = len(params.coefs) - 1
    var output = _layer(
        X, params.coefs[0], params.intercepts[0],
        output_activation if last == 0 else hidden_activation,
    )
    for layer in range(1, last + 1):
        output = _layer(
            output, params.coefs[layer], params.intercepts[layer],
            output_activation if layer == last else hidden_activation,
        )
    return output^


def _weight_sum[dtype: DType, weight_origin: ImmOrigin](
    sample_weight: Optional[ImmPointer[NDArray[dtype], weight_origin]], rows: Int,
) raises -> Float64:
    if not sample_weight:
        return Float64(rows)
    ref weights = sample_weight.value()[]
    if weights.ndim != 1 or weights.shape[0] != rows:
        raise Error("ValueError: sample weights must be a vector with one entry per row")
    var total: Float64 = 0
    for i in range(rows):
        var weight = weights.unsafe_get(i * Int(weights.strides[0]))
        if not isfinite(weight):
            raise Error("ValueError: sample weights must be finite")
        total += Float64(weight)
    if total == 0 or not isfinite(total):
        raise Error("ValueError: sample weights need a finite nonzero sum")
    return total


def _loss_and_delta[dtype: DType, weight_origin: ImmOrigin](
    prediction: NDArray[dtype], y: NDArray[dtype],
    sample_weight: Optional[ImmPointer[NDArray[dtype], weight_origin]],
    normalizer: Float64, loss: String,
    mut delta: NDArray[dtype],
) raises -> Float64:
    comptime assert dtype.is_floating_point()
    var width = prediction.shape[1]
    var total: Float64 = 0
    # sklearn clips probabilities at machine epsilon in the reported loss.
    comptime epsilon = 1.1920928955078125e-7 if dtype == DType.float32 else 2.220446049250313e-16
    for row in range(prediction.shape[0]):
        var weight = SIMD[dtype, 1](1)
        if sample_weight:
            ref weights = sample_weight.value()[]
            weight = weights.unsafe_get(row * Int(weights.strides[0]))
        for j in range(width):
            var index = row * width + j
            var target = y.unsafe_get(row * Int(y.strides[0]) + j * Int(y.strides[1]))
            var estimate = prediction.unsafe_get(index)
            var residual = estimate - target
            delta.unsafe_set(index, residual * weight)
            var contribution: Float64
            if loss == "squared_error":
                contribution = 0.5 * Float64(residual * residual)
            elif loss == "poisson":
                if target < 0:
                    raise Error("ValueError: Poisson targets must be nonnegative")
                contribution = Float64(residual)
                var ratio = target / estimate
                if target != 0 or isnan(ratio):
                    contribution += Float64(target) * Float64(_log(ratio))
            else:
                var p = clamp(estimate, SIMD[dtype, 1](epsilon), SIMD[dtype, 1](1 - epsilon))
                contribution = -Float64(target) * Float64(_log(p))
                if loss == "binary_log_loss":
                    contribution -= Float64(1 - target) * Float64(_log(1 - p))
            total += Float64(weight) * contribution
    # sklearn averages the squared loss across outputs, but uses the
    # unscaled residual for its gradient even in multioutput regression.
    if loss == "squared_error":
        total /= Float64(width)
    return total / normalizer


def _hidden_derivative[dtype: DType](
    mut delta: NDArray[dtype], activated: NDArray[dtype], activation: String,
):
    if activation == "identity":
        return
    for i in range(delta.size):
        var value = activated.unsafe_get(i)
        var derivative: SIMD[dtype, 1]
        if activation == "logistic":
            derivative = value * (1 - value)
        elif activation == "tanh":
            derivative = 1 - value * value
        else:
            derivative = SIMD[dtype, 1](Int(value > 0))
        delta.unsafe_set(i, delta.unsafe_get(i) * derivative)


def loss_gradient[dtype: DType, x_origin: ImmOrigin](
    ref[x_origin] X: NDArray[dtype], y: NDArray[dtype],
    params: LayerTensors[dtype], hidden_activation: String,
    output_activation: String, loss: String, alpha: Float64 = 0.0,
) raises -> Tuple[Float64, LayerTensors[dtype]]:
    """Return the unweighted batch loss and gradients on encoded targets."""
    var weights: Optional[ImmPointer[NDArray[dtype], x_origin]] = None
    return _loss_gradient(X, y, params, hidden_activation, output_activation, loss, alpha, weights)


def loss_gradient[dtype: DType, weight_origin: ImmOrigin](
    X: NDArray[dtype], y: NDArray[dtype], params: LayerTensors[dtype],
    hidden_activation: String, output_activation: String, loss: String,
    alpha: Float64, ref[weight_origin] sample_weight: NDArray[dtype],
) raises -> Tuple[Float64, LayerTensors[dtype]]:
    """Return the sample-weighted batch loss and gradients on encoded targets."""
    var weights: Optional[ImmPointer[NDArray[dtype], weight_origin]] = Pointer(to=sample_weight)
    return _loss_gradient(X, y, params, hidden_activation, output_activation, loss, alpha, weights)


def _loss_gradient[dtype: DType, weight_origin: ImmOrigin](
    X: NDArray[dtype], y: NDArray[dtype], params: LayerTensors[dtype],
    hidden_activation: String, output_activation: String, loss: String,
    alpha: Float64, sample_weight: Optional[ImmPointer[NDArray[dtype], weight_origin]],
) raises -> Tuple[Float64, LayerTensors[dtype]]:
    _validate_forward(X, params, hidden_activation)
    validate_matrix(y)
    var last = len(params.coefs) - 1
    if y.shape[0] != X.shape[0] or y.shape[1] != params.coefs[last].shape[1]:
        raise Error("ValueError: target shape differs from the network output shape")
    if (loss, output_activation) not in (
        ("squared_error", "identity"), ("poisson", "exp"),
        ("binary_log_loss", "logistic"), ("log_loss", "softmax"),
    ):
        raise Error("ValueError: unsupported loss and output activation pair")
    if output_activation == "softmax" and y.shape[1] < 2:
        raise Error("ValueError: categorical targets require at least two columns")
    if not isfinite(alpha) or alpha < 0:
        raise Error("ValueError: alpha must be finite and nonnegative")
    var normalizer = _weight_sum(sample_weight, X.shape[0])

    var activations = List[NDArray[dtype]]()
    activations.append(_layer(
        X, params.coefs[0], params.intercepts[0],
        output_activation if last == 0 else hidden_activation,
    ))
    for layer in range(1, last + 1):
        var activated = _layer(
            activations[layer - 1], params.coefs[layer], params.intercepts[layer],
            output_activation if layer == last else hidden_activation,
        )
        activations.append(activated^)

    var delta = zeros[dtype](y.shape)
    var objective = _loss_and_delta(
        activations[last], y, sample_weight, normalizer, loss, delta,
    )
    var units = List[Int]()
    units.append(X.shape[1])
    for coefs in params.coefs:
        units.append(coefs.shape[1])
    var gradients = allocate[dtype](LayerLayout(units))
    var squared_norm: Float64 = 0
    for layer in range(last, -1, -1):
        ref coefs = params.coefs[layer]
        var input_transposed = transpose(X) if layer == 0 else transpose(activations[layer - 1])
        var grad = input_transposed @ delta
        for i in range(coefs.shape[0]):
            for j in range(coefs.shape[1]):
                var value = coefs.unsafe_get(i * Int(coefs.strides[0]) + j * Int(coefs.strides[1]))
                squared_norm += Float64(value) * Float64(value)
                var index = i * coefs.shape[1] + j
                var numerator = grad.unsafe_get(index) + alpha.cast[dtype]() * value
                grad.unsafe_set(index, (Float64(numerator) / normalizer).cast[dtype]())
        gradients.coefs[layer] = grad^

        var bias_grad = sum_axis2d(delta, 0)
        for j in range(bias_grad.size):
            bias_grad.unsafe_set(j, (Float64(bias_grad.unsafe_get(j)) / normalizer).cast[dtype]())
        gradients.intercepts[layer] = bias_grad^
        if layer > 0:
            delta = delta @ transpose(coefs)
            _hidden_derivative(delta, activations[layer - 1], hidden_activation)
    return (objective + 0.5 * alpha * squared_norm / normalizer, gradients^)
