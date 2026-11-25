from std.math import inf, isfinite
from std.testing import assert_equal, assert_raises, assert_true
from numojo import NDArray, NDArrayStrides, Shape, zeros

from moxie._random import RandomState
from moxie.neural_network._mlp_engine import (
    allocate, forward, initialize, loss_gradient, pack, unpack,
)
from moxie.neural_network._plumbing import LayerLayout, LayerTensors


def finite_differences[dtype: DType](
    X: NDArray[dtype], y: NDArray[dtype], mut params: LayerTensors[dtype],
    output: String, loss: String, weights: NDArray[dtype], alpha: Float64,
    analytical: NDArray[DType.float64],
) raises:
    comptime step = 0.002 if dtype == DType.float32 else 0.00001
    comptime tolerance = 0.0002 if dtype == DType.float32 else 0.00000001
    var vector = pack(params)
    for i in range(vector.size):
        var original = vector.unsafe_get(i)
        vector.unsafe_set(i, original + step)
        unpack(vector, params)
        var above = loss_gradient(X, y, params, "tanh", output, loss, alpha, weights)
        vector.unsafe_set(i, original - step)
        unpack(vector, params)
        var below = loss_gradient(X, y, params, "tanh", output, loss, alpha, weights)
        vector.unsafe_set(i, original)
        var numerical = (above[0] - below[0]) / (2 * step)
        var expected = analytical.unsafe_get(i)
        assert_true(
            abs(numerical - expected) < tolerance,
            String(loss, " gradient ", i, ": finite difference ", numerical, " vs ", expected),
        )
    unpack(vector, params)


def check[dtype: DType](hidden: String, kind: Int) raises:
    var width = 1
    var output = String("identity")
    var loss = String("squared_error")
    if kind == 1:
        output = "logistic"
        loss = "binary_log_loss"
        width = 2
    elif kind == 2:
        output = "softmax"
        loss = "log_loss"
        width = 3
    elif kind == 3:
        output = "exp"
        loss = "poisson"
        width = 2
    elif kind == 4:
        width = 2

    var params = allocate[dtype](LayerLayout([2, 3, 2, width]))
    var rng = RandomState(42)
    initialize(params, rng, hidden)
    # Exercise reversed inputs, targets and weights as well as independent outputs.
    var input_data = zeros[dtype](Shape(8))
    var target_data = zeros[dtype](Shape(4 * width))
    var weight_data = zeros[dtype](Shape(4))
    var X = input_data.view_with_layout(Shape(4, 2), NDArrayStrides(-2, 1), 6)
    var y = target_data.view_with_layout(Shape(4, width), NDArrayStrides(width, -1), width - 1)
    var weights = weight_data.view_with_layout(Shape(4), NDArrayStrides(-1), 3)
    for i in range(4):
        weights.unsafe_set(-i, SIMD[dtype, 1](i - 1 if i < 2 else i))
        for j in range(2):
            X.unsafe_set(-2 * i + j, SIMD[dtype, 1](2 * i - 3 * j - 2) / 4)
        for j in range(width):
            var target = SIMD[dtype, 1](i + j) / 4
            if kind == 1:
                target = SIMD[dtype, 1]((i + j) % 2)
            elif kind == 2:
                target = SIMD[dtype, 1](Int(i % width == j))
            y.unsafe_set(i * width - j, target)

    var prediction = forward(X, params, hidden, output)
    assert_equal(prediction.shape[0], 4)
    assert_equal(prediction.shape[1], width)
    var alpha: Float64 = 0.17
    var result = loss_gradient(X, y, params, hidden, output, loss, alpha, weights)
    var packed = pack(result[1])
    var unweighted = loss_gradient(X, y, params, hidden, output, loss)
    var unweighted_grad = pack(unweighted[1])
    if hidden == "tanh" and kind != 4:
        finite_differences(X, y, params, output, loss, weights, alpha, packed)

    for i in range(prediction.size):
        print(Float64(prediction.unsafe_get(i)))
    print(result[0])
    for i in range(packed.size):
        print(packed.unsafe_get(i))
    print(unweighted[0])
    for i in range(unweighted_grad.size):
        print(unweighted_grad.unsafe_get(i))


def check_errors_and_extremes[dtype: DType]() raises:
    var params = allocate[dtype](LayerLayout([1, 2]))
    var X = zeros[dtype](Shape(1, 1))
    var y = zeros[dtype](Shape(1, 2))
    var wrong_width = zeros[dtype](Shape(1, 2))
    var bad_weights = zeros[dtype](Shape(1))
    with assert_raises():
        _ = forward(wrong_width, params, "identity", "identity")
    with assert_raises():
        _ = forward(X, params, "identity", "invalid")
    with assert_raises():
        _ = loss_gradient(X, y, params, "identity", "exp", "squared_error")
    with assert_raises():
        _ = loss_gradient(X, y, params, "identity", "identity", "squared_error", 0, bad_weights)
    with assert_raises():
        _ = loss_gradient(X, y, params, "identity", "identity", "squared_error", 0, wrong_width)
    with assert_raises():
        _ = loss_gradient(X, X, params, "identity", "identity", "squared_error")
    with assert_raises():
        _ = loss_gradient(X, y, params, "identity", "identity", "squared_error", -1)
    bad_weights.unsafe_set(0, inf[dtype]())
    with assert_raises():
        _ = loss_gradient(X, y, params, "identity", "identity", "squared_error", 0, bad_weights)
    y.unsafe_set(0, -1)
    with assert_raises():
        _ = loss_gradient(X, y, params, "identity", "exp", "poisson")

    # Noncontiguous coefficients and bias with a nonzero offset.
    var coef_data = zeros[dtype](Shape(5))
    coef_data.unsafe_set(3, 1)
    coef_data.unsafe_set(1, -1)
    params.coefs[0] = coef_data.view_with_layout(Shape(1, 2), NDArrayStrides(1, -2), 3)
    var bias_data = zeros[dtype](Shape(5))
    bias_data.unsafe_set(3, 1000)
    bias_data.unsafe_set(1, -1000)
    params.intercepts[0] = bias_data.view_with_layout(Shape(2), NDArrayStrides(-2), 3)
    X.unsafe_set(0, 2)
    var logits = forward(X, params, "identity", "identity")
    assert_equal(logits.unsafe_get(0), 1002)
    assert_equal(logits.unsafe_get(1), -1002)
    var probability = forward(X, params, "identity", "softmax")
    assert_equal(probability.unsafe_get(0), 1)
    assert_equal(probability.unsafe_get(1), 0)
    y.unsafe_set(0, 0)
    y.unsafe_set(1, 1)
    var saturated = loss_gradient(X, y, params, "identity", "logistic", "binary_log_loss")
    assert_true(isfinite(saturated[0]))
    var grad = pack(saturated[1])
    assert_equal(grad.unsafe_get(0), 2)
    assert_equal(grad.unsafe_get(1), -2)
    assert_equal(grad.unsafe_get(2), 1)
    assert_equal(grad.unsafe_get(3), -1)


def main() raises:
    check_errors_and_extremes[DType.float32]()
    check_errors_and_extremes[DType.float64]()
    var activations: List[String] = ["identity", "logistic", "tanh", "relu"]
    for hidden in activations:
        for kind in range(5):
            check[DType.float32](hidden, kind)
            check[DType.float64](hidden, kind)
