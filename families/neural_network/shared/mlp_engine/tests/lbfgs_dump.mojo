from std.math import inf
from std.testing import assert_equal, assert_raises
from numojo import NDArray, NDArrayStrides, Shape, zeros

from moxie._random import RandomState
from moxie.neural_network._mlp_engine import allocate, forward, initialize, pack
from moxie.neural_network._mlp_lbfgs import LBFGSOptions, fit_lbfgs
from moxie.neural_network._plumbing import LayerLayout


def check[dtype: DType](kind: Int, weighted: Bool, budget: Int) raises:
    var width = 1
    var hidden = String("identity")
    var output = String("identity")
    var loss = String("squared_error")
    var units: List[Int] = [2]
    if kind == 1:
        hidden = "tanh"
        output = "logistic"
        loss = "binary_log_loss"
        units.append(3)
    elif kind == 2:
        output = "softmax"
        loss = "log_loss"
        width = 3
    elif kind == 3:
        output = "exp"
        loss = "poisson"
    elif kind == 4:
        width = 2
    units.append(width)
    var params = allocate[dtype](LayerLayout(units))
    var rng = RandomState(42)
    initialize(params, rng, hidden)
    var initial = pack(params)

    var input_data = zeros[dtype](Shape(16))
    var target_data = zeros[dtype](Shape(8 * width))
    var weight_data = zeros[dtype](Shape(8))
    var X = input_data.view_with_layout(Shape(8, 2), NDArrayStrides(-2, 1), 14)
    var y = target_data.view_with_layout(Shape(8, width), NDArrayStrides(width, -1), width - 1)
    var weights = weight_data.view_with_layout(Shape(8), NDArrayStrides(-1), 7)
    for i in range(8):
        var a = SIMD[dtype, 1](i - 3) / 4
        var b = SIMD[dtype, 1](i*i % 7 - 3) / 4
        X.unsafe_set(-2 * i, a)
        X.unsafe_set(-2 * i + 1, b)
        weights.unsafe_set(-i, SIMD[dtype, 1](i % 4) / 2)
        for j in range(width):
            var value = 1 + (a - b) / SIMD[dtype, 1](j + 1)
            if kind == 1:
                value = SIMD[dtype, 1](Int(a + b > 0))
            elif kind == 2:
                value = SIMD[dtype, 1](Int(i % width == j))
            elif kind == 3:
                value = SIMD[dtype, 1](i % 3)
            y.unsafe_set(i * width - j, value)

    var options = LBFGSOptions(max_iter=100, max_fun=15000, tol=0.001)
    if budget == 1:
        options.max_iter = 2
    elif budget == 2:
        options.max_fun = 1
    elif budget == 3:
        options.tol = 100
    var alpha = Float64(0.25)
    var result = fit_lbfgs(X, y, params, hidden, output, loss, alpha, weights, options) if weighted else fit_lbfgs(
        X, y, params, hidden, output, loss, alpha, options,
    )

    var after = pack(params)
    for i in range(initial.size):
        assert_equal(after.unsafe_get(i), initial.unsafe_get(i))
    var learned = pack(result.params)
    var predictions = forward(X.contiguous().astype[DType.float64](), result.params, hidden, output)
    print("CASE", kind, Int(weighted), budget, dtype)
    print(result.n_iter, result.evaluations, result.loss)
    for i in range(learned.size):
        print(learned.unsafe_get(i))
    for i in range(predictions.size):
        print(predictions.unsafe_get(i))
    print("WARNING")
    if result.warning:
        print(result.warning)


def check_errors[dtype: DType]() raises:
    var params = allocate[dtype](LayerLayout([1, 1]))
    var X = zeros[dtype](Shape(1, 1))
    var y = zeros[dtype](Shape(1, 1))
    var weights = zeros[dtype](Shape(1))
    params.coefs[0].unsafe_set(0, 2)
    with assert_raises(contains="positive"):
        _ = fit_lbfgs(X, y, params, "identity", "identity", "squared_error", options=LBFGSOptions(max_iter=0))
    with assert_raises(contains="positive"):
        _ = fit_lbfgs(X, y, params, "identity", "identity", "squared_error", options=LBFGSOptions(max_fun=0))
    with assert_raises(contains="rank-2"):
        _ = fit_lbfgs(X, weights, params, "identity", "identity", "squared_error")
    with assert_raises(contains="finite nonzero sum"):
        _ = fit_lbfgs(X, y, params, "identity", "identity", "squared_error", 0, weights)
    assert_equal(params.coefs[0].unsafe_get(0), 2)
    params.coefs[0].unsafe_set(0, inf[dtype]())
    with assert_raises(contains="not finite"):
        _ = fit_lbfgs(X, y, params, "identity", "identity", "squared_error")


def main() raises:
    check_errors[DType.float32]()
    check_errors[DType.float64]()
    var weighting: List[Bool] = [False, True]
    for kind in range(5):
        for weighted in weighting:
            for budget in range(1 if kind == 4 else 0, 4):
                check[DType.float32](kind, weighted, budget)
                check[DType.float64](kind, weighted, budget)
