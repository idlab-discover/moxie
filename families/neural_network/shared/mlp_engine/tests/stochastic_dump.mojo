from std.math import inf
from std.testing import assert_equal, assert_raises

from numojo import NDArrayStrides, Shape, zeros

from moxie._random import RandomState
from moxie.neural_network._mlp_engine import allocate, forward, initialize, pack
from moxie.neural_network._mlp_stochastic import (
    OptimizerOptions,
    StochasticOptions,
    StochasticTrainer,
)
from moxie.neural_network._plumbing import LayerLayout


def check[dtype: DType](kind: Int, weighted: Bool, mode: Int) raises:
    var hidden = String("relu")
    var output = String("identity")
    var loss = String("squared_error")
    var width = 1
    if kind == 1:
        hidden = "tanh"
        output = "logistic"
        loss = "binary_log_loss"
    elif kind == 2:
        hidden = "logistic"
        output = "softmax"
        loss = "log_loss"
        width = 3
    elif kind == 3:
        hidden = "identity"
        output = "exp"
        loss = "poisson"
    elif kind == 4:
        width = 2
    var params = allocate[dtype](LayerLayout([2, 9, width]))
    var rng = RandomState(42)
    initialize(params, rng, hidden)
    var initial = pack(params)

    var rows = 203 if mode == 6 else 8
    var input_data = zeros[dtype](Shape(rows * 2))
    var target_data = zeros[dtype](Shape(rows * width))
    var weight_data = zeros[dtype](Shape(rows))
    var X = input_data.view_with_layout(Shape(rows, 2), NDArrayStrides(-2, 1), 2 * (rows - 1))
    var y = target_data.view_with_layout(Shape(rows, width), NDArrayStrides(width, -1), width - 1)
    var weights = weight_data.view_with_layout(Shape(rows), NDArrayStrides(-1), rows - 1)
    for i in range(rows):
        var a = SIMD[dtype, 1](i % 8 - 3) / 4
        var b = SIMD[dtype, 1](i * i % 7 - 3) / 4
        X.unsafe_set(-2 * i, a)
        X.unsafe_set(-2 * i + 1, b)
        weights.unsafe_set(-i, SIMD[dtype, 1](i % 4 + 1) / 2)
        for j in range(width):
            var value = 1 + (a - b) / SIMD[dtype, 1](j + 1)
            if kind == 1:
                value = SIMD[dtype, 1](Int(a + b > 0))
            elif kind == 2:
                value = SIMD[dtype, 1](Int(i % width == j))
            elif kind == 3:
                value = SIMD[dtype, 1](i % 3)
            y.unsafe_set(i * width - j, value)

    var optimizer = OptimizerOptions(
        solver="sgd" if mode <= 3 else "adam",
        learning_rate_init=1e-5 if mode == 3 else 0.01,
        momentum=Float64(0) if mode == 0 else 0.65,
        nesterovs_momentum=mode != 1,
        power_t=0.35, beta_1=0.8, beta_2=0.96, epsilon=1e-5,
    )
    if mode == 2:
        optimizer.learning_rate = "invscaling"
    elif mode == 3 or mode == 5:
        optimizer.learning_rate = "adaptive"
    var options = StochasticOptions(batch_size=3, max_iter=6)
    if mode == 1:
        options.batch_size = 20
        options.shuffle = False
    elif mode == 3 or mode == 5:
        options.tol = 10
        options.n_iter_no_change = 2
        options.max_iter = 15
    elif mode == 4:
        options.incremental = True
        options.max_iter = 1
    elif mode == 6:
        options.batch_size = None
        options.n_iter_no_change = None
        options.tol = 10
        options.max_iter = 4

    var trainer = StochasticTrainer(params, optimizer)
    var cycles = 3 if mode == 4 else (2 if mode == 2 else 1)
    for cycle in range(cycles):
        if cycle == 1 and mode == 2:
            trainer.reset_optimizer(optimizer)
            options.max_iter = 3
        var report = trainer.train(X, y, rng, hidden, output, loss, 0.1, weights, options) if weighted else trainer.train(
            X, y, rng, hidden, output, loss, 0.1, options,
        )

        var after = pack(params)
        for i in range(initial.size):
            assert_equal(after.unsafe_get(i), initial.unsafe_get(i))
        var params_vector = pack(trainer.parameters())
        var predictions = forward(X, trainer.parameters(), hidden, output)
        var probe = rng.copy()
        print("CASE", kind, Int(weighted), mode, dtype, cycle)
        print(
            trainer.n_iter, trainer.t, trainer.updates, trainer.no_improvement_count,
            Int(report.batch_size_clipped), Int(report.iteration_limit),
            probe.next_uint32(), trainer.learning_rate, trainer.loss, trainer.best_loss,
        )
        for value in trainer.loss_curve:
            print(value)
        for i in range(params_vector.size):
            print(params_vector.unsafe_get(i))
        for i in range(predictions.size):
            print(predictions.unsafe_get(i))


def check_errors[dtype: DType]() raises:
    var params = allocate[dtype](LayerLayout([1, 1]))
    var X = zeros[dtype](Shape(2, 1))
    var weights = zeros[dtype](Shape(2))
    var rng = RandomState(42)
    with assert_raises(contains="stochastic training"):
        _ = StochasticTrainer(params, OptimizerOptions(solver="lbfgs"))
    with assert_raises(contains="schedule"):
        _ = StochasticTrainer(params, OptimizerOptions(learning_rate="unknown"))
    with assert_raises(contains="beta"):
        _ = StochasticTrainer(params, OptimizerOptions(beta_2=1))
    with assert_raises(contains="epsilon"):
        _ = StochasticTrainer(params, OptimizerOptions(epsilon=0))
    params.intercepts[0].unsafe_set(0, inf[dtype]())
    with assert_raises(contains="nonfinite weights"):
        _ = StochasticTrainer(params)
    params.intercepts[0].unsafe_set(0, 0)
    var trainer = StochasticTrainer(params)
    with assert_raises(contains="max_iter"):
        _ = trainer.train(X, X, rng, "identity", "identity", "squared_error", options=StochasticOptions(max_iter=0))
    with assert_raises(contains="batch_size"):
        _ = trainer.train(X, X, rng, "identity", "identity", "squared_error", options=StochasticOptions(batch_size=0))
    with assert_raises(contains="one entry per row"):
        _ = trainer.train(X, X, rng, "identity", "identity", "squared_error", 0, X)
    with assert_raises(contains="finite nonzero sum"):
        _ = trainer.train(X, X, rng, "identity", "identity", "squared_error", 0, weights)
    assert_equal(trainer.updates, 0)
    assert_equal(trainer.t, 0)


def check_ownership_and_signed_weights[dtype: DType]() raises:
    var tensors = allocate[dtype](LayerLayout([1, 2]))
    var backing = zeros[dtype](Shape(4))
    var view = backing.view_with_layout(Shape(1, 2), NDArrayStrides(1, -2), 3)
    view.unsafe_set(0, 1)
    view.unsafe_set(-2, 2)
    tensors.coefs[0] = view^
    var trainer = StochasticTrainer(tensors)
    assert_equal(trainer.parameters().coefs[0].unsafe_get(0), 1)
    assert_equal(trainer.parameters().coefs[0].unsafe_get(1), 2)
    tensors.coefs[0].unsafe_set(0, 9)
    assert_equal(trainer.parameters().coefs[0].unsafe_get(0), 1)
    var X = zeros[dtype](Shape(2, 1))
    X.unsafe_set(0, 1)
    var y = zeros[dtype](Shape(2, 2))
    var weights = zeros[dtype](Shape(2))
    weights.unsafe_set(0, 1)
    weights.unsafe_set(1, -1)
    var rng = RandomState(42)
    var saved = trainer.copy()
    var report = trainer.train(
        X, y, rng, "identity", "identity", "squared_error", 0, weights,
        StochasticOptions(batch_size=1, max_iter=1, shuffle=False),
    )
    assert_equal(report.iteration_limit, True)
    assert_equal(trainer.updates, 2)
    assert_equal(saved.updates, 0)
    assert_equal(saved.parameters().coefs[0].unsafe_get(0), 1)
    assert_equal(tensors.coefs[0].unsafe_get(0), 9)


def main() raises:
    check_errors[DType.float32]()
    check_errors[DType.float64]()
    check_ownership_and_signed_weights[DType.float32]()
    check_ownership_and_signed_weights[DType.float64]()
    var weighting: List[Bool] = [False, True]
    for kind in range(5):
        for weighted in weighting:
            for mode in range(7):
                check[DType.float32](kind, weighted, mode)
                check[DType.float64](kind, weighted, mode)
