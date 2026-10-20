from numojo import NDArray, NDArrayStrides, Shape, zeros
from std.testing import assert_equal, assert_raises
from moxie.neural_network._plumbing import LayerLayout, LayerTensors, snapshot_matrix, validate_matrix, validate_incremental
from moxie.neural_network._contracts import MLPKernel, MLPOptimizer, LBFGSObjective, RBMSampling
from moxie._random import RandomState


struct RaisingEngine(MLPKernel, MLPOptimizer, LBFGSObjective, RBMSampling):
    def __init__(out self):
        pass

    comptime Matrix = NDArray[DType.float64]
    comptime Vector = NDArray[DType.float64]
    comptime Tensors = LayerTensors[DType.float64]
    comptime Workspace = List[NDArray[DType.float64]]
    comptime Input = NDArray[DType.float64]
    comptime Samples = NDArray[DType.bool]

    def initialize(self, mut params: Self.Tensors, mut rng: RandomState, activation: String) raises:
        raise Error("Contract probe: no engine")
    def forward(self, X: Self.Matrix, params: Self.Tensors, hidden: String, output: String) raises -> Self.Matrix:
        raise Error("Contract probe: no engine")
    def loss_gradient(self, X: Self.Matrix, y: Self.Matrix, weights: Optional[Self.Matrix], params: Self.Tensors, mut gradients: Self.Tensors, mut work: Self.Workspace, hidden: String, output: String, loss: String, alpha: Float64) raises -> Float64:
        raise Error("Contract probe: no engine")
    def update_params(mut self, mut params: Self.Tensors, gradients: Self.Tensors) raises:
        raise Error("Contract probe: no engine")
    def iteration_ends(mut self, time_step: Int) raises:
        raise Error("Contract probe: no engine")
    def trigger_stopping(mut self, message: String, verbose: Bool) raises -> Bool:
        raise Error("Contract probe: no engine")
    def value_gradient(mut self, packed: Self.Vector, mut gradient: Self.Vector) raises -> Float64:
        raise Error("Contract probe: no engine")
    def gibbs(mut self, X: Self.Input) raises -> Self.Samples:
        raise Error("Contract probe: no engine")
    def partial_fit(mut self, X: Self.Input, y: Optional[Self.Input] = None) raises:
        raise Error("Contract probe: no engine")


def check_dtype[dtype: DType]() raises:
    var units: List[Int] = [2, 3, 1]
    var layout = LayerLayout(units)
    units[0] = 7
    assert_equal(layout.units[0], 2)
    var coefs = List[NDArray[dtype]]()
    var biases = List[NDArray[dtype]]()
    coefs.append(zeros[dtype](Shape(2, 3)))
    coefs.append(zeros[dtype](Shape(3, 1)))
    biases.append(zeros[dtype](Shape(3)))
    biases.append(zeros[dtype](Shape(1)))
    var tensors = LayerTensors[dtype](layout, coefs, biases)
    coefs[0].unsafe_set(0, 9)
    biases[0].unsafe_set(0, 8)
    assert_equal(tensors.coefs[0].unsafe_get(0), 0)
    assert_equal(tensors.intercepts[0].unsafe_get(0), 0)
    var copied = tensors.copy()
    copied.coefs[0].unsafe_set(0, 3)
    copied.intercepts[0].unsafe_set(0, 4)
    assert_equal(tensors.coefs[0].unsafe_get(0), 0)
    assert_equal(tensors.intercepts[0].unsafe_get(0), 0)
    var X = zeros[dtype](Shape(3, 4))
    var shared = X.view()
    shared.unsafe_set(0, 2)
    assert_equal(X.unsafe_get(0), 2)
    var snap = snapshot_matrix(X)
    X.unsafe_set(0, 7)
    assert_equal(snap.unsafe_get(0), 2)
    var strided = X.view_with_layout(Shape(2, 2), NDArrayStrides(4, -1), 2)
    validate_matrix(strided)
    var strided_copy = snapshot_matrix(strided)
    strided.unsafe_set(0, 5)
    assert_equal(strided_copy.unsafe_get(0), 0)
    with assert_raises():
        validate_matrix(zeros[dtype](Shape(0, 2)))
    with assert_raises():
        validate_matrix(zeros[dtype](Shape(2, 0)))
    with assert_raises():
        validate_matrix(zeros[dtype](Shape(2)))
    X.unsafe_set(0, SIMD[dtype, 1](Float64("inf")))
    with assert_raises():
        validate_matrix(X)
    X.unsafe_set(0, SIMD[dtype, 1](Float64("nan")))
    with assert_raises():
        validate_matrix(X)
    with assert_raises():
        _ = LayerTensors[dtype](LayerLayout([2, 2]), coefs, biases)
    biases[0] = zeros[dtype](Shape(2))
    with assert_raises():
        _ = LayerTensors[dtype](layout, coefs, biases)
    biases[0] = zeros[dtype](Shape(1, 3))
    with assert_raises():
        _ = LayerTensors[dtype](layout, coefs, biases)


def main() raises:
    check_dtype[DType.float32]()
    check_dtype[DType.float64]()
    with assert_raises():
        _ = LayerLayout([2])
    with assert_raises():
        _ = LayerLayout([2, 0])
    with assert_raises():
        _ = LayerLayout([2, -1])
    validate_incremental("sgd", False)
    validate_incremental("adam", False)
    with assert_raises():
        validate_incremental("lbfgs", False)
    with assert_raises():
        validate_incremental("invalid", False)
    with assert_raises():
        validate_incremental("sgd", True)
    var engine = RaisingEngine()
    var X = zeros[DType.float64](Shape(2, 1))
    var cs = List[NDArray[DType.float64]]()
    var bs = List[NDArray[DType.float64]]()
    cs.append(zeros[DType.float64](Shape(1, 1)))
    bs.append(zeros[DType.float64](Shape(1)))
    var params = LayerTensors[DType.float64](LayerLayout([1, 1]), cs, bs)
    var gradients = params.copy()
    var work = List[NDArray[DType.float64]]()
    var rng = RandomState(42)
    with assert_raises():
        engine.initialize(params, rng, "relu")
    with assert_raises():
        _ = engine.forward(X, params, "relu", "identity")
    with assert_raises():
        _ = engine.loss_gradient(X, X, None, params, gradients, work, "relu", "identity", "squared_error", 0.0001)
    with assert_raises():
        engine.update_params(params, gradients)
    with assert_raises():
        engine.iteration_ends(2)
    with assert_raises():
        _ = engine.trigger_stopping("", False)
    var packed = zeros[DType.float64](Shape(2))
    var grad = packed.copy()
    with assert_raises():
        _ = engine.value_gradient(packed, grad)
    with assert_raises():
        _ = engine.gibbs(X)
    with assert_raises():
        engine.partial_fit(X)
    print("PASS dense plumbing and raising extension contracts")
