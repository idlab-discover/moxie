import numojo as nm
from std.testing import assert_equal, assert_raises
from moxie.linear_model._contracts import validate_xy, validate_weights, snapshot, LinearShape, LinearParameters, PreparedData, DensePreprocessor, DenseLinearPrediction, DenseSolver
from moxie._dense import row_squared_norms

def contracts[dtype: DType]() raises:
    var X = nm.ones[dtype](nm.Shape(2, 3))
    var y = nm.zeros[dtype](nm.Shape(2))
    var w = nm.ones[dtype](nm.Shape(2))
    validate_xy(X, y)
    validate_weights(w, 2)
    var copy = snapshot(X)
    var view = X.view()
    view.unsafe_set(0, 7)
    assert_equal(copy.unsafe_get(0), 1)
    assert_equal(X.unsafe_get(0), 7)
    var norms = row_squared_norms(copy)
    assert_equal(norms.unsafe_get(0), 3)
    var multi = nm.zeros[dtype](nm.Shape(2, 2))
    validate_xy(X, multi)
    var prepared = PreparedData[dtype](snapshot(X), snapshot(y), nm.zeros[dtype](nm.Shape(3)), nm.zeros[dtype](nm.Shape(1)), nm.ones[dtype](nm.Shape(3)), None, True)
    assert_equal(prepared.vector_target, True)
    var reversed_w = w.view_with_layout(nm.Shape(2), nm.NDArrayStrides(-1), 1)
    validate_weights(reversed_w, 2)
    var reversed_copy = snapshot(reversed_w)
    w.unsafe_set(1, 3)
    assert_equal(reversed_copy.unsafe_get(0), 1)
    var params = LinearParameters[dtype](nm.ones[dtype](nm.Shape(1, 3)), nm.zeros[dtype](nm.Shape(1)))
    params.validate(LinearShape(3, 1, True))
    var empty = nm.zeros[dtype](nm.Shape(0, 3))
    with assert_raises():
        validate_xy(empty, y)
    with assert_raises():
        validate_xy(y, y)
    with assert_raises():
        validate_xy(X, nm.zeros[dtype](nm.Shape(3)))
    with assert_raises():
        validate_xy(X, nm.zeros[dtype](nm.Shape(2, 0)))
    with assert_raises():
        validate_weights(w, 3)
    with assert_raises():
        validate_weights(nm.zeros[dtype](nm.Shape(2)), 2)
    w.unsafe_set(0, -1)
    with assert_raises():
        validate_weights(w, 2)
    validate_weights(w, 2, False)
    LinearShape(3, 1, True).validate()
    with assert_raises():
        LinearShape(3, 2, True).validate()

@fieldwise_init
struct EngineProbe(DensePreprocessor, DenseLinearPrediction, DenseSolver):
    comptime dtype = DType.float64
    comptime Config = Int
    comptime State = Int
    comptime Result = Int

    def prepare(self, X: nm.NDArray[Self.dtype], y: nm.NDArray[Self.dtype], weights: nm.NDArray[Self.dtype], fit_intercept: Bool, rescale_with_sw: Bool) raises -> PreparedData[Self.dtype]:
        raise Error("NotImplementedError: preprocessing probe")

    def decision(self, X: nm.NDArray[Self.dtype], coef: nm.NDArray[Self.dtype], intercept: nm.NDArray[Self.dtype]) raises -> nm.NDArray[Self.dtype]:
        raise Error("NotImplementedError: prediction probe")

    def solve(self, X: nm.NDArray[Self.dtype], y: nm.NDArray[Self.dtype], weights: nm.NDArray[Self.dtype], config: Int, mut state: Int) raises -> Int:
        raise Error("NotImplementedError: solver probe")


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    print("dense plumbing contracts passed")
    var parameters = LinearParameters[DType.float64](nm.ones[DType.float64](nm.Shape(1, 3)), nm.zeros[DType.float64](nm.Shape(1)))
    parameters.validate(LinearShape(3, 1, True))
    with assert_raises():
        parameters.validate(LinearShape(2, 1, True))
    var X = nm.ones[DType.float64](nm.Shape(2, 3))
    var y = nm.zeros[DType.float64](nm.Shape(2))
    var w = nm.ones[DType.float64](nm.Shape(2))
    var probe = EngineProbe()
    var state = 0
    with assert_raises():
        _ = probe.prepare(X, y, w, True, True)
    with assert_raises():
        _ = probe.decision(X, X, y)
    with assert_raises():
        _ = probe.solve(X, y, w, 0, state)
    w.unsafe_set(0, Float64("nan"))
    with assert_raises():
        validate_weights(w, 2)
    w.unsafe_set(0, Float64("inf"))
    with assert_raises():
        validate_weights(w, 2)
