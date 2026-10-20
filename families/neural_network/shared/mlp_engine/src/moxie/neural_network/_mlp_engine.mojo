"""Dense MLP parameter allocation, initialization and solver vector conversion.

Initialization and packing follow scikit-learn (BSD-3-Clause; see LICENSES).
"""
from numojo import NDArray, Shape, zeros
from std.math import sqrt, isfinite
from moxie._random import RandomState
from moxie.neural_network._plumbing import LayerLayout, LayerTensors


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
    if activation != "identity" and activation != "logistic" and activation != "tanh" and activation != "relu":
        raise Error("ValueError: unsupported hidden activation")
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
