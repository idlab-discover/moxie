"""Tree inspection records and native kernel mathematics.

API/mathematical reference: scikit-learn, BSD-3-Clause; see LICENSES/sklearn.txt.
"""
import numojo as nm
from std.math import cos, isfinite
# Numax fdlibm log (within one ulp): Mojo 1.0 std.math log is ~1e-10 relative in Float64.
from numax.core.libm import log

@fieldwise_init
struct NodeData(ImplicitlyCopyable, Movable):
    var idx_start: Int
    var idx_end: Int
    var is_leaf: Bool
    var radius: Float64

@fieldwise_init
struct TreeStats(ImplicitlyCopyable, Movable):
    var trims: Int
    var leaves: Int
    var splits: Int
    var calls: Int

    def __init__(out self):
        self.trims = 0
        self.leaves = 0
        self.splits = 0
        self.calls = 0

@fieldwise_init
struct TreeArrays[dtype: DType](Copyable, Movable):
    var data: nm.NDArray[Self.dtype]
    var idx_array: List[Int]
    var node_data: List[NodeData]
    var node_bounds: nm.NDArray[Self.dtype]

comptime PI = Float64(3.1415926535897932384626433832795)

def validate_kernel(h: Float64, kernel: String, atol: Float64, rtol: Float64) raises:
    if not isfinite(h) or h <= 0 or not isfinite(atol) or not isfinite(rtol) or atol < 0 or rtol < 0:
        raise Error("ValueError: bandwidth must be positive and tolerances nonnegative and finite")
    if kernel != "gaussian" and kernel != "exponential" and kernel != "tophat" and kernel != "epanechnikov" and kernel != "linear" and kernel != "cosine":
        raise Error("ValueError: unknown kernel")

def log_kernel(distance: Float64, h: Float64, kernel: String) raises -> Float64:
    var z = distance/h
    if kernel == "gaussian": return -0.5*z*z
    if kernel == "exponential": return -z
    if distance >= h: return Float64("-inf")
    if kernel == "tophat": return 0
    if kernel == "epanechnikov": return log(1-z*z)
    if kernel == "linear": return log(1-z)
    return log(cos(PI*z/2))

def log_kernel_norm(d: Int, h: Float64, kernel: String) -> Float64:
    var volume = Float64(0) if d%2 == 0 else log(Float64(2))
    for j in range(2 if d%2 == 0 else 3,d+1,2): volume += log(2*PI/Float64(j))
    var factor = volume
    if kernel == "gaussian": factor = Float64(d)*log(2*PI)/2
    elif kernel == "epanechnikov": factor += log(Float64(2)/Float64(d+2))
    elif kernel == "exponential":
        factor += log(Float64(d))
        for j in range(1,d): factor += log(Float64(j))
    elif kernel == "linear": factor -= log(Float64(d+1))
    elif kernel == "cosine":
        var value = Float64(0)
        var term = 2/PI
        for k in range(1,d+1,2):
            value += term
            term *= -Float64((d-k)*(d-k-1))*(2/PI)*(2/PI)
        factor += log(Float64(d))+log(value)
    return -factor-Float64(d)*log(h)
