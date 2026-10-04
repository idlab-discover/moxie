"""One public scaler operation on touched inputs; peak RSS is measured externally.

Usage: workload <op> <float32|float64> <rows> <cols>

Every op first allocates and touches the same inputs (an n*p table, an n-element
weight vector), so `input` is the control and any other op's extra peak is the
operation's own live storage. `full_copy` is a deliberate full-table temporary
that the acceptance gate must reject.
"""
from std.sys import argv
from numojo import NDArray, Shape, NDArrayStrides, zeros
from moxie.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
    scale, minmax_scale, maxabs_scale, robust_scale,
)


def checksum[d: DType](x: NDArray[d]) raises -> Float64:
    """Touch a few entries so results cannot be elided."""
    var total = Float64(0)
    var step = max(1, x.size // 7)
    for k in range(0, x.size, step):
        total += Float64(x.unsafe_get(k))
    return total


def run[d: DType](op: String, n: Int, p: Int) raises:
    var X = zeros[d](Shape(n, p))
    for k in range(n * p):
        # Deterministic, non-constant, NaN-free values; every page is written.
        X.unsafe_set(k, Scalar[d](Float64((k * 7919) % 1009) - 504.0))
    var w = zeros[d](Shape(n))
    for i in range(n):
        w.unsafe_set(i, Scalar[d](1.0 + Float64(i % 3)))
    # A tiny strided view (every 1000th row, first column) of the large buffer.
    var rows = max(1, n // 1000)
    var tiny = X.view_with_layout(Shape(rows, 1), NDArrayStrides(1000 * p, 1), 0)
    var out: Float64
    if op == "input":
        out = checksum(X)
    elif op == "full_copy":
        out = checksum(X.copy())
    elif op == "std_fit":
        var s = StandardScaler[d]()
        _ = s.fit(X)
        out = s.mean_.value().unsafe_get(0)
    elif op == "std_fit_weighted":
        var s = StandardScaler[d]()
        _ = s.fit(X, sample_weight=w)
        out = s.mean_.value().unsafe_get(0)
    elif op == "std_fit_scalar":
        var s = StandardScaler[d]()
        _ = s.fit(X, sample_weight=2.5)
        out = s.mean_.value().unsafe_get(0)
    elif op == "std_partial_fit":
        var s = StandardScaler[d]()
        _ = s.partial_fit(X)
        _ = s.partial_fit(X, sample_weight=w)
        out = s.var_.value().unsafe_get(0)
    elif op == "std_transform":
        var s = StandardScaler[d]()
        _ = s.fit(X)
        out = checksum(s.transform(X))
    elif op == "std_inverse":
        var s = StandardScaler[d]()
        _ = s.fit(X)
        out = checksum(s.inverse_transform(X))
    elif op == "std_inplace":
        var s = StandardScaler[d]()
        _ = s.fit(X)
        s.transform_inplace(X)
        s.inverse_transform_inplace(X)
        out = checksum(X)
    elif op == "minmax_fit":
        var m = MinMaxScaler[d]()
        _ = m.fit(X)
        _ = m.partial_fit(X)
        out = Float64(m.scale_.unsafe_get(0))
    elif op == "minmax_transform":
        var m = MinMaxScaler[d]((-1.0, 1.0), clip=True)
        _ = m.fit(X)
        out = checksum(m.transform(X))
    elif op == "minmax_inplace":
        var m = MinMaxScaler[d]()
        _ = m.fit(X)
        m.transform_inplace(X)
        out = checksum(X)
    elif op == "maxabs_fit":
        var a = MaxAbsScaler[d]()
        _ = a.fit(X)
        _ = a.partial_fit(X)
        out = Float64(a.scale_.unsafe_get(0))
    elif op == "maxabs_transform":
        var a = MaxAbsScaler[d](clip=True)
        _ = a.fit(X)
        out = checksum(a.transform(X))
    elif op == "robust_fit":
        var r = RobustScaler[d](unit_variance=True)
        _ = r.fit(X)
        out = r.scale_.value().unsafe_get(0)
    elif op == "robust_transform":
        var r = RobustScaler[d]()
        _ = r.fit(X)
        out = checksum(r.transform(X))
    elif op == "scale":
        out = checksum(scale(X))
    elif op == "scale_axis1":
        out = checksum(scale(X, axis=1))
    elif op == "minmax_scale":
        out = checksum(minmax_scale(X, axis=1))
    elif op == "maxabs_scale":
        out = checksum(maxabs_scale(X))
    elif op == "robust_scale":
        out = checksum(robust_scale(X))
    elif op == "maxabs_scale_axis1":
        out = checksum(maxabs_scale(X, axis=1))
    elif op == "robust_scale_axis1":
        out = checksum(robust_scale(X, axis=1))
    elif op == "view_fit_transform":
        var s = StandardScaler[d]()
        _ = s.fit(tiny)
        var r = RobustScaler[d]()
        _ = r.fit(tiny)
        out = checksum(s.transform(tiny)) + checksum(r.transform(tiny)) + checksum(scale(tiny))
    else:
        raise Error("unknown op " + op)
    # Inputs stay live through the op (Mojo frees at last use), keeping the
    # input-only control comparable.
    out += checksum(X) + checksum(w) + Float64(tiny.shape[0])
    print("OK", op, out)


def main() raises:
    var args = argv()
    var op = String(args[1])
    var n = Int(String(args[3]))
    var p = Int(String(args[4]))
    if String(args[2]) == "float32":
        run[DType.float32](op, n, p)
    else:
        run[DType.float64](op, n, p)
