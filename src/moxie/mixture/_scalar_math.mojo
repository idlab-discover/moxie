"""Correctly rounded Float64 scalar log/exp for mixture kernels.

The pinned stdlib `std.math.log`/`exp`/`log1p` Float64 implementations are
polynomial approximations measured at roughly 1e-10 relative error (for example
log(10.1) is off by 2.3e-10). Log-likelihood parity with NumPy needs libm
accuracy, so these helpers use the LLVM intrinsics, which lower to the
platform math library (the same choice as `moxie._random`).
"""
from std.sys.intrinsics import llvm_intrinsic

comptime LOG_2PI: Float64 = 1.8378770664093454835606594728112352797227949472755668


@always_inline
def log64(x: Float64) -> Float64:
    """Natural logarithm (libm accuracy); log(0) = -inf, log(<0) = NaN."""
    return llvm_intrinsic["llvm.log", Float64, has_side_effect=False](x)


@always_inline
def exp64(x: Float64) -> Float64:
    """Exponential (libm accuracy); exp(-inf) = 0."""
    return llvm_intrinsic["llvm.exp", Float64, has_side_effect=False](x)


@always_inline
def log1p64(x: Float64) -> Float64:
    """Return log(1 + x) accurate for tiny x (Goldberg's correction on libm log)."""
    var u = 1.0 + x
    if u == 1.0:
        return x
    return log64(u) * (x / (u - 1.0))
