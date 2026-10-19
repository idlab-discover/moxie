"""Gaussian-process positive-definite algebra over mojo-linalg factors.

Adapts the validated mojo-linalg Cholesky and triangular solves to the family
`PositiveDefiniteAlgebra` contract: lower factors, `(n, r)` right-hand sides,
owned results, explicit failures, no jitter and no inverse fallback.
"""
from numojo import Shape, zeros
from moxie._gp_linalg import NumericalError, cholesky, triangular_solve

from moxie.gaussian_process._plumbing import Dense64, PositiveDefiniteAlgebra
from moxie.gaussian_process._kernel_math import flog


def _describe(error: NumericalError, operation: String) -> Error:
    if error.kind == "not_positive_definite":
        return Error(
            "LinAlgError: "
            + operation
            + ": matrix is not positive definite (leading minor of order "
            + String(error.index + 1)
            + " is not positive)"
        )
    if error.kind == "singular":
        return Error(
            "LinAlgError: "
            + operation
            + ": singular triangular factor (zero diagonal at index "
            + String(error.index)
            + ")"
        )
    return Error(
        "ValueError: "
        + operation
        + " failed ("
        + error.kind
        + " at index "
        + String(error.index)
        + "; "
        + error.operation
        + ")"
    )


@fieldwise_init
struct CholeskyAlgebra(Copyable, Movable, PositiveDefiniteAlgebra):
    """Dense Float64 Cholesky algebra (sklearn scipy.linalg usage in GPR/GPC)."""

    def cholesky(self, matrix: Dense64) raises -> Dense64:
        """Lower L with matrix = L L.T; raises `LinAlgError` if not SPD."""
        try:
            return cholesky(matrix).lower_factor()
        except e:
            raise _describe(e, "cholesky")

    def solve_triangular(
        self, lower: Dense64, rhs: Dense64, transpose: Bool = False
    ) raises -> Dense64:
        """Solve L X = B (or L.T X = B when transpose)."""
        try:
            return triangular_solve(lower, rhs, lower=True, transpose=transpose)
        except e:
            raise _describe(e, "solve_triangular")

    def cho_solve(self, lower: Dense64, rhs: Dense64) raises -> Dense64:
        """Solve (L L.T) X = B with two triangular solves."""
        try:
            var y = triangular_solve(lower, rhs, lower=True, transpose=False)
            return triangular_solve(lower, y, lower=True, transpose=True)
        except e:
            raise _describe(e, "cho_solve")

    def log_determinant(self, lower: Dense64) raises -> Float64:
        """log det(L L.T) = 2 * sum(log(diag(L)))."""
        if lower.ndim != 2 or lower.shape[0] != lower.shape[1]:
            raise Error("ValueError: log_determinant expects a square factor")
        var n = lower.shape[0]
        var total: Float64 = 0.0
        var s0 = Int(lower.strides[0])
        var s1 = Int(lower.strides[1])
        for i in range(n):
            total += flog(Float64(lower.unsafe_get(i * (s0 + s1))))
        return 2.0 * total


def identity(n: Int) raises -> Dense64:
    """Owned `(n, n)` identity."""
    var out = zeros[DType.float64](Shape(n, n))
    for i in range(n):
        out.unsafe_set(i * n + i, 1.0)
    return out^
