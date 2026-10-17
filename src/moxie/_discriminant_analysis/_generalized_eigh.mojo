"""Generalized symmetric-definite eigenproblem A v = lambda B v (type 1).

Composition of mojo-linalg primitives, matching the contract of
`scipy.linalg.eigh(A, B)` (LAPACK sygvd, itype=1): ascending eigenvalues and
eigenvectors normalized so that V^T B V = I. Eigenvector signs and bases
within repeated eigenspaces are unspecified.

Algorithm (Golub & Van Loan, Matrix Computations 4th ed., 8.7.2):
B = L L^T (Cholesky), C = L^-1 A L^-T, C = W diag(lambda) W^T (Jacobi eigh),
V = L^-T W. A non-positive-definite B raises LinAlgError, like scipy.
"""
from ._work import (
    Work,
    get,
    put,
    matrix,
    transpose,
    checked_cholesky,
    checked_triangular_solve,
    checked_eigh,
)


@fieldwise_init
struct GeneralizedEigen(Movable):
    """Ascending eigenvalues (n,) and B-orthonormal eigenvectors (n, n) as columns."""

    var values: Work
    var vectors: Work


def generalized_symmetric_eigh(
    a: Work, b: Work, context: String
) raises -> GeneralizedEigen:
    """Solve A V = B V diag(values) for symmetric A and symmetric positive-definite B.

    Only the lower triangles of A and B are read. `context` names the caller in errors.
    """
    var n = a.shape[0]
    debug_assert(a.ndim == 2 and b.ndim == 2, "generalized eigh expects matrices")
    debug_assert(a.shape[1] == n and b.shape[0] == n and b.shape[1] == n, "square operands")
    var factor = checked_cholesky(b, context)
    var lower = factor.lower_factor()
    # Symmetrize A from its lower triangle so both solves see a symmetric input.
    var sym = matrix(n, n)
    for i in range(n):
        for j in range(i + 1):
            var x = get(a, i, j)
            put(sym, i, j, x)
            put(sym, j, i, x)
    # Y = L^-1 A, then C = L^-1 Y^T = L^-1 A L^-T (A symmetric).
    var y = checked_triangular_solve(lower, sym, transpose=False, context=context)
    var c = checked_triangular_solve(lower, transpose(y), transpose=False, context=context)
    for i in range(n):
        for j in range(i):
            var mean = 0.5 * (get(c, i, j) + get(c, j, i))
            put(c, i, j, mean)
            put(c, j, i, mean)
    var eig = checked_eigh(c, context)
    var vectors = checked_triangular_solve(
        lower, eig.eigenvectors(), transpose=True, context=context
    )
    return GeneralizedEigen(eig.eigenvalues(), vectors^)
