# Adapted from sklearn/linear_model/_omp.py, scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Count-constrained Gram OMP for dense consumers."""
from std.math import abs, isfinite, sqrt

from mojo_linalg import triangular_solve
from numojo import NDArray, NDArrayStrides, Shape, zeros
from ._linear_base import _entry, check_finite


def omp_gram_codes[dtype: DType](
    gram: NDArray[dtype], covariance: NDArray[dtype], n_nonzero_coefs: Int
) raises -> NDArray[dtype]:
    """Solve fixed-count OMP for covariance columns, returning one row per target.

    The Gram matrix represents a symmetric positive semidefinite system.
    The input arrays are borrowed and may be strided.
    """
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] == 0:
        raise Error("ValueError: OMP gram must have shape (atoms, atoms)")
    var atoms = gram.shape[0]
    if covariance.ndim != 2 or covariance.shape[0] != atoms or covariance.shape[1] == 0:
        raise Error("ValueError: OMP cov must have shape (atoms, samples)")
    if n_nonzero_coefs < 1 or n_nonzero_coefs > atoms:
        raise Error("ValueError: OMP n_nonzero_coefs must be between 1 and the number of atoms")
    check_finite(gram, "gram", "OMP")
    check_finite(covariance, "cov", "OMP")

    var samples = covariance.shape[1]
    var code = zeros[dtype](Shape(samples, atoms))
    var factor = zeros[dtype](Shape(n_nonzero_coefs, n_nonzero_coefs))
    var residual = zeros[dtype](Shape(atoms))
    var order = List[Int](capacity=atoms)
    for atom in range(atoms):
        order.append(atom)
    comptime eps = Scalar[dtype](
        1.1920928955078125e-7 if dtype == DType.float32 else 2.220446049250313e-16
    )

    for sample in range(samples):
        for atom in range(atoms):
            order[atom] = atom
            residual.unsafe_set(atom, _entry(covariance, atom, sample))

        for active in range(n_nonzero_coefs):
            # Preserve the reference's swap order when correlations tie.
            var selected = 0
            var largest = abs(residual.unsafe_get(order[0]))
            for position in range(1, atoms):
                var magnitude = abs(residual.unsafe_get(order[position]))
                if magnitude > largest:
                    largest = magnitude
                    selected = position
            if selected < active or largest * largest < eps:
                break

            var atom = order[selected]
            var pivot = _entry(gram, atom, atom)
            if active > 0:
                var previous = factor.view_with_layout(
                    Shape(active, active), NDArrayStrides(n_nonzero_coefs, 1), 0
                )
                var cross = zeros[dtype](Shape(active))
                for col in range(active):
                    cross.unsafe_set(col, _entry(gram, atom, order[col]))
                var row = triangular_solve(previous, cross)
                var row_norm = Scalar[dtype](0)
                for col in range(active):
                    var value = row.unsafe_get(col)
                    row_norm += value * value
                    factor.unsafe_set(active * n_nonzero_coefs + col, value)
                pivot -= row_norm
            if pivot <= (eps if active > 0 else Scalar[dtype](0)):
                break

            factor.unsafe_set(active * n_nonzero_coefs + active, sqrt(pivot))
            order[selected] = order[active]
            order[active] = atom

            var size = active + 1
            var lower = factor.view_with_layout(
                Shape(size, size), NDArrayStrides(n_nonzero_coefs, 1), 0
            )
            var rhs = zeros[dtype](Shape(size))
            for col in range(size):
                rhs.unsafe_set(col, _entry(covariance, order[col], sample))
            var forward = triangular_solve(lower, rhs)
            var coefficients = triangular_solve(lower, forward, transpose=True)
            for col in range(size):
                code.unsafe_set(sample * atoms + order[col], coefficients.unsafe_get(col))
            if size == n_nonzero_coefs:
                break

            for candidate in range(atoms):
                var projection = Scalar[dtype](0)
                for col in range(size):
                    projection += _entry(gram, candidate, order[col]) * coefficients.unsafe_get(col)
                var correlation = _entry(covariance, candidate, sample) - projection
                if not isfinite(correlation):
                    raise Error("ValueError: OMP residual correlations must remain finite")
                residual.unsafe_set(candidate, correlation)
    return code^
