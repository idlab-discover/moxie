"""Family-local Moore-Penrose inverse via a scaled one-sided Jacobi SVD."""
from std.math import sqrt, max, isfinite
from .matrix import ProjectionMatrix


def pseudoinverse[
    dtype: DType
](matrix: ProjectionMatrix[dtype]) raises -> ProjectionMatrix[dtype]:
    """Dense pseudoinverse with SciPy's max(shape) * dtype-epsilon cutoff.

    Rotate columns of A until orthogonal, accumulating V, then form V S^-1 U^T.
    Working on the taller orientation keeps the number of rotation columns small.
    The Gram matrix is never formed. Nonconvergence raises instead of returning
    an approximate inverse without a convergence guarantee.
    """
    matrix.validate(nonempty=True)
    var transposed = matrix.rows < matrix.cols
    var oriented = matrix.transpose() if transposed else matrix.to_dense()
    var m = oriented.rows
    var n = oriented.cols
    var scale = Float64(0)
    for value in oriented.data:
        scale = max(scale, abs(Float64(value)))
    if scale == 0:
        return ProjectionMatrix[dtype](matrix.cols, matrix.rows)
    var work = List[Float64]()
    for value in oriented.data:
        work.append(Float64(value) / scale)
    var rotations = List[Float64](length=n * n, fill=0)
    for col in range(n):
        rotations[col * n + col] = 1
    comptime eps64 = Float64(2.220446049250313e-16)
    var converged = False
    for _ in range(100):
        var changed = False
        for p in range(n):
            for q in range(p + 1, n):
                var alpha = Float64(0)
                var beta = Float64(0)
                var gamma = Float64(0)
                for row in range(m):
                    var left = work[row * n + p]
                    var right = work[row * n + q]
                    alpha += left * left
                    beta += right * right
                    gamma += left * right
                # Sub-resolution columns cannot contribute at the rank cutoff.
                if alpha <= eps64 * eps64 or beta <= eps64 * eps64:
                    continue
                if abs(gamma) <= 8.0 * eps64 * sqrt(alpha) * sqrt(beta):
                    continue
                changed = True
                var tau = (beta - alpha) / (2.0 * gamma)
                var tangent: Float64
                if abs(tau) > 1e150:
                    tangent = 0.5 / tau
                else:
                    tangent = (1.0 if tau >= 0 else -1.0) / (
                        abs(tau) + sqrt(1.0 + tau * tau)
                    )
                var cosine = 1.0 / sqrt(1.0 + tangent * tangent)
                var sine = cosine * tangent
                for row in range(m):
                    var left = work[row * n + p]
                    var right = work[row * n + q]
                    work[row * n + p] = cosine * left - sine * right
                    work[row * n + q] = sine * left + cosine * right
                for row in range(n):
                    var left = rotations[row * n + p]
                    var right = rotations[row * n + q]
                    rotations[row * n + p] = cosine * left - sine * right
                    rotations[row * n + q] = sine * left + cosine * right
        if not changed:
            converged = True
            break
    if not converged:
        raise Error("ValueError: pseudoinverse Jacobi SVD did not converge")
    var norms = List[Float64](length=n, fill=0)
    var largest = Float64(0)
    for col in range(n):
        for row in range(m):
            norms[col] += work[row * n + col] * work[row * n + col]
        largest = max(largest, sqrt(norms[col]))
    comptime epsilon = 1.1920928955078125e-7 if dtype == DType.float32 else 2.220446049250313e-16
    var cutoff = Float64(m) * epsilon * largest
    var inverse = ProjectionMatrix[dtype](n, m)
    for row in range(n):
        for col in range(m):
            var value = Float64(0)
            for k in range(n):
                if sqrt(norms[k]) > cutoff:
                    value += rotations[row * n + k] * (
                        work[col * n + k] / norms[k]
                    )
            value /= scale
            if not isfinite(value):
                raise Error("ValueError: pseudoinverse exceeds numeric range")
            inverse.data[row * m + col] = Scalar[dtype](value)
    inverse.validate()
    if transposed:
        return inverse.transpose()
    return inverse^
