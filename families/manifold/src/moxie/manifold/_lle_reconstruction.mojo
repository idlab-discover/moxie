# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Local reconstruction operators for the four dense LLE methods."""
from numojo import NDArray, Shape, ones, zeros
from std.builtin.sort import sort
from std.math import abs, isfinite, sqrt
from moxie._linalg import covariance_solve
from moxie.decomposition._svd_engine import qr_basis
from moxie.neighbors._distance import at
from mojo_linalg import svd, symmetric_eigh

comptime Matrix = NDArray[DType.float64]


@always_inline
def _add_symmetric(mut matrix: Matrix, i: Int, j: Int, value: Float64) raises:
    var n = matrix.shape[1]
    var updated = matrix.unsafe_get(i * n + j) + value
    matrix.unsafe_set(i * n + j, updated)
    matrix.unsafe_set(j * n + i, updated)


def _neighborhood(
    training: Matrix, indices: Span[Int, _], center: Int,
) raises -> Matrix:
    """Gather neighbors and subtract the sample or the neighborhood mean."""
    var features = training.shape[1]
    var local = zeros[DType.float64](Shape(len(indices), features))
    for feature in range(features):
        var origin: Float64
        if center >= 0:
            origin = at(training, center, feature)
        else:
            origin = 0
            for index in indices:
                origin += at(training, index, feature)
            origin /= Float64(len(indices))
        for neighbor in range(len(indices)):
            local.unsafe_set(
                neighbor * features + feature,
                at(training, indices[neighbor], feature) - origin,
            )
    return local^


def _normalized_weights(var weights: Matrix) raises -> Matrix:
    var total: Float64 = 0
    for i in range(weights.size):
        total += weights.unsafe_get(i)
    if total == 0 or not isfinite(total):
        raise Error("ValueError: LLE reconstruction weights have invalid sum")
    for i in range(weights.size):
        weights.unsafe_set(i, weights.unsafe_get(i) / total)
    return weights^


def barycenter(
    query: Matrix, row: Int, training: Matrix, indices: Span[Int, _], reg: Float64,
) raises -> Matrix:
    var k = len(indices)
    var features = training.shape[1]
    var differences = zeros[DType.float64](Shape(k, features))
    for j in range(k):
        for feature in range(features):
            differences.unsafe_set(
                j * features + feature,
                at(training, indices[j], feature) - at(query, row, feature),
            )
    var gram = differences @ differences.T()
    var trace: Float64 = 0
    for j in range(k):
        trace += gram.unsafe_get(j * k + j)
    var regularizer = reg * trace if trace > 0 else reg
    for j in range(k):
        var diagonal = j * k + j
        gram.unsafe_set(diagonal, gram.unsafe_get(diagonal) + regularizer)
    return _normalized_weights(covariance_solve(gram, ones[DType.float64](Shape(k))))


@fieldwise_init
struct _LocalSpectrum(Movable):
    var values: Matrix
    var vectors: Optional[Matrix]


def _local_spectrum(
    local: Matrix, *, vectors: Bool = True, full_basis: Bool = False,
) raises -> _LocalSpectrum:
    var k = local.shape[0]
    var features = local.shape[1]
    if k <= features:
        var factor = symmetric_eigh(local @ local.T())
        if vectors:
            return _LocalSpectrum(factor.eigenvalues(), factor.eigenvectors())
        return _LocalSpectrum(factor.eigenvalues(), None)

    # Preserve small singular values for the modified energy-ratio test.
    var factor = svd(local)
    var singular = factor.singular_values()
    var values = zeros[DType.float64](Shape(k))
    for d in range(features):
        var sigma = singular.unsafe_get(d)
        values.unsafe_set(k - 1 - d, sigma * sigma)
    if not vectors:
        return _LocalSpectrum(values^, None)

    var thin = factor.u()
    if not full_basis:
        var ascending = zeros[DType.float64](Shape(k, features))
        for i in range(k):
            for d in range(features):
                ascending.unsafe_set(i * features + d, at(thin, i, features - 1 - d))
        return _LocalSpectrum(values^, ascending^)

    # The unpivoted QR completion preserves the leading left singular vectors
    # up to sign and supplies an orthonormal basis for the remaining null space.
    var padded = zeros[DType.float64](Shape(k, k))
    for i in range(k):
        for d in range(features):
            padded.unsafe_set(i * k + d, at(thin, i, d))
    var basis = qr_basis(padded)
    for i in range(k):
        for d in range(k // 2):
            var left = i * k + d
            var right = i * k + k - 1 - d
            var value = basis.unsafe_get(left)
            basis.unsafe_set(left, basis.unsafe_get(right))
            basis.unsafe_set(right, value)
    return _LocalSpectrum(values^, basis^)


def _add_standard(
    mut matrix: Matrix, training: Matrix, indices: Span[Int, _], row: Int, reg: Float64,
) raises:
    var weights = barycenter(training, row, training, indices, reg)
    for j in range(len(indices)):
        var weight = weights.unsafe_get(j)
        for l in range(j + 1):
            _add_symmetric(matrix, indices[j], indices[l], weight * weights.unsafe_get(l))
        _add_symmetric(matrix, row, indices[j], (-2.0 if row == indices[j] else -1.0) * weight)
    _add_symmetric(matrix, row, row, 1)


def _add_ltsa(
    mut matrix: Matrix, indices: Span[Int, _], tangent: Matrix, dimensions: Int,
) raises:
    var k = len(indices)
    for i in range(k):
        for j in range(i + 1):
            var projection = 1.0 / Float64(k)
            for d in range(dimensions):
                var column = tangent.shape[1] - 1 - d
                projection += at(tangent, i, column) * at(tangent, j, column)
            _add_symmetric(matrix, indices[i], indices[j], (1.0 if i == j else 0.0) - projection)


def _add_hessian(
    mut matrix: Matrix, indices: Span[Int, _], tangent: Matrix,
    dimensions: Int, tolerance: Float64,
) raises:
    var k = len(indices)
    # Zero padding requests the same full, unpivoted Householder basis as QR
    # on the rectangular polynomial design followed by orthogonal completion.
    var design = zeros[DType.float64](Shape(k, k))
    var last = tangent.shape[1] - 1
    for i in range(k):
        design.unsafe_set(i * k, 1)
        for d in range(dimensions):
            design.unsafe_set(i * k + 1 + d, at(tangent, i, last - d))
        var column = 1 + dimensions
        for d in range(dimensions):
            for other in range(d, dimensions):
                design.unsafe_set(
                    i * k + column,
                    at(tangent, i, last - d) * at(tangent, i, last - other),
                )
                column += 1
    var basis = qr_basis(design)
    var start = dimensions + 1
    for d in range(start, k):
        var total: Float64 = 0
        for i in range(k):
            total += at(basis, i, d)
        if abs(total) < tolerance:
            continue
        if total == 0:
            raise Error("ValueError: LLE Hessian basis cannot be normalized")
        for i in range(k):
            var offset = i * k + d
            basis.unsafe_set(offset, basis.unsafe_get(offset) / total)

    for i in range(k):
        for j in range(i + 1):
            var update: Float64 = 0
            for d in range(start, k):
                update += at(basis, i, d) * at(basis, j, d)
            _add_symmetric(matrix, indices[i], indices[j], update)


def _add_modified(
    mut matrix: Matrix, row: Int, indices: Span[Int, _], values: Matrix,
    vectors: Matrix, nonzero_count: Int, eta: Float64, tolerance: Float64,
) raises:
    var k = len(indices)
    var first = k - nonzero_count
    var energy: Float64 = 0
    var cumulative = List[Float64](capacity=nonzero_count)
    for d in range(k - 1, first - 1, -1):
        energy += max(values.unsafe_get(d), 0)
        cumulative.append(energy)
    var regularizer = 0.001 * energy

    # The reversed prefix ratios are increasing as more tail directions enter.
    var dimension = first
    for d in range(nonzero_count - 2, -1, -1):
        var head = cumulative[d]
        if head <= 0 or energy / head - 1 >= eta:
            break
        dimension += 1
    if dimension == 0:
        return

    var sums = zeros[DType.float64](Shape(k))
    for d in range(k):
        var total: Float64 = 0
        for j in range(k):
            total += at(vectors, j, d)
        sums.unsafe_set(d, total)

    var weights = zeros[DType.float64](Shape(k))
    for d in range(k):
        var variance = max(values.unsafe_get(d), 0) if d >= first else 0
        var coefficient = sums.unsafe_get(d) / (variance + regularizer)
        for j in range(k):
            weights.unsafe_set(j, weights.unsafe_get(j) + at(vectors, j, d) * coefficient)
    weights = _normalized_weights(weights^)

    var squared_norm: Float64 = 0
    for d in range(dimension):
        var total = sums.unsafe_get(d)
        squared_norm += total * total
    var alpha = sqrt(squared_norm / Float64(dimension))
    var householder = zeros[DType.float64](Shape(dimension))
    squared_norm = 0
    for d in range(dimension):
        var value = alpha - sums.unsafe_get(d)
        householder.unsafe_set(d, value)
        squared_norm += value * value
    var norm = sqrt(squared_norm)
    if norm != 0 and norm >= tolerance:
        for d in range(dimension):
            householder.unsafe_set(d, householder.unsafe_get(d) / norm)
    else:
        householder = zeros[DType.float64](Shape(dimension))

    var residuals = zeros[DType.float64](Shape(k, dimension))
    for j in range(k):
        var projection: Float64 = 0
        for d in range(dimension):
            projection += at(vectors, j, d) * householder.unsafe_get(d)
        var total: Float64 = 0
        for d in range(dimension):
            var value = at(vectors, j, d) - 2 * projection * householder.unsafe_get(d)
            value += (1 - alpha) * weights.unsafe_get(j)
            residuals.unsafe_set(j * dimension + d, value)
            total += value
        _add_symmetric(matrix, row, indices[j], (-2.0 if row == indices[j] else -1.0) * total)
    _add_symmetric(matrix, row, row, Float64(dimension))

    for i in range(k):
        for j in range(i + 1):
            var update: Float64 = 0
            for d in range(dimension):
                update += at(residuals, i, d) * at(residuals, j, d)
            _add_symmetric(matrix, indices[i], indices[j], update)


def _modified_eta(
    X: Matrix, neighbors: Span[Int, _], k: Int, dimensions: Int,
) raises -> Float64:
    var ratios = List[Float64](capacity=X.shape[0])
    var first = k - min(k, X.shape[1])
    for row in range(X.shape[0]):
        var local = _neighborhood(X, neighbors[row * k:(row + 1) * k], row)
        var local_spectrum = _local_spectrum(local, vectors=False)
        ref values = local_spectrum.values
        var head: Float64 = 0
        var tail: Float64 = 0
        for j in range(first, k - dimensions):
            tail += max(values.unsafe_get(j), 0)
        for j in range(k - dimensions, k):
            head += max(values.unsafe_get(j), 0)
        if head <= 0:
            raise Error("ValueError: modified LLE requires nonzero neighborhood variance")
        ratios.append(tail / head)
    sort(Span(ratios))
    var half = len(ratios) // 2
    if len(ratios) % 2 == 0:
        return 0.5 * (ratios[half - 1] + ratios[half])
    return ratios[half]


def reconstruction_operator(
    X: Matrix, neighbors: Span[Int, _], k: Int, dimensions: Int, method: String,
    reg: Float64, hessian_tol: Float64, modified_tol: Float64,
) raises -> Matrix:
    var n = X.shape[0]
    var matrix = zeros[DType.float64](Shape(n, n))
    var eta: Float64 = 0
    if method == "modified":
        # Recompute local factors in the assembly pass to avoid storing N*k*k.
        eta = _modified_eta(X, neighbors, k, dimensions)

    for row in range(n):
        var indices = neighbors[row * k:(row + 1) * k]
        if method == "standard":
            _add_standard(matrix, X, indices, row, reg)
            continue
        var local = _neighborhood(X, indices, row if method == "modified" else -1)
        var spectrum = _local_spectrum(local, full_basis=method == "modified")
        ref vectors = spectrum.vectors.value()
        if method == "modified":
            _add_modified(
                matrix, row, indices, spectrum.values, vectors,
                min(k, X.shape[1]), eta, modified_tol,
            )
        elif method == "hessian":
            _add_hessian(matrix, indices, vectors, dimensions, hessian_tol)
        else:
            _add_ltsa(matrix, indices, vectors, dimensions)
    return matrix^
