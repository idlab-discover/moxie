"""NCA objective and initialization; sklearn BSD-3-Clause, LICENSES/sklearn.txt."""
import numojo as nm
from std.math import abs, sqrt, isfinite
from std.sys import llvm_intrinsic
from mojo_linalg.svd import svd, SVDFactor
from moxie._dense import sum_axis2d
from moxie.neighbors._distance import at, Metric

comptime Matrix = nm.NDArray[DType.float64]


def factor(X: Matrix) raises -> SVDFactor[DType.float64]:
    try:
        return svd(X)
    except error:
        raise Error("ValueError: NCA initialization SVD: " + String(error))


def initialize_spectral(
    X: Matrix, codes: List[Int], count: Int, k: Int, lda: Bool
) raises -> Matrix:
    var n = X.shape[0]
    var d = X.shape[1]
    if not lda:
        if k > min(n, d):
            raise Error(
                "ValueError: PCA n_components exceeds min(n_samples,"
                " n_features)"
            )
        var centered = X.copy()
        var sums = sum_axis2d(X, 0)
        for i in range(n):
            for j in range(d):
                centered.unsafe_set(
                    i * d + j, at(X, i, j) - sums.unsafe_get(j) / Float64(n)
                )
        var decomposition = factor(centered)
        var vt = decomposition.vt()
        var result = nm.zeros[DType.float64](nm.Shape(k, d))
        for c in range(k):
            var pivot = 0
            for j in range(d):
                if abs(at(vt, c, j)) > abs(at(vt, c, pivot)):
                    pivot = j
            var sign = Float64(-1) if at(vt, c, pivot) < 0 else Float64(1)
            for j in range(d):
                result.unsafe_set(c * d + j, sign * at(vt, c, j))
        return result^
    if n <= count or k > min(d, count - 1):
        raise Error(
            "ValueError: LDA requires n_samples > n_classes and n_components <="
            " min(n_features, n_classes - 1)"
        )
    var means = nm.zeros[DType.float64](nm.Shape(count, d))
    var counts = List[Int]()
    for _ in range(count):
        counts.append(0)
    for i in range(n):
        counts[codes[i]] += 1
        for j in range(d):
            means.unsafe_set(
                codes[i] * d + j, at(means, codes[i], j) + at(X, i, j)
            )
    for c in range(count):
        for j in range(d):
            means.unsafe_set(c * d + j, at(means, c, j) / Float64(counts[c]))
    var centered = nm.zeros[DType.float64](nm.Shape(n, d))
    var scale = nm.zeros[DType.float64](nm.Shape(d))
    for j in range(d):
        var total = Float64(0)
        for i in range(n):
            var delta = at(X, i, j) - at(means, codes[i], j)
            centered.unsafe_set(i * d + j, delta)
            total += delta * delta
        var sd = sqrt(total / Float64(n))
        if sd == 0:
            sd = 1
        scale.unsafe_set(j, sd)
        for i in range(n):
            centered.unsafe_set(
                i * d + j, at(centered, i, j) / (sd * sqrt(Float64(n)))
            )
    var first = factor(centered)
    var singular = first.singular_values()
    var vt = first.vt()
    var rank = 0
    for i in range(singular.size):
        if singular.unsafe_get(i) > 1e-4:
            rank += 1
    if rank == 0:
        raise Error("ValueError: LDA initialization has zero within-class rank")
    var scalings = nm.zeros[DType.float64](nm.Shape(d, rank))
    for j in range(d):
        for r in range(rank):
            scalings.unsafe_set(
                j * rank + r,
                at(vt, r, j) / scale.unsafe_get(j) / singular.unsafe_get(r),
            )
    var total = sum_axis2d(X, 0)
    var weighted = nm.zeros[DType.float64](nm.Shape(count, d))
    for c in range(count):
        for j in range(d):
            weighted.unsafe_set(
                c * d + j,
                sqrt(Float64(counts[c]) / Float64(count - 1))
                * (at(means, c, j) - total.unsafe_get(j) / Float64(n)),
            )
    var between = weighted @ scalings
    var second = factor(between)
    var second_s = second.singular_values()
    var second_vt = second.vt()
    var output_rank = 0
    for i in range(second_s.size):
        if second_s.unsafe_get(i) > 1e-4 * second_s.unsafe_get(0):
            output_rank += 1
    var axes = scalings @ second_vt.T()
    var result = nm.zeros[DType.float64](nm.Shape(min(k, output_rank), d))
    for c in range(min(k, output_rank)):
        for j in range(d):
            result.unsafe_set(c * d + j, at(axes, j, c))
    return result^


def loss_gradient(
    A: Matrix, X: Matrix, codes: List[Int]
) raises -> Tuple[Float64, Matrix]:
    """Negative expected leave-one-out accuracy and its matrix gradient."""
    var embedded = X @ A.T()
    var n = X.shape[0]
    var metric = Metric[DType.float64]("sqeuclidean")
    metric.validate_features(A.shape[0])
    var probabilities = nm.zeros[DType.float64](nm.Shape(n, n))
    var weighted = nm.zeros[DType.float64](nm.Shape(n, n))
    var loss = Float64(0)
    for i in range(n):
        var nearest = Float64("inf")
        for j in range(n):
            if i == j:
                continue
            var distance = metric.distance(embedded, i, embedded, j)
            if not isfinite(distance):
                raise Error("ValueError: nonfinite NCA squared distance")
            probabilities.unsafe_set(i * n + j, distance)
            nearest = min(nearest, distance)
        var denominator = Float64(0)
        for j in range(n):
            var value = Float64(0)
            if i != j:
                value = llvm_intrinsic[
                    "llvm.exp", Float64, has_side_effect=False
                ](nearest - at(probabilities, i, j))
            probabilities.unsafe_set(i * n + j, value)
            denominator += value
        var success = Float64(0)
        for j in range(n):
            var probability = at(probabilities, i, j) / denominator
            probabilities.unsafe_set(i * n + j, probability)
            if codes[i] == codes[j]:
                success += probability
        loss -= success
        for j in range(n):
            weighted.unsafe_set(
                i * n + j,
                at(probabilities, i, j)
                * (Float64(1 if codes[i] == codes[j] else 0) - success),
            )
    var columns = sum_axis2d(weighted, 0)
    var symmetric = nm.zeros[DType.float64](nm.Shape(n, n))
    for i in range(n):
        for j in range(n):
            symmetric.unsafe_set(
                i * n + j,
                -columns.unsafe_get(i) if i
                == j else at(weighted, i, j) + at(weighted, j, i),
            )
    var intermediate = embedded.T() @ symmetric
    var gradient = intermediate @ X
    for i in range(gradient.size):
        var value = -2 * gradient.unsafe_get(i)
        if not isfinite(value):
            raise Error("ValueError: nonfinite NCA objective gradient")
        gradient.unsafe_set(i, value)
    return (loss, gradient^)


def inner(a: Matrix, b: Matrix) -> Float64:
    var result = Float64(0)
    for i in range(a.size):
        result += a.unsafe_get(i) * b.unsafe_get(i)
    return result
