"""Gaussian sufficient statistics and log densities for all covariance types.

Translation of sklearn.mixture._gaussian_mixture (pinned 4358542a):
`_estimate_gaussian_parameters`, `_estimate_gaussian_covariances_*` and
`_estimate_log_gaussian_prob`. Both GaussianMixture and
BayesianGaussianMixture consume these kernels; they are re-exported from
`moxie.mixture._gaussian_mixture` as the plumbing contract names that owner.

Numerics: products accumulate in Float64; every intermediate that NumPy
materializes as an array (matrix products, elementwise products, differences)
is rounded to the estimator dtype at the same point, so Float64 results agree
with the reference to accumulation-order rounding and Float32 results follow
the same rounding structure. Formulas intentionally keep the reference's
expanded forms (e.g. tied covariance X'X - N mu'mu) instead of centered ones.
Inputs may be strided; outputs are owned contiguous row-major arrays.
"""
from numojo import NDArray, Shape, zeros
from moxie.mixture._contracts import covariance_shape
from moxie.mixture._scalar_math import LOG_2PI


@always_inline
def rounded[dtype: DType](value: Float64) -> Float64:
    """Round a Float64 value through the storage dtype (identity for Float64)."""
    return Float64(value.cast[dtype]())


def dtype_eps[dtype: DType]() -> Float64:
    """Machine epsilon of the storage dtype (numpy.finfo(dtype).eps)."""
    comptime if dtype == DType.float32:
        return 1.1920928955078125e-07
    else:
        return 2.220446049250313e-16


struct Matrix[dtype: DType](Copyable, Movable):
    """Borrowed strided 2-D read view over an NDArray (no copy)."""

    var s0: Int
    var s1: Int
    var rows: Int
    var cols: Int

    def __init__(out self, x: NDArray[Self.dtype]) raises:
        if x.ndim != 2:
            raise Error(String("ValueError: expected a matrix, got rank ", x.ndim))
        self.rows = x.shape[0]
        self.cols = x.shape[1]
        self.s0 = Int(x.strides[0])
        self.s1 = Int(x.strides[1])

    @always_inline
    def at(self, x: NDArray[Self.dtype], i: Int, j: Int) -> Float64:
        return Float64(x.unsafe_get(i * self.s0 + j * self.s1))


struct GaussianParameters[dtype: DType](Movable):
    """(nk, means, covariances) from `_estimate_gaussian_parameters`."""

    var nk: NDArray[Self.dtype]
    var means: NDArray[Self.dtype]
    var covariances: NDArray[Self.dtype]

    def __init__(out self, var nk: NDArray[Self.dtype], var means: NDArray[Self.dtype], var covariances: NDArray[Self.dtype]):
        self.nk = nk^
        self.means = means^
        self.covariances = covariances^


def estimate_gaussian_parameters[dtype: DType](X: NDArray[dtype], resp: NDArray[dtype], reg_covar: Float64, kind: String) raises -> GaussianParameters[dtype]:
    """Weighted counts, means and covariances of the given responsibilities.

    nk = resp.sum(0) + 10 eps; means = resp.T @ X / nk; covariance by `kind`
    (full (k,d,d), tied (d,d), diag (k,d), spherical (k,)) with reg_covar
    added to the diagonal/variances.
    """
    var xv = Matrix[dtype](X)
    var rv = Matrix[dtype](resp)
    var n = xv.rows
    var d = xv.cols
    var k = rv.cols
    if rv.rows != n:
        raise Error(String("ValueError: responsibilities have ", rv.rows, " rows for ", n, " samples"))
    var shape = covariance_shape(kind, k, d)
    var eps10 = rounded[dtype](10.0 * dtype_eps[dtype]())
    var nk = zeros[dtype](Shape(k))
    for j in range(k):
        var total = 0.0
        for i in range(n):
            total += rv.at(resp, i, j)
        nk.unsafe_set(j, (rounded[dtype](total) + eps10).cast[dtype]())
    var means = zeros[dtype](Shape(k, d))
    var acc = List[Float64](length=k * d, fill=0.0)
    for i in range(n):
        for j in range(k):
            var r = rv.at(resp, i, j)
            if r != 0.0:
                for f in range(d):
                    acc[j * d + f] += r * xv.at(X, i, f)
    for j in range(k):
        var count = Float64(nk.unsafe_get(j))
        for f in range(d):
            means.unsafe_set(j * d + f, (rounded[dtype](acc[j * d + f]) / count).cast[dtype]())
    var covariances = zeros[dtype](shape)
    if kind == "full":
        _covariances_full(X, xv, resp, rv, nk, means, reg_covar, covariances)
    elif kind == "tied":
        _covariances_tied(X, xv, nk, means, reg_covar, covariances)
    else:
        _covariances_diag(X, xv, resp, rv, nk, means, reg_covar, kind, covariances)
    return GaussianParameters[dtype](nk^, means^, covariances^)


def _covariances_full[dtype: DType](X: NDArray[dtype], xv: Matrix[dtype], resp: NDArray[dtype], rv: Matrix[dtype], nk: NDArray[dtype], means: NDArray[dtype], reg_covar: Float64, mut out: NDArray[dtype]) raises:
    # covariances[k] = (resp[:, k] * diff.T) @ diff / nk[k]; diag += reg_covar
    var n = xv.rows
    var d = xv.cols
    var diff = List[Float64](length=d, fill=0.0)
    var weighted = List[Float64](length=d, fill=0.0)
    var acc = List[Float64](length=d * d, fill=0.0)
    for k in range(rv.cols):
        for idx in range(d * d):
            acc[idx] = 0.0
        for i in range(n):
            var r = rv.at(resp, i, k)
            for f in range(d):
                diff[f] = rounded[dtype](xv.at(X, i, f) - Float64(means.unsafe_get(k * d + f)))
                weighted[f] = rounded[dtype](r * diff[f])
            for f in range(d):
                var a = weighted[f]
                if a != 0.0:
                    for g in range(d):
                        acc[f * d + g] += a * diff[g]
        var count = Float64(nk.unsafe_get(k))
        for f in range(d):
            for g in range(d):
                var value = rounded[dtype](rounded[dtype](acc[f * d + g]) / count)
                if f == g:
                    value = rounded[dtype](value + reg_covar)
                out.unsafe_set(k * d * d + f * d + g, value.cast[dtype]())


def _covariances_tied[dtype: DType](X: NDArray[dtype], xv: Matrix[dtype], nk: NDArray[dtype], means: NDArray[dtype], reg_covar: Float64, mut out: NDArray[dtype]) raises:
    # covariance = (X.T @ X - (nk * means.T) @ means) / nk.sum(); diag += reg
    var n = xv.rows
    var d = xv.cols
    var k = nk.shape[0]
    var gram = List[Float64](length=d * d, fill=0.0)
    for i in range(n):
        for f in range(d):
            var a = xv.at(X, i, f)
            for g in range(d):
                gram[f * d + g] += a * xv.at(X, i, g)
    var total = 0.0
    for j in range(k):
        total += Float64(nk.unsafe_get(j))
    total = rounded[dtype](total)
    for f in range(d):
        for g in range(d):
            var means2 = 0.0
            for j in range(k):
                means2 += rounded[dtype](Float64(nk.unsafe_get(j)) * Float64(means.unsafe_get(j * d + f))) * Float64(means.unsafe_get(j * d + g))
            var value = rounded[dtype](rounded[dtype](gram[f * d + g]) - rounded[dtype](means2))
            value = rounded[dtype](value / total)
            if f == g:
                value = rounded[dtype](value + reg_covar)
            out.unsafe_set(f * d + g, value.cast[dtype]())


def _covariances_diag[dtype: DType](X: NDArray[dtype], xv: Matrix[dtype], resp: NDArray[dtype], rv: Matrix[dtype], nk: NDArray[dtype], means: NDArray[dtype], reg_covar: Float64, kind: String, mut out: NDArray[dtype]) raises:
    # diag: resp.T @ (X * X) / nk[:, None] - means**2 + reg; spherical: row mean
    var n = xv.rows
    var d = xv.cols
    var k = rv.cols
    var acc = List[Float64](length=k * d, fill=0.0)
    for i in range(n):
        for j in range(k):
            var r = rv.at(resp, i, j)
            if r != 0.0:
                for f in range(d):
                    var x = xv.at(X, i, f)
                    acc[j * d + f] += r * rounded[dtype](x * x)
    for j in range(k):
        var count = Float64(nk.unsafe_get(j))
        var row_total = 0.0
        for f in range(d):
            var avg_x2 = rounded[dtype](rounded[dtype](acc[j * d + f]) / count)
            var mean = Float64(means.unsafe_get(j * d + f))
            var value = rounded[dtype](rounded[dtype](avg_x2 - rounded[dtype](mean * mean)) + reg_covar)
            if kind == "diag":
                out.unsafe_set(j * d + f, value.cast[dtype]())
            row_total += value
        if kind == "spherical":
            out.unsafe_set(j, rounded[dtype](rounded[dtype](row_total) / Float64(d)).cast[dtype]())


def estimate_log_gaussian_prob[dtype: DType](X: NDArray[dtype], means: NDArray[dtype], precisions_chol: NDArray[dtype], kind: String, log_det: List[Float64]) raises -> NDArray[dtype]:
    """(n, k) Gaussian log densities given precision Cholesky factors.

    `log_det` comes from `log_det_cholesky` (one value per component, a single
    value for tied). Result = -0.5 (d log(2 pi) + squared Mahalanobis) + log_det.
    """
    var xv = Matrix[dtype](X)
    var n = xv.rows
    var d = xv.cols
    var k = means.shape[0]
    if means.ndim != 2 or means.shape[1] != d:
        raise Error(String("ValueError: means shape does not match ", d, " features"))
    var mahalanobis = zeros[DType.float64](Shape(n, k))
    if kind == "full" or kind == "tied":
        _mahalanobis_triangular(X, xv, means, precisions_chol, kind, mahalanobis)
    elif kind == "diag":
        _mahalanobis_diag(X, xv, means, precisions_chol, mahalanobis)
    elif kind == "spherical":
        _mahalanobis_spherical(X, xv, means, precisions_chol, mahalanobis)
    else:
        raise Error(String("ValueError: unsupported covariance_type '", kind, "'"))
    var constant = Float64(d) * LOG_2PI
    var out = zeros[dtype](Shape(n, k))
    for i in range(n):
        for j in range(k):
            var det = log_det[0] if kind == "tied" else log_det[j]
            var value = rounded[dtype](-0.5 * rounded[dtype](constant + mahalanobis.unsafe_get(i * k + j)))
            out.unsafe_set(i * k + j, (value + det).cast[dtype]())
    return out^


def _mahalanobis_triangular[dtype: DType](X: NDArray[dtype], xv: Matrix[dtype], means: NDArray[dtype], chol: NDArray[dtype], kind: String, mut out: NDArray[DType.float64]) raises:
    var n = xv.rows
    var d = xv.cols
    var k = means.shape[0]
    var projected = List[Float64](length=d, fill=0.0)
    var mean_projected = List[Float64](length=k * d, fill=0.0)
    # mu @ U for every component (tied shares one factor).
    for j in range(k):
        var base = 0 if kind == "tied" else j * d * d
        for g in range(d):
            var total = 0.0
            for f in range(g + 1):
                total += Float64(means.unsafe_get(j * d + f)) * Float64(chol.unsafe_get(base + f * d + g))
            mean_projected[j * d + g] = rounded[dtype](total)
    if kind == "full":
        # y = X @ U - mu @ U; log_prob = sum(y**2)
        for j in range(k):
            var base = j * d * d
            for i in range(n):
                for g in range(d):
                    projected[g] = 0.0
                for f in range(d):
                    var x = xv.at(X, i, f)
                    # U is upper triangular: only columns g >= f contribute.
                    for g in range(f, d):
                        projected[g] += x * Float64(chol.unsafe_get(base + f * d + g))
                var total = 0.0
                for g in range(d):
                    var y = rounded[dtype](rounded[dtype](projected[g]) - mean_projected[j * d + g])
                    total += rounded[dtype](y * y)
                out.unsafe_set(i * k + j, rounded[dtype](total))
    else:
        # row_norms(mu_proj) - 2 (Xp @ mu_proj.T) + row_norms(Xp)[:, None]
        var mean_norms = List[Float64](length=k, fill=0.0)
        for j in range(k):
            var total = 0.0
            for g in range(d):
                total += mean_projected[j * d + g] * mean_projected[j * d + g]
            mean_norms[j] = rounded[dtype](total)
        for i in range(n):
            for g in range(d):
                projected[g] = 0.0
            for f in range(d):
                var x = xv.at(X, i, f)
                for g in range(f, d):
                    projected[g] += x * Float64(chol.unsafe_get(f * d + g))
            var norm = 0.0
            for g in range(d):
                projected[g] = rounded[dtype](projected[g])
                norm += projected[g] * projected[g]
            norm = rounded[dtype](norm)
            for j in range(k):
                var cross = 0.0
                for g in range(d):
                    cross += projected[g] * mean_projected[j * d + g]
                var value = rounded[dtype](mean_norms[j] - rounded[dtype](2.0 * rounded[dtype](cross)))
                out.unsafe_set(i * k + j, rounded[dtype](value + norm))


def _mahalanobis_diag[dtype: DType](X: NDArray[dtype], xv: Matrix[dtype], means: NDArray[dtype], chol: NDArray[dtype], mut out: NDArray[DType.float64]) raises:
    # sum(means**2 * prec, 1) - 2 X @ (means * prec).T + X**2 @ prec.T
    var n = xv.rows
    var d = xv.cols
    var k = means.shape[0]
    var precisions = List[Float64](length=k * d, fill=0.0)
    var weighted_means = List[Float64](length=k * d, fill=0.0)
    var bias = List[Float64](length=k, fill=0.0)
    for j in range(k):
        var total = 0.0
        for f in range(d):
            var c = Float64(chol.unsafe_get(j * d + f))
            var p = rounded[dtype](c * c)
            var m = Float64(means.unsafe_get(j * d + f))
            precisions[j * d + f] = p
            weighted_means[j * d + f] = rounded[dtype](m * p)
            total += rounded[dtype](rounded[dtype](m * m) * p)
        bias[j] = rounded[dtype](total)
    var squares = List[Float64](length=d, fill=0.0)
    for i in range(n):
        for f in range(d):
            var x = xv.at(X, i, f)
            squares[f] = rounded[dtype](x * x)
        for j in range(k):
            var cross = 0.0
            var quadratic = 0.0
            for f in range(d):
                cross += xv.at(X, i, f) * weighted_means[j * d + f]
                quadratic += squares[f] * precisions[j * d + f]
            var value = rounded[dtype](bias[j] - rounded[dtype](2.0 * rounded[dtype](cross)))
            out.unsafe_set(i * k + j, rounded[dtype](value + rounded[dtype](quadratic)))


def _mahalanobis_spherical[dtype: DType](X: NDArray[dtype], xv: Matrix[dtype], means: NDArray[dtype], chol: NDArray[dtype], mut out: NDArray[DType.float64]) raises:
    # sum(means**2, 1) * prec - 2 (X @ means.T * prec) + outer(row_norms(X), prec)
    var n = xv.rows
    var d = xv.cols
    var k = means.shape[0]
    var precisions = List[Float64](length=k, fill=0.0)
    var bias = List[Float64](length=k, fill=0.0)
    for j in range(k):
        var c = Float64(chol.unsafe_get(j))
        precisions[j] = rounded[dtype](c * c)
        var total = 0.0
        for f in range(d):
            var m = Float64(means.unsafe_get(j * d + f))
            total += rounded[dtype](m * m)
        bias[j] = rounded[dtype](rounded[dtype](total) * precisions[j])
    for i in range(n):
        var norm = 0.0
        for f in range(d):
            var x = xv.at(X, i, f)
            norm += x * x
        norm = rounded[dtype](norm)
        for j in range(k):
            var cross = 0.0
            for f in range(d):
                cross += xv.at(X, i, f) * Float64(means.unsafe_get(j * d + f))
            var scaled = rounded[dtype](rounded[dtype](cross) * precisions[j])
            var value = rounded[dtype](bias[j] - rounded[dtype](2.0 * scaled))
            out.unsafe_set(i * k + j, rounded[dtype](value + rounded[dtype](norm * precisions[j])))
