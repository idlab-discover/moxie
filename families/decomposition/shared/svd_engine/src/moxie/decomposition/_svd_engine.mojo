"""Native dense factorization kernels over shared NuMojo storage.

Randomized orchestration follows sklearn.utils.extmath (BSD-3-Clause;
see LICENSES/scikit-learn.txt). No Python or foreign factorization runtime.
"""
from numojo import NDArray, Shape, zeros, mean
from std.math import sqrt, abs, isfinite
from moxie._random import RandomState
from moxie.decomposition._contracts import validate_matrix, SVDResult, EigenResult


@always_inline
def _get[dtype: DType](a: NDArray[dtype], i: Int, j: Int) raises -> Float64:
    return Float64(a.unsafe_get(i * Int(a.strides[0]) + j * Int(a.strides[1])))


@always_inline
def _set[dtype: DType](mut a: NDArray[dtype], i: Int, j: Int, x: Float64) raises:
    a.unsafe_set(i * Int(a.strides[0]) + j * Int(a.strides[1]), SIMD[dtype, 1](x))


def _identity[dtype: DType](m: Int, n: Int) raises -> NDArray[dtype]:
    var q = zeros[dtype](Shape(m, n))
    for i in range(min(m, n)):
        _set(q, i, i, 1)
    return q^


def svd_flip[dtype: DType](mut u: NDArray[dtype], mut vt: NDArray[dtype], *, u_based_decision: Bool = True) raises:
    """Flip paired signs in place; first maximal absolute loading wins ties."""
    validate_matrix(u)
    validate_matrix(vt)
    if u.shape[1] != vt.shape[0]:
        raise Error("ValueError: incompatible SVD factors")
    for j in range(vt.shape[0]):
        var largest: Float64 = -1
        var chosen: Float64 = 0
        if u_based_decision:
            for i in range(u.shape[0]):
                var value = _get(u, i, j)
                if abs(value) > largest:
                    largest = abs(value)
                    chosen = value
        else:
            for i in range(vt.shape[1]):
                var value = _get(vt, j, i)
                if abs(value) > largest:
                    largest = abs(value)
                    chosen = value
        var sign: Float64 = 1.0 if chosen > 0 else (-1.0 if chosen < 0 else 0.0)
        for i in range(u.shape[0]):
            _set(u, i, j, sign * _get(u, i, j))
        for i in range(vt.shape[1]):
            _set(vt, j, i, sign * _get(vt, j, i))


def qr_basis[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    """Reduced Householder Q, including orthogonal completion at deficient rank."""
    validate_matrix(x)
    var m = x.shape[0]
    var n = x.shape[1]
    var k = min(m, n)
    var a = x.copy()
    var vectors = zeros[DType.float64](Shape(m, k))
    var active = zeros[DType.float64](Shape(k))
    for j in range(k):
        var scale: Float64 = 0
        for i in range(j, m):
            scale = max(scale, abs(_get(a, i, j)))
        if scale == 0:
            continue
        var norm: Float64 = 0
        for i in range(j, m):
            var v = _get(a, i, j) / scale
            _set(vectors, i, j, v)
            norm += v * v
        norm = sqrt(norm)
        var first = _get(vectors, j, j)
        _set(vectors, j, j, first + (norm if first >= 0 else -norm))
        var vnorm: Float64 = 0
        for i in range(j, m):
            vnorm += _get(vectors, i, j) ** 2
        vnorm = sqrt(vnorm)
        for i in range(j, m):
            _set(vectors, i, j, _get(vectors, i, j) / vnorm)
        active.unsafe_set(j, 1)
        for col in range(j, n):
            var dot: Float64 = 0
            for i in range(j, m):
                dot += _get(vectors, i, j) * _get(a, i, col)
            for i in range(j, m):
                _set(a, i, col, _get(a, i, col) - 2 * _get(vectors, i, j) * dot)
    var q = _identity[dtype](m, k)
    for rev in range(k):
        var j = k - 1 - rev
        if active.unsafe_get(j) == 0:
            continue
        for col in range(k):
            var dot: Float64 = 0
            for i in range(j, m):
                dot += _get(vectors, i, j) * _get(q, i, col)
            for i in range(j, m):
                _set(q, i, col, _get(q, i, col) - 2 * _get(vectors, i, j) * dot)
    _orthogonal(q)
    return q^


def _lu_basis[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    """Rectangular partial-pivot LU, returning P L as scipy permute_l=True."""
    validate_matrix(x)
    var m = x.shape[0]
    var n = x.shape[1]
    var k = min(m, n)
    var a = x.copy()
    var order = List[Int]()
    for i in range(m):
        order.append(i)
    for j in range(k):
        var pivot = j
        for i in range(j + 1, m):
            if abs(_get(a, i, j)) > abs(_get(a, pivot, j)):
                pivot = i
        if pivot != j:
            var oi = order[j]
            order[j] = order[pivot]
            order[pivot] = oi
            for col in range(n):
                var tmp = _get(a, j, col)
                _set(a, j, col, _get(a, pivot, col))
                _set(a, pivot, col, tmp)
        var diagonal = _get(a, j, j)
        if diagonal == 0:
            continue
        for i in range(j + 1, m):
            var multiplier = _get(a, i, j) / diagonal
            _set(a, i, j, multiplier)
            for col in range(j + 1, n):
                _set(a, i, col, _get(a, i, col) - multiplier * _get(a, j, col))
    var result = zeros[dtype](Shape(m, k))
    for i in range(m):
        for j in range(min(i + 1, k)):
            _set(result, order[i], j, 1 if i == j else _get(a, i, j))
    validate_matrix(result)
    return result^


def _complete_column(mut u: NDArray[DType.float64], j: Int) raises:
    var m = u.shape[0]
    for trial in range(m):
        for i in range(m):
            _set(u, i, j, 1.0 if i == trial else 0.0)
        for _ in range(2):
            for col in range(j):
                var dot: Float64 = 0
                for i in range(m):
                    dot += _get(u, i, col) * _get(u, i, j)
                for i in range(m):
                    _set(u, i, j, _get(u, i, j) - dot * _get(u, i, col))
        var norm: Float64 = 0
        for i in range(m):
            norm += _get(u, i, j) ** 2
        if norm > 1e-12:
            norm = sqrt(norm)
            for i in range(m):
                _set(u, i, j, _get(u, i, j) / norm)
            return
    raise Error("LinAlgError: null-space completion failed")


def _column_norm(a: NDArray[DType.float64], j: Int) raises -> Float64:
    var scale: Float64 = 0
    for i in range(a.shape[0]):
        scale = max(scale, abs(_get(a, i, j)))
    if scale == 0:
        return 0
    var ss: Float64 = 0
    for i in range(a.shape[0]):
        ss += (_get(a, i, j) / scale) ** 2
    return scale * sqrt(ss)


def _orthogonal[dtype: DType](q: NDArray[dtype]) raises:
    validate_matrix(q)
    var tolerance: Float64 = (2e-6 if dtype == DType.float32 else 2e-12) * Float64(max(q.shape[0], 1))
    for i in range(q.shape[1]):
        for j in range(i + 1):
            var dot: Float64 = 0
            for r in range(q.shape[0]):
                dot += _get(q, r, i) * _get(q, r, j)
            if abs(dot - (1.0 if i == j else 0.0)) > tolerance:
                raise Error("LinAlgError: orthogonality certificate failed")


def _certify_svd[dtype: DType](x: NDArray[dtype], result: SVDResult[dtype], *, exact: Bool, projected_on_left: Bool = True) raises:
    var k = result.singular_values.shape[0]
    if result.U.shape[0] != x.shape[0] or result.U.shape[1] != k or result.Vt.shape[0] != k or result.Vt.shape[1] != x.shape[1]:
        raise Error("LinAlgError: invalid SVD shapes")
    _orthogonal(result.U)
    var v = result.Vt.T()
    _orthogonal(v)
    var scale: Float64 = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            scale = max(scale, abs(_get(x, i, j)))
    for j in range(k):
        var sj = Float64(result.singular_values.unsafe_get(j))
        if not isfinite(sj) or sj < 0:
            raise Error("LinAlgError: invalid singular value")
        if j > 0 and sj > Float64(result.singular_values.unsafe_get(j - 1)):
            raise Error("LinAlgError: unordered singular values")
    if scale == 0:
        scale = 1
    var tolerance: Float64 = (2e-6 if dtype == DType.float32 else 2e-12) * Float64(x.shape[0] + x.shape[1])
    # The projected equation depends on which side the range finder sampled.
    if projected_on_left:
        for j in range(k):
            for col in range(x.shape[1]):
                var dot: Float64 = 0
                for i in range(x.shape[0]):
                    dot += _get(result.U, i, j) * (_get(x, i, col) / scale)
                var expected = (Float64(result.singular_values.unsafe_get(j)) / scale) * _get(result.Vt, j, col)
                if not isfinite(dot) or abs(dot - expected) > tolerance:
                    raise Error("LinAlgError: projected SVD residual certificate failed")
    else:
        for i in range(x.shape[0]):
            for j in range(k):
                var dot: Float64 = 0
                for col in range(x.shape[1]):
                    dot += (_get(x, i, col) / scale) * _get(result.Vt, j, col)
                var expected = _get(result.U, i, j) * (Float64(result.singular_values.unsafe_get(j)) / scale)
                if not isfinite(dot) or abs(dot - expected) > tolerance:
                    raise Error("LinAlgError: transposed SVD residual certificate failed")
    if exact:
        for i in range(x.shape[0]):
            for col in range(x.shape[1]):
                var value: Float64 = 0
                for j in range(k):
                    value += _get(result.U, i, j) * (Float64(result.singular_values.unsafe_get(j)) / scale) * _get(result.Vt, j, col)
                if abs(value - _get(x, i, col) / scale) > tolerance:
                    raise Error("LinAlgError: SVD reconstruction certificate failed")


def dense_svd[dtype: DType](x: NDArray[dtype], *, flip_sign: Bool = True, max_sweeps: Int = 100) raises -> SVDResult[dtype]:
    """Economy one-sided Jacobi SVD; failure to converge raises explicitly."""
    validate_matrix(x)
    if max_sweeps < 1:
        raise Error("ValueError: max_sweeps must be positive")
    if x.shape[0] < x.shape[1]:
        var xt = x.T()
        var t = dense_svd(xt, flip_sign=False, max_sweeps=max_sweeps)
        var u = t.Vt.T()
        var vt = t.U.T()
        if flip_sign:
            svd_flip(u, vt)
        return SVDResult[dtype](u^, t.singular_values.copy(), vt^)
    var m = x.shape[0]
    var n = x.shape[1]
    # Per-column norm scaling preserves tiny singular values even when the
    # input's dynamic range exceeds the ratio representable in Float64.
    var a = zeros[DType.float64](Shape(m, n))
    var v = _identity[DType.float64](n, n)
    for i in range(m):
        for j in range(n):
            _set(a, i, j, _get(x, i, j))
    var converged = False
    for _ in range(max_sweeps):
        var changed = False
        for p in range(n):
            for q in range(p + 1, n):
                var np = _column_norm(a, p)
                var nq = _column_norm(a, q)
                if not isfinite(np) or not isfinite(nq):
                    raise Error("LinAlgError: singular value exceeds Float64 workspace")
                if np == 0 or nq == 0:
                    continue
                var correlation: Float64 = 0
                for i in range(m):
                    correlation += (_get(a, i, p) / np) * (_get(a, i, q) / nq)
                if abs(correlation) <= 2e-15:
                    continue
                changed = True
                var ratio = nq / np
                var t: Float64
                if ratio < 1e-100:
                    t = -correlation * ratio
                elif ratio > 1e100:
                    t = correlation * (np / nq)
                else:
                    var tau = (ratio - 1 / ratio) / (2 * correlation)
                    t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + sqrt(1 + tau * tau))
                var c = 1 / sqrt(1 + t * t)
                var s = c * t
                for i in range(m):
                    var ap = _get(a, i, p)
                    var aq = _get(a, i, q)
                    _set(a, i, p, c * ap - s * aq)
                    _set(a, i, q, s * ap + c * aq)
                for i in range(n):
                    var vp = _get(v, i, p)
                    var vq = _get(v, i, q)
                    _set(v, i, p, c * vp - s * vq)
                    _set(v, i, q, s * vp + c * vq)
        if not changed:
            converged = True
            break
    if not converged:
        raise Error("LinAlgError: SVD did not converge")
    var norms = zeros[DType.float64](Shape(n))
    for j in range(n):
        norms.unsafe_set(j, _column_norm(a, j))
    for j in range(n):
        var best = j
        for col in range(j + 1, n):
            if norms.unsafe_get(col) > norms.unsafe_get(best):
                best = col
        var tmp = norms.unsafe_get(j)
        norms.unsafe_set(j, norms.unsafe_get(best))
        norms.unsafe_set(best, tmp)
        for i in range(m):
            tmp = _get(a, i, j)
            _set(a, i, j, _get(a, i, best))
            _set(a, i, best, tmp)
        for i in range(n):
            tmp = _get(v, i, j)
            _set(v, i, j, _get(v, i, best))
            _set(v, i, best, tmp)
    var u = zeros[dtype](Shape(m, n))
    var vt = zeros[dtype](Shape(n, n))
    var singular = zeros[dtype](Shape(n))
    for j in range(n):
        var norm = norms.unsafe_get(j)
        if norm == 0:
            _complete_column(a, j)
        else:
            for i in range(m):
                _set(a, i, j, _get(a, i, j) / norm)
        singular.unsafe_set(j, SIMD[dtype, 1](norm))
        if not isfinite(singular.unsafe_get(j)):
            raise Error("LinAlgError: singular value exceeds output dtype")
        for i in range(m):
            _set(u, i, j, _get(a, i, j))
        for i in range(n):
            _set(vt, j, i, _get(v, i, j))
    if flip_sign:
        svd_flip(u, vt)
    var result = SVDResult[dtype](u^, singular^, vt^)
    _certify_svd(x, result, exact=True)
    return result^


def _hypot(a: Float64, b: Float64) -> Float64:
    """Overflow-safe sqrt(a*a + b*b)."""
    var x = abs(a)
    var y = abs(b)
    if x > y:
        return x * sqrt(1 + (y / x) ** 2)
    if y == 0:
        return 0
    return y * sqrt(1 + (x / y) ** 2)


def _with_sign(magnitude: Float64, reference: Float64) -> Float64:
    return abs(magnitude) if reference >= 0 else -abs(magnitude)


def golub_kahan_svd[dtype: DType](x: NDArray[dtype], *, flip_sign: Bool = True, max_iterations: Int = 75) raises -> SVDResult[dtype]:
    """Economy SVD by Householder bidiagonalization and implicit-shift QR.

    This is the Golub-Kahan-Reinsch algorithm (Numer. Math. 14, 1970), the
    same method family as LAPACK's gesvd driver (bidiagonal QR iteration),
    as opposed to dense_svd's one-sided Jacobi. It backs the
    `svd_lapack_driver="gesvd"` selection. Arithmetic uses Float64 scratch;
    outputs keep the input dtype, singular values are descending and
    nonnegative, and zero singular directions receive orthogonal completion.
    max_iterations bounds the QR sweeps per singular value; exhausting it
    raises LinAlgError.
    """
    validate_matrix(x)
    if max_iterations < 1:
        raise Error("ValueError: max_iterations must be positive")
    if x.shape[0] < x.shape[1]:
        var t = golub_kahan_svd(x.T(), flip_sign=False, max_iterations=max_iterations)
        var u_wide = t.Vt.T()
        var vt_wide = t.U.T()
        if flip_sign:
            svd_flip(u_wide, vt_wide)
        return SVDResult[dtype](u_wide^, t.singular_values.copy(), vt_wide^)
    var m = x.shape[0]
    var n = x.shape[1]
    # Uniform scaling keeps squared Householder norms representable.
    var scale_all: Float64 = 0
    for i in range(m):
        for j in range(n):
            scale_all = max(scale_all, abs(_get(x, i, j)))
    if scale_all == 0:
        scale_all = 1
    var a = zeros[DType.float64](Shape(m, n))
    for i in range(m):
        for j in range(n):
            _set(a, i, j, _get(x, i, j) / scale_all)
    var w = zeros[DType.float64](Shape(n))
    var v = zeros[DType.float64](Shape(n, n))
    var rv1 = zeros[DType.float64](Shape(n))
    var g: Float64 = 0
    var scale: Float64 = 0
    var anorm: Float64 = 0
    var l = 0
    # Householder reduction to upper bidiagonal form.
    for i in range(n):
        l = i + 1
        rv1.unsafe_set(i, scale * g)
        g = 0
        scale = 0
        var s: Float64 = 0
        for k in range(i, m):
            scale += abs(_get(a, k, i))
        if scale != 0:
            for k in range(i, m):
                var value = _get(a, k, i) / scale
                _set(a, k, i, value)
                s += value * value
            var f = _get(a, i, i)
            g = -_with_sign(sqrt(s), f)
            var h = f * g - s
            _set(a, i, i, f - g)
            for j in range(l, n):
                var dot: Float64 = 0
                for k in range(i, m):
                    dot += _get(a, k, i) * _get(a, k, j)
                var factor = dot / h
                for k in range(i, m):
                    _set(a, k, j, _get(a, k, j) + factor * _get(a, k, i))
            for k in range(i, m):
                _set(a, k, i, _get(a, k, i) * scale)
        w.unsafe_set(i, scale * g)
        g = 0
        scale = 0
        s = 0
        if i != n - 1:
            for k in range(l, n):
                scale += abs(_get(a, i, k))
            if scale != 0:
                for k in range(l, n):
                    var value = _get(a, i, k) / scale
                    _set(a, i, k, value)
                    s += value * value
                var f = _get(a, i, l)
                g = -_with_sign(sqrt(s), f)
                var h = f * g - s
                _set(a, i, l, f - g)
                for k in range(l, n):
                    rv1.unsafe_set(k, _get(a, i, k) / h)
                for j in range(l, m):
                    var dot: Float64 = 0
                    for k in range(l, n):
                        dot += _get(a, j, k) * _get(a, i, k)
                    for k in range(l, n):
                        _set(a, j, k, _get(a, j, k) + dot * rv1.unsafe_get(k))
                for k in range(l, n):
                    _set(a, i, k, _get(a, i, k) * scale)
        anorm = max(anorm, abs(w.unsafe_get(i)) + abs(rv1.unsafe_get(i)))
    # Accumulate the right-hand transformations.
    for i in range(n - 1, -1, -1):
        if i < n - 1:
            if g != 0:
                for j in range(l, n):
                    _set(v, j, i, (_get(a, i, j) / _get(a, i, l)) / g)
                for j in range(l, n):
                    var dot: Float64 = 0
                    for k in range(l, n):
                        dot += _get(a, i, k) * _get(v, k, j)
                    for k in range(l, n):
                        _set(v, k, j, _get(v, k, j) + dot * _get(v, k, i))
            for j in range(l, n):
                _set(v, i, j, 0)
                _set(v, j, i, 0)
        _set(v, i, i, 1)
        g = rv1.unsafe_get(i)
        l = i
    # Accumulate the left-hand transformations.
    for i in range(n - 1, -1, -1):
        l = i + 1
        g = w.unsafe_get(i)
        for j in range(l, n):
            _set(a, i, j, 0)
        if g != 0:
            g = 1 / g
            for j in range(l, n):
                var dot: Float64 = 0
                for k in range(l, m):
                    dot += _get(a, k, i) * _get(a, k, j)
                var factor = (dot / _get(a, i, i)) * g
                for k in range(i, m):
                    _set(a, k, j, _get(a, k, j) + factor * _get(a, k, i))
            for j in range(i, m):
                _set(a, j, i, _get(a, j, i) * g)
        else:
            for j in range(i, m):
                _set(a, j, i, 0)
        _set(a, i, i, _get(a, i, i) + 1)
    # Diagonalize the bidiagonal form by implicit-shift QR sweeps.
    comptime epsilon = 2.220446049250313e-16
    for k in range(n - 1, -1, -1):
        var iteration = 0
        while True:
            var split = k
            var cancel = True
            while split >= 0:
                if split == 0 or abs(rv1.unsafe_get(split)) <= epsilon * anorm:
                    cancel = False
                    break
                if abs(w.unsafe_get(split - 1)) <= epsilon * anorm:
                    break
                split -= 1
            if cancel:
                # w[split-1] is negligible: zero rv1[split] by Givens rotations.
                var c: Float64 = 0
                var s: Float64 = 1
                var previous = split - 1
                for i in range(split, k + 1):
                    var f = s * rv1.unsafe_get(i)
                    rv1.unsafe_set(i, c * rv1.unsafe_get(i))
                    if abs(f) <= epsilon * anorm:
                        break
                    g = w.unsafe_get(i)
                    var h = _hypot(f, g)
                    w.unsafe_set(i, h)
                    h = 1 / h
                    c = g * h
                    s = -f * h
                    for j in range(m):
                        var y = _get(a, j, previous)
                        var z = _get(a, j, i)
                        _set(a, j, previous, y * c + z * s)
                        _set(a, j, i, z * c - y * s)
            var z = w.unsafe_get(k)
            if split == k:
                if z < 0:
                    w.unsafe_set(k, -z)
                    for j in range(n):
                        _set(v, j, k, -_get(v, j, k))
                break
            if iteration >= max_iterations:
                raise Error("LinAlgError: bidiagonal QR SVD did not converge")
            iteration += 1
            # Wilkinson-style shift from the trailing 2x2 block.
            var xs = w.unsafe_get(split)
            var last = k - 1
            var y = w.unsafe_get(last)
            g = rv1.unsafe_get(last)
            var h = rv1.unsafe_get(k)
            var f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y)
            g = _hypot(f, 1)
            f = ((xs - z) * (xs + z) + h * ((y / (f + _with_sign(g, f))) - h)) / xs
            var c: Float64 = 1
            var s: Float64 = 1
            for j in range(split, last + 1):
                var i = j + 1
                g = rv1.unsafe_get(i)
                y = w.unsafe_get(i)
                h = s * g
                g = c * g
                z = _hypot(f, h)
                rv1.unsafe_set(j, z)
                c = f / z
                s = h / z
                f = xs * c + g * s
                g = g * c - xs * s
                h = y * s
                y *= c
                for jj in range(n):
                    var vx = _get(v, jj, j)
                    var vz = _get(v, jj, i)
                    _set(v, jj, j, vx * c + vz * s)
                    _set(v, jj, i, vz * c - vx * s)
                z = _hypot(f, h)
                w.unsafe_set(j, z)
                if z != 0:
                    z = 1 / z
                    c = f * z
                    s = h * z
                f = c * g + s * y
                xs = c * y - s * g
                for jj in range(m):
                    var ay = _get(a, jj, j)
                    var az = _get(a, jj, i)
                    _set(a, jj, j, ay * c + az * s)
                    _set(a, jj, i, az * c - ay * s)
            rv1.unsafe_set(split, 0)
            rv1.unsafe_set(k, f)
            w.unsafe_set(k, xs)
    # Order descending; complete zero directions; publish in input dtype.
    var order = List[Int]()
    for j in range(n):
        order.append(j)
    for j in range(n):
        var best = j
        for i in range(j + 1, n):
            if w.unsafe_get(order[i]) > w.unsafe_get(order[best]):
                best = i
        var tmp = order[j]
        order[j] = order[best]
        order[best] = tmp
    var sorted_u = zeros[DType.float64](Shape(m, n))
    for j in range(n):
        for i in range(m):
            _set(sorted_u, i, j, _get(a, i, order[j]))
    for j in range(n):
        if w.unsafe_get(order[j]) == 0:
            _complete_column(sorted_u, j)
    var u = zeros[dtype](Shape(m, n))
    var vt = zeros[dtype](Shape(n, n))
    var singular = zeros[dtype](Shape(n))
    for j in range(n):
        var value = w.unsafe_get(order[j]) * scale_all
        singular.unsafe_set(j, SIMD[dtype, 1](value))
        if not isfinite(singular.unsafe_get(j)):
            raise Error("LinAlgError: singular value exceeds output dtype")
        for i in range(m):
            _set(u, i, j, _get(sorted_u, i, j))
        for i in range(n):
            _set(vt, j, i, _get(v, i, order[j]))
    if flip_sign:
        svd_flip(u, vt)
    var result = SVDResult[dtype](u^, singular^, vt^)
    _certify_svd(x, result, exact=True)
    return result^


def symmetric_eigh[dtype: DType](x: NDArray[dtype], *, max_sweeps: Int = 100) raises -> EigenResult[dtype]:
    """Symmetric cyclic Jacobi; signed eigenvalues in descending order."""
    validate_matrix(x)
    var n = x.shape[0]
    if n != x.shape[1] or max_sweeps < 1:
        raise Error("ValueError: expected square matrix and positive max_sweeps")
    var scale: Float64 = 0
    for i in range(n):
        for j in range(n):
            if _get(x, i, j) != _get(x, j, i):
                raise Error("ValueError: matrix must be symmetric")
            scale = max(scale, abs(_get(x, i, j)))
    var a = zeros[DType.float64](Shape(n, n))
    var v = _identity[DType.float64](n, n)
    if scale != 0:
        for i in range(n):
            for j in range(n):
                _set(a, i, j, _get(x, i, j))
    var converged = False
    for _ in range(max_sweeps):
        var changed = False
        for p in range(n):
            for q in range(p + 1, n):
                var apq = _get(a, p, q)
                if scale == 0 or abs(apq) / scale <= 2e-15:
                    continue
                changed = True
                var tau = (_get(a, q, q) / scale - _get(a, p, p) / scale) / (2 * (apq / scale))
                var t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + sqrt(1 + tau * tau))
                var c = 1 / sqrt(1 + t * t)
                var s = c * t
                _set(a, p, p, _get(a, p, p) - t * apq)
                _set(a, q, q, _get(a, q, q) + t * apq)
                _set(a, p, q, 0)
                _set(a, q, p, 0)
                for i in range(n):
                    if i != p and i != q:
                        var ip = _get(a, i, p)
                        var iq = _get(a, i, q)
                        _set(a, i, p, c * ip - s * iq)
                        _set(a, p, i, _get(a, i, p))
                        _set(a, i, q, s * ip + c * iq)
                        _set(a, q, i, _get(a, i, q))
                    var vp = _get(v, i, p)
                    var vq = _get(v, i, q)
                    _set(v, i, p, c * vp - s * vq)
                    _set(v, i, q, s * vp + c * vq)
        if not changed:
            converged = True
            break
    if not converged:
        raise Error("LinAlgError: eigensolver did not converge")
    var values = zeros[dtype](Shape(n))
    var vectors = zeros[dtype](Shape(n, n))
    var order = List[Int]()
    for i in range(n):
        order.append(i)
    for j in range(n):
        var best = j
        for i in range(j + 1, n):
            if _get(a, order[i], order[i]) > _get(a, order[best], order[best]):
                best = i
        var tmp = order[j]
        order[j] = order[best]
        order[best] = tmp
        var idx = order[j]
        var value = _get(a, idx, idx)
        values.unsafe_set(j, SIMD[dtype, 1](value))
        if not isfinite(values.unsafe_get(j)):
            raise Error("LinAlgError: eigenvalue exceeds output dtype")
        var pivot = 0
        for i in range(1, n):
            if abs(_get(v, i, idx)) > abs(_get(v, pivot, idx)):
                pivot = i
        var sign: Float64 = 1.0 if _get(v, pivot, idx) >= 0 else -1.0
        for i in range(n):
            _set(vectors, i, j, sign * _get(v, i, idx))
    _orthogonal(vectors)
    var divisor = scale if scale != 0 else 1.0
    var tolerance: Float64 = (2e-6 if dtype == DType.float32 else 2e-12) * Float64(n)
    for i in range(n):
        for j in range(n):
            var value: Float64 = 0
            for col in range(n):
                value += (_get(x, i, col) / divisor) * _get(vectors, col, j)
            if abs(value - _get(vectors, i, j) * (Float64(values.unsafe_get(j)) / divisor)) > tolerance:
                raise Error("LinAlgError: eigen residual certificate failed")
    return EigenResult[dtype](values^, vectors^)


def randomized_range_finder[dtype: DType](a: NDArray[dtype], mut random_state: RandomState, *, size: Int, n_iter: Int, power_iteration_normalizer: String = "auto") raises -> NDArray[dtype]:
    validate_matrix(a)
    if size < 1 or n_iter < 0:
        raise Error("ValueError: positive size and nonnegative n_iter required")
    var mode = power_iteration_normalizer.copy()
    if mode != "auto" and mode != "none" and mode != "QR" and mode != "LU":
        raise Error("ValueError: invalid power_iteration_normalizer")
    if mode == "auto":
        mode = "none" if n_iter <= 2 else "LU"
    var q = zeros[dtype](Shape(a.shape[1], size))
    for i in range(a.shape[1]):
        for j in range(size):
            _set(q, i, j, random_state.standard_normal())
    var at = a.T()
    for _ in range(n_iter):
        var y = a @ q
        if mode == "QR":
            y = qr_basis(y)
        elif mode == "LU":
            y = _lu_basis(y)
        q = at @ y
        if mode == "QR":
            q = qr_basis(q)
        elif mode == "LU":
            q = _lu_basis(q)
        validate_matrix(q)
    var sampled = a @ q
    return qr_basis(sampled)


def randomized_svd[dtype: DType](x: NDArray[dtype], n_components: Int, mut random_state: RandomState, *, n_oversamples: Int = 10, n_iter: Int = -1, power_iteration_normalizer: String = "auto", transpose: String = "auto", flip_sign: Bool = True, svd_lapack_driver: String = "gesdd") raises -> SVDResult[dtype]:
    """Seeded randomized SVD. n_iter=-1 represents auto; RNG is borrowed."""
    validate_matrix(x)
    if n_components < 1 or n_oversamples < 0 or n_iter < -1:
        raise Error("ValueError: invalid rank, oversampling or iteration count")
    if n_components > 9223372036854775807 - n_oversamples:
        raise Error("ValueError: workspace rank overflow")
    if transpose != "auto" and transpose != "true" and transpose != "false":
        raise Error("ValueError: invalid transpose")
    if svd_lapack_driver != "gesdd" and svd_lapack_driver != "gesvd":
        raise Error("ValueError: invalid svd_lapack_driver")
    var transposed = transpose == "true" or (transpose == "auto" and x.shape[0] < x.shape[1])
    var a = x.T() if transposed else x.copy()
    var iterations = n_iter
    if iterations == -1:
        iterations = 7 if Float64(n_components) < 0.1 * Float64(min(x.shape[0], x.shape[1])) else 4
    var q = randomized_range_finder(a, random_state, size=n_components + n_oversamples, n_iter=iterations, power_iteration_normalizer=power_iteration_normalizer)
    var qt = q.T()
    var b = qt @ a
    # gesdd -> one-sided Jacobi, gesvd -> Golub-Kahan bidiagonal QR (see api.md).
    var small = golub_kahan_svd(b, flip_sign=False) if svd_lapack_driver == "gesvd" else dense_svd(b, flip_sign=False)
    var lifted = q @ small.U
    var k = min(n_components, small.singular_values.shape[0])
    var u = zeros[dtype](Shape(x.shape[0], k))
    var vt = zeros[dtype](Shape(k, x.shape[1]))
    var singular = zeros[dtype](Shape(k))
    for j in range(k):
        singular.unsafe_set(j, small.singular_values.unsafe_get(j))
        for i in range(x.shape[0]):
            _set(u, i, j, _get(small.Vt, j, i) if transposed else _get(lifted, i, j))
        for i in range(x.shape[1]):
            _set(vt, j, i, _get(lifted, i, j) if transposed else _get(small.Vt, j, i))
    if flip_sign:
        svd_flip(u, vt)
    var result = SVDResult[dtype](u^, singular^, vt^)
    _certify_svd(x, result, exact=False, projected_on_left=not transposed)
    return result^


def randomized_svd[dtype: DType](x: NDArray[dtype], n_components: Int, *, random_state: Int, n_oversamples: Int = 10, n_iter: Int = -1, power_iteration_normalizer: String = "auto", transpose: String = "auto", flip_sign: Bool = True, svd_lapack_driver: String = "gesdd") raises -> SVDResult[dtype]:
    var rng = RandomState(random_state)
    return randomized_svd(x, n_components, rng, n_oversamples=n_oversamples, n_iter=n_iter, power_iteration_normalizer=power_iteration_normalizer, transpose=transpose, flip_sign=flip_sign, svd_lapack_driver=svd_lapack_driver)


def truncated_svd[dtype: DType](x: NDArray[dtype], n_components: Int) raises -> SVDResult[dtype]:
    """Deterministic leading components via economy SVD, without random draws."""
    validate_matrix(x)
    if n_components < 1 or n_components > min(x.shape[0], x.shape[1]):
        raise Error("ValueError: invalid truncated rank")
    var full = dense_svd(x)
    var u = zeros[dtype](Shape(x.shape[0], n_components))
    var vt = zeros[dtype](Shape(n_components, x.shape[1]))
    var singular = zeros[dtype](Shape(n_components))
    for j in range(n_components):
        singular.unsafe_set(j, full.singular_values.unsafe_get(j))
        for i in range(x.shape[0]):
            _set(u, i, j, _get(full.U, i, j))
        for i in range(x.shape[1]):
            _set(vt, j, i, _get(full.Vt, j, i))
    return SVDResult[dtype](u^, singular^, vt^)


def center_matrix[dtype: DType](x: NDArray[dtype]) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """Owned centered data and rank-one column means; input stays borrowed."""
    validate_matrix(x)
    var averages = mean[returned_dtype=dtype](x, axis=0)
    var centered = x - averages
    validate_matrix(centered)
    return centered^, averages^
