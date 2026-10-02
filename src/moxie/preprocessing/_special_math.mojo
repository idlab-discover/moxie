# Ports of SciPy's Cephes `ndtr`/`ndtri`/`erf`/`erfc`, SciPy's `boxcox`/
# `inv_boxcox`/`logsumexp`, and NumPy's pairwise summation, linear-interpolation
# search (`np.interp`) and linear percentile. SciPy: Copyright (c) 2001-2002
# Enthought, Inc. 2003, SciPy Developers (BSD-3-Clause, LICENSES/scipy.txt; the
# Cephes routines by Stephen L. Moshier are distributed with SciPy under it).
# NumPy: Copyright (c) 2005-2025, NumPy Developers (BSD-3-Clause,
# LICENSES/numpy.txt).
# SPDX-License-Identifier: BSD-3-Clause
"""Scalar special functions and NumPy-order reductions for the preprocessing family.

Accuracy contract (checked by tests/parity/test_quantiles.py against the
locked NumPy 2.5.3 / SciPy 1.18.1):

* `flog`, `fexp`, `fpow` lower to the LLVM `log`/`exp`/`pow` intrinsics (the
  platform math runtime); the pinned `std.math.log/exp` polynomials are off by
  up to ~1e-7 relative and are not used. `std.math.expm1` is libm-exact.
* `flog1p` is Kahan's `log(u) * x / (u - 1)` construction on `flog` (<= 2 ulp
  from libm `log1p`; `std.math.log1p` is off by ~1e-7 and is not used).
* `ndtr`, `ndtri` are the Cephes routines SciPy calls for `norm.cdf`/`norm.ppf`
  (bit-identical on the tested grid); `std.math.erf/erfc` are not used.
* `pairwise_sum` (and `pairwise_sum_in` for Float32) reproduce NumPy's
  `add.reduce` order exactly.
"""
from std.math import sqrt, isnan, isinf, abs, inf, nan, floor, expm1
from std.sys import llvm_intrinsic
from numojo import NDArray, Shape, NDArrayStrides

comptime F64_EPS = 2.220446049250313e-16
comptime F32_EPS = 1.1920928955078125e-07


@always_inline
def flog(x: Float64) -> Float64:
    """Natural logarithm via the platform math runtime (libm quality)."""
    return llvm_intrinsic["llvm.log", Float64, has_side_effect=False](x)


@always_inline
def fexp(x: Float64) -> Float64:
    return llvm_intrinsic["llvm.exp", Float64, has_side_effect=False](x)


@always_inline
def fpow(x: Float64, y: Float64) -> Float64:
    return llvm_intrinsic["llvm.pow", Float64, has_side_effect=False](x, y)


@always_inline
def fexpm1(x: Float64) -> Float64:
    return expm1(x)


def flog1p(x: Float64) -> Float64:
    """log(1 + x) accurate near zero (Kahan); NaN/inf/-1 follow libm."""
    if isinf(x):
        return x if x > 0 else nan[DType.float64]()
    var u = 1.0 + x
    if u == 1.0:
        return x
    return flog(u) * (x / (u - 1.0))


@always_inline
def sign(x: Float64) -> Float64:
    """np.sign: -1, 0 or 1; NaN propagates."""
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return x


# ---------------------------------------------------------------------------
# Cephes normal distribution (SciPy xsf/cephes ndtr.c, ndtri.c)


@always_inline
def _polevl[n: Int](x: Float64, c: InlineArray[Float64, n]) -> Float64:
    var a = c[0]
    comptime for k in range(1, n):
        a = a * x + c[k]
    return a


@always_inline
def _p1evl[n: Int](x: Float64, c: InlineArray[Float64, n]) -> Float64:
    var a = x + c[0]
    comptime for k in range(1, n):
        a = a * x + c[k]
    return a


comptime _MAXLOG = 7.09782712893383996843e2
comptime _SQRT1_2 = 0.70710678118654752440


def _erf(x: Float64) -> Float64:
    if x < 0.0:
        return -_erf(-x)
    if abs(x) > 1.0:
        return 1.0 - _erfc(x)
    var T: InlineArray[Float64, 5] = [9.60497373987051638749e0, 9.00260197203842689217e1, 2.23200534594684319226e3, 7.00332514112805075473e3, 5.55923013010394962768e4]
    var U: InlineArray[Float64, 5] = [3.35617141647503099647e1, 5.21357949780152679795e2, 4.59432382970980127987e3, 2.26290000613890934246e4, 4.92673942608635921086e4]
    var z = x * x
    return x * _polevl(z, T) / _p1evl(z, U)


def _erfc(a: Float64) -> Float64:
    var x = -a if a < 0.0 else a
    if x < 1.0:
        return 1.0 - _erf(a)
    var z = -a * a
    if z < -_MAXLOG:
        return 2.0 if a < 0 else 0.0
    z = fexp(z)
    var p: Float64
    var q: Float64
    if x < 8.0:
        var P: InlineArray[Float64, 9] = [2.46196981473530512524e-10, 5.64189564831068821977e-1, 7.46321056442269912687e0, 4.86371970985681366614e1, 1.96520832956077098242e2, 5.26445194995477358631e2, 9.34528527171957607540e2, 1.02755188689515710272e3, 5.57535335369399327526e2]
        var Q: InlineArray[Float64, 8] = [1.32281951154744992508e1, 8.67072140885989742329e1, 3.54937778887819891062e2, 9.75708501743205489753e2, 1.82390916687909736289e3, 2.24633760818710981792e3, 1.65666309194161350182e3, 5.57535340817727675546e2]
        p = _polevl(x, P)
        q = _p1evl(x, Q)
    else:
        var R: InlineArray[Float64, 6] = [5.64189583547755073984e-1, 1.27536670759978104416e0, 5.01905042251180477414e0, 6.16021097993053585195e0, 7.40974269950448939160e0, 2.97886665372100240670e0]
        var S: InlineArray[Float64, 6] = [2.26052863220117276590e0, 9.39603524938001434673e0, 1.20489539808096656605e1, 1.70814450747565897222e1, 9.60896809063285878198e0, 3.36907645100081516050e0]
        p = _polevl(x, R)
        q = _p1evl(x, S)
    var y = (z * p) / q
    if a < 0:
        y = 2.0 - y
    if y == 0.0:
        return 2.0 if a < 0 else 0.0
    return y


def ndtr(a: Float64) -> Float64:
    """Standard normal CDF (SciPy `special.ndtr`, i.e. `stats.norm.cdf`)."""
    if isnan(a):
        return a
    var x = a * _SQRT1_2
    var z = abs(x)
    if z < _SQRT1_2:
        return 0.5 + 0.5 * _erf(x)
    var y = 0.5 * _erfc(z)
    if x > 0:
        y = 1.0 - y
    return y


def ndtri(y0: Float64) -> Float64:
    """Standard normal quantile (SciPy `special.ndtri`, i.e. `stats.norm.ppf`)."""
    if y0 == 0.0:
        return -inf[DType.float64]()
    if y0 == 1.0:
        return inf[DType.float64]()
    if not (y0 > 0.0 and y0 < 1.0):
        return nan[DType.float64]()
    var code = True
    var y = y0
    if y > (1.0 - 0.13533528323661269189):
        y = 1.0 - y
        code = False
    if y > 0.13533528323661269189:
        var P0: InlineArray[Float64, 5] = [-5.99633501014107895267e1, 9.80010754185999661536e1, -5.66762857469070293439e1, 1.39312609387279679503e1, -1.23916583867381258016e0]
        var Q0: InlineArray[Float64, 8] = [1.95448858338141759834e0, 4.67627912898881538453e0, 8.63602421390890590575e1, -2.25462687854119370527e2, 2.00260212380060660359e2, -8.20372256168333339912e1, 1.59056225126211695515e1, -1.18331621121330003142e0]
        y = y - 0.5
        var y2 = y * y
        var x = y + y * (y2 * _polevl(y2, P0) / _p1evl(y2, Q0))
        return x * 2.50662827463100050242e0
    var x = sqrt(-2.0 * flog(y))
    var x0 = x - flog(x) / x
    var z = 1.0 / x
    var x1: Float64
    if x < 8.0:
        var P1: InlineArray[Float64, 9] = [4.05544892305962419923e0, 3.15251094599893866154e1, 5.71628192246421288162e1, 4.40805073893200834700e1, 1.46849561928858024014e1, 2.18663306850790267539e0, -1.40256079171354495875e-1, -3.50424626827848203418e-2, -8.57456785154685413611e-4]
        var Q1: InlineArray[Float64, 8] = [1.57799883256466749731e1, 4.53907635128879210584e1, 4.13172038254672030440e1, 1.50425385692907503408e1, 2.50464946208309415979e0, -1.42182922854787788574e-1, -3.80806407691578277194e-2, -9.33259480895457427372e-4]
        x1 = z * _polevl(z, P1) / _p1evl(z, Q1)
    else:
        var P2: InlineArray[Float64, 9] = [3.23774891776946035970e0, 6.91522889068984211695e0, 3.93881025292474443415e0, 1.33303460815807542389e0, 2.01485389549179081538e-1, 1.23716634817820021358e-2, 3.01581553508235416007e-4, 2.65806974686737550832e-6, 6.23974539184983293730e-9]
        var Q2: InlineArray[Float64, 8] = [6.02427039364742014255e0, 3.67983563856160859403e0, 1.37702099489081330271e0, 2.16236993594496635890e-1, 1.34204006088543189037e-2, 3.28014464682127739104e-4, 2.89247864745380683936e-6, 6.79019408009981274425e-9]
        x1 = z * _polevl(z, P2) / _p1evl(z, Q2)
    x = x0 - x1
    if code:
        x = -x
    return x


# ---------------------------------------------------------------------------
# Box-Cox (SciPy special `boxcox`, `inv_boxcox`; Float32 loops widen to Float64)


def boxcox(x: Float64, lmbda: Float64) -> Float64:
    var lx = flog(x)
    if abs(lmbda) < 1e-19:
        return lx
    if lmbda * lx < 709.78:
        return fexpm1(lmbda * lx) / lmbda
    return sign(lmbda) * fexp(lmbda * lx - flog(abs(lmbda))) - 1.0 / lmbda


def inv_boxcox(y: Float64, lmbda: Float64) -> Float64:
    if lmbda == 0.0:
        return fexp(y)
    var t = lmbda * y
    if isinf(t):
        # Overflowing lambda * y: SciPy's log-space branch.
        return fexp((flog(sign(lmbda) * (y - 1.0 / lmbda)) + flog(abs(lmbda))) / lmbda)
    return fexp(flog1p(t) / lmbda)


def lambertw_m1(z: Float64) raises -> Float64:
    """Real branch k=-1 of the Lambert W function for z in [-1/e, 0).

    Newton iteration on w + log(-w) = log(-z) from the asymptotic start
    log(-z) - log(-log(-z)); used only for SciPy's Box-Cox overflow guard.
    """
    if not (z < 0.0 and z >= -0.36787944117144233):
        raise Error("ValueError: lambertw(k=-1) requires -1/e <= z < 0")
    var target = flog(-z)
    var w: Float64
    if z < -0.3:
        # Branch-point series: w ~ -1 - sqrt(2 (1 + e z)).
        w = -1.0 - sqrt(2.0 * (1.0 + 2.718281828459045 * z))
    else:
        w = target - flog(-target)
    if not (w < -1.0):
        w = -1.0 - 1e-8
    for _ in range(100):
        var f = w + flog(-w) - target
        var step = f / (1.0 + 1.0 / w)
        var next = w - step
        if next >= -1.0:
            next = 0.5 * (w - 1.0)
        if abs(next - w) <= 4 * F64_EPS * abs(w):
            return next
        w = next
    return w


# ---------------------------------------------------------------------------
# NumPy-order reductions over contiguous Float64 workspaces


def pairwise_sum(values: NDArray[DType.float64], start: Int, count: Int) -> Float64:
    """`np.sum` of a contiguous float64 run: identity 0 plus NumPy's pairwise blocks."""
    return 0.0 + _pairwise(values, start, count)


def pairwise_sum_in[T: DType](values: NDArray[T], start: Int, count: Int) -> Scalar[T]:
    """NumPy's pairwise `add.reduce` order for a contiguous Float32/Float64 run,
    accumulated in T (the input precision, as NumPy reduces Float32 arrays)."""
    return _pairwise(values, start, count)


def _pairwise[T: DType](values: NDArray[T], start: Int, n: Int) -> Scalar[T]:
    if n < 8:
        var result = Scalar[T](-0.0)
        for i in range(n):
            result += values.unsafe_get(start + i)
        return result
    if n <= 128:
        var r = InlineArray[Scalar[T], 8](fill=0)
        for j in range(8):
            r[j] = values.unsafe_get(start + j)
        var i = 8
        var tail = n - n % 8
        while i < tail:
            for j in range(8):
                r[j] += values.unsafe_get(start + i + j)
            i += 8
        var result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n:
            result += values.unsafe_get(start + i)
            i += 1
        return result
    var half = n // 2
    half -= half % 8
    return _pairwise(values, start, half) + _pairwise(values, start + half, n - half)


def variance(values: NDArray[DType.float64], count: Int, mut scratch: NDArray[DType.float64]) -> Float64:
    """`np.var` (ddof=0) of the first `count` values; `scratch` holds the squares."""
    var mean = pairwise_sum(values, 0, count) / Float64(count)
    for i in range(count):
        var d = values.unsafe_get(i) - mean
        scratch.unsafe_set(i, d * d)
    return pairwise_sum(scratch, 0, count) / Float64(count)


def logsumexp(values: NDArray[DType.float64], count: Int, mut scratch: NDArray[DType.float64]) -> Float64:
    """`scipy.special.logsumexp` of the first `count` real values (no weights).

    Maximal entries are removed from the shifted sum and counted in `m`, as in
    SciPy; a non-finite result falls back to `log(sum(exp(a)))`.
    """
    var a_max = -inf[DType.float64]()
    var any_nan = False
    for i in range(count):
        var v = values.unsafe_get(i)
        if isnan(v):
            any_nan = True
        elif v > a_max:
            a_max = v
    if any_nan:
        a_max = nan[DType.float64]()
    var m = Float64(0)
    for i in range(count):
        var v = values.unsafe_get(i)
        if v == a_max:
            m += 1
            scratch.unsafe_set(i, 0.0)  # exp(-inf - a_max)
        else:
            scratch.unsafe_set(i, fexp(v - a_max))
    var s = pairwise_sum(scratch, 0, count)
    if s != 0:
        s = s / m
    var out = flog1p(s) + flog(m) + a_max
    if isnan(out) or isinf(out):
        for i in range(count):
            scratch.unsafe_set(i, fexp(values.unsafe_get(i)))
        out = flog(pairwise_sum(scratch, 0, count))
    return out


def logdiffexp(x: Float64, y: Float64) -> Float64:
    """SciPy `logsumexp([x, y], b=[1, -1], return_sign=True)[0]`: log|e^x - e^y|."""
    if isnan(x) or isnan(y):
        return nan[DType.float64]()
    var a_max = x if x > y else y
    var out: Float64
    if x == y:
        out = -inf[DType.float64]()
    elif x > y:
        # m = 1; s = -exp(y - x) (the max term is removed from the sum).
        var s = Float64(0.0) + (-fexp(y - x))
        if s < -1:
            s = -s - 2
        out = flog1p(s) + flog(1.0) + a_max
    else:
        # m = -1; s = exp(x - y) / -1; the sign is carried separately.
        var s = (fexp(x - y) + Float64(-0.0)) / -1.0
        if s < -1:
            s = -s - 2
        out = flog1p(s) + flog(1.0) + a_max
    if isnan(out) or isinf(out):
        out = flog(abs(fexp(x) - fexp(y)))
    return out


# ---------------------------------------------------------------------------
# Sorting and NumPy linear percentile on a Float64 workspace


def _sift(mut column: NDArray[DType.float64], start: Int, stop: Int):
    var root = start
    while root < stop // 2:
        var child = 2 * root + 1
        if child + 1 < stop and column.unsafe_get(child) < column.unsafe_get(child + 1):
            child += 1
        if column.unsafe_get(root) >= column.unsafe_get(child):
            return
        var tmp = column.unsafe_get(root)
        column.unsafe_set(root, column.unsafe_get(child))
        column.unsafe_set(child, tmp)
        root = child


def sort_prefix(mut column: NDArray[DType.float64], size: Int):
    """In-place ascending heap sort of the first `size` (non-NaN) values."""
    for k in range(size // 2):
        _sift(column, size // 2 - k - 1, size)
    for k in range(size - 1):
        var end = size - k - 1
        var tmp = column.unsafe_get(0)
        column.unsafe_set(0, column.unsafe_get(end))
        column.unsafe_set(end, tmp)
        _sift(column, 0, end)


def linear_percentile[dtype: DType](sorted: NDArray[DType.float64], size: Int, q: Float64) -> Float64:
    """NumPy `percentile(method="linear")` at fraction `q` of `size` sorted values.

    The neighbour difference is taken in the input dtype and interpolated in
    Float64 with NumPy's two-sided `_lerp`; no values means NaN.
    """
    if size == 0:
        return nan[DType.float64]()
    var position = Float64(size - 1) * q
    var lower = Int(floor(position))
    var upper = lower + 1
    var gamma = position - Float64(lower)
    if position >= Float64(size - 1):
        lower = size - 1
        upper = size - 1
        gamma = position + 1.0  # NumPy subtracts the -1 sentinel index
    var a = sorted.unsafe_get(lower)
    var b = sorted.unsafe_get(upper)
    var diff = Float64(b.cast[dtype]() - a.cast[dtype]())
    if gamma >= 0.5:
        return b - diff * (1 - gamma)
    return a + diff * gamma


# ---------------------------------------------------------------------------
# np.interp (NumPy `arr_interp` with `binary_search_with_guess`)


struct InterpTable[reverse: Bool](Movable):
    """Borrowed view of one fitted column for `np.interp`.

    `reverse=False` interpolates (x, quantiles -> references); `reverse=True`
    exposes NumPy's descending trick `(-quantiles[::-1], -references[::-1])`
    without materializing the negated copies. Points are row `k` of `xp`/`fp`
    columns `column`, with `n` rows.
    """
    var xp: NDArray[DType.float64]
    var fp: NDArray[DType.float64]
    var xp_col: Int
    var fp_col: Int
    var n: Int
    var guess: Int

    def __init__(out self, xp: NDArray[DType.float64], xp_col: Int, fp: NDArray[DType.float64], fp_col: Int) raises:
        self.xp = _alias(xp)
        self.fp = _alias(fp)
        self.xp_col = xp_col
        self.fp_col = fp_col
        self.n = xp.shape[0]
        self.guess = 0

    @always_inline
    def x_at(self, k: Int) -> Float64:
        comptime if Self.reverse:
            return -_cell(self.xp, self.n - 1 - k, self.xp_col)
        else:
            return _cell(self.xp, k, self.xp_col)

    @always_inline
    def y_at(self, k: Int) -> Float64:
        comptime if Self.reverse:
            return -_cell(self.fp, self.n - 1 - k, self.fp_col)
        else:
            return _cell(self.fp, k, self.fp_col)

    def _search(self, key: Float64, guess_in: Int) -> Int:
        var length = self.n
        var imin = 0
        var imax = length
        if key > self.x_at(length - 1):
            return length
        elif key < self.x_at(0):
            return -1
        if length <= 4:
            var i = 1
            while i < length and key >= self.x_at(i):
                i += 1
            return i - 1
        var guess = guess_in
        if guess > length - 3:
            guess = length - 3
        if guess < 1:
            guess = 1
        if key < self.x_at(guess):
            if key < self.x_at(guess - 1):
                imax = guess - 1
                if guess > 8 and key >= self.x_at(guess - 8):
                    imin = guess - 8
            else:
                return guess - 1
        else:
            if key < self.x_at(guess + 1):
                return guess
            elif key < self.x_at(guess + 2):
                return guess + 1
            else:
                imin = guess + 2
                if guess < length - 8 - 1 and key < self.x_at(guess + 8):
                    imax = guess + 8
        while imin < imax:
            var imid = imin + ((imax - imin) >> 1)
            if key >= self.x_at(imid):
                imin = imid + 1
            else:
                imax = imid
        return imin - 1

    def __call__(mut self, x: Float64) -> Float64:
        """One `np.interp` evaluation; carries NumPy's search guess between calls."""
        var n = self.n
        var left = self.y_at(0)
        var right = self.y_at(n - 1)
        if n == 1:
            var xp0 = self.x_at(0)
            return left if x < xp0 else (right if x > xp0 else left)
        if isnan(x):
            return x
        var j = self._search(x, self.guess)
        self.guess = j
        if j == -1:
            return left
        if j == n:
            return right
        if j == n - 1:
            return self.y_at(j)
        var xj = self.x_at(j)
        if xj == x:
            return self.y_at(j)
        var yj = self.y_at(j)
        var yj1 = self.y_at(j + 1)
        var xj1 = self.x_at(j + 1)
        var slope = (yj1 - yj) / (xj1 - xj)
        var result = slope * (x - xj) + yj
        if isnan(result):
            result = slope * (x - xj1) + yj1
            if isnan(result) and yj == yj1:
                result = yj
        return result


@always_inline
def _cell(x: NDArray[DType.float64], i: Int, j: Int) -> Float64:
    if x.ndim == 1:
        return x.unsafe_get(i * Int(x.strides.unsafe_load(0)))
    return x.unsafe_get(i * Int(x.strides.unsafe_load(0)) + j * Int(x.strides.unsafe_load(1)))


def _alias(x: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
    """Read-only shared view with x's rank-1/rank-2 layout; no copy."""
    if x.ndim == 1:
        return x.view_with_layout(Shape(x.shape[0]), NDArrayStrides(Int(x.strides[0])), x.offset)
    return x.view_with_layout(Shape(x.shape[0], x.shape[1]), NDArrayStrides(Int(x.strides[0]), Int(x.strides[1])), x.offset)
