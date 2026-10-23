"""Shared NumPy RandomState compatibility and explicitly native Philox streams.

MT19937, legacy distribution and sklearn sampling adaptations: see LICENSES.
"""
from std.math import sqrt, floor, abs, copysign, isfinite
from std.sys import llvm_intrinsic, stderr
from std.random import Random as Philox
from std.time import perf_counter_ns
from std.collections import Set
from std.atomic import Atomic
from std.ffi import _Global
from std.memory import Pointer
from std.utils import Variant
import numojo as nm
from numojo import NDArray, Shape, Item
from mojo_linalg import svd, NumericalError


def our_rand_r(mut state: UInt32) -> UInt32:
    """Advance sklearn's XorShift32 state; return its low 31 bits."""
    # Adapted from sklearn/utils/_random.pxd (BSD-3-Clause; see LICENSES).
    # Keep the full UInt32 state: only the returned value is reduced.
    if state == 0:
        state = 1
    state ^= state << 13
    state ^= state >> 17
    state ^= state << 5
    return state & UInt32(0x7fffffff)


struct NativeRandom(Copyable, Movable):
    """Isolated stdlib Philox; reproducible but NOT NumPy compatible."""
    var _generator: Philox[10]

    def __init__(out self, seed: UInt64):
        self._generator = Philox(seed=seed)

    def step(mut self) -> SIMD[DType.uint32, 4]:
        return self._generator.step()


struct RandomState(Copyable, Movable):
    """Integer-seeded MT19937 with independent deep-copy state."""
    var _key: List[UInt32]
    var _pos: Int
    var _has_gauss: Bool
    var _gauss: Float64

    def __init__(out self, seed: Int) raises:
        if seed < 0 or seed > 4294967295:
            raise Error("ValueError: seed must be between 0 and 2**32 - 1")
        self._key = List[UInt32]()
        var word = UInt32(seed)
        for i in range(624):
            self._key.append(word)
            word = UInt32(1812433253) * (word ^ (word >> 30)) + UInt32(i + 1)
        self._pos = 624
        self._has_gauss = False
        self._gauss = 0

    def next_uint32(mut self) -> UInt32:
        if self._pos == 624:
            for i in range(624):
                var y = (self._key[i] & 0x80000000) | (self._key[(i + 1) % 624] & 0x7fffffff)
                var term = UInt32(0)
                if (y & 1) != 0:
                    term = 0x9908b0df
                self._key[i] = self._key[(i + 397) % 624] ^ (y >> 1) ^ term
            self._pos = 0
        var y = self._key[self._pos]
        self._pos += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= y >> 18
        return y

    def random_sample(mut self) -> Float64:
        var a = self.next_uint32() >> 5
        var b = self.next_uint32() >> 6
        return (Float64(a) * 67108864.0 + Float64(b)) / 9007199254740992.0

    def _interval(mut self, maximum: UInt64) -> UInt64:
        if maximum == 0:
            return 0
        var mask = maximum
        mask |= mask >> 1
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16
        mask |= mask >> 32
        while True:
            var value = UInt64(self.next_uint32())
            if maximum > 4294967295:
                value = (value << 32) | UInt64(self.next_uint32())
            value &= mask
            if value <= maximum:
                return value

    def randint(mut self, low: Int, high: Int) raises -> Int:
        """Signed native Int result, exclusive high; NumPy int64 draw protocol."""
        if high <= low:
            raise Error("ValueError: low must be less than high")
        var width = UInt64(high) - UInt64(low)
        return Int(UInt64(low) + self._interval(width - 1))

    def randint(mut self, high: Int) raises -> Int:
        return self.randint(0, high)

    def uniform(mut self, low: Float64 = 0, high: Float64 = 1) raises -> Float64:
        var width = high - low
        if not (-1.7976931348623157e308 <= width <= 1.7976931348623157e308):
            raise Error("ValueError: uniform range must be finite")
        return low + width * self.random_sample()

    def standard_normal(mut self) -> Float64:
        if self._has_gauss:
            self._has_gauss = False
            var value = self._gauss
            self._gauss = 0
            return value
        while True:
            var x1 = 2.0 * self.random_sample() - 1.0
            var x2 = 2.0 * self.random_sample() - 1.0
            var r2 = x1 * x1 + x2 * x2
            if r2 < 1.0 and r2 != 0.0:
                var factor = sqrt(-2.0 * _log(r2) / r2)
                self._gauss = factor * x1
                self._has_gauss = True
                return factor * x2

    def normal(mut self, loc: Float64 = 0, scale: Float64 = 1) raises -> Float64:
        if scale < 0:
            raise Error("ValueError: scale must be nonnegative")
        return loc + scale * self.standard_normal()

    def standard_gamma(mut self, shape: Float64) raises -> Float64:
        """Legacy NumPy gamma draw with unit scale; finite nonnegative shape."""
        _gamma_parameter(shape)
        return self._standard_gamma(shape)

    def gamma(mut self, shape: Float64, scale: Float64 = 1) raises -> Float64:
        """Scalar Float64 gamma draw; zero scale still consumes a gamma draw."""
        _gamma_parameter(shape)
        _gamma_parameter(scale)
        return scale * self._standard_gamma(shape)

    def _standard_gamma(mut self, shape: Float64) -> Float64:
        # NumPy legacy-distributions.c, legacy_standard_gamma (see LICENSES).
        # Preserve rejection order and the shared polar-normal cache.
        if shape == 1.0:
            return -_log(1.0 - self.random_sample())
        elif shape == 0.0:
            return 0.0
        elif shape < 1.0:
            while True:
                var u = self.random_sample()
                var v = -_log(1.0 - self.random_sample())
                if u <= 1.0 - shape:
                    var x = _pow(u, 1.0 / shape)
                    if x <= v:
                        return x
                else:
                    var y = -_log((1.0 - u) / shape)
                    var x = _pow(1.0 - shape + shape * y, 1.0 / shape)
                    if x <= v + y:
                        return x
        else:
            var b = shape - 1.0 / 3.0
            var c = 1.0 / sqrt(9.0 * b)
            while True:
                var x = self.standard_normal()
                var v = 1.0 + c * x
                if v <= 0.0:
                    continue
                v = v * v * v
                var u = self.random_sample()
                if u < 1.0 - 0.0331 * (x * x) * (x * x):
                    return b * v
                if _log(u) < 0.5 * x * x + b * (1.0 - v + _log(v)):
                    return b * v

    def poisson(mut self, lam: Float64 = 1) raises -> Int:
        """Legacy NumPy Poisson draw, multiplication below 10 and PTRS above."""
        var maximum = 9223372036854775807.0 - 10.0 * sqrt(Float64(9223372036854775807.0))
        if not (0.0 <= lam <= maximum):
            raise Error("ValueError: poisson lam must be nonnegative and within the int64 safety limit")
        if lam == 0:
            return 0
        if lam < 10:
            var threshold = _exp(-lam)
            var product = Float64(1)
            var count = 0
            while True:
                product *= self.random_sample()
                if product <= threshold:
                    return count
                count += 1
        var b = 0.931 + 2.53 * sqrt(lam)
        var a = -0.059 + 0.02483 * b
        var inverse_alpha = 1.1239 + 1.1328 / (b - 3.4)
        var vr = 0.9277 - 3.6224 / (b - 2.0)
        var log_lam = _log(lam)
        while True:
            var u = self.random_sample() - 0.5
            var v = self.random_sample()
            var us = 0.5 - abs(u)
            if us == 0:
                continue
            var candidate = floor((2.0 * a / us + b) * u + lam + 0.43)
            if candidate < 0 or candidate >= 9223372036854775808.0:
                continue
            var k = Int(candidate)
            if us >= 0.07 and v <= vr:
                return k
            if us < 0.013 and v > us:
                continue
            if _log(v) + _log(inverse_alpha) - _log(a / (us * us) + b) <= -lam + Float64(k) * log_lam - _poisson_loggam(Float64(k) + 1.0):
                return k

    def multivariate_normal[mean_dtype: DType, cov_dtype: DType](
        mut self, mean: NDArray[mean_dtype], cov: NDArray[cov_dtype],
        size: Int = 1, tol: Float64 = 1e-8, check_valid: String = "raise",
    ) raises -> NDArray[DType.float64]:
        """Dense (size, features) draws via native SVD; see backend parity limits.

        `check_valid` follows NumPy: "raise" (this module's default) rejects a
        covariance that fails the SVD reconstruction check, "warn" (NumPy's
        default) prints a RuntimeWarning and draws with the SVD factor, and
        "ignore" skips the check.
        """
        comptime assert mean_dtype == DType.float32 or mean_dtype == DType.float64
        comptime assert cov_dtype == DType.float32 or cov_dtype == DType.float64
        if mean.ndim != 1 or cov.ndim != 2:
            raise Error("ValueError: mean must be a vector and cov a matrix")
        var n = mean.size
        if cov.shape[0] != n or cov.shape[1] != n or size < 0:
            raise Error("ValueError: incompatible covariance shape or negative size")
        if not (0.0 <= tol <= 1.7976931348623157e308):
            raise Error("ValueError: tol must be finite and nonnegative")
        if check_valid != "warn" and check_valid != "raise" and check_valid != "ignore":
            raise Error("ValueError: check_valid must equal 'warn', 'raise', or 'ignore'")
        for i in range(n):
            if not isfinite(_vector_value(mean, i)):
                raise Error("ValueError: mean must be finite")
        var covariance = nm.zeros[DType.float64](Shape(n, n))
        for i in range(n):
            for j in range(n):
                var value = Float64(cov.item(i, j))
                if not isfinite(value):
                    raise Error("ValueError: covariance must be finite")
                covariance[Item(i,j)] = value
        # Validate and factor before consuming state. This deliberately raises
        # for invalid covariance rather than NumPy's warning-and-continuing default.
        var transform = _covariance_transform(covariance, tol, check_valid)
        var z = nm.zeros[DType.float64](Shape(size, n))
        if n == 0 or size == 0:
            return z^
        for i in range(size):
            for j in range(n):
                z[Item(i,j)] = self.standard_normal()
        var result = z @ transform
        for i in range(size):
            for j in range(n):
                result[Item(i,j)] = result.item(i,j) + Float64(_vector_value(mean, j))
        return result^

    def _binomial_inversion(mut self, n: Int, p: Float64) -> Int:
        var q = 1.0 - p
        var qn = _exp(Float64(n) * _log(q))
        var np = Float64(n) * p
        var bound = min(n, Int(np + 10.0 * sqrt(np * q + 1.0)))
        var x = 0
        var px = qn
        var u = self.random_sample()
        while u > px:
            x += 1
            if x > bound:
                x = 0
                px = qn
                u = self.random_sample()
            else:
                u -= px
                px = (Float64(n - x + 1) * p * px) / (Float64(x) * q)
        return x

    def _binomial_btpe(mut self, n: Int, r: Float64) -> Int:
        var q = 1.0 - r
        var fm = Float64(n) * r + r
        var m = Int(floor(fm))
        var p1 = floor(2.195 * sqrt(Float64(n) * r * q) - 4.6 * q) + 0.5
        var xm = Float64(m) + 0.5
        var xl = xm - p1
        var xr = xm + p1
        var c = 0.134 + 20.5 / (15.3 + Float64(m))
        var a = (fm - xl) / (fm - xl * r)
        var laml = a * (1.0 + a / 2.0)
        a = (xr - fm) / (xr * q)
        var lamr = a * (1.0 + a / 2.0)
        var p2 = p1 * (1.0 + 2.0 * c)
        var p3 = p2 + c / laml
        var p4 = p3 + c / lamr
        var nrq = Float64(n) * r * q
        while True:
            var u = self.random_sample() * p4
            var v = self.random_sample()
            var y: Int
            if u <= p1:
                return Int(floor(xm - p1 * v + u))
            elif u <= p2:
                var x = xl + (u - p1) / c
                v = v * c + 1.0 - abs(Float64(m) - x + 0.5) / p1
                if v > 1.0:
                    continue
                y = Int(floor(x))
            elif u <= p3:
                if v == 0.0:
                    continue
                y = Int(floor(xl + _log(v) / laml))
                if y < 0:
                    continue
                v = v * (u - p2) * laml
            else:
                if v == 0.0:
                    continue
                y = Int(floor(xr - _log(v) / lamr))
                if y > n:
                    continue
                v = v * (u - p3) * lamr
            var k = abs(y - m)
            if k <= 20 or Float64(k) >= nrq / 2.0 - 1.0:
                var s = r / q
                a = s * Float64(n + 1)
                var f = Float64(1)
                if m < y:
                    for i in range(m + 1, y + 1):
                        f *= a / Float64(i) - s
                elif m > y:
                    for i in range(y + 1, m + 1):
                        f /= a / Float64(i) - s
                if v <= f:
                    return y
                continue
            var kd = Float64(k)
            var rho = (kd / nrq) * ((kd * (kd / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5)
            var t = -kd * kd / (2.0 * nrq)
            var av = _log(v)
            if av < t - rho:
                return y
            if av > t + rho:
                continue
            var x1 = Float64(y) + 1.0
            var f1 = Float64(m) + 1.0
            var z = Float64(n) + 1.0 - Float64(m)
            var w = Float64(n) - Float64(y) + 1.0
            # Preserve NumPy legacy's four positive Stirling corrections.
            if av <= xm * _log(f1 / x1) + (Float64(n - m) + 0.5) * _log(z / w) + Float64(y - m) * _log(w * r / (x1 * q)) + _stirling(f1) + _stirling(z) + _stirling(x1) + _stirling(w):
                return y

    def binomial(mut self, n: Int, p: Float64) raises -> Int:
        if n < 0 or not (0.0 <= p <= 1.0):
            raise Error("ValueError: invalid binomial parameters")
        var r = p
        if p > 0.5:
            r = 1.0 - p
        var value: Int
        if Float64(n) * r <= 30.0:
            value = self._binomial_inversion(n, r)
        else:
            value = self._binomial_btpe(n, r)
        if p > 0.5:
            return n - value
        return value

    def multinomial(mut self, n: Int, p: NDArray[DType.float64], size: Int = 1) raises -> NDArray[DType.int64]:
        """Dense (size, classes) rows, including size=1; last p is residual."""
        _probabilities(p, False)
        if n < 0 or size < 0:
            raise Error("ValueError: negative n or size")
        var d = p.size
        var result = nm.zeros[DType.int64](Shape(size, d))
        if d == 0:
            return result^
        for row in range(size):
            var remaining_p = Float64(1)
            var remaining_n = n
            for j in range(d - 1):
                var drawn = self.binomial(remaining_n, _vector_value(p, j) / remaining_p)
                result[Item(row, j)] = Int64(drawn)
                remaining_n -= drawn
                if remaining_n <= 0:
                    break
                remaining_p -= _vector_value(p, j)
            if remaining_n > 0:
                result[Item(row, d - 1)] = Int64(remaining_n)
        return result^

    def categorical(mut self, p: NDArray[DType.float64]) raises -> Int:
        """NumPy choice(p=...) CDF protocol, not multinomial(1, p)."""
        _probabilities(p, True)
        var total = Float64(0)
        for i in range(p.size):
            total += _vector_value(p, i)
        var u = self.random_sample()
        var cumulative = Float64(0)
        for i in range(p.size):
            cumulative += _vector_value(p, i)
            if u < cumulative / total:
                return i
        return p.size - 1

    def choice(mut self, a: Int, size: Int = 1, replace: Bool = True) raises -> NDArray[DType.int64]:
        """Uniform indices, always a vector; NumPy RandomState draw order."""
        _choice_size(a, size, replace)
        var result = nm.zeros[DType.int64](Shape(size))
        if replace:
            for i in range(size):
                result[Item(i)] = Int64(self.randint(a))
        else:
            # NumPy choice uses a complete permutation even for size=0.
            var shuffled = self.permutation(a)
            for i in range(size):
                result[Item(i)] = shuffled.item(i)
        return result^

    def choice(mut self, a: Int, p: NDArray[DType.float64], size: Int = 1, replace: Bool = True) raises -> NDArray[DType.int64]:
        """Weighted indices; without replacement follows NumPy's batch rejection."""
        _choice_size(a, size, replace)
        _probabilities(p, True)
        if p.size != a:
            raise Error("ValueError: a and p must have the same size")
        var positive = 0
        for i in range(a):
            if _vector_value(p, i) > 0:
                positive += 1
        if not replace and positive < size:
            raise Error("ValueError: fewer positive probabilities than samples")
        var result = nm.zeros[DType.int64](Shape(size))
        var cdf = nm.zeros[DType.float64](Shape(a))
        var selected = nm.zeros[DType.bool](Shape(a))
        var found = 0
        while found < size:
            var total = Float64(0)
            for i in range(a):
                if not selected.item(i):
                    total += _vector_value(p, i)
                cdf[Item(i)] = total
            for i in range(a):
                cdf[Item(i)] = cdf.item(i) / total
            var remaining = size - found
            # Keep this CDF for the entire batch; remove selected weights only
            # before the next batch, preserving first-occurrence order.
            for _ in range(remaining):
                var u = self.random_sample()
                var low = 0
                var high = a
                while low < high:
                    var mid = low + (high - low) // 2
                    if u < cdf.item(mid):
                        high = mid
                    else:
                        low = mid + 1
                if replace or not selected.item(low):
                    result[Item(found)] = Int64(low)
                    found += 1
                    if not replace:
                        selected[Item(low)] = True
        return result^

    def choice[dtype: DType](mut self, a: NDArray[dtype], size: Int = 1, replace: Bool = True) raises -> NDArray[dtype]:
        if a.ndim != 1:
            raise Error("ValueError: choice population must be one dimensional")
        var indices = self.choice(a.size, size, replace)
        var result = nm.zeros[dtype](Shape(size))
        for i in range(size):
            result[Item(i)] = _vector_value(a, Int(indices.item(i)))
        return result^

    def choice[dtype: DType](mut self, a: NDArray[dtype], p: NDArray[DType.float64], size: Int = 1, replace: Bool = True) raises -> NDArray[dtype]:
        if a.ndim != 1:
            raise Error("ValueError: choice population must be one dimensional")
        var indices = self.choice(a.size, p, size, replace)
        var result = nm.zeros[dtype](Shape(size))
        for i in range(size):
            result[Item(i)] = _vector_value(a, Int(indices.item(i)))
        return result^

    def shuffle[dtype: DType](mut self, mut x: NDArray[dtype]) raises:
        """Shuffle a vector or matrix's rows in place, including strided views."""
        if x.ndim != 1 and x.ndim != 2:
            raise Error("ValueError: shuffle supports rank one or two")
        var rows = x.shape[0]
        var cols = x.shape[1] if x.ndim == 2 else 1
        if rows <= 1 or cols == 0:
            return
        var stride0 = Int(x.strides[0])
        var stride1 = Int(x.strides[1]) if x.ndim == 2 else 0
        # Admit disjoint row/column layouts, including transposed/reversed views.
        if stride0 == 0 or (cols > 1 and (stride1 == 0 or not (
            abs(stride0) >= cols * abs(stride1) or abs(stride1) >= rows * abs(stride0)
        ))):
            raise Error("ValueError: shuffle requires disjoint rows or columns")
        for i in range(rows - 1, 0, -1):
            var j = Int(self._interval(UInt64(i)))
            for c in range(cols):
                var left = i * stride0 + c * stride1
                var right = j * stride0 + c * stride1
                var value = x.unsafe_get(left)
                x.unsafe_set(left, x.unsafe_get(right))
                x.unsafe_set(right, value)

    def permutation[dtype: DType](mut self, x: NDArray[dtype]) raises -> NDArray[dtype]:
        """Return an independent shuffled vector or matrix."""
        var result = x.copy()
        if x.ndim == 2 and x.shape[1] == 0:
            # NumPy permutes row indices even when each row is empty.
            for i in range(x.shape[0] - 1, 0, -1):
                _ = self._interval(UInt64(i))
        else:
            self.shuffle(result)
        return result^

    def permutation(mut self, n: Int) raises -> NDArray[DType.int64]:
        if n < 0:
            return nm.zeros[DType.int64](Shape(0))
        var result = nm.zeros[DType.int64](Shape(n))
        for i in range(n):
            result[Item(i)] = Int64(i)
        for i in range(n - 1, 0, -1):
            var j = Int(self._interval(UInt64(i)))
            var value = result.item(i)
            result[Item(i)] = result.item(j)
            result[Item(j)] = value
        return result^

    def sample_without_replacement(mut self, n_population: Int, n_samples: Int, method: String = "auto") raises -> NDArray[DType.int64]:
        if n_population < 0 or n_samples < 0 or n_samples > n_population:
            raise Error("ValueError: invalid population/sample count")
        if method != "auto" and method != "tracking_selection" and method != "pool" and method != "reservoir_sampling":
            raise Error("ValueError: unknown sampling method")
        var ratio = Float64(1)
        if n_population != 0:
            ratio = Float64(n_samples) / Float64(n_population)
        var result = nm.zeros[DType.int64](Shape(n_samples))
        if method == "auto" and ratio > 0.01 and ratio < 0.99:
            var pool = self.permutation(n_population)
            for i in range(n_samples):
                result[Item(i)] = pool.item(i)
        elif (method == "auto" or method == "tracking_selection") and ratio < 0.2:
            var selected = Set[Int]()
            for i in range(n_samples):
                var j = self.randint(n_population)
                while j in selected:
                    j = self.randint(n_population)
                selected.add(j)
                result[Item(i)] = Int64(j)
        elif method == "pool":
            var pool = List[Int]()
            for i in range(n_population):
                pool.append(i)
            for i in range(n_samples):
                var j = self.randint(n_population - i)
                result[Item(i)] = Int64(pool[j])
                pool[j] = pool[n_population - i - 1]
        else:
            for i in range(n_samples):
                result[Item(i)] = Int64(i)
            for i in range(n_samples, n_population):
                var j = self.randint(i + 1)
                if j < n_samples:
                    result[Item(j)] = Int64(i)
        return result^

    def state_position(self) -> Int:
        return self._pos

    def state_word(self, i: Int) raises -> UInt32:
        if i < 0 or i >= 624:
            raise Error("ValueError: state index out of range")
        return self._key[i]

    def has_gauss(self) -> Bool:
        return self._has_gauss

    def cached_gauss(self) -> Float64:
        return self._gauss


def _stirling(value: Float64) -> Float64:
    var square = value * value
    return (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / square) / square) / square) / square) / value / 166320.0


def _probabilities(p: NDArray[DType.float64], categorical: Bool) raises:
    if p.ndim != 1:
        raise Error("ValueError: probabilities must be one dimensional")
    var total = Float64(0)
    var correction = Float64(0)
    for i in range(p.size):
        var value = _vector_value(p, i)
        if not (0.0 <= value <= 1.0):
            raise Error("ValueError: probabilities must be finite in [0, 1]")
        if categorical or i < p.size - 1:
            var adjusted = value - correction
            var updated = total + adjusted
            correction = (updated - total) - adjusted
            total = updated
    if categorical:
        if p.size == 0 or abs(total - 1.0) > 1.4901161193847656e-8:
            raise Error("ValueError: probabilities must sum to one")
    elif total > 1.0 + 1e-12:
        raise Error("ValueError: sum of probabilities except last exceeds one")


def entropy_state() raises -> RandomState:
    """Create a native time-seeded context; retain and pass mutably for None semantics."""
    return RandomState(Int(UInt64(perf_counter_ns()) & 0xffffffff))


def _log(x: Float64) -> Float64:
    # Standard compiler primitive: CPU lowering uses the platform math runtime.
    return llvm_intrinsic["llvm.log", Float64, has_side_effect=False](x)


def _exp(x: Float64) -> Float64:
    return llvm_intrinsic["llvm.exp", Float64, has_side_effect=False](x)


def _pow(x: Float64, y: Float64) -> Float64:
    return llvm_intrinsic["llvm.pow", Float64, has_side_effect=False](x, y)


def _gamma_parameter(value: Float64) raises:
    # NumPy rejects negative zero too. Nonfinite parameters are explicitly
    # unsupported here, avoiding nonterminating legacy rejection loops.
    if not (0.0 <= value <= 1.7976931348623157e308) or copysign(Float64(1), value) < 0:
        raise Error("ValueError: gamma shape and scale must be finite and nonnegative")


def _choice_size(a: Int, size: Int, replace: Bool) raises:
    if size < 0 or a < 0 or (a == 0 and size != 0) or (not replace and size > a):
        raise Error("ValueError: invalid choice population or sample size")


def _vector_value[dtype: DType](a: NDArray[dtype], i: Int) raises -> Scalar[dtype]:
    # NuMojo's single-Int item overload ignores strides on rank-one views.
    return a.unsafe_get(i * Int(a.strides[0]))


def _poisson_loggam(x: Float64) -> Float64:
    # NumPy random_loggam; preserve its polynomial and evaluation order for
    # rejection decisions rather than substituting a different lgamma routine.
    if x == 1 or x == 2:
        return 0
    var n = Int(7.0 - x) if x < 7 else 0
    var x0 = x + Float64(n)
    var x2 = (1.0 / x0) * (1.0 / x0)
    var g = Float64(-1.39243221690590)
    g = g * x2 + 1.796443723688307e-1
    g = g * x2 - 2.955065359477124e-2
    g = g * x2 + 6.410256410256410e-3
    g = g * x2 - 1.917526917526918e-3
    g = g * x2 + 8.417508417508418e-4
    g = g * x2 - 5.952380952380952e-4
    g = g * x2 + 7.936507936507937e-4
    g = g * x2 - 2.777777777777778e-3
    g = g * x2 + 8.333333333333333e-2
    var result = g / x0 + 0.5 * 1.8378770664093453 + (x0 - 0.5) * _log(x0) - x0
    for _ in range(n):
        result -= _log(x0 - 1.0)
        x0 -= 1.0
    return result


def _covariance_transform(cov: NDArray[DType.float64], tol: Float64, check_valid: String = "raise") raises -> NDArray[DType.float64]:
    try:
        return _covariance_transform_impl(cov, tol, check_valid)
    except error:
        raise Error("ValueError: multivariate_normal covariance: " + String(error))


def _covariance_transform_impl(cov: NDArray[DType.float64], tol: Float64, check_valid: String) raises NumericalError -> NDArray[DType.float64]:
    var n = cov.shape[0]
    if n == 0:
        return nm.zeros[DType.float64](Shape(0, 0))
    var factor = svd(cov)
    var singular = factor.singular_values()
    var vectors = factor.vt()
    var weighted = nm.zeros[DType.float64](Shape(n, n))
    for i in range(n):
        for j in range(n):
            weighted[Item(i,j)] = singular.item(i) * vectors.item(i,j)
    if check_valid != "ignore":
        var transpose = vectors.T()
        var reconstructed = transpose @ weighted
        var invalid = -1
        for i in range(n):
            for j in range(n):
                var actual = reconstructed.item(i,j)
                var expected = cov.item(i,j)
                if invalid < 0 and (not isfinite(actual) or abs(actual - expected) > tol + tol * abs(expected)):
                    invalid = i
        if invalid >= 0:
            if check_valid == "raise":
                raise NumericalError("not_positive_semidefinite", "multivariate_normal", invalid)
            # NumPy warns and continues with the same SVD factor.
            print("RuntimeWarning: covariance is not symmetric positive-semidefinite.", file=stderr)
    for i in range(n):
        for j in range(n):
            weighted[Item(i,j)] = sqrt(singular.item(i)) * vectors.item(i,j)
    return weighted^


struct _GlobalRandomStorage(Movable):
    var state: Optional[RandomState]
    var active: Atomic[DType.int32]

    def __init__(out self):
        self.state = Optional[RandomState]()
        self.active = Atomic[DType.int32](0)


def _init_global_random_storage() -> _GlobalRandomStorage:
    return _GlobalRandomStorage()


# Distinct from std.random's Philox singleton. The compiler runtime owns and
# destroys this storage; the atomic lease protects initialization and all draws.
comptime _global_mt19937 = _Global["moxie.random.mt19937.v1", _init_global_random_storage]


struct _GlobalRandomLease(Movable):
    var storage: Pointer[_GlobalRandomStorage, MutUntrackedOrigin]

    def __init__(out self) raises:
        var storage = _global_mt19937.get_or_create_ptr()
        var expected = Int32(0)
        if not storage[].active.compare_exchange(expected, Int32(1)):
            raise Error("RuntimeError: global RandomState is already borrowed; use the active state or an independent stream")
        self.storage = storage

    def __deinit__(deinit self):
        self.storage[].active.store(Int32(0))


struct RandomStateHandle[origin: MutOrigin](Movable):
    """Move-only normalized RNG; access with handle[] or a scoped with block.

    An owned seed creates fresh state. A borrowed pointer retains the caller's
    origin. A default stream additionally holds an exclusive process-wide lease.
    """
    var _state: Variant[RandomState, Pointer[RandomState, Self.origin]]
    var _lease: Optional[_GlobalRandomLease]

    def __init__(out self, var state: RandomState):
        self._state = Variant[RandomState, Pointer[RandomState, Self.origin]](state^)
        self._lease = Optional[_GlobalRandomLease]()

    def __init__(out self, state: Pointer[RandomState, Self.origin]):
        self._state = Variant[RandomState, Pointer[RandomState, Self.origin]](state)
        self._lease = Optional[_GlobalRandomLease]()

    def __init__(out self, state: Pointer[RandomState, Self.origin], var lease: _GlobalRandomLease):
        self._state = Variant[RandomState, Pointer[RandomState, Self.origin]](state)
        self._lease = Optional[_GlobalRandomLease](lease^)

    def __getitem__(mut self) -> ref[origin_of(self, Self.origin)] RandomState:
        # Narrow, never extend, the referent's lifetime. Owned state belongs to
        # self; a borrowed origin parameter outlives self; global state is leased
        # for self's lifetime. No raw untracked pointer escapes this boundary.
        if self._state.isa[RandomState]():
            return Pointer(to=self._state[RandomState]).unsafe_origin_cast[origin_of(self, Self.origin)]()[]
        return self._state[Pointer[RandomState, Self.origin]].unsafe_origin_cast[origin_of(self, Self.origin)]()[]

    def __enter__(mut self) -> Pointer[RandomState, origin_of(self, Self.origin)]:
        return Pointer(to=self[])

    def __exit__(mut self):
        # Fields are destroyed on both normal and exceptional scope exit.
        pass


def check_random_state(seed: Int) raises -> RandomStateHandle[MutUntrackedOrigin]:
    """Normalize an integer to a fresh, owned MT19937 stream."""
    return RandomStateHandle[MutUntrackedOrigin](RandomState(seed))


def check_random_state(mut state: RandomState) -> RandomStateHandle[origin_of(state)]:
    """Borrow an existing stream without copying or resetting it."""
    return RandomStateHandle[origin_of(state)](Pointer(to=state))


def check_random_state(seed: NoneType = None) raises -> RandomStateHandle[MutUntrackedOrigin]:
    """Lease the process default stream; initialize once on first use."""
    var lease = _GlobalRandomLease()
    if not lease.storage[].state:
        lease.storage[].state = entropy_state()
    var state = Pointer(to=lease.storage[].state.value())
    return RandomStateHandle[MutUntrackedOrigin](state, lease^)


def check_random_state(seed: Optional[Int]) raises -> RandomStateHandle[MutUntrackedOrigin]:
    """Normalize an optional integer configuration; an empty value means None."""
    if seed:
        return check_random_state(seed.value())
    return check_random_state()


def seed_global_random_state(seed: Int) raises:
    """Reset the process default MT19937, including its Gaussian cache.

    Invalid seeds or an outstanding default-stream lease leave it unchanged.
    This does not seed std.random, NumPy, or any independently owned state.
    """
    var state = RandomState(seed)
    var lease = _GlobalRandomLease()
    lease.storage[].state = state^
