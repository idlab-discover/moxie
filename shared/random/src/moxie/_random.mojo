"""Shared NumPy RandomState compatibility and explicitly native Philox streams.

MT19937, legacy distribution and sklearn sampling adaptations: see LICENSES.
"""
from std.math import sqrt, floor, abs
from std.sys import llvm_intrinsic
from std.random import Random as Philox
from std.time import perf_counter_ns
from std.collections import Set
import numojo as nm
from numojo import NDArray, Shape, Item


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
                var drawn = self.binomial(remaining_n, p.item(j) / remaining_p)
                result[Item(row, j)] = Int64(drawn)
                remaining_n -= drawn
                if remaining_n <= 0:
                    break
                remaining_p -= p.item(j)
            if remaining_n > 0:
                result[Item(row, d - 1)] = Int64(remaining_n)
        return result^

    def categorical(mut self, p: NDArray[DType.float64]) raises -> Int:
        """NumPy choice(p=...) CDF protocol, not multinomial(1, p)."""
        _probabilities(p, True)
        var total = Float64(0)
        for i in range(p.size):
            total += p.item(i)
        var u = self.random_sample()
        var cumulative = Float64(0)
        for i in range(p.size):
            cumulative += p.item(i)
            if u < cumulative / total:
                return i
        return p.size - 1

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
        var value = p.item(i)
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
