# Adapted from scikit-learn sklearn/preprocessing/_polynomial.py; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Spline evaluation ports SciPy v1.18.1 scipy/interpolate/src/__fitpack.cc
# (`_find_interval`, `_deBoor_D`, `_evaluate_spline`), BSD-3-Clause, Copyright (c)
# 2001-2002 Enthought, Inc. 2003, SciPy Developers; see LICENSES/scipy.txt.
# Knot placement reproduces NumPy 2.5 `linspace`, `nanpercentile(method="linear")`
# and `remainder` arithmetic, BSD-3-Clause, Copyright (c) 2005-2025 NumPy
# Developers; see LICENSES/numpy.txt.
"""PolynomialFeatures and SplineTransformer on dense NuMojo arrays.

See docs/api/polynomial.md for the native signatures, adaptations, memory
bounds and verified tolerances.
"""
from std.math import isnan, isfinite, floor, nan
from std.memory import bitcast
from numojo import NDArray, Shape, full
from moxie.preprocessing._base import Preprocessor
from moxie.preprocessing._common import (
    at, at1, put, matrix, vector, check_fitted, check_n_features, check_set_output,
    check_names_length, feature_names_in, dtype_name,
)
from moxie.preprocessing._validation import require_dense_output
from moxie.preprocessing._percentiles import (
    linspace as _linspace, weighted_percentile_column as _weighted_percentile_column,
)


# ---------------------------------------------------------------------------
# Shared helpers


def _check_order(order: String, owner: String) raises:
    if order != "C" and order != "F":
        raise Error("ValueError: The 'order' parameter of " + owner + " must be a str among {'C', 'F'}. Got '" + order + "' instead.")


def _output[d: DType](rows: Int, cols: Int, order: String) raises -> NDArray[d]:
    """Zero-filled owned result in the requested memory layout ('C' or 'F')."""
    if rows < 0 or cols < 0:
        raise Error("ValueError: negative output dimensions")
    if cols > 0 and rows > (Int.MAX // cols):
        raise Error("ValueError: output of shape (" + String(rows) + ", " + String(cols) + ") is too large to allocate")
    return full[d](Shape(rows, cols), Scalar[d](0), order=order)


def _check_shape[d: DType](X: NDArray[d], min_samples: Int, owner: String) raises:
    """sklearn check_array dimensions (Float32/Float64 rank 2, minimum sizes)."""
    comptime assert d == DType.float32 or d == DType.float64, "Float32/Float64 only"
    if X.ndim != 2:
        raise Error("ValueError: Expected 2D array, got " + String(X.ndim) + "D array instead")
    var by = String(" by ") + owner if min_samples > 1 else String("")
    if X.shape[0] < min_samples:
        raise Error("ValueError: Found array with " + String(X.shape[0]) + " sample(s) (shape=(" + String(X.shape[0]) + ", " + String(X.shape[1]) + ")) while a minimum of " + String(min_samples) + " is required" + by + ".")
    if X.shape[1] < 1:
        raise Error("ValueError: Found array with 0 feature(s) (shape=(" + String(X.shape[0]) + ", 0)) while a minimum of 1 is required" + by + ".")


def _check_finite[d: DType](X: NDArray[d], nan_message: String, input_name: String) raises:
    """sklearn `_assert_all_finite` for one named input: NaN, then infinity."""
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            var v = at(X, i, j)
            if not isfinite(v):
                if isnan(v):
                    raise Error(nan_message)
                raise Error("ValueError: Input " + input_name + "contains infinity or a value too large for dtype('" + dtype_name[d]() + "').")


# ---------------------------------------------------------------------------
# Exact big-integer counting (sklearn uses Python integers via scipy comb)


struct _BigUInt(Copyable, Movable):
    """Nonnegative integer, little-endian base 2^32 limbs; empty means zero."""
    var limbs: List[UInt32]

    def __init__(out self, value: UInt64):
        self.limbs = List[UInt32]()
        var v = value
        while v != 0:
            self.limbs.append(UInt32(v & 0xFFFFFFFF))
            v >>= 32

    def _trim(mut self):
        while len(self.limbs) > 0 and self.limbs[len(self.limbs) - 1] == 0:
            _ = self.limbs.pop()

    def is_zero(self) -> Bool:
        return len(self.limbs) == 0

    def mul_small(mut self, factor: UInt64):
        var carry = UInt128(0)
        for i in range(len(self.limbs)):
            var v = UInt128(self.limbs[i]) * UInt128(factor) + carry
            self.limbs[i] = UInt32(v & 0xFFFFFFFF)
            carry = v >> 32
        while carry != 0:
            self.limbs.append(UInt32(carry & 0xFFFFFFFF))
            carry >>= 32
        self._trim()

    def div_small(mut self, divisor: UInt64) -> UInt64:
        """Floor division in place; returns the remainder."""
        var rem = UInt128(0)
        var i = len(self.limbs) - 1
        while i >= 0:
            var cur = (rem << 32) | UInt128(self.limbs[i])
            self.limbs[i] = UInt32(cur // UInt128(divisor))
            rem = cur % UInt128(divisor)
            i -= 1
        self._trim()
        return UInt64(rem)

    def add(mut self, other: Self):
        var carry = UInt64(0)
        var n = max(len(self.limbs), len(other.limbs))
        for i in range(n):
            var a = UInt64(self.limbs[i]) if i < len(self.limbs) else UInt64(0)
            var b = UInt64(other.limbs[i]) if i < len(other.limbs) else UInt64(0)
            var s = a + b + carry
            if i < len(self.limbs):
                self.limbs[i] = UInt32(s & 0xFFFFFFFF)
            else:
                self.limbs.append(UInt32(s & 0xFFFFFFFF))
            carry = s >> 32
        if carry != 0:
            self.limbs.append(UInt32(carry))

    def sub(mut self, other: Self):
        """self -= other; callers guarantee self >= other."""
        var borrow = Int64(0)
        for i in range(len(self.limbs)):
            var b = Int64(other.limbs[i]) if i < len(other.limbs) else Int64(0)
            var s = Int64(self.limbs[i]) - b - borrow
            borrow = 0
            if s < 0:
                s += Int64(1) << 32
                borrow = 1
            self.limbs[i] = UInt32(s)
        self._trim()

    def fits_int64(self) -> Bool:
        if len(self.limbs) <= 1:
            return True
        if len(self.limbs) > 2:
            return False
        return self.limbs[1] < 0x80000000

    def to_int(self) -> Int:
        var v = 0
        var i = len(self.limbs) - 1
        while i >= 0:
            v = (v << 32) | Int(self.limbs[i])
            i -= 1
        return v

    def decimal(self) -> String:
        if self.is_zero():
            return "0"
        var work = self.copy()
        var chunks = List[UInt64]()
        while not work.is_zero():
            chunks.append(work.div_small(1_000_000_000))
        var text = String(chunks[len(chunks) - 1])
        var i = len(chunks) - 2
        while i >= 0:
            var part = String(chunks[i])
            while part.byte_length() < 9:
                part = "0" + part
            text += part
            i -= 1
        return text


def _comb(n: UInt64, k: UInt64) -> _BigUInt:
    """Exact binomial coefficient C(n, k) (0 when k > n), scipy comb(exact=True)."""
    if k > n:
        return _BigUInt(0)
    var r = k if k <= n - k else n - k
    var result = _BigUInt(1)
    for i in range(1, Int(r) + 1):
        result.mul_small(n - r + UInt64(i))
        _ = result.div_small(UInt64(i))
    return result^


def _num_combinations(n: Int, min_degree: Int, max_degree: Int, interaction_only: Bool, include_bias: Bool) -> _BigUInt:
    """sklearn PolynomialFeatures._num_combinations with exact integers."""
    var total = _BigUInt(0)
    if interaction_only:
        var start = max(1, min_degree)
        var stop = min(max_degree, n)
        if start <= stop:
            var term = _comb(UInt64(n), UInt64(start))
            total.add(term)
            for i in range(start + 1, stop + 1):
                term.mul_small(UInt64(n - i + 1))
                _ = term.div_small(UInt64(i))
                total.add(term)
    else:
        total = _comb(UInt64(n) + UInt64(max_degree), UInt64(max_degree))
        total.sub(_BigUInt(1))
        if min_degree > 0:
            var d = min_degree - 1
            var lower = _comb(UInt64(n) + UInt64(d), UInt64(d))
            lower.sub(_BigUInt(1))
            total.sub(lower)
    if include_bias:
        total.add(_BigUInt(1))
    return total^


def _next_combination(mut combo: List[Int], n: Int, with_replacement: Bool) -> Bool:
    """Advance to the next itertools combination (lexicographic); False when done."""
    var r = len(combo)
    var i = r - 1
    if with_replacement:
        while i >= 0 and combo[i] == n - 1:
            i -= 1
        if i < 0:
            return False
        var value = combo[i] + 1
        for j in range(i, r):
            combo[j] = value
        return True
    while i >= 0 and combo[i] == i + n - r:
        i -= 1
    if i < 0:
        return False
    combo[i] += 1
    for j in range(i + 1, r):
        combo[j] = combo[j - 1] + 1
    return True


def _expansion_plan(p: Int, max_degree: Int, interaction_only: Bool, include_bias: Bool, n_full: Int) raises -> List[Int]:
    """sklearn's dense column bookkeeping: blocks XP[:, start:end] * X[:, feature].

    Returns flat (destination, start, end, feature) quadruples. Stops once a
    degree adds no columns: every later degree would add none either.
    """
    var plan = List[Int]()
    var current = 1 if include_bias else 0
    if max_degree == 0:
        return plan^
    var index = List[Int]()
    for j in range(p):
        index.append(current + j)
    current += p
    index.append(current)
    var degree = 2
    while degree <= max_degree:
        var new_index = List[Int]()
        var end = index[len(index) - 1]
        var added = False
        for feature in range(p):
            if feature + 1 >= len(index):
                raise Error("ValueError: internal polynomial expansion index out of range")
            var start = index[feature]
            new_index.append(current)
            if interaction_only:
                start += index[feature + 1] - index[feature]
            var next_col = current + end - start
            if next_col <= current:
                break
            plan.append(current)
            plan.append(start)
            plan.append(end)
            plan.append(feature)
            current = next_col
            added = True
        new_index.append(current)
        index = new_index^
        if not added:
            break
        degree += 1
    if current != n_full:
        raise Error("ValueError: internal polynomial expansion width mismatch")
    return plan^


# ---------------------------------------------------------------------------
# PolynomialFeatures


@fieldwise_init
struct PolynomialFeaturesParams(Copyable, Movable):
    var degree: Int
    var degree_range: Optional[Tuple[Int, Int]]
    var interaction_only: Bool
    var include_bias: Bool
    var order: String


struct PolynomialFeatures[dtype: DType = DType.float64](Preprocessor):
    """Generate polynomial and interaction features of a dense matrix.

    `degree` is the maximal degree; the `(min_degree, max_degree)` tuple form is
    the `degree_range` field (set by the tuple constructor overload). Output
    columns follow sklearn's itertools combination order, and each column is
    built with sklearn's multiplication order, so values are bit-identical.
    """
    comptime Params = PolynomialFeaturesParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var degree: Int
    var degree_range: Optional[Tuple[Int, Int]]
    var interaction_only: Bool
    var include_bias: Bool
    var order: String
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    var n_output_features_: Int
    var _fitted: Bool
    var _min_degree: Int
    var _max_degree: Int
    var _interaction_only: Bool
    var _include_bias: Bool
    var _order: String
    var _n_out_full: Int  # -1 when the intermediate width does not fit Int
    var _plan: List[Int]

    def __init__(out self, degree: Int = 2, *, interaction_only: Bool = False, include_bias: Bool = True, order: String = "C"):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 only"
        self.degree = degree
        self.degree_range = None
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self.n_output_features_ = 0
        self._fitted = False
        self._min_degree = 0
        self._max_degree = 0
        self._interaction_only = False
        self._include_bias = True
        self._order = "C"
        self._n_out_full = 0
        self._plan = List[Int]()

    def __init__(out self, degree: Tuple[Int, Int], *, interaction_only: Bool = False, include_bias: Bool = True, order: String = "C"):
        """sklearn `degree=(min_degree, max_degree)`."""
        self = Self(degree[1], interaction_only=interaction_only, include_bias=include_bias, order=order)
        self.degree_range = degree

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.degree, self.degree_range, self.interaction_only, self.include_bias, self.order)

    def set_params(mut self, params: Self.Params) raises:
        self.degree = params.degree
        self.degree_range = params.degree_range
        self.interaction_only = params.interaction_only
        self.include_bias = params.include_bias
        self.order = params.order

    def set_params(
        mut self, *, degree: Optional[Int] = None, interaction_only: Optional[Bool] = None,
        include_bias: Optional[Bool] = None, order: Optional[String] = None,
    ) raises:
        """Keyword updates; an integer `degree` replaces a tuple degree."""
        if degree:
            self.degree = degree.value()
            self.degree_range = None
        if interaction_only:
            self.interaction_only = interaction_only.value()
        if include_bias:
            self.include_bias = include_bias.value()
        if order:
            self.order = order.value()

    def set_params(mut self, *, degree: Tuple[Int, Int]) raises:
        """sklearn `set_params(degree=(min_degree, max_degree))`."""
        self.degree = degree[1]
        self.degree_range = degree

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    # -- fitting ------------------------------------------------------------

    def _fit(mut self, X: Self.Input) raises:
        # Parameter constraints precede data validation, as in sklearn.
        if not self.degree_range and self.degree < 0:
            raise Error("ValueError: The 'degree' parameter of PolynomialFeatures must be an int in the range [0, inf) or an array-like. Got " + String(self.degree) + " instead.")
        _check_order(self.order, "PolynomialFeatures")
        _check_shape(X, 1, "PolynomialFeatures")
        _check_finite(X, "ValueError: Input X contains NaN.", "X ")
        var p = X.shape[1]
        var min_degree = 0
        var max_degree = self.degree
        if not self.degree_range:
            if max_degree == 0 and not self.include_bias:
                raise Error("ValueError: Setting degree to zero and include_bias to False would result in an empty output array.")
        else:
            min_degree = self.degree_range.value()[0]
            max_degree = self.degree_range.value()[1]
            if not (min_degree >= 0 and min_degree <= max_degree):
                raise Error("ValueError: degree=(min_degree, max_degree) must be non-negative integers that fulfil min_degree <= max_degree, got (" + String(min_degree) + ", " + String(max_degree) + ").")
            if max_degree == 0 and not self.include_bias:
                raise Error("ValueError: Setting both min_degree and max_degree to zero and include_bias to False would result in an empty output array.")
        var n_out = _num_combinations(p, min_degree, max_degree, self.interaction_only, self.include_bias)
        if not n_out.fits_int64():
            raise Error("ValueError: The output that would result from the current configuration would have " + n_out.decimal() + " features which is too large to be indexed by int64. Please change some or all of the following:\n- The number of features in the input, currently n_features=" + String(p) + "\n- The range of degrees to calculate, currently [" + String(min_degree) + ", " + String(max_degree) + "]\n- Whether to include only interaction terms, currently " + String(self.interaction_only) + "\n- Whether to include a bias term, currently " + String(self.include_bias) + ".")
        var full_count = _num_combinations(p, 0, max_degree, self.interaction_only, self.include_bias)
        var n_full = full_count.to_int() if full_count.fits_int64() else -1
        var plan = List[Int]()
        if n_full >= 0:
            plan = _expansion_plan(p, max_degree, self.interaction_only, self.include_bias, n_full)
        self.n_output_features_ = n_out.to_int()
        self._n_out_full = n_full
        self._plan = plan^
        self._min_degree = min_degree
        self._max_degree = max_degree
        self._interaction_only = self.interaction_only
        self._include_bias = self.include_bias
        self._order = self.order
        self.n_features_in_ = p
        self.feature_names_in_ = None
        self._fitted = True

    def fit(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises:
        self._fit(X)

    def fit_named(mut self, X: Self.Input, feature_names: List[String], y: Optional[Self.Target] = None) raises:
        _check_shape(X, 1, "PolynomialFeatures")
        check_names_length(feature_names, X.shape[1])
        self._fit(X)
        self.feature_names_in_ = feature_names.copy()

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self._fit(X)
        return self.transform(X)

    # -- transformation -------------------------------------------------------

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """Rows are expanded one at a time in an O(n_out_full) workspace."""
        check_fitted(self._fitted, "PolynomialFeatures")
        _check_shape(X, 1, "PolynomialFeatures")
        _check_finite(X, "ValueError: Input X contains NaN.", "X ")
        check_n_features("PolynomialFeatures", self.n_features_in_, X.shape[1])
        if self._n_out_full < 0:
            raise Error("ValueError: the intermediate expansion up to degree " + String(self._max_degree) + " is too large to compute")
        var n = X.shape[0]
        var p = X.shape[1]
        var n_out = self.n_output_features_
        var n_full = self._n_out_full
        var out = _output[Self.dtype](n, n_out, self._order)
        var work = List[Scalar[Self.dtype]](length=n_full, fill=0)
        var first = 1 if self._include_bias else 0
        # Columns of degree < min_degree are intermediates only (sklearn slices them off).
        var skip = n_full - n_out if self._min_degree > 1 else 0
        var ops = len(self._plan) // 4
        for i in range(n):
            if self._include_bias:
                work[0] = 1
            if self._max_degree > 0:
                for j in range(p):
                    work[first + j] = at(X, i, j)
                for o in range(ops):
                    var dest = self._plan[4 * o]
                    var start = self._plan[4 * o + 1]
                    var stop = self._plan[4 * o + 2]
                    var factor = work[first + self._plan[4 * o + 3]]
                    for c in range(stop - start):
                        work[dest + c] = work[start + c] * factor
            if skip > 0 and self._include_bias:
                put(out, i, 0, Scalar[Self.dtype](1))
                for c in range(1, n_out):
                    put(out, i, c, work[skip + c])
            else:
                for c in range(n_out):
                    put(out, i, c, work[skip + c])
        return out^

    # -- metadata -----------------------------------------------------------

    def _enumerate(self, mut names: List[String], mut powers: List[Int64], input_names: List[String], want_names: Bool) raises:
        """Walk sklearn `_combinations`, emitting names or bincount rows."""
        var p = self.n_features_in_
        var with_replacement = not self._interaction_only
        if self._include_bias:
            if want_names:
                names.append("1")
            else:
                for _ in range(p):
                    powers.append(0)
        for degree in range(max(1, self._min_degree), self._max_degree + 1):
            if not with_replacement and degree > p:
                break
            var combo = List[Int]()
            for k in range(degree):
                combo.append(0 if with_replacement else k)
            while True:
                if want_names:
                    var name = String("")
                    var k = 0
                    while k < degree:
                        var feature = combo[k]
                        var exponent = 1
                        while k + exponent < degree and combo[k + exponent] == feature:
                            exponent += 1
                        if name.byte_length() > 0:
                            name += " "
                        name += input_names[feature]
                        if exponent != 1:
                            name += "^" + String(exponent)
                        k += exponent
                    names.append(name^)
                else:
                    var base = len(powers)
                    for _ in range(p):
                        powers.append(0)
                    for k in range(degree):
                        powers[base + combo[k]] += 1
                if not _next_combination(combo, p, with_replacement):
                    break

    def powers_(self) raises -> NDArray[DType.int64]:
        """sklearn `powers_`: (n_output_features_, n_features_in_) Int64 exponents.

        Computed on each call, as sklearn's property is, so the fitted state
        stays O(degree * n_features).
        """
        check_fitted(self._fitted, "PolynomialFeatures")
        var names = List[String]()
        var powers = List[Int64]()
        self._enumerate(names, powers, List[String](), False)
        var p = self.n_features_in_
        var result = matrix[DType.int64](len(powers) // p, p)
        for k in range(len(powers)):
            result.unsafe_set(k, powers[k])
        return result^

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        """Names such as `x0^2 x1`; `1` for the bias column."""
        check_fitted(self._fitted, "PolynomialFeatures")
        var inputs = feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)
        var names = List[String]()
        var powers = List[Int64]()
        self._enumerate(names, powers, inputs, True)
        return names^


# ---------------------------------------------------------------------------
# Exact float helpers for knot placement and periodic mapping


def _pow2(exponent: Int) -> Float64:
    """2**exponent for -1074 <= exponent <= 1023, exactly."""
    if exponent >= -1022:
        return bitcast[DType.float64](UInt64(exponent + 1023) << 52)
    return bitcast[DType.float64](UInt64(1) << UInt64(exponent + 1074))


def _split(x: Float64) -> Tuple[UInt64, Int]:
    """|x| = significand * 2**exponent for finite nonzero x."""
    var bits = bitcast[DType.uint64](x)
    var biased = Int((bits >> 52) & 0x7FF)
    var fraction = bits & 0xFFFFFFFFFFFFF
    if biased == 0:
        return (fraction, -1074)
    return (fraction | (UInt64(1) << 52), biased - 1075)


def _fmod(a: Float64, b: Float64) -> Float64:
    """C `fmod`: the exact remainder with the sign of `a` (binary long division).

    With |a| = sa * 2**ea >= |b| = sb * 2**eb (so ea >= eb), the remainder is
    (sa * 2**(ea - eb) mod sb) * 2**eb, accumulated one doubling at a time.
    """
    if isnan(a) or isnan(b) or not isfinite(a) or b == 0:
        return nan[DType.float64]()
    if not isfinite(b) or a == 0 or abs(a) < abs(b):
        return a
    var ua = _split(a)
    var ub = _split(b)
    var r = ua[0] % ub[0]
    for _ in range(ua[1] - ub[1]):
        r = (r << 1) % ub[0]
    if r == 0:
        return 0.0 if a > 0 else -0.0
    var magnitude = Float64(r) * _pow2(ub[1])
    return magnitude if a > 0 else -magnitude


def _remainder(a: Float64, b: Float64) -> Float64:
    """NumPy `np.remainder` for float64 (Python `%`): result has the sign of b."""
    var mod = _fmod(a, b)
    if b == 0:
        return mod
    if mod != 0:
        if (b < 0) != (mod < 0):
            mod += b
    else:
        mod = 0.0 if b > 0 else -0.0
    return mod


def _percentile_ranks(n_knots: Int) -> List[Float64]:
    """100 * np.linspace(0, 1, n_knots)."""
    var zero: List[Float64] = [0.0]
    var one: List[Float64] = [1.0]
    var rows = _linspace[DType.float64](zero, one, n_knots)
    var ranks = List[Float64]()
    for i in range(n_knots):
        ranks.append(100 * rows[i][0])
    return ranks^


def _sift(mut values: List[Float64], root: Int, size: Int):
    var r = root
    while True:
        var child = 2 * r + 1
        if child >= size:
            return
        if child + 1 < size and values[child] < values[child + 1]:
            child += 1
        if values[r] < values[child]:
            var tmp = values[r]
            values[r] = values[child]
            values[child] = tmp
            r = child
        else:
            return


def _sort_floats(mut values: List[Float64]):
    """Ascending heap sort (values contain no NaN)."""
    var n = len(values)
    var start = n // 2 - 1
    while start >= 0:
        _sift(values, start, n)
        start -= 1
    var end = n - 1
    while end > 0:
        var tmp = values[0]
        values[0] = values[end]
        values[end] = tmp
        _sift(values, 0, end)
        end -= 1


def _nanpercentile_column[d: DType](X: NDArray[d], feature: Int, ranks: List[Float64]) raises -> List[Float64]:
    """np.nanpercentile(X[:, feature], ranks) with NumPy's linear `_lerp`.

    The difference `next - previous` is taken in X's dtype, as NumPy does.
    """
    var values = List[Float64]()
    for i in range(X.shape[0]):
        var v = at(X, i, feature)
        if not isnan(v):
            values.append(Float64(v))
    _sort_floats(values)
    var n = len(values)
    var result = List[Float64]()
    for rank in ranks:
        if n == 0:
            result.append(nan[DType.float64]())
            continue
        var q = rank / 100.0
        var virtual = Float64(n - 1) * q
        if virtual >= Float64(n - 1):
            result.append(values[n - 1])
            continue
        var previous = floor(virtual)
        var gamma = virtual - previous
        var a = values[Int(previous)]
        var b = values[Int(previous) + 1]
        var diff = Float64(Scalar[d](b) - Scalar[d](a))
        if gamma >= 0.5:
            result.append(b - diff * (1 - gamma))
        else:
            result.append(a + diff * gamma)
    return result^


def _weighted_quantile_knots[d: DType](X: NDArray[d], n_knots: Int, weights: List[Scalar[d]]) raises -> List[List[Scalar[d]]]:
    """`_weighted_percentile(X, sample_weight, ranks).T` in X's dtype."""
    var p = X.shape[1]
    var ranks = _percentile_ranks(n_knots)
    var rows = List[List[Scalar[d]]]()
    for _ in range(n_knots):
        rows.append(List[Scalar[d]](length=p, fill=0))
    for j in range(p):
        var column = _weighted_percentile_column(X, j, weights, ranks)
        for i in range(n_knots):
            rows[i][j] = column[i]
    return rows^


def _check_weights_nonzero[d: DType](weights: List[Scalar[d]]) raises:
    for w in weights:
        if w != 0:
            return
    raise Error("ValueError: Sample weights must contain at least one non-zero number.")


# ---------------------------------------------------------------------------
# B-spline basis (SciPy BSpline with an identity-like coefficient matrix)


def _find_interval(t: NDArray[DType.float64], k: Int, x: Float64, prev_l: Int, extrapolate: Bool) raises -> Int:
    """SciPy `_find_interval`: t[l] <= x < t[l+1], or -1 for NaN / out of range."""
    var n = t.shape[0] - k - 1
    var tb = t.unsafe_get(k)
    var te = t.unsafe_get(n)
    if x != x:
        return -1
    if (x < tb or x > te) and not extrapolate:
        return -1
    var l = prev_l if (k < prev_l and prev_l < n) else k
    while x < t.unsafe_get(l) and l != k:
        l -= 1
    l += 1
    while x >= t.unsafe_get(l) and l != n:
        l += 1
    return l - 1


def _de_boor(t: NDArray[DType.float64], x: Float64, k: Int, ell: Int, m: Int, mut h: List[Float64]):
    """SciPy `_deBoor_D`: h[0..k] = the k+1 nonzero B-splines (m-th derivative) at x.

    `h` has length 2k+2; its upper half is SciPy's scratch `hh`.
    """
    var hh = k + 1
    h[0] = 1.0
    for j in range(1, k - m + 1):
        for q in range(j):
            h[hh + q] = h[q]
        h[0] = 0.0
        for n in range(1, j + 1):
            var ind = ell + n
            var xb = t.unsafe_get(ind)
            var xa = t.unsafe_get(ind - j)
            if xb == xa:
                h[n] = 0.0
                continue
            var w = h[hh + n - 1] / (xb - xa)
            h[n - 1] += w * (xb - x)
            h[n] = w * (x - xa)
    for j in range(k - m + 1, k + 1):
        for q in range(j):
            h[hh + q] = h[q]
        h[0] = 0.0
        for n in range(1, j + 1):
            var ind = ell + n
            var xb = t.unsafe_get(ind)
            var xa = t.unsafe_get(ind - j)
            if xb == xa:
                h[n] = 0.0
                continue
            var w = Float64(j) * h[hh + n - 1] / (xb - xa)
            h[n - 1] -= w
            h[n] = w


struct BSpline(Copyable, Movable):
    """Fitted per-feature spline basis: the native view of a SciPy `BSpline`.

    `t` holds the knots (n_knots + 2 * k of them) and `k` the degree. sklearn's
    coefficient matrix `c` is the identity of size `n_splines` (periodic splines
    append its first `k` rows); `coefficients()` materializes it on demand.
    """
    var t: NDArray[DType.float64]
    var k: Int
    var extrapolate: Bool
    var periodic: Bool

    def __init__(out self, var t: NDArray[DType.float64], k: Int, extrapolate: Bool, periodic: Bool):
        self.t = t^
        self.k = k
        self.extrapolate = extrapolate
        self.periodic = periodic

    def n_splines(self) raises -> Int:
        """`c.shape[1]`: the number of basis functions."""
        var n = self.t.shape[0] - self.k - 1
        return n - self.k if self.periodic else n

    def coefficients(self) raises -> NDArray[DType.float64]:
        """sklearn's coefficient matrix `c`, shape (n_knots + k - 1, n_splines)."""
        var cols = self.n_splines()
        var rows = self.t.shape[0] - self.k - 1
        var c = matrix[DType.float64](rows, cols)
        for r in range(rows):
            c.unsafe_set(r * cols + (r if r < cols else r - cols), 1.0)
        return c^

    def _row(self, x: Float64, nu: Int, extrapolate: Bool, mut interval: Int, mut work: List[Float64], mut values: List[Float64]) raises:
        """SciPy `_evaluate_spline` for one x into values[0..n_splines)."""
        var cols = self.n_splines()
        interval = _find_interval(self.t, self.k, x, interval, extrapolate)
        if interval < 0:
            for c in range(cols):
                values[c] = nan[DType.float64]()
            return
        _de_boor(self.t, x, self.k, interval, nu, work)
        var finite = True
        for a in range(self.k + 1):
            if not isfinite(work[a]):
                finite = False
        for c in range(cols):
            values[c] = 0.0
        if finite:
            for a in range(self.k + 1):
                var r = interval + a - self.k
                values[r if r < cols else r - cols] += work[a]
            return
        # Non-finite basis values: SciPy's 0 * value terms reach every column.
        for c in range(cols):
            var s = Float64(0)
            for a in range(self.k + 1):
                var r = interval + a - self.k
                var coef = Float64(1) if (r if r < cols else r - cols) == c else Float64(0)
                s += coef * work[a]
            values[c] = s

    def evaluate(self, x: Float64, nu: Int = 0) raises -> NDArray[DType.float64]:
        """`spl(x, nu)` with SciPy semantics: all basis values (or derivatives) at x.

        NaN is returned outside the base interval unless `extrapolate`; a
        periodic spline evaluates x as given (SplineTransformer maps it first).
        """
        if nu < 0:
            raise Error("ValueError: Cannot do derivative order nu= " + String(nu))
        var cols = self.n_splines()
        var result = vector[DType.float64](cols)
        if nu > self.k + 1:
            return result^
        var work = List[Float64](length=2 * self.k + 2, fill=0)
        var values = List[Float64](length=cols, fill=0)
        var interval = self.k
        self._row(x, nu, self.extrapolate, interval, work, values)
        for c in range(cols):
            result.unsafe_set(c, values[c])
        return result^


# ---------------------------------------------------------------------------
# SplineTransformer


@fieldwise_init
struct SplineTransformerParams(Copyable, Movable):
    var n_knots: Int
    var degree: Int
    var knots: String
    var knot_positions: Optional[NDArray[DType.float64]]
    var extrapolation: String
    var include_bias: Bool
    var order: String
    var handle_missing: String
    var sparse_output: Bool


struct SplineTransformer[dtype: DType = DType.float64](Preprocessor):
    """Univariate B-spline bases for each feature (dense output only).

    `knots` is `"uniform"` or `"quantile"`; explicit knot positions are the
    `knot_positions` field, set by the constructor overload taking an
    `NDArray[float64]` of shape (n_knots, n_features). `bsplines_` holds one
    `BSpline` (knots, degree) per feature.
    """
    comptime Params = SplineTransformerParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    var n_knots: Int
    var degree: Int
    var knots: String
    var knot_positions: Optional[NDArray[DType.float64]]
    var extrapolation: String
    var include_bias: Bool
    var order: String
    var handle_missing: String
    var sparse_output: Bool
    var bsplines_: List[BSpline]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]
    var n_features_out_: Int
    var _fitted: Bool
    var _extrapolation: String
    var _include_bias: Bool
    var _order: String
    var _handle_missing: String

    def __init__(out self, n_knots: Int = 5, degree: Int = 3, *, knots: String = "uniform", extrapolation: String = "constant", include_bias: Bool = True, order: String = "C", handle_missing: String = "error", sparse_output: Bool = False):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "Float32/Float64 only"
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.knot_positions = None
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.order = order
        self.handle_missing = handle_missing
        self.sparse_output = sparse_output
        self.bsplines_ = List[BSpline]()
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self.n_features_out_ = 0
        self._fitted = False
        self._extrapolation = "constant"
        self._include_bias = True
        self._order = "C"
        self._handle_missing = "error"

    def __init__(out self, n_knots: Int = 5, degree: Int = 3, *, knots: NDArray[DType.float64], extrapolation: String = "constant", include_bias: Bool = True, order: String = "C", handle_missing: String = "error", sparse_output: Bool = False):
        """Explicit sorted knot positions, shape (n_knots, n_features); `n_knots` is ignored."""
        self = Self(n_knots, degree, extrapolation=extrapolation, include_bias=include_bias, order=order, handle_missing=handle_missing, sparse_output=sparse_output)
        self.knot_positions = knots.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.n_knots, self.degree, self.knots, self.knot_positions.copy(), self.extrapolation, self.include_bias, self.order, self.handle_missing, self.sparse_output)

    def set_params(mut self, params: Self.Params) raises:
        self.n_knots = params.n_knots
        self.degree = params.degree
        self.knots = params.knots
        self.knot_positions = params.knot_positions.copy()
        self.extrapolation = params.extrapolation
        self.include_bias = params.include_bias
        self.order = params.order
        self.handle_missing = params.handle_missing
        self.sparse_output = params.sparse_output

    def set_params(
        mut self, *, n_knots: Optional[Int] = None, degree: Optional[Int] = None,
        knots: Optional[String] = None, extrapolation: Optional[String] = None,
        include_bias: Optional[Bool] = None, order: Optional[String] = None,
        handle_missing: Optional[String] = None, sparse_output: Optional[Bool] = None,
    ) raises:
        """Keyword updates; a string `knots` replaces explicit knot positions."""
        if n_knots:
            self.n_knots = n_knots.value()
        if degree:
            self.degree = degree.value()
        if knots:
            self.knots = knots.value()
            self.knot_positions = None
        if extrapolation:
            self.extrapolation = extrapolation.value()
        if include_bias:
            self.include_bias = include_bias.value()
        if order:
            self.order = order.value()
        if handle_missing:
            self.handle_missing = handle_missing.value()
        if sparse_output:
            self.sparse_output = sparse_output.value()

    def set_params(mut self, *, knots: NDArray[DType.float64]) raises:
        """sklearn `set_params(knots=array)`."""
        self.knot_positions = knots.copy()

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        check_set_output(transform)

    # -- fitting ------------------------------------------------------------

    def _validate_params(self) raises:
        if self.n_knots < 2:
            raise Error("ValueError: The 'n_knots' parameter of SplineTransformer must be an int in the range [2, inf). Got " + String(self.n_knots) + " instead.")
        if self.degree < 0:
            raise Error("ValueError: The 'degree' parameter of SplineTransformer must be an int in the range [0, inf). Got " + String(self.degree) + " instead.")
        if not self.knot_positions and self.knots != "uniform" and self.knots != "quantile":
            raise Error("ValueError: The 'knots' parameter of SplineTransformer must be a str among {'quantile', 'uniform'} or an array-like. Got '" + self.knots + "' instead.")
        var e = self.extrapolation
        if e != "error" and e != "constant" and e != "linear" and e != "continue" and e != "periodic":
            raise Error("ValueError: The 'extrapolation' parameter of SplineTransformer must be a str among {'constant', 'continue', 'error', 'linear', 'periodic'}. Got '" + e + "' instead.")
        _check_order(self.order, "SplineTransformer")
        if self.handle_missing != "error" and self.handle_missing != "zeros":
            raise Error("ValueError: The 'handle_missing' parameter of SplineTransformer must be a str among {'error', 'zeros'}. Got '" + self.handle_missing + "' instead.")
        require_dense_output(self.sparse_output)

    def _check_X(self, X: Self.Input, min_samples: Int, handle_missing: String, fitting: Bool) raises:
        _check_shape(X, min_samples, "SplineTransformer")
        if handle_missing == "zeros":
            return  # sklearn: ensure_all_finite=False (NaN and infinity pass)
        if fitting:
            _check_finite(X, "ValueError: Input X contains NaN values and `SplineTransformer` is configured to error in this case (handle_missing='error'). To avoid this error, set handle_missing='zeros' to encode missing values as splines with value 0 or ensure no missing values in X.", "X ")
        else:
            _check_finite(X, "ValueError: Input X contains NaN.", "X ")

    def _base_knots(self, X: Self.Input, weights: Optional[List[Scalar[Self.dtype]]]) raises -> List[List[Float64]]:
        """Float64 base knots (n_knots rows): explicit, uniform or unweighted quantile."""
        var p = X.shape[1]
        var rows = List[List[Float64]]()
        if self.knot_positions:
            ref K = self.knot_positions.value()
            if K.ndim != 2:
                raise Error("ValueError: Expected 2D array, got " + String(K.ndim) + "D array instead")
            if K.shape[0] < 1:
                raise Error("ValueError: Found array with 0 sample(s) (shape=(0, " + String(K.shape[1]) + ")) while a minimum of 1 is required.")
            if K.shape[1] < 1:
                raise Error("ValueError: Found array with 0 feature(s) (shape=(" + String(K.shape[0]) + ", 0)) while a minimum of 1 is required.")
            _check_finite(K, "ValueError: Input contains NaN.", "")
            if K.shape[0] < 2:
                raise Error("ValueError: Number of knots, knots.shape[0], must be >= 2.")
            if K.shape[1] != p:
                raise Error("ValueError: knots.shape[1] == n_features is violated.")
            for i in range(1, K.shape[0]):
                for j in range(p):
                    if not (at(K, i, j) - at(K, i - 1, j) > 0):
                        raise Error("ValueError: knots must be sorted without duplicates.")
            for i in range(K.shape[0]):
                var row = List[Float64]()
                for j in range(p):
                    row.append(at(K, i, j))
                rows.append(row^)
            return rows^
        if self.knots == "quantile":
            var ranks = _percentile_ranks(self.n_knots)
            for _ in range(self.n_knots):
                rows.append(List[Float64](length=p, fill=0))
            for j in range(p):
                var column = _nanpercentile_column(X, j, ranks)
                for i in range(self.n_knots):
                    rows[i][j] = column[i]
            return rows^
        # uniform: nanmin/nanmax over positive-weight rows; all-missing -> 0.
        var lo = List[Float64](length=p, fill=0)
        var hi = List[Float64](length=p, fill=0)
        for j in range(p):
            var seen = False
            for i in range(X.shape[0]):
                if weights and not (weights.value()[i] > 0):
                    continue
                var v = Float64(at(X, i, j))
                if isnan(v):
                    continue
                if not seen:
                    lo[j] = v
                    hi[j] = v
                    seen = True
                else:
                    if v < lo[j]:
                        lo[j] = v
                    if v > hi[j]:
                        hi[j] = v
        return _linspace[DType.float64](lo, hi, self.n_knots)

    def _splines[kd: DType](self, base: List[List[Scalar[kd]]], p: Int) raises -> List[BSpline]:
        """Extend base knots by `degree` knots on each side (sklearn fit), per feature."""
        var n_knots = len(base)
        var degree = self.degree
        var periodic = self.extrapolation == "periodic"
        if periodic and n_knots <= degree:
            raise Error("ValueError: Periodic splines require degree < n_knots. Got n_knots=" + String(n_knots) + " and degree=" + String(degree) + ".")
        var rows = List[List[Scalar[kd]]]()
        if periodic:
            var period = List[Scalar[kd]]()
            for j in range(p):
                period.append(base[n_knots - 1][j] - base[0][j])
            for i in range(n_knots - degree - 1, n_knots - 1):
                var row = List[Scalar[kd]]()
                for j in range(p):
                    row.append(base[i][j] - period[j])
                rows.append(row^)
            for i in range(n_knots):
                rows.append(base[i].copy())
            for i in range(1, degree + 1):
                var row = List[Scalar[kd]]()
                for j in range(p):
                    row.append(base[i][j] + period[j])
                rows.append(row^)
        else:
            var left_start = List[Scalar[kd]]()
            var left_stop = List[Scalar[kd]]()
            var right_start = List[Scalar[kd]]()
            var right_stop = List[Scalar[kd]]()
            for j in range(p):
                var dist_min = base[1][j] - base[0][j]
                var dist_max = base[n_knots - 1][j] - base[n_knots - 2][j]
                left_start.append(base[0][j] - Scalar[kd](degree) * dist_min)
                left_stop.append(base[0][j] - dist_min)
                right_start.append(base[n_knots - 1][j] + dist_max)
                right_stop.append(base[n_knots - 1][j] + Scalar[kd](degree) * dist_max)
            rows = _linspace[kd](left_start, left_stop, degree)
            for i in range(n_knots):
                rows.append(base[i].copy())
            var right = _linspace[kd](right_start, right_stop, degree)
            for i in range(len(right)):
                rows.append(right[i].copy())
        var extrapolate = periodic or self.extrapolation == "continue"
        var splines = List[BSpline]()
        var n_t = len(rows)
        for j in range(p):
            var t = vector[DType.float64](n_t)
            for i in range(n_t):
                t.unsafe_set(i, Float64(rows[i][j]))
            splines.append(BSpline(t^, degree, extrapolate, periodic))
        return splines^

    def _fit(mut self, X: Self.Input, weights: Optional[List[Scalar[Self.dtype]]]) raises:
        var p = X.shape[1]
        var splines: List[BSpline]
        if weights and not self.knot_positions and self.knots == "quantile":
            # sklearn keeps _weighted_percentile's dtype (X's) for the knot arithmetic.
            var base = _weighted_quantile_knots[Self.dtype](X, self.n_knots, weights.value())
            splines = self._splines[Self.dtype](base, p)
        else:
            var base = self._base_knots(X, weights)
            splines = self._splines[DType.float64](base, p)
        var n_splines = splines[0].n_splines()
        self.bsplines_ = splines^
        self.n_features_out_ = p * n_splines - p * (0 if self.include_bias else 1)
        self._extrapolation = self.extrapolation
        self._include_bias = self.include_bias
        self._order = self.order
        self._handle_missing = self.handle_missing
        self.n_features_in_ = p
        self.feature_names_in_ = None
        self._fitted = True

    def _weights(self, X: Self.Input, sample_weight: NDArray[Self.dtype]) raises -> List[Scalar[Self.dtype]]:
        """sklearn `_check_sample_weight`: 1-D, matching length, finite, not all zero."""
        if sample_weight.ndim != 1:
            raise Error("ValueError: Sample weights must be 1D array or scalar, got " + String(sample_weight.ndim) + "D array. Expected either a scalar value or a 1D array of length " + String(X.shape[0]) + ".")
        if sample_weight.shape[0] != X.shape[0]:
            raise Error("ValueError: sample_weight.shape == (" + String(sample_weight.shape[0]) + ",), expected (" + String(X.shape[0]) + ",)!")
        var weights = List[Scalar[Self.dtype]]()
        for i in range(X.shape[0]):
            var w = at1(sample_weight, i)
            if not isfinite(w):
                if isnan(w):
                    raise Error("ValueError: Input sample_weight contains NaN.")
                raise Error("ValueError: Input sample_weight contains infinity or a value too large for dtype('" + dtype_name[Self.dtype]() + "').")
            weights.append(w)
        _check_weights_nonzero(weights)
        return weights^

    def _start(self, X: Self.Input) raises:
        self._validate_params()
        self._check_X(X, 2, self.handle_missing, True)

    def fit(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises:
        self._start(X)
        self._fit(X, None)

    def fit(mut self, X: Self.Input, y: Optional[Self.Target], sample_weight: NDArray[Self.dtype]) raises:
        self._start(X)
        var weights = self._weights(X, sample_weight)
        self._fit(X, weights^)

    def fit(mut self, X: Self.Input, *, sample_weight: NDArray[Self.dtype]) raises:
        self.fit(X, None, sample_weight)

    def fit(mut self, X: Self.Input, *, sample_weight: Float64) raises:
        """Scalar weight, broadcast as sklearn `_check_sample_weight` does."""
        self._start(X)
        var weights = List[Scalar[Self.dtype]](length=X.shape[0], fill=Scalar[Self.dtype](sample_weight))
        _check_weights_nonzero(weights)
        self._fit(X, weights^)

    def fit_named(mut self, X: Self.Input, feature_names: List[String], y: Optional[Self.Target] = None) raises:
        _check_shape(X, 2, "SplineTransformer")
        check_names_length(feature_names, X.shape[1])
        self.fit(X, y)
        self.feature_names_in_ = feature_names.copy()

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.fit(X, y)
        return self.transform(X)

    def fit_transform(mut self, X: Self.Input, *, sample_weight: NDArray[Self.dtype]) raises -> Self.Transformed:
        self.fit(X, None, sample_weight)
        return self.transform(X)

    # -- transformation -------------------------------------------------------

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """Spline values are computed in Float64 and stored in X's dtype."""
        check_fitted(self._fitted, "SplineTransformer")
        self._check_X(X, 1, self._handle_missing, False)
        check_n_features("SplineTransformer", self.n_features_in_, X.shape[1])
        var n = X.shape[0]
        var p = X.shape[1]
        var n_splines = self.bsplines_[0].n_splines()
        var keep = n_splines - (0 if self._include_bias else 1)
        var out = _output[Self.dtype](n, p * keep, self._order)
        var degree = self.bsplines_[0].k
        var work = List[Float64](length=2 * degree + 2, fill=0)
        var values = List[Float64](length=n_splines, fill=0)
        var mode = self._extrapolation
        for j in range(p):
            ref spl = self.bsplines_[j]
            var base = j * keep
            var interval = degree
            if mode == "continue" or mode == "error" or mode == "periodic":
                var m = spl.t.shape[0] - degree - 1
                var t_k = spl.t.unsafe_get(degree)
                var period = spl.t.unsafe_get(m) - t_k
                for i in range(n):
                    var raw = Float64(at(X, i, j))
                    var x = raw
                    if mode == "periodic":
                        x = t_k + _remainder(raw - t_k, period) if period > 0 else 0.0
                    spl._row(x, 0, spl.extrapolate, interval, work, values)
                    if isnan(raw):
                        for c in range(n_splines):
                            values[c] = 0.0
                    if mode == "error":
                        for c in range(n_splines):
                            if isnan(values[c]):
                                raise Error("ValueError: `X` contains values beyond the limits of the knots.")
                    for c in range(keep):
                        put(out, i, base + c, Scalar[Self.dtype](values[c]))
                continue
            # constant / linear: evaluate inside [xmin, xmax], extrapolate outside.
            var xmin = spl.t.unsafe_get(degree)
            var xmax = spl.t.unsafe_get(spl.t.shape[0] - degree - 1)
            var f_min = spl.evaluate(xmin)
            var f_max = spl.evaluate(xmax)
            var linear = mode == "linear"
            var fp_min = spl.evaluate(xmin, 1) if linear else vector[DType.float64](0)
            var fp_max = spl.evaluate(xmax, 1) if linear else vector[DType.float64](0)
            var span = degree + 1 if degree <= 1 else degree
            for i in range(n):
                var x = Float64(at(X, i, j))
                for c in range(n_splines):
                    values[c] = 0.0
                if xmin <= x and x <= xmax:
                    spl._row(x, 0, False, interval, work, values)
                elif linear:
                    for q in range(span):
                        if x < xmin:
                            values[q] = f_min.unsafe_get(q) + (x - xmin) * fp_min.unsafe_get(q)
                        if x > xmax:
                            var c = n_splines - 1 - q
                            values[c] = f_max.unsafe_get(c) + (x - xmax) * fp_max.unsafe_get(c)
                else:
                    if x < xmin:
                        for q in range(degree):
                            values[q] = f_min.unsafe_get(q)
                    if x > xmax:
                        for q in range(n_splines - degree, n_splines):
                            values[q] = f_max.unsafe_get(q)
                for c in range(keep):
                    put(out, i, base + c, Scalar[Self.dtype](values[c]))
        return out^

    # -- metadata -----------------------------------------------------------

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        """`<feature>_sp_<j>` for each kept spline of each feature."""
        check_fitted(self._fitted, "SplineTransformer")
        var inputs = feature_names_in(self.n_features_in_, self.feature_names_in_, input_features)
        var n_splines = self.bsplines_[0].n_splines()
        var keep = n_splines - (0 if self._include_bias else 1)
        var names = List[String]()
        for i in range(self.n_features_in_):
            for j in range(keep):
                names.append(inputs[i] + "_sp_" + String(j))
        return names^
