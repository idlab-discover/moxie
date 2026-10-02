# Ports sklearn sklearn/utils/stats.py `_weighted_percentile` (authors: the
# scikit-learn developers; BSD-3-Clause, LICENSES/sklearn.txt) and NumPy 2.5
# `linspace` and `percentile`/`quantile` methods from numpy/_core/function_base.py
# and numpy/lib/_function_base_impl.py (Copyright (c) 2005-2025, NumPy
# Developers; BSD-3-Clause, LICENSES/numpy.txt).
# SPDX-License-Identifier: BSD-3-Clause
"""NumPy-exact linspace and percentiles shared by the preprocessing family.

Consumers: SplineTransformer (`_polynomial`) and KBinsDiscretizer
(`_discretization`). Every helper reproduces the reference operation order and
precision, so results match NumPy/sklearn bit for bit on the locked toolchain.
"""
from std.math import isnan, floor
from std.memory import bitcast
from numojo import NDArray
from moxie.preprocessing._common import at


@always_inline
def min_subnormal[d: DType]() -> Scalar[d]:
    """np.nextafter(0, 1) in dtype d."""
    comptime if d == DType.float32:
        return rebind[Scalar[d]](bitcast[DType.float32](UInt32(1)))
    else:
        return rebind[Scalar[d]](bitcast[DType.float64](UInt64(1)))


@always_inline
def machine_eps[d: DType]() -> Scalar[d]:
    """np.finfo(d).eps."""
    comptime if d == DType.float32:
        return Scalar[d](1.1920928955078125e-07)
    else:
        return Scalar[d](2.220446049250313e-16)


def linspace[kd: DType](start: List[Scalar[kd]], stop: List[Scalar[kd]], num: Int) -> List[List[Scalar[kd]]]:
    """np.linspace(start_vector, stop_vector, num) rows, including its step logic.

    NumPy switches every column to `(i / div) * delta` when any column's step is
    zero; this is reproduced because it changes the rounding of all columns.
    """
    var p = len(start)
    var div = num - 1
    var rows = List[List[Scalar[kd]]]()
    var delta = List[Scalar[kd]]()
    var step = List[Scalar[kd]]()
    var any_zero = False
    for j in range(p):
        delta.append(stop[j] - start[j])
        if div > 0:
            step.append(delta[j] / Scalar[kd](div))
            if step[j] == 0:
                any_zero = True
    for i in range(num):
        var row = List[Scalar[kd]]()
        for j in range(p):
            var y: Scalar[kd]
            if div > 0:
                if any_zero:
                    y = (Scalar[kd](i) / Scalar[kd](div)) * delta[j]
                else:
                    y = Scalar[kd](i) * step[j]
            else:
                y = Scalar[kd](i) * delta[j]
            row.append(y + start[j])
        rows.append(row^)
    if num > 1:
        for j in range(p):
            rows[num - 1][j] = stop[j]
    return rows^


def linspace_scalar[kd: DType](start: Scalar[kd], stop: Scalar[kd], num: Int) -> List[Scalar[kd]]:
    """np.linspace(start, stop, num) for scalar bounds, evaluated in kd."""
    var lo: List[Scalar[kd]] = [start]
    var hi: List[Scalar[kd]] = [stop]
    var rows = linspace[kd](lo, hi, num)
    var result = List[Scalar[kd]](capacity=num)
    for i in range(num):
        result.append(rows[i][0])
    return result^


def sort_indices[d: DType](X: NDArray[d], feature: Int, mut order: List[Int]):
    """Stable bottom-up merge sort of row indices by X[:, feature], NaN last."""
    var n = len(order)
    var scratch = List[Int](length=n, fill=0)
    var width = 1
    while width < n:
        var left = 0
        while left < n:
            var mid = min(left + width, n)
            var right = min(left + 2 * width, n)
            var a = left
            var b = mid
            var k = left
            while a < mid and b < right:
                var va = at(X, order[a], feature)
                var vb = at(X, order[b], feature)
                if (not isnan(vb)) and (isnan(va) or vb < va):
                    scratch[k] = order[b]
                    b += 1
                else:
                    scratch[k] = order[a]
                    a += 1
                k += 1
            while a < mid:
                scratch[k] = order[a]
                a += 1
                k += 1
            while b < right:
                scratch[k] = order[b]
                b += 1
                k += 1
            left += 2 * width
        for i in range(n):
            order[i] = scratch[i]
        width *= 2


def _search[d: DType](cdf: List[Scalar[d]], key: Scalar[d], right: Bool) -> Int:
    """NumPy's single-key `searchsorted` binary search (side left or right)."""
    var lo = 0
    var hi = len(cdf)
    while lo < hi:
        var mid = lo + ((hi - lo) >> 1)
        var go_right = cdf[mid] <= key if right else cdf[mid] < key
        if go_right:
            lo = mid + 1
        else:
            hi = mid
    return lo


def weighted_percentile_column[d: DType](
    X: NDArray[d], feature: Int, weights: List[Scalar[d]], ranks: List[Float64], average: Bool = False,
) raises -> List[Scalar[d]]:
    """sklearn `_weighted_percentile(X[:, feature], weights, ranks, average)` in X's dtype.

    `average=False` is "inverted_cdf", `average=True` "averaged_inverted_cdf".
    NaN values sort last with zero weight; cumulative weights are summed
    sequentially and searched with NumPy's binary search. Equal values keep
    their row order (stable); NumPy's unstable argsort may order ties
    differently, which can change the rounding of the cumulative weights.
    """
    var n = X.shape[0]
    var order = List[Int]()
    for i in range(n):
        order.append(i)
    sort_indices(X, feature, order)
    var sorted_weights = List[Scalar[d]](capacity=n)
    var cdf = List[Scalar[d]](capacity=n)
    var running = Scalar[d](0)
    for k in range(n):
        var i = order[k]
        var w = weights[i]
        if isnan(at(X, i, feature)):
            w = 0
        sorted_weights.append(w)
        running = running + w if k > 0 else w
        cdf.append(running)
    var total = cdf[n - 1]
    var result = List[Scalar[d]]()
    for rank in ranks:
        var adjusted = Scalar[d](rank) / Scalar[d](100) * total
        if adjusted == 0:
            adjusted = min_subnormal[d]()
        var idx = min(_search(cdf, adjusted, False), n - 1)
        var value = at(X, order[idx], feature)
        if average:
            var above = (cdf[idx] - adjusted) > machine_eps[d]()
            var plus = min(idx + 1, n - 1)
            if sorted_weights[plus] == 0:
                # Next index with a larger cumulative weight (skip zero weights).
                var following = _search(cdf, cdf[idx], True)
                plus = idx if following > n - 1 else following
            if not above:
                value = (value + at(X, order[plus], feature)) / Scalar[d](2)
        result.append(value)
    return result^


# ---------------------------------------------------------------------------
# Unweighted np.percentile methods on a sorted workspace


def quantile_method_code(method: String) -> Int:
    """Index of a NumPy H&F percentile method KBinsDiscretizer accepts, or -1."""
    if method == "inverted_cdf":
        return 0
    if method == "averaged_inverted_cdf":
        return 1
    if method == "closest_observation":
        return 2
    if method == "interpolated_inverted_cdf":
        return 3
    if method == "hazen":
        return 4
    if method == "weibull":
        return 5
    if method == "linear":
        return 6
    if method == "median_unbiased":
        return 7
    if method == "normal_unbiased":
        return 8
    return -1


def _floor_mod2(x: Float64) -> Float64:
    """Python/NumPy floored `x % 2` for an integral float."""
    return x - 2.0 * floor(x / 2.0)


def percentile_sorted[dtype: DType](sorted: NDArray[DType.float64], size: Int, q: Float64, method: Int) -> Float64:
    """`np.percentile(a, 100 * q, method=...)` of `size >= 1` sorted dtype values.

    `sorted` holds the values exactly in Float64; `method` is a
    `quantile_method_code`. Discrete methods return an order statistic. The
    others follow `_get_indexes`/`_get_gamma`/`_lerp`: the neighbour difference
    is taken in `dtype`, the blend in Float64 (gamma uses the -1/0 sentinel
    indices exactly as NumPy subtracts them).
    """
    var nn = Float64(size)
    if method == 0 or method == 2:
        # _discrete_interpolation_to_boundaries
        var index = nn * q - 1.0
        if method == 2:
            index = index - 0.5
        var previous = floor(index)
        var gamma = index - previous
        var take_previous = gamma == 0.0
        if method == 2:
            take_previous = take_previous and _floor_mod2(previous) == 1.0
        var k = Int(previous) if take_previous else Int(previous) + 1
        return sorted.unsafe_get(max(k, 0))
    var virtual: Float64
    if method == 1:
        virtual = nn * q - 1.0
    elif method == 6:
        virtual = Float64(size - 1) * q
    else:
        var alpha: Float64
        var beta: Float64
        if method == 3:
            alpha = 0.0
            beta = 1.0
        elif method == 4:
            alpha = 0.5
            beta = 0.5
        elif method == 5:
            alpha = 0.0
            beta = 0.0
        elif method == 7:
            alpha = 1.0 / 3.0
            beta = 1.0 / 3.0
        else:
            alpha = 3.0 / 8.0
            beta = 3.0 / 8.0
        # _compute_virtual_index
        virtual = (nn * q + (alpha + q * ((1.0 - alpha) - beta))) - 1.0
    var previous = floor(virtual)
    var next = previous + 1.0
    if virtual >= Float64(size - 1):
        previous = -1.0
        next = -1.0
    if virtual < 0.0:
        previous = 0.0
        next = 0.0
    var gamma = virtual - previous
    if method == 1:
        gamma = 0.5 if gamma == 0.0 else 1.0
    var lower = Int(previous)
    var upper = Int(next)
    if lower < 0:
        lower += size
    if upper < 0:
        upper += size
    var a = sorted.unsafe_get(lower)
    var b = sorted.unsafe_get(upper)
    var diff = Float64(Scalar[dtype](b) - Scalar[dtype](a))
    if gamma >= 0.5:
        return b - diff * (1.0 - gamma)
    return a + diff * gamma
