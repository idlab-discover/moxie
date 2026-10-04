# Adapted from scikit-learn sklearn/covariance/_shrunk_covariance.py;
# authors: scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Ledoit-Wolf and Oracle Approximating Shrinkage (sklearn `LedoitWolf`, `OAS`).

Shared shrinkage statistics (`ShrinkageStatistics`, family contract) are
available to native engine consumers with widened accumulation:
trace = tr(S), squared_frobenius = sum S_ij^2, fourth_moment_sum =
sum_i (sum_j Z_ij^2)^2 for centered Z and S = Zᵀ Z / n. `block_size` only
partitions the Gram accumulation; it never changes the formula. Public Float32
estimators separately preserve the reference intermediate arithmetic.
"""
import numojo as nm
from numojo import NDArray, Shape
from numojo.routines.linalg import matmul
from std.sys import stderr
from moxie.covariance._base import CovarianceEstimator
from moxie.covariance._engines import ShrinkageEngine, ShrinkageStatistics
from moxie.covariance._empirical_covariance import (
    CovarianceModel, at, centered_columns, column_dot, column_means, shape_text, to_array, validate_samples,
)


# Float32 reductions follow NumPy's pairwise order (loops_utils.h.src, v2.4.0).
# Copyright (c) 2005-2025, NumPy Developers. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the NumPy Developers nor the names of any contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
def _sum32(x: NDArray[DType.float32], start: Int, count: Int, stride: Int = 1) -> Float32:
    if count < 8:
        var total = Float32(-0.0)
        for i in range(count):
            total += x.unsafe_get(start + i * stride)
        return total
    if count > 128:
        var half = (count // 2) // 8 * 8
        return _sum32(x, start, half, stride) + _sum32(x, start + half * stride, count - half, stride)
    var lanes = SIMD[DType.float32, 8](0)
    comptime for j in range(8):
        lanes[j] = x.unsafe_get(start + j * stride)
    var i = 8
    while i + 8 <= count:
        comptime for j in range(8):
            lanes[j] += x.unsafe_get(start + (i + j) * stride)
        i += 8
    var total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
    while i < count:
        total += x.unsafe_get(start + i * stride)
        i += 1
    return total


def _lw32(centered: NDArray[DType.float32], block_size: Int) raises -> Float64:
    """Keep NumPy scalar promotion and blocked moments in Float32."""
    var n = centered.shape[0]
    var p = centered.shape[1]
    if n == 1:
        print("UserWarning: Only one sample available. You may want to reshape your data array", file=stderr)
    var squared = centered * centered
    var diagonal = nm.zeros[DType.float32](Shape(p))
    for j in range(p):
        var total: Float32 = 0
        for i in range(n):
            total += squared.unsafe_get(i * p + j)
        diagonal.unsafe_set(j, total / Float32(n))
    var trace = Float64(_sum32(diagonal, 0, p))
    var mu = trace / Float64(p)
    var beta_sum: Float32 = 0
    var delta_sum: Float32 = 0
    # Full tiles followed by the trailing row/column, including empty tails.
    var splits = p // block_size
    for rb in range(splits + 1):
        var rows = min(block_size, p - rb * block_size)
        for cb in range(splits + 1):
            var cols = min(block_size, p - cb * block_size)
            if rows == 0 or cols == 0:
                continue
            var left = nm.zeros[DType.float32](Shape(rows, n))
            var right = nm.zeros[DType.float32](Shape(n, cols))
            for a in range(rows):
                for i in range(n):
                    left.unsafe_set(a * n + i, centered.unsafe_get(i * p + rb * block_size + a))
            for i in range(n):
                for b in range(cols):
                    right.unsafe_set(i * cols + b, centered.unsafe_get(i * p + cb * block_size + b))
            var product = matmul(left, right)
            var product_squared = product * product
            delta_sum += _sum32(product_squared, 0, rows * cols)
            var fourth = matmul(left * left, right * right)
            beta_sum += _sum32(fourth, 0, rows * cols)
    delta_sum /= Float32(n * n)
    var beta = Float32(1.0 / (Float64(p) * Float64(n))) * (beta_sum / Float32(n) - delta_sum)
    var delta = delta_sum - Float32(2.0 * mu * trace) + Float32(Float64(p) * (mu * mu))
    delta /= Float32(p)
    # Python min keeps its first argument for unordered comparisons.
    if delta < beta:
        beta = delta
    return 0.0 if beta == 0 else Float64(beta / delta)


def _estimate32(centered: NDArray[DType.float32], is_oas: Bool, block_size: Int = 1000) raises -> Tuple[NDArray[DType.float32], Float64]:
    var n = centered.shape[0]
    var p = centered.shape[1]
    var empirical = matmul(centered.T(), centered)
    for k in range(p * p):
        empirical.unsafe_set(k, empirical.unsafe_get(k) / Float32(n))
    var mu = Float64(_sum32(empirical, 0, p, p + 1)) / Float64(p)
    var shrinkage: Float64
    if is_oas:
        if n == 1:
            print("UserWarning: Only one sample available. You may want to reshape your data array", file=stderr)
        var squared = empirical * empirical
        var alpha = Float64(_sum32(squared, 0, p * p) / Float32(p * p))
        var mu_squared = mu * mu
        var denominator = Float64(n + 1) * (alpha - mu_squared / Float64(p))
        var ratio = (alpha + mu_squared) / denominator if denominator != 0 else Float64(1)
        shrinkage = 1.0 if ratio > 1.0 else ratio
    else:
        shrinkage = _lw32(centered, block_size)
    # LW's coefficient is a NumPy Float32 scalar; OAS's is a Python float.
    var scale = Float32(1) - Float32(shrinkage) if not is_oas else Float32(1.0 - shrinkage)
    var addition = Float32(shrinkage) * Float32(mu) if not is_oas else Float32(shrinkage * mu)
    for k in range(p * p):
        empirical.unsafe_set(k, scale * empirical.unsafe_get(k))
    for j in range(p):
        empirical.unsafe_set(j * p + j, empirical.unsafe_get(j * p + j) + addition)
    return (empirical^, shrinkage)


def _check_block_size(block_size: Int, owner: String) raises:
    if block_size < 1:
        raise Error(String("InvalidParameterError: The 'block_size' parameter of ", owner,
                           " must be an int in the range [1, inf). Got ", block_size, " instead."))


def shrinkage_statistics[dtype: DType](centered: NDArray[dtype], block_size: Int = 1000) raises -> ShrinkageStatistics[dtype]:
    """Shrinkage statistics of already-centered samples (n, p), Float64 accumulation.

    The Gram matrix is visited in block_size x block_size tiles in sklearn's
    order; the empirical covariance is returned in the input dtype.
    """
    _check_block_size(block_size, "shrinkage_statistics")
    var n = centered.shape[0]
    if n == 1:
        print("UserWarning: Only one sample available. You may want to reshape your data array", file=stderr)
    var p = centered.shape[1]
    var columns = centered_columns(centered, nm.zeros[DType.float64](Shape(p)))
    var empirical = nm.zeros[DType.float64](Shape(p, p))
    var squared_frobenius: Float64 = 0
    var starts = List[Int]()
    var start = 0
    while start < p:
        starts.append(start)
        start += block_size
    for rb in range(len(starts)):
        for cb in range(len(starts)):
            var block: Float64 = 0
            for a in range(starts[rb], min(starts[rb] + block_size, p)):
                for b in range(starts[cb], min(starts[cb] + block_size, p)):
                    var value = column_dot(columns, a, b, n)
                    empirical.unsafe_set(a * p + b, value / Float64(n))
                    block += value * value
            squared_frobenius += block
    squared_frobenius /= Float64(n) * Float64(n)
    var fourth: Float64 = 0
    var trace: Float64 = 0
    for i in range(n):
        var row: Float64 = 0
        for j in range(p):
            var z = columns.unsafe_get(j * n + i)
            row += z * z
        fourth += row * row
    for j in range(p):
        trace += empirical.unsafe_get(j * p + j)
    return ShrinkageStatistics[dtype](to_array[dtype](empirical), n, trace, squared_frobenius, fourth)


def ledoit_wolf_coefficient[dtype: DType](statistics: ShrinkageStatistics[dtype]) raises -> Float64:
    """Ledoit-Wolf shrinkage from the shared statistics (sklearn formula)."""
    var p = Float64(statistics.empirical.shape[0])
    var n = Float64(statistics.n_samples)
    var mu = statistics.trace / p
    var beta = 1.0 / (p * n) * (statistics.fourth_moment_sum / n - statistics.squared_frobenius)
    var delta = statistics.squared_frobenius - 2.0 * mu * statistics.trace + p * mu * mu
    delta /= p
    beta = min(beta, delta)
    if beta == 0:
        return 0.0
    return beta / delta


def oas_coefficient[dtype: DType](statistics: ShrinkageStatistics[dtype]) raises -> Float64:
    """Oracle Approximating Shrinkage coefficient (sklearn `_oas` formula)."""
    var p = Float64(statistics.empirical.shape[0])
    var n = Float64(statistics.n_samples)
    var alpha = statistics.squared_frobenius / (p * p)
    var mu = statistics.trace / p
    var mu_squared = mu * mu
    var numerator = alpha + mu_squared
    var denominator = (n + 1.0) * (alpha - mu_squared / p)
    if denominator == 0:
        return 1.0
    var ratio = numerator / denominator
    # Preserve NaN from overflowing moments; min(NaN, 1) selects 1 in Mojo.
    return 1.0 if ratio > 1.0 else ratio


struct AdaptiveShrinkageEngine[engine_dtype: DType](ShrinkageEngine):
    """Concrete family ShrinkageEngine used by LedoitWolf and OAS."""

    comptime dtype = Self.engine_dtype

    @staticmethod
    def statistics(centered: NDArray[Self.dtype], block_size: Int = 1000) raises -> ShrinkageStatistics[Self.dtype]:
        return shrinkage_statistics(centered, block_size)

    @staticmethod
    def ledoit_wolf(statistics: ShrinkageStatistics[Self.dtype]) raises -> Float64:
        return ledoit_wolf_coefficient(statistics)

    @staticmethod
    def oas(statistics: ShrinkageStatistics[Self.dtype]) raises -> Float64:
        return oas_coefficient(statistics)


def _shrink[dtype: DType](statistics: ShrinkageStatistics[dtype], shrinkage: Float64) raises -> NDArray[dtype]:
    """(1 - shrinkage) S + shrinkage * mu I with mu = tr(S)/p."""
    var p = statistics.empirical.shape[0]
    var mu = statistics.trace / Float64(p)
    var result = nm.zeros[dtype](Shape(p, p))
    for k in range(p * p):
        var value = (1.0 - shrinkage) * Float64(statistics.empirical.unsafe_get(k))
        if k // p == k % p:
            value += shrinkage * mu
        result.unsafe_set(k, value.cast[dtype]())
    return result^


def _centered_by[dtype: DType](X: NDArray[dtype], location: NDArray[dtype]) raises -> NDArray[dtype]:
    """X - location in the input dtype (sklearn subtracts before shrinkage)."""
    var n = X.shape[0]
    var p = X.shape[1]
    var rs = Int(X.strides[0])
    var cs = Int(X.strides[1])
    var result = nm.zeros[dtype](Shape(n, p))
    for i in range(n):
        for j in range(p):
            result.unsafe_set(i * p + j, at(X, i, j, rs, cs).cast[dtype]() - location.unsafe_get(j))
    return result^


def _single_feature[dtype: DType](centered: NDArray[dtype]) raises -> NDArray[dtype]:
    """sklearn's one-feature branch: mean(X**2) as a (1, 1) covariance."""
    # The compiler needs rebind to express equality inside dtype specializations.
    comptime if dtype == DType.float32:
        var squared = centered * centered
        var result = nm.zeros[dtype](Shape(1, 1))
        result.unsafe_set(0, (_sum32(rebind[NDArray[DType.float32]](squared), 0, centered.shape[0]) / Float32(centered.shape[0])).cast[dtype]())
        return result^
    var total: Float64 = 0
    for i in range(centered.shape[0]):
        var z = Float64(centered.unsafe_get(i))
        total += z * z
    var result = nm.zeros[dtype](Shape(1, 1))
    result.unsafe_set(0, (total / Float64(centered.shape[0])).cast[dtype]())
    return result^


def _location[dtype: DType](X: NDArray[dtype], assume_centered: Bool) raises -> NDArray[dtype]:
    if assume_centered:
        return nm.zeros[dtype](Shape(X.shape[1]))
    comptime if dtype == DType.float32:
        var n = X.shape[0]
        var p = X.shape[1]
        var result = nm.zeros[dtype](Shape(p))
        for j in range(p):
            var total: Float32 = 0
            if X.strides[0] == 1:
                total = _sum32(rebind[NDArray[DType.float32]](X), j * Int(X.strides[1]), n)
            else:
                for i in range(n):
                    total += Float32(X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1])))
            result.unsafe_set(j, (total / Float32(n)).cast[dtype]())
        return result^
    return to_array[dtype](column_means(X))


def ledoit_wolf_shrinkage[dtype: DType](X: NDArray[dtype], assume_centered: Bool = False, block_size: Int = 1000) raises -> Float64:
    """Ledoit-Wolf shrinkage coefficient of X (sklearn `ledoit_wolf_shrinkage`).

    X is (n_samples, n_features), finite; a single feature returns 0.0.
    """
    _check_block_size(block_size, "ledoit_wolf_shrinkage")
    validate_samples(X, "ledoit_wolf_shrinkage")
    if X.shape[1] == 1:
        return 0.0
    var centered = _centered_by(X, _location(X, assume_centered))
    comptime if dtype == DType.float32:
        return _lw32(rebind[NDArray[DType.float32]](centered), block_size)
    return ledoit_wolf_coefficient(shrinkage_statistics(centered, block_size))


@fieldwise_init
struct LedoitWolfParams(Copyable, Movable, Writable):
    var store_precision: Bool
    var assume_centered: Bool
    var block_size: Int


@fieldwise_init
struct OASParams(Copyable, Movable, Writable):
    var store_precision: Bool
    var assume_centered: Bool


struct LedoitWolf[dtype: DType = DType.float64](CovarianceEstimator, Copyable):
    """Ledoit-Wolf shrunk covariance estimator (sklearn `LedoitWolf`).

    Defaults: store_precision=True, assume_centered=False, block_size=1000.
    Learned: `location_()`, `covariance_()`, `precision_()`, `shrinkage_()`,
    `n_features_in_()`.
    """

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Precision = NDArray[Self.dtype]
    comptime Distances = NDArray[Self.dtype]
    comptime Params = LedoitWolfParams
    comptime NAME = "LedoitWolf"

    var store_precision: Bool
    var assume_centered: Bool
    var block_size: Int
    var _model: CovarianceModel[Self.dtype]
    var _shrinkage: Optional[Float64]

    def __init__(out self, *, store_precision: Bool = True, assume_centered: Bool = False, block_size: Int = 1000):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.block_size = block_size
        self._model = CovarianceModel[Self.dtype]()
        self._shrinkage = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.store_precision, self.assume_centered, self.block_size)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.store_precision = params.store_precision
        self.assume_centered = params.assume_centered
        self.block_size = params.block_size
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, store_precision: Optional[Bool] = None, assume_centered: Optional[Bool] = None,
                   block_size: Optional[Int] = None) raises -> ref[self] Self:
        if store_precision:
            self.store_precision = store_precision.value()
        if assume_centered:
            self.assume_centered = assume_centered.value()
        if block_size:
            self.block_size = block_size.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(store_precision=self.store_precision, assume_centered=self.assume_centered, block_size=self.block_size)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._model.is_fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        _check_block_size(self.block_size, Self.NAME)
        validate_samples(X, Self.NAME + ".fit")
        var location = _location(X, self.assume_centered)
        self._model.update_location(location)
        var centered = _centered_by(X, location)
        var covariance: NDArray[Self.dtype]
        var shrinkage: Float64
        if X.shape[1] == 1:
            covariance = _single_feature(centered)
            shrinkage = 0.0
        else:
            if X.shape[0] == 1:
                print("UserWarning: Only one sample available. You may want to reshape your data array", file=stderr)
            comptime if Self.dtype == DType.float32:
                var estimate = _estimate32(rebind[NDArray[DType.float32]](centered), False, self.block_size)
                covariance = rebind[NDArray[Self.dtype]](estimate[0]).copy()
                shrinkage = estimate[1]
            else:
                var statistics = shrinkage_statistics(centered, self.block_size)
                shrinkage = ledoit_wolf_coefficient(statistics)
                covariance = _shrink(statistics, shrinkage)
        self._shrinkage = shrinkage
        self._model.install(location, covariance, self.store_precision, Self.NAME)
        return self

    def shrinkage_(self) raises -> Float64:
        if not self._shrinkage:
            self._model.check_fitted(Self.NAME)
        return self._shrinkage.value()

    def get_precision(self) raises -> Self.Precision:
        return self._model.get_precision(Self.NAME)

    def mahalanobis(self, X: Self.Input) raises -> Self.Distances:
        return self._model.mahalanobis(X, Self.NAME)

    def score(self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Float64:
        return self._model.score(X, Self.NAME)

    def error_norm(self, comp_cov: Self.Input, norm: String = "frobenius", scaling: Bool = True, squared: Bool = True) raises -> Float64:
        return self._model.error_norm(comp_cov, Self.NAME, norm, scaling, squared)

    def location_(self) raises -> NDArray[Self.dtype]:
        return self._model.location(Self.NAME)

    def covariance_(self) raises -> NDArray[Self.dtype]:
        return self._model.covariance(Self.NAME)

    def precision_(self) raises -> Optional[NDArray[Self.dtype]]:
        return self._model.precision(Self.NAME)

    def n_features_in_(self) raises -> Int:
        return self._model.n_features_in(Self.NAME)


struct OAS[dtype: DType = DType.float64](CovarianceEstimator, Copyable):
    """Oracle Approximating Shrinkage estimator (sklearn `OAS`).

    Defaults: store_precision=True, assume_centered=False. Learned attributes
    as in LedoitWolf, including `shrinkage_()`.
    """

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Precision = NDArray[Self.dtype]
    comptime Distances = NDArray[Self.dtype]
    comptime Params = OASParams
    comptime NAME = "OAS"

    var store_precision: Bool
    var assume_centered: Bool
    var _model: CovarianceModel[Self.dtype]
    var _shrinkage: Optional[Float64]

    def __init__(out self, *, store_precision: Bool = True, assume_centered: Bool = False):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self._model = CovarianceModel[Self.dtype]()
        self._shrinkage = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.store_precision, self.assume_centered)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.store_precision = params.store_precision
        self.assume_centered = params.assume_centered
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, store_precision: Optional[Bool] = None, assume_centered: Optional[Bool] = None) raises -> ref[self] Self:
        if store_precision:
            self.store_precision = store_precision.value()
        if assume_centered:
            self.assume_centered = assume_centered.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(store_precision=self.store_precision, assume_centered=self.assume_centered)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._model.is_fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        validate_samples(X, Self.NAME + ".fit")
        var location = _location(X, self.assume_centered)
        self._model.update_location(location)
        var centered = _centered_by(X, location)
        var covariance: NDArray[Self.dtype]
        var shrinkage: Float64
        if X.shape[1] == 1:
            covariance = _single_feature(centered)
            shrinkage = 0.0
        else:
            comptime if Self.dtype == DType.float32:
                var estimate = _estimate32(rebind[NDArray[DType.float32]](centered), True)
                covariance = rebind[NDArray[Self.dtype]](estimate[0]).copy()
                shrinkage = estimate[1]
            else:
                var statistics = shrinkage_statistics(centered)
                shrinkage = oas_coefficient(statistics)
                covariance = _shrink(statistics, shrinkage)
        self._shrinkage = shrinkage
        self._model.install(location, covariance, self.store_precision, Self.NAME)
        return self

    def shrinkage_(self) raises -> Float64:
        if not self._shrinkage:
            self._model.check_fitted(Self.NAME)
        return self._shrinkage.value()

    def get_precision(self) raises -> Self.Precision:
        return self._model.get_precision(Self.NAME)

    def mahalanobis(self, X: Self.Input) raises -> Self.Distances:
        return self._model.mahalanobis(X, Self.NAME)

    def score(self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Float64:
        return self._model.score(X, Self.NAME)

    def error_norm(self, comp_cov: Self.Input, norm: String = "frobenius", scaling: Bool = True, squared: Bool = True) raises -> Float64:
        return self._model.error_norm(comp_cov, Self.NAME, norm, scaling, squared)

    def location_(self) raises -> NDArray[Self.dtype]:
        return self._model.location(Self.NAME)

    def covariance_(self) raises -> NDArray[Self.dtype]:
        return self._model.covariance(Self.NAME)

    def precision_(self) raises -> Optional[NDArray[Self.dtype]]:
        return self._model.precision(Self.NAME)

    def n_features_in_(self) raises -> Int:
        return self._model.n_features_in(Self.NAME)


def ledoit_wolf[dtype: DType](X: NDArray[dtype], *, assume_centered: Bool = False, block_size: Int = 1000) raises -> Tuple[NDArray[dtype], Float64]:
    """Return (shrunk covariance, shrinkage) as sklearn `ledoit_wolf`."""
    var model = LedoitWolf[dtype](store_precision=False, assume_centered=assume_centered, block_size=block_size)
    model.fit(X)
    return (model.covariance_(), model.shrinkage_())


def oas[dtype: DType](X: NDArray[dtype], *, assume_centered: Bool = False) raises -> Tuple[NDArray[dtype], Float64]:
    """Return (shrunk covariance, shrinkage) as sklearn `oas`."""
    var model = OAS[dtype](assume_centered=assume_centered)
    model.fit(X)
    return (model.covariance_(), model.shrinkage_())
