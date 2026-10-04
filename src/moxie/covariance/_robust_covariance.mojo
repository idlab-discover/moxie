# Adapted from scikit-learn sklearn/covariance/_robust_covariance.py;
# authors: scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Minimum Covariance Determinant: FastMCD (Rousseeuw & Van Driessen, 1999).

Ports `c_step`, `select_candidates`, `fast_mcd`, the chi-squared consistency
factor and `MinCovDet` (raw estimate, consistency-corrected distances and
reweighting). Random subsets use the NumPy legacy `RandomState` stream from
`moxie._random`, so integer seeds reproduce sklearn's draws. All work is
Float64 (sklearn's np.cov path); results are rounded to the estimator dtype.
Ties in distances and determinants are broken by the lower sample/trial
index (NumPy's partition/argsort leave them unspecified).
"""
import numojo as nm
from moxie.covariance._param_update import NullableUpdate
from numojo import NDArray, Shape
from std.math import ceil, isinf, abs
from std.sys import stderr
from mojo_linalg import gammainc, gammaincinv, gammainccinv, symmetric_eigh
from moxie._random import RandomState, entropy_state
from moxie.covariance._base import CovarianceEstimator
from moxie.covariance._engines import RobustCovarianceState
from moxie.covariance._empirical_covariance import (
    CovarianceModel, fast_logdet, pinvh, to_array, to_float64, validate_samples, gram, centered_columns,
)


# ---------------------------------------------------------------------------
# Chi-squared helpers (scipy.stats.chi2 through regularized incomplete gamma)
# ---------------------------------------------------------------------------


def chi2_cdf(x: Float64, df: Int) raises -> Float64:
    return gammainc(Float64(df) / 2.0, x / 2.0)


def chi2_ppf(probability: Float64, df: Int) raises -> Float64:
    return 2.0 * gammaincinv(Float64(df) / 2.0, probability)


def chi2_isf(probability: Float64, df: Int) raises -> Float64:
    return 2.0 * gammainccinv(Float64(df) / 2.0, probability)


def consistency_factor(n_features: Int, alpha: Float64) raises -> Float64:
    """alpha / chi2.cdf(chi2.ppf(alpha, p), p + 2) (sklearn `_consistency_factor`)."""
    var quantile = chi2_ppf(alpha, n_features)
    return alpha / chi2_cdf(quantile, n_features + 2)


# ---------------------------------------------------------------------------
# Float64 subset kernels on a contiguous (n, p) work copy
# ---------------------------------------------------------------------------


def subset_mean(X: NDArray[DType.float64], indices: List[Int]) raises -> NDArray[DType.float64]:
    var p = X.shape[1]
    var result = nm.zeros[DType.float64](Shape(p))
    for j in range(p):
        var total: Float64 = 0
        for i in indices:
            total += X.unsafe_get(i * p + j)
        result.unsafe_set(j, total / Float64(len(indices)))
    return result^


def subset_covariance(X: NDArray[DType.float64], indices: List[Int], location: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
    """MLE covariance (divisor h) of the selected rows around `location`."""
    var p = X.shape[1]
    var h = len(indices)
    var centered = nm.zeros[DType.float64](Shape(h, p))
    for r in range(h):
        for j in range(p):
            centered.unsafe_set(r * p + j, X.unsafe_get(indices[r] * p + j) - location.unsafe_get(j))
    var result = nm.zeros[DType.float64](Shape(p, p))
    for a in range(p):
        for b in range(a + 1):
            var total: Float64 = 0
            for r in range(h):
                total += centered.unsafe_get(r * p + a) * centered.unsafe_get(r * p + b)
            result.unsafe_set(a * p + b, total / Float64(h))
            result.unsafe_set(b * p + a, total / Float64(h))
    return result^


def distances(X: NDArray[DType.float64], location: NDArray[DType.float64], precision: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
    """Squared Mahalanobis distances of every row: ((X - m) P * (X - m)).sum(1)."""
    var n = X.shape[0]
    var p = X.shape[1]
    var result = nm.zeros[DType.float64](Shape(n))
    var centered = nm.zeros[DType.float64](Shape(p))
    for i in range(n):
        for j in range(p):
            centered.unsafe_set(j, X.unsafe_get(i * p + j) - location.unsafe_get(j))
        var total: Float64 = 0
        for b in range(p):
            var column: Float64 = 0
            for a in range(p):
                column += centered.unsafe_get(a) * precision.unsafe_get(a * p + b)
            total += column * centered.unsafe_get(b)
        result.unsafe_set(i, total)
    return result^


def _less(values: NDArray[DType.float64], a: Int, b: Int) -> Bool:
    """Strict order by value then index; NaN sorts last as in NumPy."""
    var x = values.unsafe_get(a)
    var y = values.unsafe_get(b)
    if x != x:
        return False if y != y else False
    if y != y:
        return True
    if x != y:
        return x < y
    return a < b


def argsort(values: NDArray[DType.float64]) -> List[Int]:
    """Stable ascending order (merge sort), NaN last."""
    var n = values.size
    var order = List[Int]()
    for i in range(n):
        order.append(i)
    var buffer = order.copy()
    var width = 1
    while width < n:
        var start = 0
        while start < n:
            var middle = min(start + width, n)
            var end = min(start + 2 * width, n)
            var i = start
            var j = middle
            var k = start
            while i < middle and j < end:
                if _less(values, order[j], order[i]):
                    buffer[k] = order[j]
                    j += 1
                else:
                    buffer[k] = order[i]
                    i += 1
                k += 1
            while i < middle:
                buffer[k] = order[i]
                i += 1
                k += 1
            while j < end:
                buffer[k] = order[j]
                j += 1
                k += 1
            start += 2 * width
        order = buffer.copy()
        width *= 2
    return order^


def smallest(values: NDArray[DType.float64], count: Int) -> List[Int]:
    """Indices of the `count` smallest values, returned in ascending index order."""
    var order = argsort(values)
    var chosen = List[Bool]()
    for _ in range(values.size):
        chosen.append(False)
    for k in range(count):
        chosen[order[k]] = True
    var result = List[Int]()
    for i in range(values.size):
        if chosen[i]:
            result.append(i)
    return result^


def _is_inf(value: Float64) -> Bool:
    return isinf(value)


def _allclose(a: Float64, b: Float64) -> Bool:
    """np.allclose(a, b) for scalars (rtol=1e-5, atol=1e-8, equal infinities)."""
    if a == b:
        return True
    if isinf(a) or isinf(b):
        return False
    return abs(a - b) <= 1e-8 + 1e-5 * abs(b)


@fieldwise_init
struct CStepResult(Copyable, Movable):
    """One concentration-step outcome: estimates, log-determinant, support, distances."""

    var location: NDArray[DType.float64]
    var covariance: NDArray[DType.float64]
    var det: Float64
    var support: List[Int]
    var dist: NDArray[DType.float64]


@fieldwise_init
struct InitialEstimate(Copyable, Movable):
    var location: NDArray[DType.float64]
    var covariance: NDArray[DType.float64]


def c_step(X: NDArray[DType.float64], n_support: Int, mut random_state: RandomState, remaining_iterations: Int = 30,
           initial: Optional[InitialEstimate] = None) raises -> CStepResult:
    """Concentration steps from a random subset or from initial estimates (sklearn `_c_step`)."""
    var n = X.shape[0]
    var dist = nm.zeros[DType.float64](Shape(n))
    var dist_is_inf = True
    var support = List[Int]()
    var precision = nm.zeros[DType.float64](Shape(X.shape[1], X.shape[1]))
    if not initial:
        var permutation = random_state.permutation(n)
        var picked = List[Bool]()
        for _ in range(n):
            picked.append(False)
        for k in range(n_support):
            picked[Int(permutation.unsafe_get(k))] = True
        for i in range(n):
            if picked[i]:
                support.append(i)
    else:
        precision = pinvh(initial.value().covariance)
        dist = distances(X, initial.value().location, precision)
        dist_is_inf = False
        support = smallest(dist, n_support)
    var location = subset_mean(X, support)
    var covariance = subset_covariance(X, support, location)
    var det = fast_logdet(covariance)
    if _is_inf(det):
        precision = pinvh(covariance)
    var previous_det = Float64.MAX * 2.0
    var previous_location = location.copy()
    var previous_covariance = covariance.copy()
    var previous_support = support.copy()
    var remaining = remaining_iterations
    while det < previous_det and remaining > 0 and not _is_inf(det):
        previous_location = location.copy()
        previous_covariance = covariance.copy()
        previous_det = det
        previous_support = support.copy()
        precision = pinvh(covariance)
        dist = distances(X, location, precision)
        dist_is_inf = False
        support = smallest(dist, n_support)
        location = subset_mean(X, support)
        covariance = subset_covariance(X, support, location)
        det = fast_logdet(covariance)
        remaining -= 1
    var previous_dist = dist.copy()
    var previous_dist_inf = dist_is_inf
    dist = distances(X, location, precision)
    var use_previous = False
    if not _is_inf(det) and not _allclose(det, previous_det) and det > previous_det:
        print("RuntimeWarning: Determinant has increased; this should not happen: log(det) > log(previous_det) (",
              det, ">", previous_det, "). You may want to try with a higher value of support_fraction (current value:",
              Float64(n_support) / Float64(n), ").", file=stderr)
        use_previous = True
    if remaining == 0:
        use_previous = False
    if use_previous:
        if previous_dist_inf:
            for i in range(n):
                previous_dist.unsafe_set(i, Float64.MAX * 2.0)
        return CStepResult(previous_location^, previous_covariance^, previous_det, previous_support^, previous_dist^)
    return CStepResult(location^, covariance^, det, support^, dist^)


def select_candidates(X: NDArray[DType.float64], n_support: Int, n_trials: Int, select: Int, n_iter: Int,
                      mut random_state: RandomState) raises -> List[CStepResult]:
    """Best `select` of `n_trials` random-start c_step runs, by log-determinant."""
    var results = List[CStepResult]()
    for _ in range(n_trials):
        results.append(c_step(X, n_support, random_state, n_iter))
    return _best(results, select)


def select_candidates(X: NDArray[DType.float64], n_support: Int, initial: List[InitialEstimate], select: Int, n_iter: Int,
                      mut random_state: RandomState) raises -> List[CStepResult]:
    """Best `select` c_step runs started from the given estimates."""
    var results = List[CStepResult]()
    for k in range(len(initial)):
        results.append(c_step(X, n_support, random_state, n_iter, initial[k].copy()))
    return _best(results, select)


def _best(results: List[CStepResult], select: Int) raises -> List[CStepResult]:
    var dets = nm.zeros[DType.float64](Shape(len(results)))
    for k in range(len(results)):
        dets.unsafe_set(k, results[k].det)
    var order = argsort(dets)
    var best = List[CStepResult]()
    for k in range(min(select, len(results))):
        best.append(results[order[k]].copy())
    return best^


@fieldwise_init
struct FastMCDResult[dtype: DType](Copyable, Movable):
    """fast_mcd output: location (p,), covariance (p, p), support mask (n,), distances (n,)."""

    var location: NDArray[Self.dtype]
    var covariance: NDArray[Self.dtype]
    var support: NDArray[DType.bool]
    var dist: NDArray[Self.dtype]


def _rows(X: NDArray[DType.float64], indices: List[Int]) raises -> NDArray[DType.float64]:
    var p = X.shape[1]
    var result = nm.zeros[DType.float64](Shape(len(indices), p))
    for r in range(len(indices)):
        for j in range(p):
            result.unsafe_set(r * p + j, X.unsafe_get(indices[r] * p + j))
    return result^


def _mask(n: Int, indices: List[Int]) raises -> NDArray[DType.bool]:
    var mask = nm.zeros[DType.bool](Shape(n))
    for i in indices:
        mask.unsafe_set(i, True)
    return mask^


def _fast_mcd64(X: NDArray[DType.float64], support_fraction: Optional[Float64], mut random_state: RandomState, round32: Bool = False) raises -> Tuple[NDArray[DType.float64], NDArray[DType.float64], List[Int], NDArray[DType.float64]]:
    var n = X.shape[0]
    var p = X.shape[1]
    var n_support: Int
    if support_fraction:
        n_support = Int(support_fraction.value() * Float64(n))
    else:
        n_support = min(Int(ceil(0.5 * Float64(n + p + 1))), n)
    if n_support < 1:
        raise Error(String("ValueError: fast_mcd: support_fraction gives an empty support (n_support=", n_support, ")"))
    if p == 1:
        var values = nm.zeros[DType.float64](Shape(n))
        for i in range(n):
            values.unsafe_set(i, X.unsafe_get(i))
        var location = nm.zeros[DType.float64](Shape(1))
        var support = List[Int]()
        if n_support < n:
            var order = argsort(values)
            var sorted = List[Float64]()
            for i in range(n):
                sorted.append(values.unsafe_get(order[i]))
            var smallest_width = Float64.MAX * 2.0
            for k in range(n - n_support):
                smallest_width = min(smallest_width, sorted[n_support + k] - sorted[k])
            var total: Float64 = 0
            var count = 0
            for k in range(n - n_support):
                if sorted[n_support + k] - sorted[k] == smallest_width:
                    total += _round(0.5 * _round(sorted[n_support + k] + sorted[k], round32), round32)
                    count += 1
            var center = _round(total / Float64(count), round32)
            location.unsafe_set(0, center)
            var deviation = nm.zeros[DType.float64](Shape(n))
            for i in range(n):
                # Endpoints of the shortest half tie exactly in real arithmetic;
                # sklearn resolves them in the input dtype's rounding.
                deviation.unsafe_set(i, abs(_round(values.unsafe_get(i) - center, round32)))
            support = smallest(deviation, n_support)
            var mean: Float64 = 0
            for i in support:
                mean += values.unsafe_get(i)
            mean /= Float64(len(support))
            var variance: Float64 = 0
            for i in support:
                variance += (values.unsafe_get(i) - mean) * (values.unsafe_get(i) - mean)
            var covariance = nm.zeros[DType.float64](Shape(1, 1))
            covariance.unsafe_set(0, variance / Float64(len(support)))
            var precision = pinvh(covariance)
            return (location^, covariance.copy(), support^, distances(X, location, precision))
        for i in range(n):
            support.append(i)
        var mean = subset_mean(X, support)
        var covariance = subset_covariance(X, support, mean)
        var precision = pinvh(covariance)
        var dist = distances(X, mean, precision)
        return (mean^, covariance^, support^, dist^)
    if n > 500:
        var n_subsets = n // 300
        var n_samples_subsets = n // n_subsets
        var samples_shuffle = random_state.permutation(n)
        var h_subset = Int(ceil(Float64(n_samples_subsets) * (Float64(n_support) / Float64(n))))
        var n_trials = max(10, 500 // n_subsets)
        var n_best_sub = 10
        var all_best = List[InitialEstimate]()
        for i in range(n_subsets):
            var indices = List[Int]()
            for k in range(i * n_samples_subsets, (i + 1) * n_samples_subsets):
                indices.append(Int(samples_shuffle.unsafe_get(k)))
            var subset = _rows(X, indices)
            var best = select_candidates(subset, h_subset, n_trials, n_best_sub, 2, random_state)
            for k in range(len(best)):
                all_best.append(InitialEstimate(best[k].location.copy(), best[k].covariance.copy()))
        var n_samples_merged = min(1500, n)
        var h_merged = Int(ceil(Float64(n_samples_merged) * (Float64(n_support) / Float64(n))))
        var n_best_merged = 10 if n > 1500 else 1
        var permutation = random_state.permutation(n)
        var selection = List[Int]()
        for k in range(n_samples_merged):
            selection.append(Int(permutation.unsafe_get(k)))
        var merged = select_candidates(_rows(X, selection), h_merged, all_best, n_best_merged, 30, random_state)
        if n < 1500:
            var support = List[Int]()
            var dist = nm.zeros[DType.float64](Shape(n))
            var selected_support = merged[0].support.copy()
            for r in selected_support:
                support.append(selection[r])
            for r in range(len(selection)):
                dist.unsafe_set(selection[r], merged[0].dist.unsafe_get(r))
            var sorted_support = smallest_indices_sorted(support)
            return (merged[0].location.copy(), merged[0].covariance.copy(), sorted_support^, dist^)
        var starts = List[InitialEstimate]()
        for k in range(len(merged)):
            starts.append(InitialEstimate(merged[k].location.copy(), merged[k].covariance.copy()))
        var full = select_candidates(X, n_support, starts, 1, 30, random_state)
        return (full[0].location.copy(), full[0].covariance.copy(), full[0].support.copy(), full[0].dist.copy())
    var best = select_candidates(X, n_support, 30, 10, 2, random_state)
    var starts = List[InitialEstimate]()
    for k in range(len(best)):
        starts.append(InitialEstimate(best[k].location.copy(), best[k].covariance.copy()))
    var full = select_candidates(X, n_support, starts, 1, 30, random_state)
    return (full[0].location.copy(), full[0].covariance.copy(), full[0].support.copy(), full[0].dist.copy())


def _round(value: Float64, round32: Bool) -> Float64:
    """Round to Float32 when emulating sklearn's Float32 univariate arithmetic."""
    if round32:
        return Float64(Float32(value))
    return value


def smallest_indices_sorted(indices: List[Int]) -> List[Int]:
    """Ascending copy of a list of sample indices (insertion sort)."""
    var result = indices.copy()
    for i in range(1, len(result)):
        var value = result[i]
        var k = i - 1
        while k >= 0 and result[k] > value:
            result[k + 1] = result[k]
            k -= 1
        result[k + 1] = value
    return result^


def fast_mcd[dtype: DType](X: NDArray[dtype], support_fraction: Optional[Float64], mut random_state: RandomState) raises -> FastMCDResult[dtype]:
    """FastMCD raw robust location/covariance (sklearn `fast_mcd`), shared RNG stream."""
    validate_samples(X, "fast_mcd", min_samples=2)
    var result = _fast_mcd64(to_float64(X), support_fraction, random_state, dtype == DType.float32)
    return FastMCDResult[dtype](to_array[dtype](result[0]), to_array[dtype](result[1]), _mask(X.shape[0], result[2]),
                                to_array[dtype](result[3]))


def fast_mcd[dtype: DType](X: NDArray[dtype], support_fraction: Optional[Float64] = None, random_state: Optional[Int] = None) raises -> FastMCDResult[dtype]:
    """FastMCD with an integer seed (fresh RandomState) or None (entropy-seeded state)."""
    var state = RandomState(random_state.value()) if random_state else entropy_state()
    return fast_mcd(X, support_fraction, state)


# ---------------------------------------------------------------------------
# MinCovDet
# ---------------------------------------------------------------------------


@fieldwise_init
struct MinCovDetParams(Copyable, Movable, Writable):
    var store_precision: Bool
    var assume_centered: Bool
    var support_fraction: Optional[Float64]
    var random_state: Optional[Int]


struct RobustFit[dtype: DType](Copyable, Movable):
    """MinCovDet learned state shared with EllipticEnvelope: model + raw snapshots."""

    var model: CovarianceModel[Self.dtype]
    var robust: Optional[RobustCovarianceState[Self.dtype]]

    def __init__(out self):
        self.model = CovarianceModel[Self.dtype]()
        self.robust = None

    def fit(mut self, X: NDArray[Self.dtype], store_precision: Bool, assume_centered: Bool, support_fraction: Optional[Float64],
            mut random_state: RandomState, owner: String) raises:
        """sklearn MinCovDet.fit: raw FastMCD, consistency correction, reweighting."""
        if support_fraction:
            var fraction = support_fraction.value()
            if not (fraction > 0.0 and fraction <= 1.0):
                raise Error(String("InvalidParameterError: The 'support_fraction' parameter of ", owner,
                                   " must be a float in the range (0.0, 1.0] or None. Got ", fraction, " instead."))
        validate_samples(X, owner + ".fit", min_samples=2)
        var X64 = to_float64(X)
        var n = X64.shape[0]
        var p = X64.shape[1]
        var scatter = gram(centered_columns(X64, nm.zeros[DType.float64](Shape(p))), 1.0)
        var eigenvalues = symmetric_eigh(scatter).eigenvalues()
        var rank = 0
        for j in range(p):
            if abs(eigenvalues.unsafe_get(j)) > 1e-8:
                rank += 1
        if rank != p:
            print("UserWarning: The covariance matrix associated to your dataset is not full rank", file=stderr)
        var raw = _fast_mcd64(X64, support_fraction, random_state, Self.dtype == DType.float32)
        var raw_location = raw[0].copy()
        var raw_covariance = raw[1].copy()
        var raw_support = raw[2].copy()
        var raw_dist = raw[3].copy()
        if assume_centered:
            raw_location = nm.zeros[DType.float64](Shape(p))
            raw_covariance = subset_covariance(X64, raw_support, raw_location)
            raw_dist = distances(X64, raw_location, pinvh(raw_covariance))
        self.model.update_location(to_array[Self.dtype](raw_location))
        self.robust = RobustCovarianceState[Self.dtype](
            self.model.state.copy(), to_array[Self.dtype](raw_location), to_array[Self.dtype](raw_covariance),
            _mask(n, raw_support), _mask(n, raw_support), to_array[Self.dtype](raw_dist))
        # correct_covariance: only the distances are rescaled in this sklearn revision.
        var n_support = len(raw_support)
        if n_support < n:
            var all_zero = True
            for k in range(p * p):
                if abs(raw_covariance.unsafe_get(k)) > 1e-8:
                    all_zero = False
            if all_zero:
                raise Error("ValueError: The covariance matrix of the support data is equal to 0, try to increase support_fraction")
        var raw_factor = consistency_factor(p, Float64(n_support) / Float64(n))
        var corrected = nm.zeros[DType.float64](Shape(n))
        for i in range(n):
            corrected.unsafe_set(i, raw_dist.unsafe_get(i) / raw_factor)
        self.robust.value().dist_ = to_array[Self.dtype](corrected)
        # reweight_covariance
        var threshold = chi2_isf(0.025, p)
        var mask = List[Int]()
        for i in range(n):
            if corrected.unsafe_get(i) < threshold:
                mask.append(i)
        if len(mask) == 0:
            raise Error(String("ValueError: ", owner, ": no sample passes the reweighting threshold"))
        var location: NDArray[DType.float64]
        if assume_centered:
            location = nm.zeros[DType.float64](Shape(p))
        else:
            location = subset_mean(X64, mask)
        var covariance = subset_covariance(X64, mask, location)
        var factor = consistency_factor(p, 1.0 - 0.025)
        for k in range(p * p):
            covariance.unsafe_set(k, covariance.unsafe_get(k) * factor)
        var location_out = to_array[Self.dtype](location)
        var candidate = CovarianceModel[Self.dtype]()
        candidate.install_float64(location_out, covariance, store_precision, owner)
        var final_dist = distances(X64, to_float64_vector(location_out), candidate.precision64())
        self.robust = RobustCovarianceState[Self.dtype](candidate.state.copy(), to_array[Self.dtype](raw_location),
                                                        to_array[Self.dtype](raw_covariance), _mask(n, raw_support),
                                                        _mask(n, mask), to_array[Self.dtype](final_dist))
        self.model = candidate^

    def correct_covariance(mut self, data: NDArray[Self.dtype], owner: String) raises -> NDArray[Self.dtype]:
        """Return corrected raw covariance and rescale current distances; data is unused."""
        self.check_fitted(owner)
        var snapshot = self.state(owner)
        var n = snapshot.dist_.size
        var p = snapshot.raw_covariance_.shape[0]
        var h = 0
        for i in range(n):
            if snapshot.support_.unsafe_get(i):
                h += 1
        var covariance = to_float64(snapshot.raw_covariance_)
        var all_zero = True
        for k in range(p * p):
            if abs(covariance.unsafe_get(k)) > 1e-8:
                all_zero = False
        if h < n and all_zero:
            raise Error("ValueError: The covariance matrix of the support data is equal to 0, try to increase support_fraction")
        var factor = consistency_factor(p, Float64(h) / Float64(n))
        for k in range(p * p):
            covariance.unsafe_set(k, covariance.unsafe_get(k) * factor)
        for i in range(n):
            snapshot.dist_.unsafe_set(i, (Float64(snapshot.dist_.unsafe_get(i)) / factor).cast[Self.dtype]())
        var result = to_array[Self.dtype](covariance)
        self.robust = snapshot^
        return result^

    def reweight_covariance(mut self, data: NDArray[Self.dtype], store_precision: Bool, assume_centered: Bool,
                            owner: String) raises -> Tuple[NDArray[Self.dtype], NDArray[Self.dtype], NDArray[DType.bool]]:
        """Return location, unscaled covariance and support; install scaled covariance."""
        self.model.check_features(data, owner)
        var snapshot = self.state(owner)
        if data.shape[0] != snapshot.dist_.size:
            raise Error("ValueError: reweight_covariance requires the training sample count")
        var X = to_float64(data)
        var n = X.shape[0]
        var p = X.shape[1]
        var threshold = chi2_isf(0.025, p)
        var selected = List[Int]()
        for i in range(n):
            if Float64(snapshot.dist_.unsafe_get(i)) < threshold:
                selected.append(i)
        if len(selected) == 0:
            raise Error(String("ValueError: ", owner, ": no sample passes the reweighting threshold"))
        var location = nm.zeros[DType.float64](Shape(p)) if assume_centered else subset_mean(X, selected)
        var covariance = subset_covariance(X, selected, location)
        var returned_covariance = to_array[Self.dtype](covariance)
        var factor = consistency_factor(p, 0.975)
        for k in range(p * p):
            covariance.unsafe_set(k, covariance.unsafe_get(k) * factor)
        var location_out = to_array[Self.dtype](location)
        var support = _mask(n, selected)
        var candidate = CovarianceModel[Self.dtype]()
        candidate.install_float64(location_out, covariance, store_precision, owner)
        snapshot.estimate = candidate.state.copy()
        snapshot.support_ = support.copy()
        snapshot.dist_ = to_array[Self.dtype](distances(X, to_float64_vector(location_out), candidate.precision64()))
        self.robust = snapshot^
        self.model = candidate^
        return (location_out^, returned_covariance^, support^)

    def check_fitted(self, owner: String) raises:
        self.model.check_fitted(owner)

    def state(self, owner: String) raises -> RobustCovarianceState[Self.dtype]:
        """Independent snapshot of the raw and reweighted robust state."""
        self.model.check_fitted(owner)
        return self.robust.value().copy()


def to_float64_vector[dtype: DType](x: NDArray[dtype]) raises -> NDArray[DType.float64]:
    var result = nm.zeros[DType.float64](Shape(x.shape[0]))
    for k in range(x.shape[0]):
        result.unsafe_set(k, Float64(x.unsafe_get(k * Int(x.strides[0]))))
    return result^


struct MinCovDet[dtype: DType = DType.float64](CovarianceEstimator, Copyable):
    """Minimum Covariance Determinant robust estimator (sklearn `MinCovDet`).

    Defaults: store_precision=True, assume_centered=False, support_fraction=None,
    random_state=None (entropy-seeded; pass an Int for reproducible fits, or
    call `fit(X, random_state)` with a shared mutable `RandomState`).
    Learned: `raw_location_()`, `raw_covariance_()`, `raw_support_()`,
    `location_()`, `covariance_()`, `precision_()`, `support_()`, `dist_()`.
    """

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Precision = NDArray[Self.dtype]
    comptime Distances = NDArray[Self.dtype]
    comptime Params = MinCovDetParams
    comptime NAME = "MinCovDet"

    var store_precision: Bool
    var assume_centered: Bool
    var support_fraction: Optional[Float64]
    var random_state: Optional[Int]
    var _fit: RobustFit[Self.dtype]

    def __init__(out self, *, store_precision: Bool = True, assume_centered: Bool = False,
                 support_fraction: Optional[Float64] = None, random_state: Optional[Int] = None):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.random_state = random_state
        self._fit = RobustFit[Self.dtype]()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.store_precision, self.assume_centered, self.support_fraction, self.random_state)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.store_precision = params.store_precision
        self.assume_centered = params.assume_centered
        self.support_fraction = params.support_fraction
        self.random_state = params.random_state
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *,
        store_precision: Optional[Bool] = None,
        assume_centered: Optional[Bool] = None,
        support_fraction: NullableUpdate[Float64] = NullableUpdate[Float64](),
        random_state: NullableUpdate[Int] = NullableUpdate[Int](),
    ) raises -> ref[self] Self:
        """Update supplied constructor parameters without changing fitted state."""
        if store_precision:
            self.store_precision = store_precision.value().copy()
        if assume_centered:
            self.assume_centered = assume_centered.value().copy()
        if support_fraction.supplied:
            self.support_fraction = support_fraction.value.copy()
        if random_state.supplied:
            self.random_state = random_state.value.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(store_precision=self.store_precision, assume_centered=self.assume_centered,
                    support_fraction=self.support_fraction, random_state=self.random_state)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fit.model.is_fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        var state = RandomState(self.random_state.value()) if self.random_state else entropy_state()
        self._fit.fit(X, self.store_precision, self.assume_centered, self.support_fraction, state, Self.NAME)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, mut random_state: RandomState) raises -> ref[self] Self:
        """Fit drawing from a caller-owned stream (sklearn: a RandomState instance)."""
        self._fit.fit(X, self.store_precision, self.assume_centered, self.support_fraction, random_state, Self.NAME)
        return self

    def correct_covariance(mut self, data: Self.Input) raises -> NDArray[Self.dtype]:
        """Rescale dist_ and return corrected raw covariance without installing it."""
        return self._fit.correct_covariance(data, Self.NAME)

    def reweight_covariance(mut self, data: Self.Input) raises -> Tuple[NDArray[Self.dtype], NDArray[Self.dtype], NDArray[DType.bool]]:
        """Reweight training rows; return (location, unscaled covariance, support)."""
        return self._fit.reweight_covariance(data, self.store_precision, self.assume_centered, Self.NAME)

    def get_precision(self) raises -> Self.Precision:
        return self._fit.model.get_precision(Self.NAME)

    def mahalanobis(self, X: Self.Input) raises -> Self.Distances:
        return self._fit.model.mahalanobis(X, Self.NAME)

    def score(self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Float64:
        return self._fit.model.score(X, Self.NAME)

    def error_norm(self, comp_cov: Self.Input, norm: String = "frobenius", scaling: Bool = True, squared: Bool = True) raises -> Float64:
        return self._fit.model.error_norm(comp_cov, Self.NAME, norm, scaling, squared)

    def location_(self) raises -> NDArray[Self.dtype]:
        return self._fit.model.location(Self.NAME)

    def covariance_(self) raises -> NDArray[Self.dtype]:
        return self._fit.model.covariance(Self.NAME)

    def precision_(self) raises -> Optional[NDArray[Self.dtype]]:
        return self._fit.model.precision(Self.NAME)

    def n_features_in_(self) raises -> Int:
        return self._fit.model.n_features_in(Self.NAME)

    def raw_location_(self) raises -> NDArray[Self.dtype]:
        return self._fit.state(Self.NAME).raw_location_.copy()

    def raw_covariance_(self) raises -> NDArray[Self.dtype]:
        return self._fit.state(Self.NAME).raw_covariance_.copy()

    def raw_support_(self) raises -> NDArray[DType.bool]:
        return self._fit.state(Self.NAME).raw_support_.copy()

    def support_(self) raises -> NDArray[DType.bool]:
        return self._fit.state(Self.NAME).support_.copy()

    def dist_(self) raises -> NDArray[Self.dtype]:
        return self._fit.state(Self.NAME).dist_.copy()
