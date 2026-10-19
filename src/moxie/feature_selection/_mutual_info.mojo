# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense featurewise mutual information (Kraskov/Ross/contingency estimates)."""
from moxie._labels import StringTarget
from moxie.feature_selection._univariate_scores import _class_codes
import numojo as nm
from std.math import sqrt, abs, max, min, floor, isfinite
from std.utils import Variant
from std.memory import bitcast
from std.sys import llvm_intrinsic
from moxie._dense import sum_axis2d, row_squared_norms
from moxie._random import RandomState, check_random_state
from moxie.neighbors._unsupervised import NearestNeighbors
from moxie.feature_selection._univariate_scores import _validate_x, _validate_y

comptime DiscreteFeatures = Variant[String, Bool, List[Bool], List[Int]]


def _discrete_mask(features: DiscreteFeatures, p: Int) raises -> List[Bool]:
    var mask = List[Bool]()
    var all_discrete = False
    if features.isa[String]():
        if features[String] != "auto":
            raise Error("ValueError: discrete_features string must be auto")
    elif features.isa[Bool]():
        all_discrete = features[Bool]
    elif features.isa[List[Bool]]():
        if len(features[List[Bool]]) != p:
            raise Error("ValueError: discrete_features mask must match feature count")
        return features[List[Bool]].copy()
    for _ in range(p):
        mask.append(all_discrete)
    if features.isa[List[Int]]():
        if len(features[List[Int]]) == 0:
            raise Error("ValueError: discrete feature indices cannot be empty; use False")
        for index in features[List[Int]]:
            var j = index if index >= 0 else index + p
            if j < 0 or j >= p:
                raise Error("ValueError: discrete feature index out of range")
            mask[j] = True
    return mask^


def _harmonics(n: Int) raises -> nm.NDArray[DType.float64]:
    # H[m-1] = psi(m) + EulerGamma; constants cancel in the MI formulas.
    var h = nm.zeros[DType.float64](nm.Shape(n + 1))
    for i in range(2, n + 1):
        h.unsafe_set(i, h.unsafe_get(i - 1) + 1 / Float64(i - 1))
    return h^


def _inside_radius(radius: Float64) -> Float64:
    # nextafter(nonnegative finite radius, 0); includes exact duplicates at zero.
    if radius == 0:
        return 0
    return bitcast[DType.float64, 1](bitcast[DType.uint64, 1](radius) - UInt64(1))


def _column[dt: DType](x: nm.NDArray[dt], j: Int) raises -> nm.NDArray[DType.float64]:
    var out = nm.zeros[DType.float64](nm.Shape(x.shape[0], 1))
    for i in range(x.shape[0]):
        out.unsafe_set(i, Float64(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))))
    return out^


def _radius_count(nn: NearestNeighbors[DType.float64], value: Float64, radius: Float64) raises -> Int:
    var query = nm.full[DType.float64](nm.Shape(1, 1), value)
    var found = nn.radius_neighbors(query^, radius=radius, return_distance=False)
    return len(found.indices)


def _compute_mi_cc(x: nm.NDArray[DType.float64], y: nm.NDArray[DType.float64], k: Int, h: nm.NDArray[DType.float64]) raises -> Float64:
    var n = x.shape[0]
    var xy = nm.zeros[DType.float64](nm.Shape(n, 2))
    for i in range(n):
        xy.unsafe_set(i * 2, x.unsafe_get(i))
        xy.unsafe_set(i * 2 + 1, y.unsafe_get(i))
    var joint = NearestNeighbors[DType.float64](k, metric="chebyshev")
    _ = joint.fit(xy)
    var nearest = joint.kneighbors()
    var nx = NearestNeighbors[DType.float64](metric="chebyshev")
    var ny = NearestNeighbors[DType.float64](metric="chebyshev")
    _ = nx.fit(x)
    _ = ny.fit(y)
    var total = Float64(0)
    for i in range(n):
        var radius = _inside_radius(nearest.distances.value().unsafe_get(i * k + k - 1))
        var cx = _radius_count(nx, x.unsafe_get(i), radius)
        var cy = _radius_count(ny, y.unsafe_get(i), radius)
        total += h.unsafe_get(cx) + h.unsafe_get(cy)
    return max(Float64(0), h.unsafe_get(n) + h.unsafe_get(k) - total / Float64(n))


def _compute_mi_cd(c: nm.NDArray[DType.float64], d: nm.NDArray[DType.float64], k: Int, h: nm.NDArray[DType.float64]) raises -> Float64:
    var n = c.shape[0]
    var counts = List[Int]()
    var radii = nm.zeros[DType.float64](nm.Shape(n))
    var retained = List[Int]()
    for i in range(n):
        var count = 0
        for j in range(n):
            if d.unsafe_get(i) == d.unsafe_get(j):
                count += 1
        counts.append(count)
        if count > 1:
            retained.append(i)
    if len(retained) == 0:
        raise Error("ValueError: no repeated labels for continuous/discrete MI")
    var visited = List[Bool]()
    for _ in range(n):
        visited.append(False)
    for i in range(n):
        if visited[i] or counts[i] <= 1:
            continue
        var members = List[Int]()
        var subset = nm.zeros[DType.float64](nm.Shape(counts[i], 1))
        for j in range(n):
            if d.unsafe_get(i) == d.unsafe_get(j):
                subset.unsafe_set(len(members), c.unsafe_get(j))
                members.append(j)
                visited[j] = True
        var local_k = min(k, counts[i] - 1)
        var nn = NearestNeighbors[DType.float64](local_k, metric="chebyshev")
        _ = nn.fit(subset)
        var nearest = nn.kneighbors()
        for j in range(len(members)):
            radii.unsafe_set(members[j], _inside_radius(nearest.distances.value().unsafe_get(j * local_k + local_k - 1)))
    var valid = nm.zeros[DType.float64](nm.Shape(len(retained), 1))
    for i in range(len(retained)):
        valid.unsafe_set(i, c.unsafe_get(retained[i]))
    var nn = NearestNeighbors[DType.float64](metric="chebyshev")
    _ = nn.fit(valid)
    var total = Float64(0)
    for index in retained:
        var m = _radius_count(nn, c.unsafe_get(index), radii.unsafe_get(index))
        total += h.unsafe_get(min(k, counts[index] - 1)) - h.unsafe_get(counts[index]) - h.unsafe_get(m)
    return max(Float64(0), h.unsafe_get(len(retained)) + total / Float64(len(retained)))


def _compute_mi_dd(x: nm.NDArray[DType.float64], y: nm.NDArray[DType.float64]) raises -> Float64:
    var n = x.shape[0]
    var total = Float64(0)
    # Each observation contributes log(pxy/(px*py))/n, avoiding a dense table.
    for i in range(n):
        var nx = 0
        var ny = 0
        var nxy = 0
        for j in range(n):
            var same_x = x.unsafe_get(i) == x.unsafe_get(j)
            var same_y = y.unsafe_get(i) == y.unsafe_get(j)
            nx += Int(same_x)
            ny += Int(same_y)
            nxy += Int(same_x and same_y)
        total += llvm_intrinsic["llvm.log", Float64, has_side_effect=False](Float64(nxy) * Float64(n) / (Float64(nx) * Float64(ny)))
    return max(Float64(0), total / Float64(n))


def _scale[dt: DType](mut values: nm.NDArray[dt], scalar_scale: Bool = False) raises:
    """Population standard deviation without centering the returned data."""
    var sums = sum_axis2d(values, 0)
    var means = nm.zeros[dt](nm.Shape(1, values.shape[1]))
    for j in range(values.shape[1]):
        means.unsafe_set(j, sums.unsafe_get(j) / SIMD[dt, 1](values.shape[0]))
    var centered = values - means
    var transposed = centered.T()
    var squares = row_squared_norms(transposed)
    var eps = Float64(2.220446049250313e-16)
    comptime if dt == DType.float32:
        eps = 1.1920928955078125e-7
    for j in range(values.shape[1]):
        var scale = sqrt(squares.unsafe_get(j) / SIMD[dt, 1](values.shape[0]))
        if scale == 0 or (not scalar_scale and Float64(scale) < 10 * eps):
            scale = 1
        for i in range(values.shape[0]):
            var offset = i * values.shape[1] + j
            values.unsafe_set(offset, values.unsafe_get(offset) / scale)


def _estimate_mi[dt: DType, yt: DType](mut X: nm.NDArray[dt], y: nm.NDArray[yt], mut rng: RandomState, discrete_features: DiscreteFeatures, discrete_target: Bool, n_neighbors: Int, copy: Bool, n_jobs: Optional[Int]) raises -> nm.NDArray[DType.float64]:
    comptime assert yt == DType.float32 or yt == DType.float64
    _validate_x(X)
    _validate_y(y, X.shape[0])
    if n_neighbors < 1:
        raise Error("ValueError: n_neighbors must be positive")
    if n_jobs and n_jobs.value() == 0:
        raise Error("ValueError: n_jobs cannot be zero")
    var mask = _discrete_mask(discrete_features, X.shape[1])
    var n = X.shape[0]
    var p = X.shape[1]
    var continuous = List[Int]()
    for j in range(p):
        if not mask[j]:
            continuous.append(j)
    var target = nm.zeros[yt](nm.Shape(n, 1))
    for i in range(n):
        var value = y.unsafe_get(i * Int(y.strides[0]))
        if discrete_target and value != floor(value):
            raise Error("ValueError: classification target contains continuous values")
        target.unsafe_set(i, value)
    var work = nm.zeros[DType.float64](nm.Shape(n, p))
    for i in range(n):
        for j in range(p):
            work.unsafe_set(i * p + j, Float64(X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))))
    if len(continuous) > 0:
        var scaled = nm.zeros[DType.float64](nm.Shape(n, len(continuous)))
        for i in range(n):
            for j in range(len(continuous)):
                scaled.unsafe_set(i * len(continuous) + j, Float64(X.unsafe_get(i * Int(X.strides[0]) + continuous[j] * Int(X.strides[1]))))
        _scale(scaled)
        var magnitudes = nm.zeros[DType.float64](nm.Shape(n, len(continuous)))
        for i in range(n * len(continuous)):
            magnitudes.unsafe_set(i, abs(scaled.unsafe_get(i)))
        var sums = sum_axis2d(magnitudes, 0)
        for i in range(n):
            for j in range(len(continuous)):
                var noise = 1e-10 * max(Float64(1), sums.unsafe_get(j) / Float64(n)) * rng.standard_normal()
                var value = scaled.unsafe_get(i * len(continuous) + j) + noise
                if not isfinite(value):
                    raise Error("ValueError: nonfinite scaled feature")
                work.unsafe_set(i * p + continuous[j], value)
                comptime if dt == DType.float64:
                    if not copy:
                        X.unsafe_set(i * Int(X.strides[0]) + continuous[j] * Int(X.strides[1]), SIMD[dt, 1](value))
    if not discrete_target:
        _scale(target, scalar_scale=True)
        var magnitudes = nm.zeros[yt](nm.Shape(n, 1))
        for i in range(n):
            magnitudes.unsafe_set(i, abs(target.unsafe_get(i)))
        var sums = sum_axis2d(magnitudes, 0)
        var magnitude = max(Float64(1), Float64(sums.unsafe_get(0) / SIMD[yt, 1](n)))
        for i in range(n):
            target.unsafe_set(i, SIMD[yt, 1](Float64(target.unsafe_get(i)) + 1e-10 * magnitude * rng.standard_normal()))
    var target64 = target.astype[DType.float64]()
    var h = _harmonics(n)
    var scores = nm.zeros[DType.float64](nm.Shape(p))
    for j in range(p):
        var feature = _column(work, j)
        var value: Float64
        if mask[j] and discrete_target:
            value = _compute_mi_dd(feature, target64)
        elif mask[j]:
            value = _compute_mi_cd(target64, feature, n_neighbors, h)
        elif discrete_target:
            value = _compute_mi_cd(feature, target64, n_neighbors, h)
        else:
            value = _compute_mi_cc(feature, target64, n_neighbors, h)
        scores.unsafe_set(j, value)
    return scores^


def mutual_info_classif[dt: DType, yt: DType](mut X: nm.NDArray[dt], y: nm.NDArray[yt], *, discrete_features: DiscreteFeatures = DiscreteFeatures(String("auto")), n_neighbors: Int = 3, copy: Bool = True, random_state: Optional[Int] = None, n_jobs: Optional[Int] = None) raises -> nm.NDArray[DType.float64]:
    var rng = check_random_state(random_state)
    _validate_x(X)
    var codes = _class_codes(y, X.shape[0])
    return _estimate_mi(X, codes, rng[], discrete_features, True, n_neighbors, copy, n_jobs)


def mutual_info_classif[dt: DType, yt: DType](mut X: nm.NDArray[dt], y: nm.NDArray[yt], *, discrete_features: DiscreteFeatures = DiscreteFeatures(String("auto")), n_neighbors: Int = 3, copy: Bool = True, mut random_state: RandomState, n_jobs: Optional[Int] = None) raises -> nm.NDArray[DType.float64]:
    _validate_x(X)
    var codes = _class_codes(y, X.shape[0])
    return _estimate_mi(X, codes, random_state, discrete_features, True, n_neighbors, copy, n_jobs)


def mutual_info_classif[dt: DType](mut X: nm.NDArray[dt], y: StringTarget, *, discrete_features: DiscreteFeatures = DiscreteFeatures(String("auto")), n_neighbors: Int = 3, copy: Bool = True, random_state: Optional[Int] = None, n_jobs: Optional[Int] = None) raises -> nm.NDArray[DType.float64]:
    var rng = check_random_state(random_state)
    _validate_x(X)
    var codes = _class_codes(y, X.shape[0])
    return _estimate_mi(X, codes, rng[], discrete_features, True, n_neighbors, copy, n_jobs)


def mutual_info_classif[dt: DType](mut X: nm.NDArray[dt], y: StringTarget, *, discrete_features: DiscreteFeatures = DiscreteFeatures(String("auto")), n_neighbors: Int = 3, copy: Bool = True, mut random_state: RandomState, n_jobs: Optional[Int] = None) raises -> nm.NDArray[DType.float64]:
    _validate_x(X)
    var codes = _class_codes(y, X.shape[0])
    return _estimate_mi(X, codes, random_state, discrete_features, True, n_neighbors, copy, n_jobs)


def mutual_info_regression[dt: DType, yt: DType](mut X: nm.NDArray[dt], y: nm.NDArray[yt], *, discrete_features: DiscreteFeatures = DiscreteFeatures(String("auto")), n_neighbors: Int = 3, copy: Bool = True, random_state: Optional[Int] = None, n_jobs: Optional[Int] = None) raises -> nm.NDArray[DType.float64]:
    var rng = check_random_state(random_state)
    return _estimate_mi(X, y, rng[], discrete_features, False, n_neighbors, copy, n_jobs)


def mutual_info_regression[dt: DType, yt: DType](mut X: nm.NDArray[dt], y: nm.NDArray[yt], *, discrete_features: DiscreteFeatures = DiscreteFeatures(String("auto")), n_neighbors: Int = 3, copy: Bool = True, mut random_state: RandomState, n_jobs: Optional[Int] = None) raises -> nm.NDArray[DType.float64]:
    return _estimate_mi(X, y, random_state, discrete_features, False, n_neighbors, copy, n_jobs)
