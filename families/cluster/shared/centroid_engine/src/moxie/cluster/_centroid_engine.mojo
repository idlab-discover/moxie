"""Private dense centroid kernels on the shared NuMojo storage contract.

Algorithms adapted from scikit-learn (BSD-3-Clause); see THIRD_PARTY_NOTICES.
"""
import numojo as nm
from std.math import sqrt, isfinite
from std.utils.numerics import inf
from moxie._dense import row_squared_norms
from moxie.cluster._centroid_selection import relocation_order
from moxie.cluster._contracts import (
    Labels, AssignmentResult, CenterUpdate, CentroidAssignment, CentroidUpdate,
    validate_matrix, validate_labels,
)


@always_inline
def _at[dtype: DType](a: nm.NDArray[dtype], i: Int, j: Int) raises -> SIMD[dtype, 1]:
    return a.unsafe_get(i * Int(a.strides[0]) + j * Int(a.strides[1]))


@always_inline
def _weight[dtype: DType](a: nm.NDArray[dtype], i: Int) raises -> SIMD[dtype, 1]:
    return a.unsafe_get(i * Int(a.strides[0]))


def _finite[dtype: DType](a: nm.NDArray[dtype]) raises:
    if a.ndim == 1:
        for i in range(a.shape[0]):
            if not isfinite(_weight(a, i)):
                raise Error("ValueError: nonfinite centroid input")
    else:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if not isfinite(_at(a, i, j)):
                    raise Error("ValueError: nonfinite centroid input")


def _validate[dtype: DType](x: nm.NDArray[dtype], centers: nm.NDArray[dtype]) raises:
    validate_matrix(centers)
    if x.ndim != 2 or x.shape[1] != centers.shape[1]:
        raise Error("ValueError: sample/center dimensions disagree")
    _finite(x)
    _finite(centers)


def _weights[dtype: DType](weights: nm.NDArray[dtype], n: Int) raises:
    if weights.ndim != 1 or weights.size != n:
        raise Error("ValueError: weights must have shape (n_samples,)")
    _finite(weights)


@always_inline
def _squared[dtype: DType](x: nm.NDArray[dtype], i: Int, c: nm.NDArray[dtype], j: Int) raises -> SIMD[dtype, 1]:
    var total = SIMD[dtype, 1](0)
    for f in range(x.shape[1]):
        var delta = _at(x, i, f) - _at(c, j, f)
        total += delta * delta
    return total


def distances[dtype: DType](x: nm.NDArray[dtype], centers: nm.NDArray[dtype], squared: Bool = True) raises -> nm.NDArray[dtype]:
    """Owned (n,k) distances; squared=False returns Euclidean units."""
    _validate(x, centers)
    var out_array = nm.zeros[dtype](nm.Shape(x.shape[0], centers.shape[0]))
    for i in range(x.shape[0]):
        for j in range(centers.shape[0]):
            var value = _squared(x, i, centers, j)
            if not squared:
                value = sqrt(value)
            out_array.unsafe_set(i * centers.shape[0] + j, value)
    return out_array^


def inertia[dtype: DType](x: nm.NDArray[dtype], centers: nm.NDArray[dtype], weights: nm.NDArray[dtype], labels: Labels, single_label: Int = -1) raises -> Float64:
    _validate(x, centers)
    _weights(weights, x.shape[0])
    validate_labels(labels, x.shape[0], centers.shape[0])
    var total = SIMD[dtype, 1](0)
    for i in range(x.shape[0]):
        var j = Int(_weight(labels, i))
        if single_label < 0 or j == single_label:
            total += _squared(x, i, centers, j) * _weight(weights, i)
    return Float64(total)


def assign[dtype: DType](x: nm.NDArray[dtype], centers: nm.NDArray[dtype], weights: nm.NDArray[dtype]) raises -> AssignmentResult:
    """Lloyd E-step; first center wins equal norm/dot scores."""
    _validate(x, centers)
    _weights(weights, x.shape[0])
    var labels = nm.zeros[DType.int64](nm.Shape(x.shape[0]))
    var norms = row_squared_norms(centers)
    comptime if dtype == DType.float32:
        # sklearn's assignment uses same-dtype center norms. Widened norms
        # change ties even for identical unit rows after streaming rounding.
        for j in range(centers.shape[0]):
            var norm = SIMD[dtype, 1](0)
            for f in range(centers.shape[1]):
                var value = _at(centers, j, f)
                norm += value * value
            norms.unsafe_set(j, norm)
    for i in range(x.shape[0]):
        var best = 0
        var best_score = SIMD[dtype, 1](0)
        for j in range(centers.shape[0]):
            var dot = SIMD[dtype, 1](0)
            for f in range(x.shape[1]):
                dot += _at(x, i, f) * _at(centers, j, f)
            var score = norms.unsafe_get(j) - 2 * dot
            if j == 0 or score < best_score:
                best = j
                best_score = score
        labels.unsafe_set(i, Int64(best))
    var loss = inertia(x, centers, weights, labels)
    return AssignmentResult(labels^, loss)


def update[dtype: DType](x: nm.NDArray[dtype], labels: Labels, weights: nm.NDArray[dtype], previous: nm.NDArray[dtype], *, allow_singleton: Bool = False) raises -> CenterUpdate[dtype]:
    """Weighted M-step including relocation; borrows all input buffers."""
    _validate(x, previous)
    _weights(weights, x.shape[0])
    var n = x.shape[0]
    var k = previous.shape[0]
    var d = x.shape[1]
    if n == 0 or (k > n and not (allow_singleton and n == 1 and k == 2)):
        raise Error("ValueError: center update requires n_samples >= n_centers")
    validate_labels(labels, n, k)
    var centers = nm.zeros[dtype](nm.Shape(k, d))
    var masses = nm.zeros[dtype](nm.Shape(k))
    for i in range(n):
        var j = Int(_weight(labels, i))
        var w = _weight(weights, i)
        masses.unsafe_set(j, masses.unsafe_get(j) + w)
        for f in range(d):
            centers.unsafe_set(j * d + f, centers.unsafe_get(j * d + f) + w * _at(x, i, f))
    var empty = List[Int]()
    for j in range(k):
        if masses.unsafe_get(j) == 0:
            empty.append(j)
    if len(empty):
        var residual = nm.zeros[dtype](nm.Shape(n))
        var largest = SIMD[dtype, 1](0)
        for i in range(n):
            var value = _squared(x, i, previous, Int(_weight(labels, i)))
            residual.unsafe_set(i, value)
            if value > largest:
                largest = value
        if largest > 0:
            var selected = relocation_order(residual, len(empty))
            for position in range(len(empty)):
                var j = empty[position]
                var far = Int(selected.unsafe_get(position))
                var old = Int(_weight(labels, far))
                var w = _weight(weights, far)
                for f in range(d):
                    var value = _at(x, far, f) * w
                    centers.unsafe_set(old * d + f, centers.unsafe_get(old * d + f) - value)
                    centers.unsafe_set(j * d + f, value)
                masses.unsafe_set(j, w)
                masses.unsafe_set(old, masses.unsafe_get(old) - w)
    var biggest = 0
    for j in range(1, k):
        if masses.unsafe_get(j) > masses.unsafe_get(biggest):
            biggest = j
    # Preserve sklearn's ordered normalization/fallback semantics.
    for j in range(k):
        if masses.unsafe_get(j) > 0:
            var alpha = 1 / masses.unsafe_get(j)
            for f in range(d):
                centers.unsafe_set(j * d + f, centers.unsafe_get(j * d + f) * alpha)
        else:
            for f in range(d):
                centers.unsafe_set(j * d + f, centers.unsafe_get(biggest * d + f))
    return CenterUpdate[dtype](centers^, masses^)


def center_shift[dtype: DType](old: nm.NDArray[dtype], new: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
    _validate(old, new)
    if old.shape[0] != new.shape[0]:
        raise Error("ValueError: center counts disagree")
    var shifts = nm.zeros[dtype](nm.Shape(old.shape[0]))
    for j in range(old.shape[0]):
        shifts.unsafe_set(j, sqrt(_squared(old, j, new, j)))
    return shifts^


def minibatch_update[dtype: DType](x: nm.NDArray[dtype], labels: Labels, weights: nm.NDArray[dtype], previous: nm.NDArray[dtype], prior_masses: nm.NDArray[dtype]) raises -> CenterUpdate[dtype]:
    """Return new centers and cumulative mass without mutating prior state."""
    _validate(x, previous)
    _weights(weights, x.shape[0])
    _weights(prior_masses, previous.shape[0])
    validate_labels(labels, x.shape[0], previous.shape[0])
    var k = previous.shape[0]
    var d = x.shape[1]
    var centers = nm.zeros[dtype](nm.Shape(k, d))
    var masses = nm.zeros[dtype](nm.Shape(k))
    for j in range(k):
        var batch_mass = SIMD[dtype, 1](0)
        for i in range(x.shape[0]):
            if Int(_weight(labels, i)) == j:
                batch_mass += _weight(weights, i)
        var mass = _weight(prior_masses, j)
        if batch_mass > 0:
            mass += batch_mass
            if mass <= 0:
                raise Error("ValueError: nonpositive cumulative mini-batch mass")
            for f in range(d):
                var value = _at(previous, j, f) * _weight(prior_masses, j)
                for i in range(x.shape[0]):
                    if Int(_weight(labels, i)) == j:
                        value += _at(x, i, f) * _weight(weights, i)
                centers.unsafe_set(j * d + f, value * (1 / mass))
        else:
            for f in range(d):
                centers.unsafe_set(j * d + f, _at(previous, j, f))
        masses.unsafe_set(j, mass)
    return CenterUpdate[dtype](centers^, masses^)


@fieldwise_init
struct Lloyd[dtype_: DType](CentroidAssignment, CentroidUpdate):
    """Stateless adapter implementing the family's assignment/update seams."""
    comptime dtype = Self.dtype_

    def assign(self, x: nm.NDArray[Self.dtype], centers: nm.NDArray[Self.dtype], weights: nm.NDArray[Self.dtype]) raises -> AssignmentResult:
        return assign(x, centers, weights)

    def update(self, x: nm.NDArray[Self.dtype], labels: Labels, weights: nm.NDArray[Self.dtype], previous: nm.NDArray[Self.dtype]) raises -> CenterUpdate[Self.dtype]:
        return update(x, labels, weights, previous)


struct ElkanBounds[dtype: DType](Copyable, Movable):
    """Persistent bounds for one immutable sample set and owned center snapshot.

    Call advance after each M-step; reinitialize for different samples.
    """
    var labels: Labels
    var upper: nm.NDArray[Self.dtype]
    var lower: nm.NDArray[Self.dtype]
    var centers: nm.NDArray[Self.dtype]
    var half_distances: nm.NDArray[Self.dtype]
    var next_center: nm.NDArray[Self.dtype]

    def __init__(out self, x: nm.NDArray[Self.dtype], centers: nm.NDArray[Self.dtype]) raises:
        _validate(x, centers)
        var n = x.shape[0]
        var k = centers.shape[0]
        self.centers = centers.copy()
        self.labels = nm.zeros[DType.int64](nm.Shape(n))
        self.upper = nm.zeros[Self.dtype](nm.Shape(n))
        self.lower = nm.zeros[Self.dtype](nm.Shape(n, k))
        self.half_distances = distances(centers, centers, False)
        self.next_center = nm.zeros[Self.dtype](nm.Shape(k))
        self._center_bounds()
        for i in range(n):
            var best = 0
            var distance = sqrt(_squared(x, i, centers, 0))
            self.lower.unsafe_set(i * k, distance)
            for j in range(1, k):
                if distance > self.half_distances.unsafe_get(best * k + j):
                    var candidate = sqrt(_squared(x, i, centers, j))
                    self.lower.unsafe_set(i * k + j, candidate)
                    if candidate < distance:
                        distance = candidate
                        best = j
            self.labels.unsafe_set(i, Int64(best))
            self.upper.unsafe_set(i, distance)

    def _center_bounds(mut self) raises:
        var k = self.centers.shape[0]
        for j in range(k):
            var nearest = inf[Self.dtype]()
            for other in range(k):
                var half = self.half_distances.unsafe_get(j * k + other) * 0.5
                self.half_distances.unsafe_set(j * k + other, half)
                if j != other and half < nearest:
                    nearest = half
            self.next_center.unsafe_set(j, nearest)

    def assign(mut self, x: nm.NDArray[Self.dtype], weights: nm.NDArray[Self.dtype]) raises -> AssignmentResult:
        _validate(x, self.centers)
        _weights(weights, x.shape[0])
        if x.shape[0] != self.labels.size:
            raise Error("ValueError: Elkan sample count changed; initialize new bounds")
        var k = self.centers.shape[0]
        for i in range(x.shape[0]):
            var label = Int(self.labels.unsafe_get(i))
            var upper = self.upper.unsafe_get(i)
            var tight = False
            if self.next_center.unsafe_get(label) < upper:
                for j in range(k):
                    if j != label and upper > self.lower.unsafe_get(i * k + j) and upper > self.half_distances.unsafe_get(label * k + j):
                        if not tight:
                            upper = sqrt(_squared(x, i, self.centers, label))
                            self.lower.unsafe_set(i * k + label, upper)
                            tight = True
                        if upper > self.lower.unsafe_get(i * k + j) or upper > self.half_distances.unsafe_get(label * k + j):
                            var distance = sqrt(_squared(x, i, self.centers, j))
                            self.lower.unsafe_set(i * k + j, distance)
                            if distance < upper:
                                upper = distance
                                label = j
                self.labels.unsafe_set(i, Int64(label))
                self.upper.unsafe_set(i, upper)
        var loss = inertia(x, self.centers, weights, self.labels)
        return AssignmentResult(self.labels.copy(), loss)

    def advance(mut self, new_centers: nm.NDArray[Self.dtype]) raises:
        """Shift bounds by center movement and replace the owned snapshot."""
        var shifts = center_shift(self.centers, new_centers)
        var next_centers = new_centers.copy()
        var half = distances(new_centers, new_centers, False)
        var k = self.centers.shape[0]
        for i in range(self.labels.size):
            var label = Int(self.labels.unsafe_get(i))
            self.upper.unsafe_set(i, self.upper.unsafe_get(i) + shifts.unsafe_get(label))
            for j in range(k):
                var bound = self.lower.unsafe_get(i * k + j) - shifts.unsafe_get(j)
                if bound < 0:
                    bound = 0
                self.lower.unsafe_set(i * k + j, bound)
        self.centers = next_centers^
        self.half_distances = half^
        self._center_bounds()
