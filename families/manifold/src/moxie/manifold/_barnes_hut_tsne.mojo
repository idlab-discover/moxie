# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Barnes–Hut forces over dense embeddings and fixed-k input neighborhoods."""
from numojo import NDArray
from std.builtin.sort import sort
from std.math import abs, isfinite
from numax.core.libm import log
from moxie.neighbors._distance import metric_power

comptime _Point = SIMD[DType.float32, 4]
comptime _Pair = Tuple[Int, Int, Float64]
comptime _DUPLICATE_TOLERANCE: Float32 = 1e-6
comptime _FLOAT32_TINY: Float32 = 1.1754943508222875e-38
comptime _EPS: Float64 = 2.220446049250313e-16


def _squared_norm(delta: _Point) -> Float32:
    # Keep the reference's Float32 accumulation order for up to three axes.
    var squares = delta * delta
    return (squares[0] + squares[1]) + squares[2]


def _weight(squared: Float32, dof: Float32) -> Float64:
    var base = Float64(dof / (dof + squared))
    return base if dof == 1 else metric_power(base, 1.5)


struct _Cell(Copyable, Movable):
    var lower: _Point
    var upper: _Point
    var middle: _Point
    var squared_width: Float32
    var mean: _Point
    var count: Int
    var leaf: Bool
    var children: InlineArray[Int, 8]

    def __init__(out self, lower: _Point, upper: _Point, mean: _Point, count: Int):
        self.lower = lower
        self.upper = upper
        self.middle = (lower + upper) * 0.5
        for axis in range(3):
            # Force the split to progress when the bounds are adjacent floats.
            if self.middle[axis] <= lower[axis] and upper[axis] > lower[axis]:
                self.middle[axis] = upper[axis]
        var width = (upper - lower).reduce_max()
        self.squared_width = width * width
        self.mean = mean
        self.count = count
        self.leaf = True
        self.children = InlineArray[Int, 8](fill=-1)


struct _Tree(Movable):
    var cells: List[_Cell]
    var dimensions: Int

    def __init__(out self, points: List[_Point], dimensions: Int) raises:
        self.dimensions = dimensions
        self.cells = List[_Cell]()
        var lower = points[0]
        var upper = lower
        for point in points:
            lower = min(lower, point)
            upper = max(upper, point)
        for axis in range(dimensions):
            # Match sklearn's padded upper bound for positive and negative axes.
            var factor: Float32 = 1.001 if upper[axis] >= 0 else 0.999
            upper[axis] = max(upper[axis] * factor, upper[axis] + 0.001)
        self.cells.append(_Cell(lower, upper, points[0], 1))
        if not isfinite(self.cells[0].squared_width * 3):
            raise Error("ValueError: t-SNE Barnes–Hut coordinate range exceeds Float32")
        for i in range(1, len(points)):
            self._insert(points[i])

    def _child_slot(self, point: _Point, middle: _Point) -> Int:
        var slot = 0
        for axis in range(self.dimensions):
            slot = (slot << 1) + Int(point[axis] >= middle[axis])
        return slot

    def _add_child(mut self, parent: Int, slot: Int, point: _Point, count: Int):
        var lower = self.cells[parent].lower
        var upper = self.cells[parent].upper
        var middle = self.cells[parent].middle
        for axis in range(self.dimensions):
            if slot & (1 << (self.dimensions - axis - 1)):
                lower[axis] = middle[axis]
            else:
                upper[axis] = middle[axis]
        self.cells[parent].children[slot] = len(self.cells)
        self.cells[parent].leaf = False
        self.cells.append(_Cell(lower, upper, point, count))

    def _insert(mut self, point: _Point):
        var node = 0
        while True:
            var count = self.cells[node].count
            var mean = self.cells[node].mean
            var middle = self.cells[node].middle
            if self.cells[node].leaf:
                if abs(point - mean).reduce_max() <= _DUPLICATE_TOLERANCE:
                    self.cells[node].count += 1
                    return
                var slot = self._child_slot(mean, middle)
                self._add_child(node, slot, mean, count)
            self.cells[node].mean = (mean * Float32(count) + point) / Float32(count + 1)
            self.cells[node].count += 1
            var slot = self._child_slot(point, middle)
            var child = self.cells[node].children[slot]
            if child < 0:
                self._add_child(node, slot, point, 1)
                return
            node = child

    def repulsion(
        self, node: Int, point: _Point, dof: Float32, squared_angle: Float32,
        mut normalizer: Float64, mut force: _Point,
    ):
        ref cell = self.cells[node]
        var delta = point - cell.mean
        if cell.leaf and abs(delta).reduce_max() <= _DUPLICATE_TOLERANCE:
            return
        var squared = _squared_norm(delta)
        if cell.leaf or (squared > 0 and cell.squared_width / squared < squared_angle):
            var weight = _weight(squared, dof)
            var mass = Float64(cell.count) * weight
            normalizer += mass
            force += Float32(mass * weight) * delta
            return
        for slot in range(1 << self.dimensions):
            var child = cell.children[slot]
            if child >= 0:
                self.repulsion(child, point, dof, squared_angle, normalizer, force)


def _merge_neighbor_probabilities(var directed: List[_Pair]) raises -> List[_Pair]:
    """Combine directed k-neighbor weights by unordered endpoints."""
    sort(Span(directed))
    var total: Float64 = 0
    var end = 0
    var start = 0
    while start < len(directed):
        var left, right, probability = directed[start]
        start += 1
        if start < len(directed) and directed[start][0] == left and directed[start][1] == right:
            probability += directed[start][2]
            start += 1
        directed[end] = (left, right, probability)
        total += probability
        end += 1
    var joint = List[_Pair](capacity=end)
    var normalizer = max(2 * total, _EPS)
    for i in range(end):
        var left, right, probability = directed[i]
        if probability > 0:
            joint.append((left, right, probability / normalizer))
    return joint^


def _barnes_hut_objective[dtype: DType](
    coordinates: NDArray[dtype], pairs: List[_Pair], probability_scale: Float64,
    mut gradient: NDArray[DType.float64], angle: Float32, compute_error: Bool,
) raises -> Float64:
    var n = coordinates.shape[0]
    var dimensions = coordinates.shape[1]
    var points = List[_Point](capacity=n)
    for i in range(n):
        var point = _Point(0)
        for axis in range(dimensions):
            var value = Float32(coordinates.unsafe_get(i * dimensions + axis))
            if not isfinite(value):
                raise Error("ValueError: t-SNE Barnes–Hut coordinates must fit in Float32")
            point[axis] = value
        points.append(point)
    var tree = _Tree(points, dimensions)
    var dof = Float32(max(dimensions - 1, 1))
    var normalizer: Float64 = 0
    var negative = List[_Point](capacity=n)
    var positive = List[_Point](capacity=n)
    for i in range(n):
        var force = _Point(0)
        tree.repulsion(0, points[i], dof, angle * angle, normalizer, force)
        negative.append(force)
        positive.append(_Point(0))
    normalizer = max(normalizer, _EPS)

    var loss: Float32 = 0
    for pair in pairs:
        var left, right, probability = pair
        var delta = points[left] - points[right]
        var weight = Float32(_weight(_squared_norm(delta), dof))
        var p = Float32(probability * probability_scale)
        var force = p * weight * delta
        positive[left] += force
        positive[right] -= force
        if compute_error:
            var q = weight / Float32(normalizer)
            var ratio = max(p, _FLOAT32_TINY) / max(q, _FLOAT32_TINY)
            loss = Float32(Float64(loss) + 2 * Float64(p) * log(Float64(ratio)))
    for i in range(n):
        for axis in range(dimensions):
            var force = Float32(Float64(positive[i][axis]) - Float64(negative[i][axis]) / normalizer)
            gradient.unsafe_set(i * dimensions + axis, Float64(force))
    return Float64(loss)
