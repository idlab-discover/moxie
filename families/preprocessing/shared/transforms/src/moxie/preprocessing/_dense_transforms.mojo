"""Borrowed, dtype-preserving dense transform primitives. No estimator validation.

Reference formulas: sklearn preprocessing/_data.py (BSD-3-Clause).
See LICENSES/sklearn.txt and docs/transforms.md for ownership and rounding.
"""
from numojo import NDArray, Shape
from std.math import abs, sqrt, isnan

comptime DEFAULT_TILE_BYTES = 65536


def _matrix[d: DType](x: NDArray[d]) raises:
    comptime assert d == DType.float32 or d == DType.float64, "Float32/64 only"
    if x.ndim != 2:
        raise Error("ValueError: expected a rank-2 dense array")


def _tile(bytes: Int) raises:
    if bytes < 8 or bytes > 8388608:
        raise Error("ValueError: tile_bytes must be between 8 and 8 MiB")


def _gcd(a: Int, b: Int) -> Int:
    var u = abs(a)
    var v = abs(b)
    while v != 0:
        var t = u % v
        u = v
        v = t
    return u


def _writable[d: DType](x: NDArray[d]) raises:
    _matrix(x)
    if not x.flags.WRITEABLE:
        raise Error("ValueError: destination is not writable; no fallback copy")
    var n = Int(x.shape[0])
    var p = Int(x.shape[1])
    if n == 0 or p == 0:
        return
    var a = abs(Int(x.strides[0]))
    var b = abs(Int(x.strides[1]))
    if (n > 1 and a == 0) or (p > 1 and b == 0):
        raise Error("ValueError: overlapping destination elements")
    if n > 1 and p > 1:
        var g = _gcd(a, b)
        if b // g < n and a // g < p:
            raise Error("ValueError: overlapping destination elements")


def _lo[d: DType](x: NDArray[d]) raises -> Int:
    var offset = 0
    for i in range(x.ndim):
        var delta = (Int(x.shape[i]) - 1) * Int(x.strides[i])
        if delta < 0:
            offset += delta
    return Int(x.unsafe_ptr()) + offset * Int(4 if d == DType.float32 else 8)


def _hi[d: DType](x: NDArray[d]) raises -> Int:
    var offset = 0
    for i in range(x.ndim):
        var delta = (Int(x.shape[i]) - 1) * Int(x.strides[i])
        if delta > 0:
            offset += delta
    return Int(x.unsafe_ptr()) + (offset + 1) * Int(4 if d == DType.float32 else 8)


def _intersects[d: DType, e: DType](a: NDArray[d], b: NDArray[e]) raises -> Bool:
    return a.size > 0 and b.size > 0 and _lo(a) < _hi(b) and _lo(b) < _hi(a)


def _pair[d: DType](x: NDArray[d], y: NDArray[d]) raises:
    _matrix(x)
    _writable(y)
    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        raise Error("ValueError: output shape mismatch")
    var identical = Int(x.unsafe_ptr()) == Int(y.unsafe_ptr()) and x.strides[0] == y.strides[0] and x.strides[1] == y.strides[1]
    if _intersects(x, y) and not identical:
        raise Error("ValueError: potentially overlapping input/output; use disjoint storage")


def _parameters[d: DType, s: DType](x: NDArray[d], shift: NDArray[s], scale: NDArray[s], axis: Int, operation: Int) raises:
    comptime assert s == DType.float32 or s == DType.float64, "Float32/64 parameters only"
    _matrix(x)
    if axis != 0 and axis != 1:
        raise Error("ValueError: axis must be 0 or 1")
    if operation < 0 or operation > 5:
        raise Error("ValueError: unknown transform operation")
    if operation < 4:
        var count = Int(x.shape[1-axis])
        if shift.ndim != 1 or scale.ndim != 1 or shift.size != count or scale.size != count:
            raise Error("ValueError: parameter length mismatch")


def _value[d: DType, s: DType](value: SIMD[d, 1], shift: SIMD[s, 1], scale: SIMD[s, 1], operation: Int, lower: SIMD[d, 1], upper: SIMD[d, 1]) -> SIMD[d, 1]:
    comptime calc = DType.float64 if d == DType.float64 or s == DType.float64 else DType.float32
    var v = value
    if operation == 0:
        v = (v.cast[calc]() - shift.cast[calc]()).cast[d]()
        v = (v.cast[calc]() / scale.cast[calc]()).cast[d]()
    elif operation == 1:
        v = (v.cast[calc]() * scale.cast[calc]()).cast[d]()
        v = (v.cast[calc]() + shift.cast[calc]()).cast[d]()
    elif operation == 2:
        v = (v.cast[calc]() * scale.cast[calc]()).cast[d]()
        v = (v.cast[calc]() + shift.cast[calc]()).cast[d]()
        if isnan(lower):
            v = lower
        elif isnan(upper):
            v = upper
        else:
            if v < lower:
                v = lower
            if v > upper:
                v = upper
    elif operation == 3:
        v = (v.cast[calc]() - shift.cast[calc]()).cast[d]()
        v = (v.cast[calc]() / scale.cast[calc]()).cast[d]()
    elif operation == 4:
        if isnan(lower):
            v = lower
        elif isnan(upper):
            v = upper
        else:
            if v < lower:
                v = lower
            if v > upper:
                v = upper
    else:
        v = 1 if v > lower else 0
    return v


def transform_into[d: DType, s: DType](x: NDArray[d], mut y: NDArray[d], shift: NDArray[s], scale: NDArray[s], *, operation: Int = 0, axis: Int = 0, lower: SIMD[d, 1] = 0, upper: SIMD[d, 1] = 1, tile_bytes: Int = DEFAULT_TILE_BYTES) raises:
    """0 center/divide; 1 multiply/add; 2 multiply/add/clip; 3 minmax inverse;
    4 clip; 5 threshold. axis=0 parameters per feature, axis=1 per row.
    """
    _tile(tile_bytes)
    _parameters(x, shift, scale, axis, operation)
    _pair(x, y)
    if operation < 4 and (_intersects(shift, y) or _intersects(scale, y)):
        raise Error("ValueError: parameters alias destination")
    var p = Int(x.shape[1])
    var width = tile_bytes // Int(4 if d == DType.float32 else 8)
    for r in range(Int(x.shape[0])):
        for start in range(0, p, width):
            for c in range(start, min(start + width, p)):
                var k = c if axis == 0 else r
                var a = SIMD[s, 1](0)
                var b = SIMD[s, 1](1)
                if operation < 4:
                    a = shift.unsafe_get(k * Int(shift.strides[0]))
                    b = scale.unsafe_get(k * Int(scale.strides[0]))
                var v = x.unsafe_get(r * Int(x.strides[0]) + c * Int(x.strides[1]))
                y.unsafe_set(r * Int(y.strides[0]) + c * Int(y.strides[1]), _value(v, a, b, operation, lower, upper))


def transform_inplace[d: DType, s: DType](mut x: NDArray[d], shift: NDArray[s], scale: NDArray[s], *, operation: Int = 0, axis: Int = 0, lower: SIMD[d, 1] = 0, upper: SIMD[d, 1] = 1, tile_bytes: Int = DEFAULT_TILE_BYTES) raises:
    var borrowed = x.view()
    transform_into(borrowed, x, shift, scale, operation=operation, axis=axis, lower=lower, upper=upper, tile_bytes=tile_bytes)


def transform[d: DType, s: DType](x: NDArray[d], shift: NDArray[s], scale: NDArray[s], *, operation: Int = 0, axis: Int = 0, lower: SIMD[d, 1] = 0, upper: SIMD[d, 1] = 1, tile_bytes: Int = DEFAULT_TILE_BYTES) raises -> NDArray[d]:
    _tile(tile_bytes)
    _parameters(x, shift, scale, axis, operation)
    var result = NDArray[d](Shape(Int(x.shape[0]), Int(x.shape[1])))
    transform_into(x, result, shift, scale, operation=operation, axis=axis, lower=lower, upper=upper, tile_bytes=tile_bytes)
    return result^


def _norm_args[d: DType](x: NDArray[d], axis: Int, norm: Int, tile_bytes: Int) raises:
    _matrix(x)
    _tile(tile_bytes)
    if axis != 0 and axis != 1:
        raise Error("ValueError: axis must be 0 or 1")
    if norm < 1 or norm > 3:
        raise Error("ValueError: norm is 1=L1, 2=L2, 3=max")
    if x.shape[axis] == 0:
        raise Error("ValueError: empty reduction dimension")


def _norm[d: DType](x: NDArray[d], k: Int, axis: Int, norm: Int, tile_bytes: Int) raises -> SIMD[d, 1]:
    var total = SIMD[d, 1](0)
    var width = tile_bytes // Int(4 if d == DType.float32 else 8)
    for start in range(0, Int(x.shape[axis]), width):
        for j in range(start, min(start + width, Int(x.shape[axis]))):
            var v = x.unsafe_get(k * Int(x.strides[1-axis]) + j * Int(x.strides[axis]))
            if norm == 1:
                total += abs(v)
            elif norm == 2:
                total += v * v
            elif abs(v) > total or isnan(v):
                total = abs(v)
    if norm == 2:
        total = sqrt(total)
    var eps = SIMD[d, 1](2.220446049250313e-16)
    comptime if d == DType.float32:
        eps = 1.1920928955078125e-7
    if total < 10 * eps:
        total = 1
    return total


def norms_into[d: DType](x: NDArray[d], mut norms: NDArray[d], *, axis: Int = 1, norm: Int = 2, tile_bytes: Int = DEFAULT_TILE_BYTES) raises:
    _norm_args(x, axis, norm, tile_bytes)
    if norms.ndim != 1 or norms.size != Int(x.shape[1-axis]) or not norms.flags.WRITEABLE or (norms.size > 1 and norms.strides[0] == 0):
        raise Error("ValueError: invalid writable norm vector")
    if _intersects(x, norms):
        raise Error("ValueError: norms must not alias input")
    for k in range(norms.size):
        norms.unsafe_set(k * Int(norms.strides[0]), _norm(x, k, axis, norm, tile_bytes))


def normalize_into[d: DType](x: NDArray[d], mut y: NDArray[d], *, axis: Int = 1, norm: Int = 2, tile_bytes: Int = DEFAULT_TILE_BYTES) raises:
    _norm_args(x, axis, norm, tile_bytes)
    _pair(x, y)
    var width = tile_bytes // Int(4 if d == DType.float32 else 8)
    for k in range(Int(x.shape[1-axis])):
        var divisor = _norm(x, k, axis, norm, tile_bytes)
        for start in range(0, Int(x.shape[axis]), width):
            for j in range(start, min(start + width, Int(x.shape[axis]))):
                var v = x.unsafe_get(k * Int(x.strides[1-axis]) + j * Int(x.strides[axis]))
                y.unsafe_set(k * Int(y.strides[1-axis]) + j * Int(y.strides[axis]), v / divisor)


def normalize_inplace[d: DType](mut x: NDArray[d], *, axis: Int = 1, norm: Int = 2, tile_bytes: Int = DEFAULT_TILE_BYTES) raises:
    var borrowed = x.view()
    normalize_into(borrowed, x, axis=axis, norm=norm, tile_bytes=tile_bytes)


def normalize[d: DType](x: NDArray[d], *, axis: Int = 1, norm: Int = 2, tile_bytes: Int = DEFAULT_TILE_BYTES) raises -> NDArray[d]:
    _norm_args(x, axis, norm, tile_bytes)
    var result = NDArray[d](Shape(Int(x.shape[0]), Int(x.shape[1])))
    normalize_into(x, result, axis=axis, norm=norm, tile_bytes=tile_bytes)
    return result^


trait TileSink:
    """Synchronous callback. A sink must consume each borrowed tile before return."""
    def write[d: DType](mut self, row: Int, col: Int, tile: NDArray[d]) raises: ...


def transform_sink[d: DType, s: DType, S: TileSink](x: NDArray[d], mut sink: S, shift: NDArray[s], scale: NDArray[s], *, operation: Int = 0, axis: Int = 0, lower: SIMD[d, 1] = 0, upper: SIMD[d, 1] = 1, tile_bytes: Int = DEFAULT_TILE_BYTES) raises:
    _tile(tile_bytes)
    _parameters(x, shift, scale, axis, operation)
    var p = Int(x.shape[1])
    var width = min(p, tile_bytes // Int(4 if d == DType.float32 else 8))
    if width == 0:
        return
    var tile = NDArray[d](Shape(1, width))
    for r in range(Int(x.shape[0])):
        for start in range(0, p, width):
            var count = min(width, p-start)
            for j in range(count):
                var k = start+j if axis == 0 else r
                var a = SIMD[s, 1](0)
                var b = SIMD[s, 1](1)
                if operation < 4:
                    a = shift.unsafe_get(k * Int(shift.strides[0]))
                    b = scale.unsafe_get(k * Int(scale.strides[0]))
                var v = x.unsafe_get(r * Int(x.strides[0]) + (start+j) * Int(x.strides[1]))
                tile.unsafe_set(j, _value(v, a, b, operation, lower, upper))
            var chunk = tile.view_with_layout(Shape(1, count), tile.strides, 0)
            sink.write(r, start, chunk)


def normalize_sink[d: DType, S: TileSink](x: NDArray[d], mut sink: S, *, axis: Int = 1, norm: Int = 2, tile_bytes: Int = DEFAULT_TILE_BYTES) raises:
    _norm_args(x, axis, norm, tile_bytes)
    var width = min(Int(x.shape[axis]), tile_bytes // Int(4 if d == DType.float32 else 8))
    var tile = NDArray[d](Shape(1, width) if axis == 1 else Shape(width, 1))
    for k in range(Int(x.shape[1-axis])):
        var divisor = _norm(x, k, axis, norm, tile_bytes)
        for start in range(0, Int(x.shape[axis]), width):
            var count = min(width, Int(x.shape[axis])-start)
            for j in range(count):
                var v = x.unsafe_get(k * Int(x.strides[1-axis]) + (start+j) * Int(x.strides[axis]))
                tile.unsafe_set(j, v / divisor)
            var chunk = tile.view_with_layout(Shape(1, count) if axis == 1 else Shape(count, 1), tile.strides, 0)
            sink.write(k if axis == 1 else start, start if axis == 1 else k, chunk)


struct FeatureNorms[d: DType](Copyable, Movable):
    """One feature-norm pass over borrowed row chunks, then replay to divide.

    Chunk boundaries do not finalize norms. No chunk is retained. Accumulation
    order is rows in arrival order and input dtype, including overflow and NaNs.
    """
    var values: NDArray[Self.d]
    var norm: Int
    var finalized: Bool
    var rows_seen: Int

    def __init__(out self, features: Int, norm: Int = 2) raises:
        comptime assert Self.d == DType.float32 or Self.d == DType.float64, "Float32/64 only"
        if features < 0 or norm < 1 or norm > 3:
            raise Error("ValueError: invalid features/norm")
        self.values = NDArray[Self.d](Shape(features))
        for c in range(features):
            self.values.unsafe_set(c, 0)
        self.norm = norm
        self.finalized = False
        self.rows_seen = 0

    def update(mut self, chunk: NDArray[Self.d], *, tile_bytes: Int = DEFAULT_TILE_BYTES) raises:
        _matrix(chunk)
        _tile(tile_bytes)
        if self.finalized or Int(chunk.shape[1]) != self.values.size:
            raise Error("ValueError: finalized state or feature mismatch")
        if _intersects(chunk, self.values):
            raise Error("ValueError: norm state aliases input")
        var width = tile_bytes // Int(4 if Self.d == DType.float32 else 8)
        for r in range(Int(chunk.shape[0])):
            for start in range(0, self.values.size, width):
                for c in range(start, min(start+width, self.values.size)):
                    var v = chunk.unsafe_get(r*Int(chunk.strides[0])+c*Int(chunk.strides[1]))
                    var total = self.values.unsafe_get(c)
                    if self.norm == 1:
                        total += abs(v)
                    elif self.norm == 2:
                        total += v*v
                    elif abs(v) > total or isnan(v):
                        total = abs(v)
                    self.values.unsafe_set(c, total)
        self.rows_seen += Int(chunk.shape[0])

    def finish(mut self) raises:
        if self.finalized or self.rows_seen == 0:
            raise Error("ValueError: no rows or already finalized")
        var eps = SIMD[Self.d, 1](2.220446049250313e-16)
        comptime if Self.d == DType.float32:
            eps = 1.1920928955078125e-7
        for c in range(self.values.size):
            var v = self.values.unsafe_get(c)
            if self.norm == 2:
                v = sqrt(v)
            if v < 10*eps:
                v = 1
            self.values.unsafe_set(c,v)
        self.finalized = True

    def transform_into(self, chunk: NDArray[Self.d], mut destination: NDArray[Self.d], *, tile_bytes: Int = DEFAULT_TILE_BYTES) raises:
        _tile(tile_bytes)
        _pair(chunk,destination)
        if not self.finalized or Int(chunk.shape[1]) != self.values.size:
            raise Error("ValueError: unfinished norms or feature mismatch")
        if _intersects(self.values,destination):
            raise Error("ValueError: norm state aliases destination")
        var width = tile_bytes // Int(4 if Self.d == DType.float32 else 8)
        for r in range(Int(chunk.shape[0])):
            for start in range(0,self.values.size,width):
                for c in range(start,min(start+width,self.values.size)):
                    var v=chunk.unsafe_get(r*Int(chunk.strides[0])+c*Int(chunk.strides[1]))
                    destination.unsafe_set(r*Int(destination.strides[0])+c*Int(destination.strides[1]),v/self.values.unsafe_get(c))
