from numojo import NDArray, NDArrayStrides, Shape
from moxie.preprocessing._dense_transforms import transform, transform_into, transform_inplace, transform_sink, normalize, normalize_into, normalize_inplace, normalize_sink, norms_into, TileSink, FeatureNorms
from std.testing import assert_equal, assert_raises, assert_true
from std.math import abs, isnan

struct CheckSink(TileSink):
    var expected: NDArray[DType.float64]
    var elements: Int
    var largest: Int
    def __init__(out self, var expected: NDArray[DType.float64]):
        self.expected = expected^
        self.elements = 0
        self.largest = 0
    def write[d: DType](mut self, row: Int, col: Int, tile: NDArray[d]) raises:
        self.elements += tile.size
        self.largest = max(self.largest, tile.size)
        for r in range(Int(tile.shape[0])):
            for c in range(Int(tile.shape[1])):
                var wanted = self.expected.item(row+r, col+c)
                var got = Float64(tile.unsafe_get(r * Int(tile.strides[0])+c*Int(tile.strides[1])))
                assert_true(abs(got-wanted)<1e-6)


def contracts[d: DType]() raises:
    var x = NDArray[d](Shape(3, 4))
    for j in range(12):
        x.unsafe_set(j, SIMD[d, 1](j-5))
    var a = NDArray[d](Shape(4))
    var b = NDArray[d](Shape(4))
    for j in range(4):
        a.unsafe_set(j, SIMD[d, 1](j))
        b.unsafe_set(j, SIMD[d, 1](j+1))
    var output = NDArray[d](Shape(3, 4))
    var pointer = Int(output.unsafe_ptr())
    for tile in range(8, 33, 8):
        for mode in range(6):
            var copied = transform(x, a, b, operation=mode, lower=-2, upper=2, tile_bytes=tile)
            transform_into(x, output, a, b, operation=mode, lower=-2, upper=2, tile_bytes=tile)
            var inplace = x.copy()
            transform_inplace(inplace, a, b, operation=mode, lower=-2, upper=2, tile_bytes=tile)
            for j in range(12):
                assert_equal(copied.unsafe_get(j), output.unsafe_get(j))
                assert_equal(copied.unsafe_get(j), inplace.unsafe_get(j))
                assert_equal(x.unsafe_get(j), SIMD[d, 1](j-5))
            assert_equal(Int(output.unsafe_ptr()), pointer)
        for axis in range(2):
            for norm in range(1, 4):
                var result = normalize(x, axis=axis, norm=norm, tile_bytes=tile)
                normalize_into(x, output, axis=axis, norm=norm, tile_bytes=tile)
                var inplace = x.copy()
                normalize_inplace(inplace, axis=axis, norm=norm, tile_bytes=tile)
                var norms = NDArray[d](Shape(Int(x.shape[1-axis])))
                norms_into(x, norms, axis=axis, norm=norm, tile_bytes=tile)
                for j in range(12):
                    assert_equal(result.unsafe_get(j), output.unsafe_get(j))
                    assert_equal(result.unsafe_get(j), inplace.unsafe_get(j))
    var back = transform(transform(x, a, b), a, b, operation=1)
    for j in range(12):
        assert_true(abs(back.unsafe_get(j)-x.unsafe_get(j)) < 1e-5)
    var view = x.view_with_layout(Shape(3, 4), NDArrayStrides(-4, -1), 11)
    var reversed = transform(view, a, b)
    assert_equal(reversed.item(0, 0), 6)
    assert_equal(x.unsafe_get(11), 6)
    with assert_raises():
        transform_inplace(view, a, b)
    # Explicit writable view fixture, as supported by NuMojo public flags.
    # Production code must have ownership permission before granting write access.
    view.flags.WRITEABLE = True
    normalize_inplace(view, norm=3)
    assert_equal(x.unsafe_get(11), 1)
    var big = NDArray[d](Shape(30))
    for j in range(30):
        big.unsafe_set(j, SIMD[d, 1](j))
    var src = big.view_with_layout(Shape(2, 4), NDArrayStrides(4, 1), 0)
    var dst = big.view_with_layout(Shape(2, 4), NDArrayStrides(4, 1), 16)
    dst.flags.WRITEABLE = True
    transform_into(src, dst, a, b, operation=5)
    assert_equal(big.unsafe_get(17), 1)
    var overlap = big.view_with_layout(Shape(2, 4), NDArrayStrides(4, 1), 1)
    overlap.flags.WRITEABLE = True
    with assert_raises():
        transform_into(src, overlap, a, b)
    assert_equal(big.unsafe_get(1), 1)
    var repeated = big.view_with_layout(Shape(2, 4), NDArrayStrides(2, 1), 0)
    repeated.flags.WRITEABLE = True
    with assert_raises():
        transform_inplace(repeated, a, b)
    var aliased_parameter = dst.view_with_layout(Shape(4), NDArrayStrides(1), 16)
    with assert_raises():
        transform_into(src, dst, aliased_parameter, b)
    with assert_raises():
        _ = normalize(src, axis=2)
    with assert_raises():
        _ = normalize(src, norm=0)
    with assert_raises():
        _ = transform(src, a, b, tile_bytes=0)
    var clipped = transform(src, a, b, operation=4, lower=2, upper=1)
    assert_equal(clipped.unsafe_get(0), 1)
    var bound_zero=SIMD[d,1](0)
    var nan_clipped=transform(src,a,b,operation=4,lower=bound_zero/bound_zero)
    assert_true(isnan(nan_clipped.unsafe_get(0)))
    with assert_raises():
        transform_into(src, output, a, b)
    with assert_raises():
        _ = normalize(a)
    var empty = NDArray[d](Shape(0, 4))
    var empty_result = transform(empty, a, b)
    assert_equal(empty_result.size, 0)
    with assert_raises():
        _ = normalize(empty, axis=0)
    var zero = SIMD[d, 1](0)
    x.unsafe_set(0, zero/zero)
    var nan_result = transform(x, a, b)
    assert_true(isnan(nan_result.unsafe_get(0)))
    var nan_norm = normalize(x)
    assert_true(isnan(nan_norm.unsafe_get(1)))
    for j in range(12):
        x.unsafe_set(j, 0)
    normalize_inplace(x)
    assert_equal(x.unsafe_get(0), 0)


def streams() raises:
    var x = NDArray[DType.float64](Shape(3, 5))
    var a = NDArray[DType.float64](Shape(5))
    var b = NDArray[DType.float64](Shape(5))
    for j in range(15):
        x.unsafe_set(j, Float64(j-5))
    for j in range(5):
        a.unsafe_set(j, 1)
        b.unsafe_set(j, 2)
    for mode in range(6):
        var sink = CheckSink(transform(x, a, b, operation=mode))
        transform_sink(x, sink, a, b, operation=mode, tile_bytes=16)
        assert_equal(sink.elements, 15)
        assert_true(sink.largest<=2)
    for axis in range(2):
        for norm in range(1,4):
            var sink = CheckSink(normalize(x, axis=axis, norm=norm))
            normalize_sink(x, sink, axis=axis, norm=norm, tile_bytes=16)
            assert_equal(sink.elements, 15)
            assert_true(sink.largest<=2)
    # Borrowed row chunks apply the SAME full-data feature norms.
    var norms = NDArray[DType.float64](Shape(5))
    norms_into(x, norms, axis=0)
    for j in range(5):
        a.unsafe_set(j, 0)
    var expected = normalize(x, axis=0)
    for row in range(3):
        var chunk = x.view_with_layout(Shape(1,5), NDArrayStrides(5,1), row*5)
        var result = transform(chunk, a, norms)
        for c in range(5):
            assert_equal(result.item(0,c), expected.item(row,c))


def feature_stream[d: DType]() raises:
    var x=NDArray[d](Shape(7,5))
    for j in range(35):
        x.unsafe_set(j,SIMD[d,1](j-12))
    for norm in range(1,4):
        var state=FeatureNorms[d](5,norm)
        var y=NDArray[d](Shape(7,5))
        with assert_raises():
            state.transform_into(x,y)
        for start in range(0,7,3):
            var chunk=x.view_with_layout(Shape(min(3,7-start),5),NDArrayStrides(5,1),start*5)
            state.update(chunk,tile_bytes=16)
        state.finish()
        with assert_raises():
            state.update(x)
        with assert_raises():
            state.finish()
        for start in range(0,7,2):
            var chunk=x.view_with_layout(Shape(min(2,7-start),5),NDArrayStrides(5,1),start*5)
            var output=y.view_with_layout(chunk.shape,NDArrayStrides(5,1),start*5)
            output.flags.WRITEABLE=True
            state.transform_into(chunk,output,tile_bytes=8)
        var expected=normalize(x,axis=0,norm=norm)
        for j in range(35):
            assert_equal(y.unsafe_get(j),expected.unsafe_get(j))


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    streams()
    feature_stream[DType.float32]()
    feature_stream[DType.float64]()
    print("PASS ownership, aliases, layouts, tiles, axes, norms, streaming, errors")
