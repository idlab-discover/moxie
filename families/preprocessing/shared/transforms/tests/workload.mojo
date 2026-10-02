from numojo import NDArray, NDArrayStrides, Shape
from moxie.preprocessing._dense_transforms import transform, transform_into, transform_inplace, transform_sink, normalize, normalize_into, normalize_inplace, normalize_sink, TileSink, FeatureNorms
from std.sys import argv
from std.time import perf_counter_ns

struct SumSink(TileSink):
    var total: Float64
    var largest: Int
    def __init__(out self):
        self.total=0
        self.largest=0
    def write[d: DType](mut self, row: Int, col: Int, tile: NDArray[d]) raises:
        self.largest=max(self.largest,tile.size)
        for r in range(Int(tile.shape[0])):
            for c in range(Int(tile.shape[1])):
                self.total+=Float64(tile.unsafe_get(r*Int(tile.strides[0])+c*Int(tile.strides[1])))

@no_inline
def checksum[d: DType](x: NDArray[d]) raises -> Float64:
    var total=Float64(0)
    for r in range(Int(x.shape[0])):
        for c in range(Int(x.shape[1])):
            total+=Float64(x.unsafe_get(r*Int(x.strides[0])+c*Int(x.strides[1])))
    return total


def run[d: DType](mode: String,n: Int,p: Int,layout: Int,tile: Int,reps: Int) raises:
    var capacity=n*p
    if layout==2:
        capacity=max(8388608,n*p)
    var backing=NDArray[d](Shape(capacity))
    for j in range(capacity):
        backing.unsafe_set(j,SIMD[d,1](Float64(j%31-15)/8))
    if layout==3:
        var zero=SIMD[d,1](0)
        backing.unsafe_set(0,zero/zero)
    var x=backing.view_with_layout(Shape(n,p),NDArrayStrides(p,1),0)
    if layout==1:
        x=backing.view_with_layout(Shape(n,p),NDArrayStrides(-p,-1),n*p-1)
    x.flags.WRITEABLE=True
    var a=NDArray[d](Shape(p))
    var b=NDArray[d](Shape(p))
    for j in range(p):
        a.unsafe_set(j,0.25)
        b.unsafe_set(j,2)
    var destination=NDArray[d](Shape(0,0))
    if mode=="into" or mode=="control_into" or mode=="norm_into":
        destination=NDArray[d](Shape(n,p))
        for j in range(n*p):
            destination.unsafe_set(j,1)
    var total=Float64(0)
    var sink=SumSink()
    var start=perf_counter_ns()
    for rep in range(reps):
        if mode=="copy":
            var y=transform(x,a,b,tile_bytes=tile)
            total+=checksum(y)
        elif mode=="norm_copy":
            var y=normalize(x,tile_bytes=tile)
            total+=checksum(y)
        elif mode=="into":
            transform_into(x,destination,a,b,tile_bytes=tile)
            total+=checksum(destination)
        elif mode=="norm_into":
            normalize_into(x,destination,tile_bytes=tile)
            total+=checksum(destination)
        elif mode=="inplace":
            transform_inplace(x,a,b,tile_bytes=tile)
            total+=checksum(x)
        elif mode=="norm_inplace":
            normalize_inplace(x,tile_bytes=tile)
            total+=checksum(x)
        elif mode=="sink":
            transform_sink(x,sink,a,b,tile_bytes=tile)
        elif mode=="norm_sink":
            normalize_sink(x,sink,tile_bytes=tile)
        elif mode=="baseline":
            var y=NDArray[d](Shape(n,p))
            for r in range(n):
                for c in range(p):
                    var v=x.unsafe_get(r*Int(x.strides[0])+c*Int(x.strides[1]))
                    v=v-a.unsafe_get(c)
                    v=v/b.unsafe_get(c)
                    y.unsafe_set(r*p+c,v)
            total+=checksum(y)
        elif mode=="features":
            var state=FeatureNorms[d](p)
            for start_row in range(0,n,64):
                var chunk=x.view_with_layout(Shape(min(64,n-start_row),p),x.strides,x.offset+start_row*Int(x.strides[0]))
                state.update(chunk,tile_bytes=tile)
            state.finish()
            for c in range(p):
                total+=Float64(state.values.unsafe_get(c))
        elif mode=="negative":
            # Deliberate prohibited table temporary: gate must reject it.
            var extra=NDArray[d](Shape(n,p))
            for j in range(n*p):
                extra.unsafe_set(j, SIMD[d,1](Float64(j%7)))
            transform_inplace(x,a,b,tile_bytes=tile)
            total+=checksum(x)+checksum(extra)
        else:
            total+=checksum(x)
    var elapsed=perf_counter_ns()-start
    print("seconds",Float64(elapsed)/1e9)
    print("checksum",total+sink.total)
    print("largest_tile",sink.largest)
    # Keep caller-owned setup alive until the timed region has ended.
    print("keep",backing.unsafe_get(capacity-1),destination.size)

def main() raises:
    var args=argv()
    if args[1]=="f32":
        run[DType.float32](args[2],Int(args[3]),Int(args[4]),Int(args[5]),Int(args[6]),Int(args[7]))
    else:
        run[DType.float64](args[2],Int(args[3]),Int(args[4]),Int(args[5]),Int(args[6]),Int(args[7]))
