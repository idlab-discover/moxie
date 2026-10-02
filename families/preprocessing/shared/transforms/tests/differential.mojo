from numojo import NDArray, NDArrayStrides, Shape
from moxie.preprocessing._dense_transforms import transform, normalize, norms_into, FeatureNorms
from std.sys import argv

def run[d: DType, s: DType](n: Int, p: Int, layout: Int, mode: Int, axis: Int, norm: Int, tile: Int, scenario: Int) raises:
    var backing = NDArray[d](Shape(2*n*p+7))
    for j in range(backing.size):
        backing.unsafe_set(j, 99)
    var x = backing.view_with_layout(Shape(n,p), NDArrayStrides(2*p,2), 3)
    if layout == 1:
        x = backing.view_with_layout(Shape(n,p), NDArrayStrides(-2*p,-2), 3+2*(n*p-1))
    for r in range(n):
        for c in range(p):
            var v = SIMD[d,1](Float64((r*p+c)*37%101-50)/7)
            if r==0:
                v=0
            if scenario == 1:
                v = SIMD[d,1](1e20)
                comptime if d == DType.float64:
                    v = 1e200
            elif scenario == 2:
                v = SIMD[d,1](1e-30)
            elif scenario == 3:
                v = SIMD[d,1](1e-8)
            elif scenario == 4 and c == 0:
                var zero=SIMD[d,1](0)
                v=zero/zero
            x.unsafe_set(r*Int(x.strides[0])+c*Int(x.strides[1]),v)
    var count = p if axis==0 else n
    var a = NDArray[s](Shape(count))
    var b = NDArray[s](Shape(count))
    for j in range(count):
        a.unsafe_set(j, SIMD[s,1](Float64(j%5-2)/3))
        b.unsafe_set(j, SIMD[s,1](Float64(j%7+1)/7))
    var y = NDArray[d](Shape(0,0))
    if mode==7:
        var state=FeatureNorms[d](p,norm)
        for start in range(0,n,3):
            var chunk=x.view_with_layout(Shape(min(3,n-start),p),x.strides,x.offset+start*Int(x.strides[0]))
            state.update(chunk,tile_bytes=tile)
        state.finish()
        y=NDArray[d](Shape(n,p))
        for start in range(0,n,2):
            var chunk=x.view_with_layout(Shape(min(2,n-start),p),x.strides,x.offset+start*Int(x.strides[0]))
            var dst=y.view_with_layout(chunk.shape,y.strides,start*p)
            dst.flags.WRITEABLE=True
            state.transform_into(chunk,dst,tile_bytes=tile)
        for j in range(p):
            print("N",state.values.unsafe_get(j))
    elif mode==6:
        y=normalize(x,axis=axis,norm=norm,tile_bytes=tile)
        var norms=NDArray[d](Shape(Int(x.shape[1-axis])))
        norms_into(x,norms,axis=axis,norm=norm,tile_bytes=tile)
        for j in range(norms.size):
            print("N", norms.unsafe_get(j))
    else:
        var lower=SIMD[d,1](-0.5)
        var upper=SIMD[d,1](0.75)
        if scenario==5:
            var zero=SIMD[d,1](0)
            lower=zero/zero
        elif scenario==6:
            lower=0.75
            upper=-0.5
        y=transform(x,a,b,operation=mode,axis=axis,lower=lower,upper=upper,tile_bytes=tile)
    for j in range(y.size):
        print("V",y.unsafe_get(j))

def main() raises:
    var args=argv()
    if args[1]=="f32":
        run[DType.float32,DType.float32](Int(args[2]),Int(args[3]),Int(args[4]),Int(args[5]),Int(args[6]),Int(args[7]),Int(args[8]),Int(args[9]))
    elif args[1]=="mixed":
        run[DType.float32,DType.float64](Int(args[2]),Int(args[3]),Int(args[4]),Int(args[5]),Int(args[6]),Int(args[7]),Int(args[8]),Int(args[9]))
    else:
        run[DType.float64,DType.float64](Int(args[2]),Int(args[3]),Int(args[4]),Int(args[5]),Int(args[6]),Int(args[7]),Int(args[8]),Int(args[9]))
