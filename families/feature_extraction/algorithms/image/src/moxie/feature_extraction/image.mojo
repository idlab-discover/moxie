"""Dense image feature extraction on shared NuMojo arrays."""
# Algorithms adapted from scikit-learn (BSD-3-Clause); see LICENSES.
from std.sys import size_of
from numojo import NDArray, Shape, zeros
from moxie.feature_extraction._image_contracts import GridGeometry, checked_extent_product
from moxie.feature_extraction._validation import require_dense_graph_output


def _logical[dtype: DType](x: NDArray[dtype], index: Int) raises -> SIMD[dtype, 1]:
    var remaining = index
    var offset = 0
    for k in range(x.ndim):
        var axis = x.ndim - k - 1
        offset += (remaining % x.shape[axis]) * Int(x.strides[axis])
        remaining //= x.shape[axis]
    return x.unsafe_get(offset)


def _vertices[mask_dtype: DType](geometry: GridGeometry, mask: NDArray[mask_dtype]) raises -> List[Int]:
    var count = geometry.vertices()
    if mask.size != count:
        raise Error("ValueError: mask size must equal grid size")
    var vertices = List[Int]()
    for i in range(count):
        if Bool(_logical(mask, i)):
            vertices.append(i)
    return vertices^


def _adjacent(a: Int, b: Int, n_y: Int, n_z: Int) -> Bool:
    var ax = a // (n_y * n_z)
    var bx = b // (n_y * n_z)
    var ay = (a // n_z) % n_y
    var by = (b // n_z) % n_y
    var az = a % n_z
    var bz = b % n_z
    return abs(ax-bx) + abs(ay-by) + abs(az-bz) == 1


def grid_to_graph[dtype: DType = DType.int64, mask_dtype: DType = DType.bool](
    n_x: Int, n_y: Int, n_z: Int = 1, *, mask: NDArray[mask_dtype],
    return_as: String = "sparse"
) raises -> NDArray[dtype]:
    require_dense_graph_output(return_as)
    if n_x <= 0 or n_y <= 0 or n_z <= 0:
        raise Error("ValueError: grid dimensions must be positive")
    var geometry = GridGeometry(n_x, n_y, n_z)
    var vertices = _vertices(geometry, mask)
    var n = len(vertices)
    _ = checked_extent_product(checked_extent_product(n, n), size_of[SIMD[dtype, 1]]())
    var result = zeros[dtype](geometry.dense_shape(n))
    for i in range(n):
        result.unsafe_set(i*n+i, 1)
        for j in range(i+1, n):
            if _adjacent(vertices[i], vertices[j], n_y, n_z):
                result.unsafe_set(i*n+j, 1)
                result.unsafe_set(j*n+i, 1)
    return result^


def grid_to_graph[dtype: DType = DType.int64](
    n_x: Int, n_y: Int, n_z: Int = 1, *, return_as: String = "sparse"
) raises -> NDArray[dtype]:
    require_dense_graph_output(return_as)
    var geometry = GridGeometry(n_x, n_y, n_z)
    var n = geometry.vertices()
    _ = checked_extent_product(checked_extent_product(n,n),size_of[SIMD[dtype,1]]())
    var mask = zeros[DType.bool](Shape(n))
    for i in range(mask.size):
        mask.unsafe_set(i, True)
    return grid_to_graph[dtype](n_x, n_y, n_z, mask=mask, return_as=return_as)


def img_to_graph[dtype: DType, //, output_dtype: DType = dtype, mask_dtype: DType = DType.bool](
    img: NDArray[dtype], *, mask: NDArray[mask_dtype], return_as: String = "sparse"
) raises -> NDArray[output_dtype]:
    require_dense_graph_output(return_as)
    if img.ndim != 2 and img.ndim != 3:
        raise Error("ValueError: image graph requires rank 2 or 3")
    comptime if dtype == DType.bool:
        raise Error("TypeError: boolean image subtraction is unsupported")
    else:
        var nz = img.shape[2] if img.ndim == 3 else 1
        var geometry = GridGeometry(img.shape[0], img.shape[1], nz)
        var vertices = _vertices(geometry, mask)
        var n = len(vertices)
        _ = checked_extent_product(checked_extent_product(n, n), size_of[SIMD[output_dtype, 1]]())
        var result = zeros[output_dtype](geometry.dense_shape(n))
        for i in range(n):
            result.unsafe_set(i*n+i, _logical(img, vertices[i]).cast[output_dtype]())
            for j in range(i+1, n):
                if _adjacent(vertices[i], vertices[j], img.shape[1], nz):
                    var delta = _logical(img, vertices[i]) - _logical(img, vertices[j])
                    var weight = abs(delta).cast[output_dtype]()
                    result.unsafe_set(i*n+j, weight)
                    result.unsafe_set(j*n+i, weight)
        return result^



def img_to_graph[dtype: DType, //, output_dtype: DType = dtype](
    img: NDArray[dtype], *, return_as: String = "sparse"
) raises -> NDArray[output_dtype]:
    require_dense_graph_output(return_as)
    _ = checked_extent_product(checked_extent_product(img.size,img.size),size_of[SIMD[output_dtype,1]]())
    var mask = zeros[DType.bool](Shape(img.size))
    for i in range(mask.size):
        mask.unsafe_set(i, True)
    return img_to_graph[output_dtype=output_dtype](img, mask=mask, return_as=return_as)

from std.math import isfinite
from moxie._random import RandomState
from moxie.feature_extraction._image_contracts import PatchGeometry


struct PatchLimit(Copyable, Movable):
    """Tagged max_patches: kind 0=None, 1=integer, 2=fraction."""
    var kind: Int
    var count: Int
    var fraction: Float64

    def __init__(out self, kind: Int = 0, count: Int = 0, fraction: Float64 = 0):
        self.kind = kind
        self.count = count
        self.fraction = fraction

    def validate(self) raises:
        if self.kind < 0 or self.kind > 2:
            raise Error("ValueError: invalid patch limit tag")
        if self.kind == 1 and self.count < 1:
            raise Error("ValueError: max_patches must be positive")
        if self.kind == 2 and not (self.fraction > 0 and self.fraction < 1):
            raise Error("ValueError: max_patches fraction must lie in (0,1)")

    def selected(self, total: Int) raises -> Int:
        self.validate()
        if self.kind == 1:
            return min(self.count, total)
        if self.kind == 2:
            return Int(self.fraction * Float64(total))
        return total


def _finite[dtype: DType](x: NDArray[dtype]) raises:
    for i in range(x.size):
        if not isfinite(Float64(_logical(x, i))):
            raise Error("ValueError: image contains NaN or infinity")


def extract_patches_2d[dtype: DType](
    image: NDArray[dtype], patch_size: Tuple[Int, Int], *,
    mut random_state: RandomState, max_patches: PatchLimit = PatchLimit()
) raises -> NDArray[dtype]:
    if image.ndim != 2 and image.ndim != 3:
        raise Error("ValueError: patch image requires rank 2 or 3")
    var channels = image.shape[2] if image.ndim == 3 else 1
    var g = PatchGeometry(image.shape[0], image.shape[1], patch_size[0], patch_size[1], channels)
    _finite(image)
    var n = max_patches.selected(g.positions())
    var patch_elements = checked_extent_product(checked_extent_product(g.patch_height,g.patch_width), channels)
    _ = checked_extent_product(checked_extent_product(n,patch_elements),size_of[SIMD[dtype,1]]())
    var result = zeros[dtype](g.output_shape(n))
    var rows = List[Int]()
    var cols = List[Int]()
    var nh = g.height-g.patch_height+1
    var nw = g.width-g.patch_width+1
    for i in range(n):
        rows.append(random_state.randint(nh) if max_patches.kind != 0 else i // nw)
    for i in range(n):
        cols.append(random_state.randint(nw) if max_patches.kind != 0 else i % nw)
    for k in range(n):
        for y in range(g.patch_height):
            for x in range(g.patch_width):
                for c in range(channels):
                    var index = ((rows[k]+y)*g.width+cols[k]+x)*channels+c
                    result.unsafe_set(k*patch_elements+(y*g.patch_width+x)*channels+c, _logical(image,index))
    return result^


def extract_patches_2d[dtype: DType](
    image: NDArray[dtype], patch_size: Tuple[Int, Int], *,
    random_state: Int = -1, max_patches: PatchLimit = PatchLimit()
) raises -> NDArray[dtype]:
    if random_state == -1 and max_patches.kind != 0:
        raise Error("ValueError: sampling with random_state=None requires an explicit mutable context")
    var rng = RandomState(0 if random_state == -1 else random_state)
    return extract_patches_2d(image, patch_size, random_state=rng, max_patches=max_patches)


def reconstruct_from_patches_2d[dtype: DType](
    patches: NDArray[dtype], image_size: Shape
) raises -> NDArray[DType.float64]:
    if patches.ndim != 3 and patches.ndim != 4:
        raise Error("ValueError: patches require rank 3 or 4")
    if len(image_size) != 2 and len(image_size) != 3:
        raise Error("ValueError: image_size requires rank 2 or 3")
    var channels = image_size[2] if len(image_size) == 3 else 1
    var pc = patches.shape[3] if patches.ndim == 4 else 1
    if pc != channels:
        raise Error("ValueError: channels do not match")
    var g = PatchGeometry(image_size[0],image_size[1],patches.shape[1],patches.shape[2],channels)
    _ = checked_extent_product(checked_extent_product(checked_extent_product(g.height,g.width),channels),8)
    var result = zeros[DType.float64](image_size)
    var nw = g.width-g.patch_width+1
    var nh = g.height-g.patch_height+1
    for k in range(min(patches.shape[0],g.positions())):
        for y in range(g.patch_height):
            for x in range(g.patch_width):
                for c in range(channels):
                    var dst = (((k//nw)+y)*g.width+(k%nw)+x)*channels+c
                    var src = ((k*g.patch_height+y)*g.patch_width+x)*channels+c
                    result.unsafe_set(dst,result.unsafe_get(dst)+Float64(_logical(patches,src)))
    for y in range(g.height):
        for x in range(g.width):
            var count = min(min(y+1,g.patch_height),min(nh,g.height-y))*min(min(x+1,g.patch_width),min(nw,g.width-x))
            for c in range(channels):
                var index = (y*g.width+x)*channels+c
                result.unsafe_set(index,result.unsafe_get(index)/Float64(count))
    return result^

from moxie.feature_extraction._base import FeatureExtractor


@fieldwise_init
struct PatchParams(Copyable, Movable):
    var patch_size: Tuple[Int, Int]
    var max_patches: PatchLimit
    var random_state: Int


struct PatchExtractor[dtype: DType = DType.float64](FeatureExtractor, Copyable):
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[DType.float64]
    comptime Params = PatchParams
    var patch_size: Tuple[Int, Int]
    var max_patches: PatchLimit
    var random_state: Int

    def __init__(out self, *, patch_size: Tuple[Int,Int] = (-1,-1),
                 max_patches: PatchLimit = PatchLimit(), random_state: Int = -1):
        self.patch_size = patch_size
        self.max_patches = max_patches.copy()
        self.random_state = random_state

    def get_params(self, deep: Bool = True) raises -> PatchParams:
        return PatchParams(self.patch_size, self.max_patches.copy(), self.random_state)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        self.patch_size = params.patch_size
        self.max_patches = params.max_patches.copy()
        self.random_state = params.random_state
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, patch_size: Optional[Tuple[Int, Int]] = None,
                   max_patches: Optional[PatchLimit] = None,
                   random_state: Optional[Int] = None) raises -> ref[self] Self:
        if patch_size:
            self.patch_size = patch_size.value()
        if max_patches:
            self.max_patches = max_patches.value().copy()
        if random_state:
            self.random_state = random_state.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(patch_size=self.patch_size,max_patches=self.max_patches,random_state=self.random_state)

    def __sklearn_is_fitted__(self) -> Bool:
        return True

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> ref[self] Self:
        self.max_patches.validate()
        if self.random_state != -1:
            _ = RandomState(self.random_state)
        return self

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        if self.random_state == -1 and self.max_patches.kind != 0:
            raise Error("ValueError: sampling with random_state=None requires an explicit mutable context")
        var rng = RandomState(0 if self.random_state == -1 else self.random_state)
        return self.transform(X, random_state=rng)

    def transform(self, X: Self.Input, *, mut random_state: RandomState) raises -> Self.Transformed:
        if X.ndim != 3 and X.ndim != 4:
            raise Error("ValueError: expected (images,height,width[,channels])")
        if X.shape[0] < 1:
            raise Error("ValueError: batch must contain images")
        _finite(X)
        var channels = X.shape[3] if X.ndim == 4 else 1
        var ph = X.shape[1]//10 if self.patch_size == (-1,-1) else self.patch_size[0]
        var pw = X.shape[2]//10 if self.patch_size == (-1,-1) else self.patch_size[1]
        var g = PatchGeometry(X.shape[1],X.shape[2],ph,pw,channels)
        var n = self.max_patches.selected(g.positions())
        var count = checked_extent_product(n,X.shape[0])
        _ = checked_extent_product(checked_extent_product(count,checked_extent_product(checked_extent_product(ph,pw),channels)),8)
        var result = zeros[DType.float64](g.output_shape(count))
        var image_elements = checked_extent_product(checked_extent_product(g.height,g.width),channels)
        var patch_elements = checked_extent_product(checked_extent_product(ph,pw),channels)
        # One bounded image scratch, avoiding a full-batch reshape/cast temporary.
        var scratch = zeros[Self.dtype](Shape(g.height,g.width,channels))
        for i in range(X.shape[0]):
            for j in range(image_elements):
                scratch.unsafe_set(j,_logical(X,i*image_elements+j))
            var patches = extract_patches_2d(scratch,(ph,pw),max_patches=self.max_patches,random_state=random_state)
            for j in range(n*patch_elements):
                result.unsafe_set(i*n*patch_elements+j,Float64(patches.unsafe_get(j)))
        return result^

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X,y)
        return self.transform(X)

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None,
                      *, mut random_state: RandomState) raises -> Self.Transformed:
        _ = self.fit(X,y)
        return self.transform(X,random_state=random_state)
