# Image patches and dense lattice graphs

Import helpers and `PatchExtractor` from `moxie.feature_extraction.image`.
The graph helpers are also exported by `moxie.feature_extraction`.
All data use the shared NuMojo `NDArray[dtype]`; outputs own their storage.
Inputs may have nonzero offsets or negative strides. Fresh outputs are C-order.
See [the executable example](../examples/patches.mojo).

## Patches

`extract_patches_2d(image, (height,width))` returns every overlapping patch,
ordered top to bottom then left to right. Inputs have shape `(H,W[,C])`;
outputs have shape `(N,h,w[,C])`, dropping a singleton channel. Raw extraction
preserves dtype. Images must be nonempty and finite, and patch extents positive
and no larger than the image. These are rank-specific native interfaces;
arbitrary higher-rank arrays and Python array-like coercion are not provided.

`max_patches=PatchLimit()` represents None. `PatchLimit(1, count)` selects an
integer maximum, capped at the number of positions. `PatchLimit(2, 0, fraction)`
selects a fraction in (0,1), rounded down. Both sample with replacement, including
when the integer maximum reaches all positions. A fraction may select zero.
Use `random_state=42` for repeatable results, or pass a mutable shared
`RandomState` to advance an application-owned stream. All row positions are
drawn before column positions, matching the legacy NumPy stream.

The native None sentinel is `random_state=-1`. Exhaustive enumeration needs no
stream; sampling with this default raises unless the caller passes an explicit
mutable context (for example one initialized with `moxie._random.entropy_state`).
This follows the shared RNG contract instead of an implicit process-global state.
Other negative integer seeds raise. The sentinel is a documented language
adaptation, not a valid NumPy seed.

`reconstruct_from_patches_2d(patches, Shape(H,W[,C]))` returns Float64. It adds
patches in enumeration order and divides by the theoretical full overlap count.
Partial input therefore leaves attenuated regions; excess patches are ignored.
It cannot undo random sampling without known patch positions. Channels must
match explicitly; NumPy's incidental broadcasting of mismatched patch/image
ranks is not exposed by this native shape contract.

## Stateless estimator

`PatchExtractor[dtype]` implements the family `FeatureExtractor` trait.
Input is `(images,H,W[,C])`, output is Float64 and image-major. Constructor fields
are `patch_size=(-1,-1)` (None), `max_patches=PatchLimit()` (None), and
`random_state=-1` (None). The default patch dimensions are `(H//10,W//10)`.
`fit` validates configuration and returns a mutable borrow of the estimator. `transform`
works before fitting. `fit_transform` calls both. There is no learned state.

`get_params()` returns `PatchParams`; edit that independent record and pass it
to `set_params`. Named updates such as `set_params(patch_size=(2,2),
random_state=42)` preserve omitted parameters. Reset defaults with
`patch_size=(-1,-1)`, `max_patches=PatchLimit()`, and `random_state=-1`;
`None` in a named update means leave that parameter unchanged. Updates defer
validation until fitting or transformation. Both forms return a mutable borrow,
so `model.set_params(random_state=42).fit(X).transform(X)` is supported.
`moxie.base.clone` copies constructor configuration. As a
stateless estimator, `__sklearn_is_fitted__` returns true. Each integer-seeded
transform resets the stream once, then shares it across batch images. For mutable
state or the None context use `transform(X, random_state=context)`; that explicit
argument takes precedence over the constructor's seed. Shared references are
passed per call rather than retained inside an estimator.
`fit_transform(X, random_state=context)` also borrows and advances the context,
after validating configuration. Repeated calls continue the stream. This keyword
is a native adaptation of retaining a RandomState in the Python estimator.

## Graphs

`grid_to_graph(n_x,n_y,n_z=1, return_as="dense")` creates axial adjacency with
unit diagonal and edge weights. `img_to_graph(img, return_as="dense")` uses
pixel values on the diagonal and absolute differences on edges. Rank-three
images are spatial volumes here, not channel images. C-order vertex index is
`(x*n_y+y)*n_z+z`. A `mask=` array compacts retained vertices in this order,
including isolated vertices. Masks must contain exactly the grid's vertex count;
they may be flat or shaped. Nonzero numeric values select vertices.

Grid output defaults to Int64; choose a compile-time dtype, for example
`grid_to_graph[DType.float32](...)`. Image output defaults to input dtype;
`img_to_graph[output_dtype=DType.float64](...)` selects another output dtype.
Differences are computed in the input dtype before casting, preserving integer
wraparound. Boolean image subtraction raises, as in NumPy; Boolean grid outputs
are supported. Image graphs do not reject nonfinite pixel values.

`return_as="sparse"` is the preserved default and raises a clear unsupported
error. Explicit dense output allocates O(V²) entries directly, without a sparse
intermediate; size overflow is checked before allocating adjacency. Pairwise
neighbor checks take O(V²) time. Patch output takes O(N*h*w*C) storage/time;
batch extraction uses one image scratch plus one image's patches, and
reconstruction uses O(H*W*C) output storage.

Python reflection, metadata routing, serialization, sparse containers and
Python exception classes are outside the binding-free native interface.
Mojo errors carry ValueError/TypeError/NotImplementedError prefixes.
