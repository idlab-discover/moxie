# Dense transform primitives

Import from `moxie.preprocessing._dense_transforms`; no public estimators are
exported. Assemble this project's live modules with the foundation assembler.
Inputs are borrowed NuMojo rank-2 Float32/64 arrays with valid metadata, including
signed strides and offsets. Outputs preserve the input dtype and logical shape.
No input-sized conversion or Boolean mask is constructed. Estimator validation
(including whether to reject NaNs/infinity) belongs to callers.

## Affine, clipping and threshold operations

`transform(x, shift, scale, *, operation=0, axis=0, lower=0, upper=1,
tile_bytes=65536)` returns one new, row-major logical output. `shift` and `scale`
are borrowed rank-1 arrays with one entry per feature (`axis=0`) or row (`axis=1`).
Parameters may be Float32 or Float64, including Float64 parameters with Float32
input. Both parameter vectors share a dtype. Arithmetic promotes to the wider
dtype, then rounds back to input dtype **after each stage**, matching separate
NumPy in-place operations. StandardScaler consumers must cast their parameters
to the input dtype as the pinned reference does. Never contract to an FMA.

| operation | Stages |
| --- | --- |
| 0 | subtract shift, divide scale (standard/robust forward) |
| 1 | multiply scale, add shift (inverse, or unclipped minmax forward) |
| 2 | multiply scale, add shift, clip to [lower,upper] |
| 3 | subtract shift, divide scale (minmax inverse) |
| 4 | clip only; parameter arrays unused |
| 5 | value > lower becomes 1, otherwise 0; parameter arrays unused |

Missing centering/scaling is represented by neutral parameters (0/1). Clipping
propagates NaN bounds and produces the upper bound for reversed intervals, as
NumPy does; consumers validate estimator feature ranges. NaNs in data pass through affine/clip; threshold
comparison with NaN produces 0. These are kernel operations, not an assertion
that a public Binarizer should accept NaNs. Divide-by-zero/overflow are not healed.
Typed dtypes, integer operation/norm selectors, and explicit mutation methods
are native adaptations covered by contracts. Sparse arrays are unavailable by type.

`transform_into(x, destination, shift, scale, ...)` borrows caller storage;
`transform_inplace(x, shift, scale, ...)` mutates the input without a result.
These are the native `copy=False` modes: unsupported layouts raise and never
fall back to a copy. Same-dtype input/output is enforced by the type system.
Shape, writable flag, self-overlap, parameter aliases, and potentially hazardous
input/output overlap are checked before writes. Exact input/output aliases are
supported; use the in-place method to satisfy Mojo borrow checking. Disjoint
address spans are supported, including views into one allocation. Interleaved
views whose address spans overlap are conservatively rejected even if their
individual elements do not overlap. Concurrent mutation is unsupported.

Pinned NuMojo creates shared views with `WRITEABLE=False`. Read-only views work
as inputs. The engine does not override this flag. A storage owner that has
independently established write permission can expose a valid writable view;
tests explicitly grant the flag for such fixtures. Never grant it for genuinely
read-only memory. There is no hidden `.copy()` of a backing allocation.

## Norms and streaming

`normalize`, `normalize_into`, and `normalize_inplace` accept `axis=1`, `norm=2`,
and `tile_bytes=65536`. Axis 1 normalizes each row; axis 0 each feature. Norms are
1=L1, 2=L2, 3=max. Reduction and division are separate passes over each vector,
with O(1) scalar scratch. An empty reduction dimension raises. An empty set of
vectors returns an empty result. `norms_into(x, vector, ...)` writes requested
norms to a caller-owned rank-1 output of length n (axis 1) or p (axis 0).
As in sklearn, norms below 10 times the input dtype epsilon are replaced by 1.
Returned norms are these guarded divisors, not unguarded mathematical norms.

Norm accumulation stays in the input dtype and propagates NaNs and overflow.
Reduction order can differ from NumPy's pairwise/SIMD order. This is an explicit
operation-specific exception to the shared widened `row_squared_norms`, whose
Float32 overflow behavior cannot reproduce sklearn normalization. The shared
kernel is not forked or replaced. Parity tolerances are 4e-6 for Float32 and
2e-13 for Float64 on the deterministic bounded suite; no bitwise or arbitrary
ill-conditioned-data guarantee is made.

`transform_sink` and `normalize_sink` invoke a native `TileSink.write(row, col,
borrowed_tile)` synchronously. Tiles never exceed `tile_bytes`; the engine reuses
one buffer and retains no previous tile. A sink must finish consuming a tile
before returning, must not mutate inputs/parameters through another alias, and
must not retain its view. Sink exceptions leave previously emitted output intact.
For normalization, axis-0 tiles are column panels and axis-1 tiles row panels;
callbacks receive logical offsets, not a promise of row-major emission ordering.

For externally streamed row chunks, invoke `transform_into`/`transform_sink`
with the same globally fitted feature parameters for every chunk. Row-wise
normalization can process each complete-row chunk independently. Feature-wise
normalization requires **two passes**: construct `FeatureNorms[dtype](p, norm)`,
`update(chunk)` in row order for every bounded chunk, call `finish()` once, replay
the chunks, and call `state.transform_into(chunk, destination)`. The state owns
one p-element vector and retains no chunks. Its `values` are read-only to callers.
Calling transform before finish, update after finish, or finish twice raises.
The stream must be replayable (or externally stored); normalizing each row chunk
with its own feature norms is incorrect. For an already borrowed full matrix,
`norms_into(...,axis=0)` is an alternative global norm pass.
