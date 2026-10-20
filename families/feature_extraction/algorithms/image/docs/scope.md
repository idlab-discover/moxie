# Validated dense image scope

The implemented public exports in `moxie.feature_extraction.image` are
`PatchExtractor`, `extract_patches_2d`, `reconstruct_from_patches_2d`,
`img_to_graph`, and `grid_to_graph`. The graph helpers are also exported at the
family root. See [API contracts and language adaptations](api.md).

Supported modes include exhaustive and sampled patches, integer/fraction/None
limits, integer seeds and borrowed mutable RNG contexts, grayscale and color
images, singleton channels, stateless batch transform, clone, fluent fit and typed or named parameter
updates, partial/full/excess reconstruction, and masked or unmasked dense 2-D
and 3-D axial graphs. Outputs preserve C-order indexing. Positive/negative
strides and nonzero offsets are tested. Dense graphs preserve isolated vertices,
empty selections, input-dtype differences and integer overflow before output
conversion. Raw patches preserve dtype; batch extraction and reconstruction
return Float64.

Differential cases cover Float32/Float64, signed/unsigned 8/16/32/64-bit image
graphs, integer and Boolean patches, and integer/Boolean/Float32/Float64 grid
outputs. Native contracts additionally exercise UInt64 precision beyond 2**53,
nonfinite validation/propagation, strided ownership, invalid shapes/parameters,
RNG advancement, repeated fitting, clone independence and default handling.
Boolean image subtraction raises, consistent with NumPy.

## Explicit exclusions

Sparse graph output, including scikit-learn's default `return_as=coo_array`,
is deferred. The native preserved default `return_as="sparse"` raises;
callers must select `return_as="dense"`. Construction is directly dense and
never densifies sparse input. CSR/CSC/COO types, sparse conversions and the
family's deferred text, TF-IDF and FeatureHasher implementations are unavailable.
Source: `sklearn/feature_extraction/image.py` and family exports at reference
revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

Native rank-specific shapes, typed parameters, compile-time dtype selection,
in-place fit, explicit None RNG context and error prefixes are documented in
api.md and tested. Python array-like coercion, reflection, metadata routing,
serialization, complex arrays and incidental mismatched-rank broadcasting are
not exposed by the shared real NDArray interface. No private n-dimensional
`_extract_patches` export is promised.

## Validation

Required commands (run with project-local `PIXI_CACHE_DIR`):

```sh
pixi run build
pixi run test
pixi run parity
```

Build compiles the owned module, dependencies, tests, example and two isolated
consumers through the pinned assembler. Test runs native contracts, both
consumers, shared-source identity checks, sparse-default rejection, invalid
return selectors, invalid masks and Boolean image rejection. Parity executes
263 cases against AST-selected function/class bodies from the clean pinned
local sklearn checkout, with NumPy/SciPy infrastructure. The fixture strips
Python decorators; native validation is tested separately rather than claiming
parity with Python's decorator machinery. No installed sklearn wheel supplies
the algorithm. Shapes and RNG words are exact; bounded numeric cases use 1e-12
absolute tolerance, with native exact assertions for large UInt64 values.

Repeatable baseline commands:

```sh
pixi run python tools/benchmark.py --processes 2 --values 3 --warmups 1 --min-time 0.02 -o benchmarks/image.json
pixi run python tools/memory.py
```

Raw logs, compiler/hardware/thread metadata, timings and memory measurements
remain ignored under `builds/` and `benchmarks/`. The short pyperf run is a
baseline with stability warnings, not a speed or scalability claim.
