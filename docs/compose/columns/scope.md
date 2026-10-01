# Scope

Dense rank-2 `NDArray[float32/float64]` or native `ColumnTable` input, dense
output of the same dtype. Named tables support name/list/slice selection,
dtype selectors, feature names and name-aligned transform. `n_jobs` uses
MAX CPU workers with declaration-order output and transactional failed fits.
Components with `NDArray[dtype]` input, target and output; up to 8 entries per
constructor call (more through explicit `ColumnCons` lists).

Not provided: sparse inputs, sparse component outputs and `sparse_threshold`
density decisions (`_hstack` sparse branch), mixed-type or nonnumeric native
columns, regex patterns, metadata routing and HTML display. Pandas/polars
inputs and outputs are intentionally excluded. Components with other target/output types (e.g. boolean masks) are
rejected at compile time.

Dependencies: compose family `_base` and `_plumbing` (live; `ColumnSelection`
for the remainder, `BlockLayout`/`OutputSpan` for output slices,
`DenseBlockAssembler` implemented by `DenseHStack`), and the live
foundation and pipeline dependencies in Moxie’s `DEPENDENCIES.json`.

Validation: `pixi run test` and `pixi run parity`; named-column parity covers
14 Float32/Float64 cases using pandas solely as a reference input adapter.
Native contracts cover ownership, missing/duplicate names, selectors, refits,
parallel equivalence and worker failure without publishing partial state.
