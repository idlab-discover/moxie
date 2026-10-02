# Dense scaler memory contract

`pixi run memory` (here or in the family root) runs
`tests/memory/acceptance.py`. It compiles `tests/memory/workload.mojo` against
the family package and runs each public operation in its own native process
under a systemd memory cap when available. It compares GNU time peak RSS with an
input-only control whose inputs stay live for the whole run. Budgets are the
formulas below plus a fixed 32 MiB allowance and 1%. The Mojo runtime allocator
keeps up to about 24 MiB of freed blocks resident: excess grows with freed size
up to that cap and does not depend on n. Copying operations must show at least
98% of their logical result, and a deliberate full-table temporary (`full_copy`)
must be rejected. Configurations are Float64 1e6x50, 1e7x5 and 2000x25000, and
Float32 1e6x50, each with a 1/1000-row strided view of the large buffer. Raw
results, commands, Mojo version and CPU go to ignored `benchmarks/memory/`.

## Required public modes and budgets

Use the live statistics and transform modules on borrowed Float32/64 NuMojo
arrays. Preserve strides and offsets, including tiny views of larger buffers.
The intended default aggregate transform tile budget is 65,536 bytes, independent
of table size, with at most 8 MiB explicitly selectable. Moments traversal uses
256-row tiles by default, no heap tile scratch and a fixed 1,600-byte stack
workspace for pairwise counts. Both owners are serial. Input-precision mean/std
calls return scalar records and allocate no arrays; their bounded reduction
stack is documented by the statistics owner.

For n rows, p features, and d bytes per input element:

| Operation | Required additional live storage |
| --- | --- |
| Standard/minmax/maxabs fit and partial_fit | O(p) feature state plus fixed tile budget |
| Copying transform and inverse | One n*p*d result plus O(p) feature state and tile budget |
| In-place, caller-output, synchronous sink | O(p) feature state and tile budget; caller destination excluded |
| Exact robust quantile fit | O(p) state, one reusable 8*n-byte column, and tile budget |

Current owner allocations are four p-element Float64 vectors for MomentsStateV1
(count, mean, variance, new_sum), three for ExtremaStateV1 (minimum, maximum,
max_abs), and three for RobustQuantilesV1 (lower, median, upper). They live with
the state/result and own their storage. Quantile fitting additionally allocates
one reusable Float64 column for the duration of the call. A scalar weight is
never expanded. Estimator inventory, all O(p): StandardScaler keeps its
MomentsStateV1, `mean_`/`var_`/`scale_`, an n_samples_seen_ vector and two
dtype-d transform vectors, and partial_fit updates a copy of the state, so a
failed update leaves fitted state intact. MinMaxScaler keeps ExtremaStateV1 plus
five dtype-d vectors, MaxAbsScaler ExtremaStateV1 plus three, and RobustScaler
four (center_, scale_ and the two transform vectors).

Copying transforms must preserve input and allocate exactly the logical result.
Explicit mutation, caller-output, and sink APIs must reuse globally fitted
statistics, validate writable layout/overlap before writing, and retain no prior
chunks. Raw-batch moments require replay for correction passes. Exact robust
quantiles require replay per feature and a sufficient single-column budget.
No full-table cast, mask, broadcast weights, residuals or squared residuals may
be allocated. NuMojo view `.copy()` can copy its backing allocation; it must not
be used as a column gather.

## Measured public bounds

| Operation | Additional live storage |
| --- | --- |
| Standard/minmax/maxabs fit and partial_fit (vector or scalar weights) | O(p) state |
| RobustScaler fit | O(p) state plus one 8*n-byte Float64 column |
| Copying transforms, inverses, axis-0 functions | one n*p*d result plus O(p) |
| `transform_inplace` / `inverse_transform_inplace` | O(p) state |
| `scale(axis=1)` | result plus three n-element vectors in X's dtype |
| `minmax_scale` / `maxabs_scale` with axis=1 | result plus two n-element vectors in X's dtype |
| `robust_scale(axis=1)` | result plus two n-element Float64 vectors |

The axis=1 functions use per-row parameters, the O(n) analogue of per-feature
state. They keep only the two (or three) vectors the affine pass reads, and
release owner state before allocating the result. They never keep a fitted
estimator alive.

## Still open

Caller-output and synchronous sink scaler methods are not exposed (the transform
owner provides `transform_into`/`transform_sink`). The tile-budget selection is
not public. No throughput comparison against a native loop or sklearn exists,
so no speed claims are made.
