# Dense statistics engine, API version 1

Import helpers from `moxie.preprocessing._moments`. This private engine implements
statistics for scaler consumers; it exports no public scaler estimator.
`MOMENTS_API_VERSION == 1`. `MODULE_EXPORTS.json` lists the exact module symbols;
`EXPORTS.json` deliberately adds no `moxie.preprocessing` package exports.
The original scaffold `_contracts.FeatureMoments` and `FeatureStatistics` remain
unchanged. Scalers should use this versioned API instead of compute/merge summaries.

## Logical moments updates

Construct `MomentsStateV1(features, track_variance=True)` to start with zero
counts, means and variances. All four owned vectors (`count`, `mean`, `variance`,
`new_sum`) are compact Float64 NDArrays of length `features`. Copies of the state
own independent vectors. A fresh state resets fitting; updates mutate an existing
state. Consumers may seed elements with their previous state, including NaNs,
negative variances and signed counts. Set `weighted` appropriately when importing
weighted state. Replacing vector layouts or creating aliases among state fields
is unsupported. Input and weights must be disjoint from the owned state.

| Operation | Contract |
| --- | --- |
| `update_moments(x, state, tile_rows=256)` | Unweighted logical batch, Float32/64 rank-2 input |
| `update_moments_weighted(x, weights, state, tile_rows=256)` | Borrowed Float32/64 rank-1 weights, one per row; either weight dtype may accompany either input dtype |
| `update_moments_scalar(x, sample_weight, state, tile_rows=256)` | Scalar Float64 argument cast once to input dtype, never expanded |
| `state.constant_at(j)` | Reference Float64 two-pass error bound; raises when variance is disabled or the index is invalid |
| `state.scalar_count()` | True when reference max(count)==min(count), including False for a single NaN |
| `state.scalar_count_value()` | Equal count as Float64; raises when vector counts are needed |
| `state.integer_count_at(j)` | Unweighted integral count as Int64; rejects weighted or unrepresentable counts |

The weighted vector entry point consumes already chosen weight precision. To
match StandardScaler validation, provide weights in `x`'s dtype; to call the
reference private update directly, use its weight dtype. No hidden cast array is
created. Optional weights use separate typed entry points, avoiding owned
`Optional[NDArray]` copies. `weighted` records whether any update was weighted;
consumers select floating or integer learned attributes and the scalar/vector
shape from these projections. For mixed weighted/unweighted estimator calls,
any reference count-dtype conversion at the estimator boundary is the consumer's
responsibility. The engine preserves the supplied old Float64 state exactly.

Each call is one logical batch. The engine scans the original borrowed data for
raw sums/counts, then scans again for centered residuals and the two-pass correction.
It combines the corrected variance with the old state using the reference
Youngs–Cramer equations. The raw incoming sum survives in `new_sum` independently
of `mean * count`. It describes the most recent batch, not a cumulative sum.
`tile_rows` changes traversal boundaries only, with no allocation or intermediate
partial-fit update. Do not split batches into separate calls to save memory.
Both passes require replayable data; a non-replayable stream is not this API.

Weighted effective counts follow the pinned NumPy reduction layout: pairwise
eight-lane/128-element leaves for single-column and column-major inputs, serial
reduction for row-major multi-column inputs. Float32 widening preserves the
default 8192-element reduction buffer boundary. A fixed 1600-byte stack workspace
computes the pairwise tree directly from borrowed weights and missingness; no
mask, product array or recursive call stack is allocated. This behavior is
independent of `tile_rows`. Custom NumPy reduction-buffer settings are outside
the reference configuration. Effective counts are compared exactly in parity.

For example, eight rows weighted `[1e16,1,-1e16,1,0,0,0,0]` in a single feature
have an effective incoming count of zero. Serial summation would produce one,
changing the mean denominator and the variance's NaN classification. Regression
coverage includes neighboring lengths, reduction/widening boundaries, missing
values, both dtypes, strides, and logical update partitions.

For old `(count=1, mean=0, variance=0)`, the weighted batches `[1,2]` and `[2,4]`,
each with weights `[1,-1]`, produce means `-1` and `-2`, respectively. Their
isolated `(count,mean,variance)` summaries are both `(0,-inf,NaN)`. The two isolated
consumer fixtures compile this distinction using the same live engine.

NaNs in input are missing. Infinity in input is rejected. Finite vector weights
may be negative, including zero-effective sums, but cannot all be zero. Following
the pinned validator's scalar branch, scalar NaN/infinity are accepted and
propagate; a scalar that casts to zero is rejected. Missing entries multiplied
by such nonfinite scalar weights retain the reference `nonfinite * 0` behavior.
A weight vector containing NaN/infinity is rejected before state mutation.

Old-state NaNs are never repaired. In particular, `old_mean * old_count` remains
NaN when the old mean is NaN and the old count is zero. The reference's separate
zero-old-count variance branch is preserved. Zero new counts can poison variance,
and signed weights can yield negative variance; neither is clamped. Disabling
variance leaves its vector unchanged and marks it unavailable to constant tests.
All invalid shape/weight/tile validation completes before updating state.

## Extrema and exact robust quantiles

`ExtremaStateV1(features)` owns Float64 `minimum`, `maximum`, and `max_abs` vectors.
`update_extrema(x, state)` ignores NaNs within one batch, returns NaN for an
all-missing feature, and propagates NaNs between logical updates, as reference
minimum/maximum merging does. No weights apply. Construct a new state to reset.
Returned values represent input-dtype extrema exactly; consumers choose their
learned/output dtype.

`robust_quantiles(x, q_min=25, q_max=75, column_budget_bytes=67108864)` returns
`RobustQuantilesV1` with independent Float64 `lower`, `median`, `upper` vectors.
Quantiles are exact linear order-statistic quantiles. One logical column is
gathered, missing values are filtered, and an in-place heapsort orders it; the
same column workspace serves every feature and all three statistics. Time is
O(p n log n); storage is O(p+n). All-missing features return NaN. Quantile ranges
must satisfy `0 <= q_min <= q_max <= 100`, including equal bounds and endpoints.
The column budget is distinct from tile scratch and fails clearly when too small.

Float32 median addition/division and percentile subtraction retain the reference
input precision, including overflow; interpolation for the two-value percentile
request uses Float64. This matches RobustScaler's vector quantile request, which
can differ from a NumPy scalar-percentile request. Unit-variance normal-quantile
adjustments and near-zero scale replacement belong to the scaler consumer.

## Layout, compatibility and evidence

Input and vector weights are borrowed, not modified or retained. Valid NuMojo
positive/negative strides and nonzero offsets are supported. State owns compact
vectors; no `.copy()` gathers a strided view. Concurrent mutation is unsupported.
Errors are native Mojo `Error` with `ValueError` prefixes. Python warning emission,
dynamic attribute unions, metadata routing and bindings are outside this engine.

Run `pixi run build`, `pixi run test`, `pixi run parity`, and `pixi run memory`.
Parity executes original AST function bodies from the clean pinned scikit-learn
checkout at `4358542a1bd39af2e7e1cbace8b9fed067473c5b`, verifying hashes and code
filenames. NumPy provides namespace infrastructure and robust order statistics;
no installed sklearn estimator substitutes for the pinned update. Comparisons
use relative tolerance 3e-11 and absolute 2e-10 for Float64 reduction-order rounding;
NaN/infinity classes, blocker means and count projections must agree. The kernel
uses deterministic serial accumulation for weighted value/residual products.
Unweighted sums, centered corrections and squared residuals follow NumPy's
nansum order: single-feature and column-major inputs use the eight-lane pairwise
tree (Float32 sums add 8192-element casting-buffer boundaries); row-major
multi-feature inputs sum serially. The finite Float64 column
`[1e308,0,1e308,-1e308,-1e308,0,0,0]` therefore has mean zero, as in the
reference, instead of overflowing;
bitwise identity to every BLAS configuration is not promised. Count reductions
preserve the reference order described above. See [memory.md](memory.md) for allocation
acceptance, performance limits, and [scope.md](scope.md) for dense exclusions.

## Functional scale: separate input-precision API version 1

`INPUT_PRECISION_API_VERSION == 1` names an independent contract. It does not
change `MOMENTS_API_VERSION`, Float64 accumulation, or incremental updates.
Import these helpers explicitly from the same private `_moments` module:

| API | Result and use |
| --- | --- |
| `nanmean_at(x, axis, index)` | One input-dtype scalar mean; use again after centering or scaling for correction passes |
| `nanmean_std_at(x, axis, index)` | `InputPrecisionStatsV1[dtype]` with `count: Int`, `mean`, `stddev`; floating fields have input dtype |

Both accept borrowed Float32/64 rank-2 NDArrays. Axis 0 reduces one column
(`index < p`); axis 1 reduces one row (`index < n`). Negative axis aliases -2/-1
are supported. Invalid rank, axis or index raises before reading elements.
An empty reduced dimension or all-NaN slice returns count zero and NaN statistics;
a single finite value returns itself and zero standard deviation. There are no
weights, ddof options or incremental state in this contract. Population ddof=0
and input precision are intentional requirements of functional `scale`.

Sum, mean rounding, residual subtraction, residual squares and sqrt preserve
NumPy's input precision. Division of a floating sum by the integer count uses
Float64 before casting back, following NumPy's mixed-type division. The variance
is uncorrected: it does not subtract the sum-of-residuals correction used by the
separate StandardScaler moments algorithm. In particular, Float32 columns
`[100000000,100000008,100000016]` and `[100000000,100000016,100000008]`
have stddevs `10.32795524597168` and `6.531972885131836`, respectively.

Missing entries contribute zero to sums and squared residuals and do not count.
Overflow, underflow, infinity and generated NaNs retain IEEE behavior. The
reduction API accepts infinity so that correction passes can consume overflowed
intermediates. Estimator entry points must still call the shared
`validate_dense(x, allow_nan=True)` to reject infinity in original user input.
The engine emits no Python warnings; consumers can inspect zero counts and their
correction means to implement their own warning policy.

Calls allocate no arrays or result vectors. Iterate over columns, or over bounded
row panels, and retain only the statistics that the consumer needs. The caller
controls result storage. Input must remain available for repeated scans and must
not be mutated concurrently. Each call reduces the complete logical row/column;
never split it into separate partial means. Offsets, gapped, reversed and
column-major layouts use borrowed loads. The effective reduction order follows
the K-order copy used internally by NumPy's nan reductions: fast axes use an
eight-lane pairwise tree with 128-element leaves, slow axes use serial addition.
The default NumPy reduction environment and pinned CPU/toolchain are the target.

MAX generator reductions supply valid counts; standard `isnan`, `sqrt` and SIMD arithmetic are reused. Floating
sums need the small compatibility tree because MAX's reduction order differs on
cancellation cases and its floating reduction flushes subnormal tails. No generic numerical container, mask or centered array is
introduced. See [memory.md](memory.md) for the fixed scratch bound and acceptance.

A scaler consumer should obtain the original mean/stddev, subtract the mean in
input precision if requested, and call `nanmean_at` on that result. Follow the
reference's global `allclose(mean_1, 0)` correction decision, then divide by the
reference-adjusted stddev and repeat the mean reduction for the second correction.
For axis 1, call these helpers with axis 1 on the original matrix; there is no
need to materialize NumPy's rolled view. Correction decisions and affine updates
belong to the consumer. Tests compose those steps without exporting an estimator.
The original rank/layout must be retained when reducing a row or column: passing
an extracted single-column array would change the reference's reduction order.

Acceptance includes 1,440 input-precision comparisons against the pinned dense
`scale` function body and NumPy reductions, plus the 1,038 earlier moments,
extrema and quantile comparisons. New reduction/count/correction results compare
exactly after input-dtype rounding; final test-composed scale outputs use relative
tolerance 2e-7 for Float32 and 3e-15 for Float64, zero absolute tolerance, and
matching NaN/infinity classes. Boundary lengths include 7/8, 127/128/129,
8191/8192/8193 and 16385, with C/F, gapped and signed strides. Overflow,
subnormal sums/squares, all-missing inputs, and both correction decisions are
covered. Native contracts additionally cover empty reductions, invalid axes,
rank/index errors, broadcast views and infinite correction intermediates.
Both isolated consumers import the same live API while composing
unchanged scaffold contracts. `tests/functional_parity.py` and the memory drivers
are reproducible engine acceptance, not public estimator acceptance.
