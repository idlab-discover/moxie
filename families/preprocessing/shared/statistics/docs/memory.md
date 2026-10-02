# Statistics memory contract

The engine borrows caller-owned input and weights. It retains no input views and
never materializes a table-sized cast, mask, product, centered copy or squared
residual. Shape/stride access uses the numerical owner's NuMojo storage.
One worker performs all operations. The total reduction scratch budget B is
**1600 bytes** on the stack: three 64-element Int/Float64 stacks and eight Float64
accumulator lanes. Scalar locals and compiler call frames are additional constant
overhead. No heap scratch is allocated. `DEFAULT_TILE_ROWS=256` only groups
iteration, and every positive tile size has the same memory bound. A tile is
never a logical moments update.

## Additional storage

Let n be rows and p features. These are payload bytes, excluding caller input
and existing state. NuMojo supplies vector metadata, reference counts and allocator
rounding; these fixed per-array costs are checked separately with native tooling.

| Operation | Owned payload and lifetime | Additional update scratch |
| --- | --- | --- |
| Moments state construction | Four p-element Float64 vectors: count, mean, variance, latest raw new_sum; 32p bytes until state destruction | Constant scalar locals |
| Moments update | Reuses those four vectors; no new heap array, including mean-only mode | At most 1600 bytes fixed stack workspace plus scalar locals |
| Extrema state construction | Three p-element Float64 vectors: minimum, maximum, max_abs; 24p bytes until state destruction | Constant scalar locals |
| Extrema update | Reuses those three vectors | B=0 |
| Robust quantiles | Three p-element Float64 result vectors, 24p bytes transferred to caller | One n-element Float64 column, 8n bytes, freed on return; in-place heapsort has constant scratch |
| Explicit state copy | Another complete state, intentionally owned by caller | Never performed during update |
| Scalar/vector count projection | Scalar values or existing count vector, no new array | B=0 |

`column_budget_bytes` defaults to 64 MiB (8,388,608 rows for the Float64 column),
independently of B. A caller may choose a smaller or larger budget. Validation
rejects an insufficient budget before workspace allocation; no approximation or
parallel column workers are substituted. Missing-value filtering uses the front
of the same column. Median and both quantiles reuse its single sort.

Logical elements are gathered through offsets/strides. The engine never calls
`NDArray.copy()` on a column or small view, which could copy an entire backing
allocation. Both Float32/64 inputs and weights support negative/positive strides.
All fitting inputs remain unchanged. These kernels require replayable input;
streaming transform destinations and in-place/output aliases belong to the
separate transform owner and downstream estimators, not this fit-only engine.

## Executable acceptance

`pixi run memory` compiles first, runs `tests/memory.py` and saves raw logs under
`benchmarks/memory/` and `profiling/`. It exits nonzero on any violation.
`tests/memory_probe.mojo` runs native operations in separate processes with the
same touched input/weight buffers and runtime setup as input-only controls.

The gate combines three independent forms of evidence:

1. The engine's single allocation site emits byte events only when
   `MOXIE_STATISTICS_ALLOC_TRACE=1`. Exact event counts/payload sums must equal
   the table above. A source gate rejects untraced allocation/copy operations.
2. Valgrind Callgrind observes the actual `TCMallocGlobals::tc_new` calls used
   by NuMojo. Operation-minus-control counts must stay constant across row
   counts 32 and 257 at four features, and 32, 257 and 16385 at one feature,
   for both dtypes. The single-feature cases exercise pairwise count reductions
   and widening-buffer boundaries. Instrumentation must
   see the negative control too; a profiler missing this allocator fails.
3. GNU time records each process's peak RSS. Operation-minus-input-control RSS
   must be no more than its payload budget plus 8 MiB fixed overhead/noise.
   This allowance covers runtime startup, allocation rounding/caching and RSS
   granularity; it is not permission to allocate untraced scratch. Two input
   controls are run, taking the smaller peak to make the gate conservative.

The declared per-process RAM ceiling is 512 MiB. Workloads include 65,536×64,
131,072×64 and 8,192×1,024 tables, signed weights, NaNs, tile sizes 1 and 257,
reverse strides, and a 5×3 view of an 8,388,608-element backing buffer. Quantile
sweeps use n=8,192/65,536 and p=32/64. Both dtypes run every configuration.
The smallest large input is 16 MiB, above the fixed RSS allowance. A deliberate
owned Float64 table/backing copy is touched and kept live; the same threshold
must reject it. Inputs and weights are checked elementwise after each operation.
This also catches mutations without allocating a preservation snapshot.

The measured initial run completed 84 isolated workloads. Native allocation-call
increments were 20 for moments, 15 for extrema, 20 for quantiles and 5 for the
forbidden copy, unchanged by row count or dtype. Each NuMojo array entails five
allocator calls (payload plus metadata). The byte trace distinguishes feature
vectors from columns/tables; allocator counts alone would not do so.
At 131,072×64 Float64, input controls used about 84–85 MiB, moments about 85 MiB,
and the forbidden-copy control about 144 MiB. At 65,536×64, quantiles traced
525,824 payload bytes (24p+8n). On the tiny view, quantiles traced only 112 bytes.
Raw results, not these rounded illustrative values, govern each rerun.

RSS alone cannot prove the absence of hidden copies. Allocation tracing alone
cannot see a bypassed allocation. Callgrind counts do not measure allocation
sizes. All three checks, current storage/source inspection, scaling sweeps and
the deliberately failing workload are mandatory parts of this gate.

## Timing baseline and limitations

After memory acceptance, `tests/performance.py` uses pyperf to time moments,
extrema and quantiles on 8×3 and 4,096×64 inputs in both dtypes. Native timed
regions exclude compilation, process startup and input generation, and include
state/workspace allocation. Comparisons use a simple native two-pass loop and
the pinned sklearn moments function; extrema/quantile reference expressions
follow the pinned scaler call sites. Raw pyperf results are stored in
`benchmarks/performance.json`. Flags are `-O3 --fp-mode=contract=off`, with one
worker and `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. Inputs are deterministic.
Compiler, hardware, seeds, commands, thread settings and tool metadata accompany
the local artifacts. Reproduce with `pixi run memory`.

The initial Float64 4,096×64 baseline measured about 10.7 ms for native moments,
4.5 ms for the simple native loop and 1.0 ms for the pinned NumPy-backed update.
Native extrema measured about 5.4 ms versus 0.57 ms; exact quantiles about 22 ms
versus 9.9 ms. The native path trades throughput for the proven memory bound;
these are observed slowdowns, not speedup claims. Concurrent runner peers and
short runs can distort timings, especially tiny cases. Repeat isolated runs
before tuning or making performance claims. No universal throughput acceptance
threshold is asserted.

Engine checks establish only engine bounds. Scaler consumers must separately
validate fit/partial_fit and their allocating, in-place, caller-output and
streaming transforms. Exact quantile fitting inherently needs one column, even
when all later transforms stream their outputs.

## Input-precision mean/std extension

`nanmean_at` and `nanmean_std_at` return scalar values/records and allocate **zero
heap payload or metadata**. There are no owned O(n) or O(p) result vectors.
A caller requesting all results may store p column records or bounded row panels;
that caller storage is explicit. Input and any weights used by other APIs remain
borrowed. No full-input validation scan is repeated per index: these helpers
validate shape/axis/index and permit IEEE infinity in correction intermediates.
Public consumers perform finite validation before their first reduction.

The compatibility tree uses two 64-element Int stacks and one 64-element
input-dtype stack, plus eight accumulator lanes: 1,600 bytes for Float64 and
1,312 bytes for Float32, within the existing B=1,600 byte fixed scratch budget.
MAX count reductions run sequentially with the tree, using
constant SIMD/call-frame space; ordinary scalar locals and backend call frames
are additional fixed overhead. MAX's sum generator executes a single reduction
worker. There is no table materialization or column workspace for this API.

The memory driver runs `precision0` and `precision1` for both dtypes on all large,
tall, wide, reversed and tiny-view configurations. Each performs mean/stddev plus
an additional mean-only pass, preserving the input elementwise. Traced allocations
must be zero; Callgrind operation-minus-control NuMojo allocator calls must also
be zero across the n/p sweeps. RSS must stay within the same fixed allowance,
and the deliberately forbidden table remains the mandatory negative control.
Both isolated consumers use the new shared module and distinguish the functional
scale collision while retaining the earlier signed-weight tests.

The pyperf baseline now includes both axes of mean/std plus a correction-mean
pass, corresponding NumPy nan reductions, and a simple input-precision native
column loop. Native setup and process startup remain outside timed regions.
Raw results are in `benchmarks/performance.json`; no speedup is promised.

The extension acceptance run completed **116 isolated workloads**. Both new
operations traced zero payload bytes in every workload, with zero extra native
allocator calls for both dtypes at (n,p)=(32,4), (257,4), (32,1), (257,1), and
(16385,1). The large 131072×64 runs used approximately 53 MiB process peak RSS
for Float32 and 85 MiB for Float64, including caller input and weights. All
input-control deltas met the fixed allowance; deliberate table copies failed the
same budget as required. Exact per-process values are recorded in the raw report.

On the 4096×64 workload, Float32 column/row reductions plus the extra mean pass
measured about 4.32/5.02 ms versus NumPy's 1.39/1.24 ms. Float64 measured about
5.53/8.75 ms versus 1.48/1.68 ms. The simple native column mean/std loop measured
0.51 ms (Float32) and 1.06 ms (Float64); that simpler baseline omits the separate
mean-only pass and does not preserve the full layout-dependent reduction order.
These short pyperf measurements have stability warnings and may include peer
contention. They establish a reproducible baseline and expose overhead; they are
not performance equivalence or speedup claims. The compatibility implementation
retains bounded storage while future tuning remains possible within this owner.
