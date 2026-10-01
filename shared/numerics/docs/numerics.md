# Numerical storage and operations

Decision: selectively adopt pinned NuMojo dense storage, the tested operation
subset below, and two shared native reduction kernels. This is a private engine
with a restricted dense contract, not blanket NumPy compatibility.
No estimator has been migrated and no parent Moxie API has changed.

## Reproduce the candidate

Use `pixi install --locked`, then `pixi run build`, `pixi run test`, and
`pixi run parity`. NuMojo 0.10.0 is pinned to source revision
`515fb2856f0ecf3d2740a34d958fe168183e1129`; its package constraints require
Mojo **1.0.0** and **max-core 26.5.0**, both from the stable Modular channel.
The full resolved environment, including package hashes, is in `pixi.lock`.
Do not replace this combination with whichever stable version is newest.

`DEPENDENCIES.lock.json` records source provenance and hashes. `tools/check.py`
checks the NuMojo revision and every consumed source file, then materializes
verified copies under ignored `builds/deps/`. Source drift fails the build.
The local source checkout must be at the recorded relative path. This is a
bootstrap for this project, not a family dependency assembler. Parent contracts
are read-only context; no parent runtime implementation is linked here.

Python, NumPy and SciPy are validation tools. Native library operations use
NuMojo and Mojo only, with MAX available to NuMojo's parallel matrix product.
NuMojo Python conversion and I/O helpers are outside the approved runtime path.

## Dense conventions

Use `numojo.NDArray[DType.float32]` or `NDArray[DType.float64]` directly;
`numojo.Shape` and `numojo.Item` supply shape/index metadata. Samples occupy
rows and features columns. Targets remain separately shaped by each estimator.
Dtype is a compile-time parameter, not a runtime promotion policy. Kernels
require valid NuMojo metadata; manually corrupting shape/strides is outside the
contract. No replacement generic container is introduced.

Borrow input arrays for computation. Use `.copy()` for independent fitted
snapshots. Explicit `.view()` shares mutation and retains buffer lifetime after
the original array is destroyed. Both behaviors have Float32/Float64 tests. A strided view copy preserves its
layout and copies the upstream backing buffer; do not budget that copy solely
from the smaller logical view size.
An owned return moves between consumer modules; the consumer
probe verifies shared-view mutation remains visible to its producer.

Slicing, `.reshape(...)`, `.T()` and `.astype[...]()` allocate independent data
in the tested cases. They must not be assumed to be free views. A real strided
view uses `.view_with_layout(shape, strides, offset)`; strides are **element
counts** in the tested implementation. Positive and negative strides, and nonzero offsets, are tested for
mutation, copying and numerical operations. Keep arbitrary layout construction
in the storage-owning layer. Public estimators should document any
materialization and avoid retaining mutable caller aliases as fitted snapshots.
Concurrent mutation and overlapping in-place operations are not approved.

## Operation matrix

| Operation | Evaluation result and use |
| --- | --- |
| Creation, deep copy, explicit shared views | Adopted: native ownership/lifetime tests pass in both dtypes |
| Elementwise multiply and column broadcast subtraction | Adopted for finite matrices, including singleton/tall/wide/strided inputs; eager centering is a concise baseline, not the fastest measured path |
| Global `numojo.sum` | Ordinary finite cases and empty sum pass; accumulation stays in input dtype; not a stable/widened reduction guarantee |
| `numojo.mean` | Explicit `returned_dtype=dtype` is required for same-dtype centering; output default is Float64, but intermediate sum still uses input dtype |
| `numojo.sum(A, axis)` | Excluded as a general operation: 1-D axis use raises and a singleton output can terminate the process |
| `moxie._dense.sum_axis2d` | Matrix-only axis reduction, axes -2 through 1; Float64 accumulation and input-dtype output, supports empty reduced dimensions |
| `moxie._dense.row_squared_norms` | Matrix-only fused squared norms; no matrix-sized intermediate; Float64 products/accumulation, input-dtype output |
| Two-dimensional matrix multiplication | Adopted as a correctness-tested baseline for finite rank-2 inputs: Gram and rectangular products pass; throughput is below NumPy on measured large cases |
| Vector/batched matrix products | Outside the approved rank-2 subset: vector-vector `@` returns a `(1, 1)` matrix rather than a NumPy scalar (contract-tested); general batch broadcasting is not promised |
| `numojo.dot` | Excluded: vectors yield elementwise products instead of a reduced scalar |
| `solve`, inverse, unpivoted LU | Excluded as general solvers: a nonsingular zero-pivot case produces NaNs |
| Variance/ddof | Excluded: validates ddof against unrelated dimensions; do not infer sklearn variance semantics |
| NuMojo random sampling | Excluded for estimator random_state: caller-seeded API unavailable and sampling reseeds internally |
| Sparse, QR/SVD/Cholesky/eigen/least squares | Not supplied by this engine; see ownership below |

The two custom kernels use validated matrix loop bounds, NuMojo strides and
NuMojo's offset-aware access. They allocate only the result array. Widening
Float32 arithmetic is an explicit adaptation: results can differ in low bits
from NumPy's Float32 accumulation. NaNs and infinities propagate; callers apply
estimator-specific finite-input rules. Kernels raise Mojo `Error` with a
`ValueError` prefix for invalid rank/axis rather than Python exception objects.

This is not a replacement for sklearn's `row_norms` API: sparse dispatch,
unsquared norms and array-API namespaces are not exposed. No general solver,
RNG or sparse API returns a placeholder result.

## Evidence and limits

`tests/parity.py` executes the actual `row_norms` function body from the pinned
local sklearn checkout using AST extraction, with NumPy namespace plumbing.
It verifies Git revision, source cleanliness and the compiled code filename;
it does not silently use a wheel's implementation. This checks only its dense
NumPy branch. Other candidate operations are compared directly with NumPy.
Float32 uses rtol/atol 3e-5 and Float64 1e-12 for the bounded deterministic test
inputs; these tolerate summation-order differences and do not certify arbitrary
ill-conditioned data. `tests/edges.mojo` separately checks cancellation, NaN,
infinity, empty dimensions, negative axes, copies and capability errors.

`tests/negative_contracts.py` isolates the upstream singleton reduction crash
and missing seeded RNG API. Passing those negative tests means the exclusion
is reproduced, not that the candidate operation is fixed.

Performance commands (raw output stays ignored):

```sh
pixi run python tools/benchmark.py --processes 2 --values 3 --warmups 1 \
  --min-time 0.02 --affinity 0 -o benchmarks/norms.json
pixi run python tools/profile.py
pixi run python tools/allocations.py
```

The benchmark measures a native timed region after data creation, with output
allocation and repeated changing input included. Python/pyperf orchestrates
subprocesses but process startup is outside the returned elapsed time. It
compares a checked List loop, the shared fused kernel and the eager NuMojo
expression. CPU affinity, compiler, hashes and raw profiler output must travel
with results. Short samples are a baseline, not a precise speed guarantee.


## Measured tradeoffs

On this AMD Ryzen 9 5950X/Linux x86-64 host, Mojo 1.0.0 `-O3`, with one logical
CPU pinned, the row-norm workload measured the following mean microseconds per
call. Each result uses two pyperf worker processes, three measured values and
one warmup; the short run emitted stability warnings. Results are directional
baselines, not portable latency guarantees.

| Dtype / shape, contiguous | List baseline | Shared kernel | NuMojo eager expression |
| --- | ---: | ---: | ---: |
| Float32, 2×4 | 0.038 | 0.244 | 2.204 |
| Float32, 64×32 | 2.172 | 1.423 | 32.377 |
| Float32, 256×256 | 67.665 | 50.394 | 521.721 |
| Float64, 2×4 | 0.038 | 0.250 | 2.197 |
| Float64, 64×32 | 1.830 | 1.144 | 29.248 |
| Float64, 256×256 | 62.013 | 53.271 | 672.471 |

The shared kernel retains NuMojo storage while avoiding eager intermediates.
The List baseline has no general shape/view/ownership interface. Its tiny-input
advantage does not justify a second array container: the absolute shared-kernel
overhead is about 0.2 microseconds here, and representative/large cases improve.
The first checked-index implementation took about 3.9 ms for Float64 256×256;
source inspection identified per-element index-list allocation. Using validated
bounds and cached strides reduced that to about 53 microseconds. This is evidence
for changing access strategy before rejecting storage.

The eager norm expression is a one-line operation; the shared implementation is
one small function with rank checking and stride-aware iteration. Its caller is
still one line. Counting nonblank, non-comment function source lines (including
signatures/docstrings), the eager expression uses 2, the List control 11, the
shared norm kernel 20 and the axis kernel 19. The added native code buys the
tested singleton/stride semantics and measured temporary reduction. Both use NuMojo's existing container; no buffer implementation
was copied and no parent code was removed. Centering remains a short expression,
but its mean and temporaries cost time. The parent has no numerical baseline;
the List loop is a readable control, not an existing estimator implementation.

Callgrind's custom-allocator entry counts for a 64×32 workload were 1 per call
for List, 6 for the shared path, and 461/469 for contiguous/strided eager paths,
for both dtypes. Counts include output metadata and the changing-input driver.
Zero named `memcpy` deltas exclude inlined copies and do not establish zero
copying. Mutation tests and source inspection establish API copy behavior.
At 2048×512 Float64, whole-process RSS was about 42.6 MiB for the shared path
and 78.5 MiB for the expression; common setup includes both input representations.
Float32 corresponding figures were about 30.9 and 44.6 MiB.

For Float64 256×256, the complete Gram expression took about 3.21 ms on one
CPU and 1.55 ms on four; centering instead increased from 262 to 916 microseconds.
Do not turn on broad parallelism for every operation. Three repeated 30-product
runs averaged 102 ms on one CPU, 46 ms with four CPUs, and 123 ms per caller when
four callers shared those four CPUs. This demonstrates contention; downstream
outer parallel loops must coordinate their CPU budgets with MAX's inner work.
Affinity controls available CPUs, not a promise of a particular worker count.

A single-BLAS-thread NumPy context baseline at 256×256 Float64 measured about
21 microseconds for norms, 41 for centering and 388 for Gram products. That
protocol includes Python dispatch, whereas the Mojo timings return native elapsed
time, so it is not an exact pure-kernel ratio. It still exposes a substantial
throughput gap in eager centering/products. NuMojo transpose and noncontiguous
matmul paths materialize data, and its product uses its own SIMD/row-parallel
implementation rather than the NumPy BLAS implementation. The selected paths
are correct native baselines, not a claim of competitive BLAS throughput.
Optimize future hot kernels through this shared owner while preserving storage.

Raw timings and commands are in ignored `benchmarks/`; profiler outputs and
metadata are in ignored `profiling/`. `tools/workloads.py`,
`tools/numpy_baseline.py`, `tools/scaling.py`, `tools/call_costs.py`, and
`tools/memory.py` reproduce the additional comparisons. Memcheck passed 23 ownership, layout and workload cases with zero reported
errors after fetching matching host debug data into the project-local cache.
The original startup failure is retained alongside the successful run. On this
Arch Linux host, reproduce it with
`DEBUGINFOD_URLS=https://debuginfod.archlinux.org pixi run python tools/memory_checks.py`.
DHAT and default Massif do not observe the native TCMalloc allocation path reliably;
those results are explicitly inconclusive. Allocation evidence above comes from
Callgrind, and memory scale from GNU time. No memory-safety certification is made.

## Planned shared ownership

This numerical project owns future `moxie._sparse` CSR/CSC contracts,
`moxie._linalg` factorization/solve kernels and `moxie._random` reproducible RNG
contracts. These names are planned modules, not importable capabilities.
A separate implementation task must validate each before families use it.
Families should report the missing capability and depend on its single owner;
a missing factorization is not a reason to create another dense array type.

Advanced factorizations, sparse operations and sklearn-compatible random streams
remain separate implementation tasks. Their absence is an explicit exclusion
from this dense-storage decision. Family projects consume this policy and pin
these shared modules; they do not repeat storage selection. The initializer in
`src/moxie/__init__.mojo` is an empty local package root. A future family assembler
should contribute `_dense.mojo` through the export manifest while retaining its
own parent package initializer; adding competing `moxie` include paths is not
an assembly mechanism.
