# Shared dense storage and operations

Owner: `shared/numerics`, module `moxie._dense`. Follow
[live-source policy](../../../docs/family-development.md). Builds use current
NuMojo sources through shared assembly; Pixi locks retain compatible Mojo/MAX.
Numax remains preferred CPU numerical backend; this contract defines accepted
NuMojo storage and operation-specific fallbacks, not universal NumPy parity.

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

## Scalar transcendental imports

Mojo 1.0 `std.math.exp`, `log` and `pow` can lose roughly 1e-10 relative
accuracy in Float64. Numax `core.libm.exp` and `log` provide dedicated
Float64 implementations and delegate other dtypes to the standard library.
Use `numax.core.libm.exp(Float64)` and `numax.core.libm.log(Float64)` for
CPU scalar Float64 calculations. The
[shared validation](../../special/docs/special.md#scalar-float64-exp-and-log)
requires agreement within one ULP of the rounded reference on its finite test
cases and checks signed zeros, infinities, NaNs and subnormal boundaries.
Check caller-specific edge semantics and pinned scikit-learn parity when
migrating callers. The scalar tolerance does not determine the tolerance of
an estimator or a composition.

Use `moxie._dense.pow_float64(base, exponent)` for runtime scalar Float64
powers on CPU. It uses [LLVM's power intrinsic](https://releases.llvm.org/18.1.4/docs/LangRef.html#llvm-pow-intrinsic),
which follows the platform libm result semantics. Numax `core.libm` has no
`pow` API. Composing rounded `log`, multiplication, and `exp` can amplify
rounding error for large products and requires extra logic for negative bases
and signed zero.

Run `python shared/numerics/tools/check.py pow-accuracy` from the repository
root with the locked toolchain. The probe checks agreement within one ULP of
the 100-digit Decimal reference rounded to Float64. It includes subnormal
bases and results, representable exponents around underflow and overflow,
and negative bases with integer exponents. Zero signs, infinities, and NaN
classification are checked separately. This bound describes the sampled CPU
results. Keep approximate floating-point flags disabled for numerical parity.
Neighbors uses the shared primitive for general Minkowski powers and verifies
weighted and unweighted queries against the pinned reference. ClassicalMDS
uses it for Minkowski dissimilarities and verifies distances, eigenvalues and
coordinates against the same reference.

Run `python tools/check_std_math_precision.py` from the repository root to
check direct `from std.math import` declarations in production Mojo sources.
It detects new imports and reports stale baseline entries so they can be
removed in the same change as the migration. The inventory includes these
remaining imports from tracking PR #10 part 5:

| Owner | Remaining users |
| --- | --- |
| Cluster | K-means++ local-trial count. |
| Linear model | Bayesian regression scores. |
| Multioutput | Classifier-chain log probabilities. |
| Preprocessing | Unused scaler `log` import. |
| SVM | libsvm probability calibration, liblinear logistic optimization, and C/nu-SVC log probabilities. One-class SVM also has an unused `log` import. |

## Validation

Run `pixi run build`, `pixi run test`, `pixi run parity` here.
Parity executes original sklearn `row_norms` from the recorded reference;
Python/NumPy/SciPy remain tooling. Tests cover ownership, strides, empty axes,
NaNs and numerical errors. Float32 tolerances: 3e-5; Float64: 1e-12 for tested
bounded inputs. Negative probes retain singleton-axis reduction and seeded-RNG
exclusions. New compositions need owner-local evidence.

Benchmarks/profiling: existing `tools/benchmark.py`, `profile.py`,
`allocations.py`, `memory_checks.py`, `scaling.py`, `call_costs.py`.
Keep commands, flags, versions, hardware and raw results under ignored
`benchmarks/` or `profiling/`. Historical timings are not current guarantees.

## Other owners

- [RNG](../../random/docs/random.md): `moxie._random`.
- [Covariance solves](../../linalg/docs/linalg.md): `moxie._linalg`.
- [Labels](../../labels/docs/labels.md): `moxie._labels`.
- [Special functions](../../special/docs/special.md): `moxie._special`.
- [Categorical columns](../../../docs/categorical.md): `moxie._categorical`.
- [Assembly](../../assembly/docs/assembly.md): one composed namespace.

Sparse storage remains unavailable here. Each owner documents its supported
operations; importing shared storage does not approve other NuMojo routines.
