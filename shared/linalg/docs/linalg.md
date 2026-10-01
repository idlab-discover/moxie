# Covariance solve adapter

`moxie._linalg` owns the shared covariance adapter over NuMojo
`NDArray[DType.float32]` and `NDArray[DType.float64]`. Numerical factorization and
substitution are delegated to the native `mojo-linalg` implementation.
There is no Python, NumPy, SciPy or foreign numerical runtime in library calls.

```mojo
from moxie._linalg import covariance_factor, covariance_inverse, covariance_solve
from numojo import Shape, zeros

var covariance = zeros[DType.float64](Shape(2, 2))
covariance.unsafe_set(0, 2)
covariance.unsafe_set(3, 4)
var rhs = zeros[DType.float64](Shape(2))
rhs.unsafe_set(0, 6)
rhs.unsafe_set(1, 8)
var factor = covariance_factor(covariance)
var solution = factor.solve(rhs)  # [3, 2]
var precision = factor.inverse()  # diagonal [0.5, 0.25]
```

## Calls and contracts

| Call | Result |
| --- | --- |
| `covariance_factor(V)` | Owned `CovarianceFactor[dtype]`, reusable across RHS inputs |
| `factor.solve(B)` / `covariance_solve(V, B)` | Owned solution of `V @ X = B` |
| `factor.inverse()` / `covariance_inverse(V)` | Owned precision matrix, computed by solving against identity |
| `factor.lower_factor()` | Independent lower-triangular factor snapshot |
| `factor.order()` | Runtime covariance dimension |

Covariance must be rank two, square, finite in **both** triangles, and exactly
symmetric. Positive-definiteness is checked by the delegated Cholesky kernel.
Singular, indefinite and asymmetric inputs raise; no jitter, regularization,
symmetrization, pseudo-inverse, or solver switch is applied. This private adapter
is explicitly restricted to SPD covariance matrices. It is not a general
`numpy.linalg.inv` replacement, which can accept invertible nonsymmetric inputs.
Nearly symmetric caller data must be handled explicitly by its statistical owner.

RHS shape `(n,)` returns `(n,)`; `(n,k)` returns `(n,k)`, including `k=0`.
RHS must be finite and have the covariance's dtype. Dtype conversion is explicit
caller work. Empty `(0,0)` covariance and compatible empty RHS shapes are valid
linear-algebra operations; estimators enforce their own nonempty-data rules.
Unsupported dtypes fail at compilation. Runtime failures use Mojo `Error` with
a `ValueError` prefix, retaining the backend's failure kind and operation.

Inputs are borrowed and preserved on both success and failure. Positive/negative
strides, transposes and nonzero offsets are supported with valid NuMojo metadata.
The factor owns compact `n*n` storage. Each solve allocates a compact logical RHS
copy, not a copy of a sliced input's whole backing allocation. Inversion allocates
one `n*n` identity/result buffer and solves in place inside that owned buffer.
`lower_factor()` and `.copy()` provide independent snapshots. No caller-owned
array is mutated, and no input aliases survive in the returned factor/result.
Concurrent mutation of borrowed inputs is outside this contract.

Arithmetic remains in the selected dtype. There is no condition estimate or
promise of reliable forward error for arbitrary ill-conditioned matrices.
Nonpositive pivots and nonfinite intermediate/output values raise explicitly;
very small positive pivots may factor or fail according to dtype rounding.
Prefer `solve` when a consumer needs a solution rather than the full inverse.

## Why this backend

Numax remains the preferred numerical dependency. At the evaluated revision
`8fae39ae48febc240f125f9afb1721de46d86618`, its public Tensor Cholesky/solve
entry points require compile-time dimensions. Runtime-sized lower-level panel
primitives were also examined: their import fails on the accepted Mojo 1.0.0 /
MAX 26.5.0 environment because `layout.tile_tensor.DefaultEngine` is unavailable.
They require storage/toolchain integration before they can serve this adapter.
This is an operation-specific fallback, not a claim that Numax lacks Cholesky.

The existing `mojo-linalg` revision
`d89dd6a1a2508a7767dab415f841121b9941b9d3` already uses the accepted NuMojo
storage and compiler. Its scalar Cholesky/triangular solve is consumed directly;
no factorization is copied or reimplemented here. The wrapper adds full
covariance validation, consistent errors, and inverse-via-solve composition.
NuMojo's general solver exclusions remain in force.

## Consumption and validation

Use current `shared/assembly/tools/consume.py`, the dependency manifest and
`EXPORTS.json` under the [live-source policy](../../../docs/family-development.md).
Backend revisions above identify evaluated evidence, not build gates. Covariance estimation,
sample normalization, metric evaluation and estimator APIs remain with their
respective owners. In particular, sklearn's automatic Mahalanobis covariance
uses sample covariance (`ddof=1`); this adapter takes an already computed V.

Run `pixi install --locked`, `pixi run build`, `pixi run test`, and
`pixi run parity`. Tests cover both dtypes, vector/multiple RHS, empty dimensions,
strides/offsets, alias independence, invalid inputs, singularity and numerical
failure. Parity compares to NumPy/SciPy solves and executes the pinned local
sklearn `_precompute_metric_params` body for the Mahalanobis precision fixture.
Tolerances and case conditioning are recorded by `tests/parity.py`.
The two isolated consumers exercise inverse preparation and factor reuse.

To measure adapter validation overhead against the same backend, run:

```sh
pixi run python tools/benchmark.py --processes 1 --values 2 --warmups 1 \
  --min-time 0.01 --affinity 0 -o benchmarks/covariance.json
pixi run python tools/memory.py
```

The pyperf native region includes factor allocation and a four-RHS solve, with
startup and input construction excluded. Both modes use identical changing
inputs. GNU time records whole-process RSS, including runtime/setup. These short
runs characterize wrapper cost; they do not establish competitive BLAS/LAPACK
throughput or a universal performance guarantee. Use a CPU affinity available
on your host and longer runs for deployment decisions.
