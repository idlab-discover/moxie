# Multioutput plumbing

The family owns the four estimator wrappers, their capability traits and the
`moxie._multioutput` helpers. Global estimator contracts, numerical storage and
random state come from current Moxie shared owners. The family is the unit of
development; no separate child repository is required.

The wrappers live in `src/moxie/multioutput.mojo` and are exported through
`moxie.multioutput`. See [API usage](api-surface.md) and [scope](scope.md) for
supported calls, dtype requirements and current limitations.

## Exact helper interfaces and layouts

All helpers borrow arguments, return owned results, and raise Mojo `Error` with
`ValueError` prefixes for invalid shape/order. They do not implement fit, predict,
scoring, label encoding, CV, response selection or concurrency. `dtype` is a
compile-time `DType`; no implicit dtype promotion occurs.

| Interface | Contract |
| --- | --- |
| `validate_targets[dtype](y: NDArray[dtype], rows: Int)` | Nonempty rank-2 `(rows, outputs)`, outputs >= 1; accepts one output only when rank 2; does not impose finiteness policy |
| `clone_outputs[E: BaseEstimator & Deinitable](estimator: E, count: Int) -> List[E]` | count > 0; invokes foundation `clone` once per output; constructor-only independent state, subject to estimator clone contract; Deinitable is required for exception-safe list cleanup; respects explicit FrozenEstimator exceptions |
| `ChainOrder(order: List[Int], outputs: Int)` | Owns copied permutation and inverse; rejects duplicates, missing/out-of-range indices; `column(position)` and `position(column)` check bounds |
| `target_column[dtype](y, column) -> NDArray[dtype]` | Matrix to independent contiguous rank-1 `(n,)` copy; valid zero rows permitted |
| `stack_predictions[dtype](columns: List[NDArray[dtype]]) -> NDArray[dtype]` | Nonempty equal-length rank-1 columns to owned row-major `(n, q)`; zero rows permitted; input order is output order |
| `augment_features[dtype](x, responses, prefix: Int) -> NDArray[dtype]` | Both rank 2 with equal rows; `0 <= prefix <= response_width`; independent row-major `[x, responses[:,:prefix]]`, `(n,p+prefix)` |
| `ChainResponses` | Associated `Estimator: BaseEstimator`, `Input`, `Target`, `Response: Movable`; `cross_val_response(self, estimator, X, y, method: String) raises -> Response` |

NuMojo owns allocation, shape, strides, offsets and lifetime. Input arrays may be
strided (including negative strides and nonzero offsets), with valid metadata.
Helpers copy logical elements using NuMojo offset-aware access. They are indexing
adapters, not alternative numerical kernels: no reductions or replacement
container. Work/memory: column O(n), stack O(nq), augmentation O(n(p+k)), permutation
O(q). No speed claim is made; any future replacement of an accepted numerical
operation requires the shared evaluation/measurement process.

Use `numojo.NDArray[DType.float32/float64]`, `Shape`, `Item` directly for numeric
features/responses. Numeric label layout helpers also preserve Int64 exactly.
Associated types retain nonnumeric labels in the independent owner; never cast
string labels or large integers through floats. Ragged per-output classes use
`List[Classes]`; independent probabilities use `List[NDArray[DType.float64]]` with
element j of shape `(n, len(classes_[j]))`, including a single output. Chain
probabilities, log probabilities and decisions are Float64 matrices `(n,q)`, not
that ragged list.

`.view()` aliases and retains lifetime; `.copy()` owns an independent snapshot.
Slicing/transpose/reshape/conversion can allocate under the shared numerical policy; a
strided copy may copy the backing allocation. Helpers here always materialize
contiguous outputs. Concurrent mutation and overlapping in-place operations
are not supported. Input finiteness is base-estimator dependent (independent
wrappers must preserve pipeline handling of NaNs).

Accepted operations and exclusions are those in
[shared numerical policy](../../shared/numerics/docs/numerics.md), summarized
in AGENTS.md. Use `sum_axis2d` / `row_squared_norms` for those operations, not the
excluded general NuMojo axis sum/dot/solvers/variance/RNG. Shared random state
provides `permutation(q)` as an Int64 NDArray; the chain owner converts its q
indices to the owned `ChainOrder`. Integer seeds reset per reference operation;
mutable states advance. Native `None` uses an application-owned entropy context.

## CV response contract

`ChainResponses` specializes Input to a dense feature matrix, Target to a
single target vector and Response to an owned feature-dtype vector. A provider
must return exactly one out-of-fold response for each original sample, in its
original position. It owns splitting, fold cloning, coverage validation and
leakage prevention. Unsupported modes must raise rather than fabricate values.
The wrappers fit a separate full-data component at every position and request
CV responses only for positions preceding the final link.

Classifier response extraction selects the last column of binary probability
or log-probability matrices. Feature responses and requested output responses
are separate buffers, and output columns are restored to original target order.

## Assembly and validation

Moxie's root `tools/check.py` includes `tools/multioutput_checks.py` in build,
native tests and parity. `DEPENDENCIES.json` declares core module ownership;
current shared assembly reads live sources and records consumed bytes.
No historical assembler or framework cache is used.

The two isolated consumers compile all four estimators, incremental updates,
probabilities, chains and explicit CV providers. They also exercise production
`DummyRegressor` components. Tests compare assembled modules with their owners,
reject duplicate module owners and conflicting exports, and reject unavailable
component capabilities at compilation. No sibling checkout is required.

Run `pixi run build`, `pixi run test`, and `pixi run parity` from Moxie.
For focused checks use `pixi run python tools/multioutput_checks.py test` and
`pixi run python tools/multioutput_checks.py parity-only` after building.
Generated artifacts and comparison records remain in ignored `builds/`.

Native compatibility is Mojo 1.0.0, MAX 26.5.0 and NuMojo 0.10.0 at
`515fb2856f0ecf3d2740a34d958fe168183e1129`. Builds use
`-O3 --fp-mode=contract=off`.

Reference: sklearn `4358542a1bd39af2e7e1cbace8b9fed067473c5b`,
`sklearn/multioutput.py`, `sklearn/base.py`, `sklearn/utils/_response.py`,
`sklearn/tests/test_multioutput.py`, and `doc/modules/multiclass.rst`.
The chain formulation follows
[Read et al. (2009)](https://ml.cms.waikato.ac.nz/publications/2009/chains.pdf).
