# Dense affinity propagation

Import `AffinityPropagation` and `affinity_propagation` from `moxie.cluster`.
The implementation uses native Mojo, NuMojo storage and the shared Moxie RNG.
See [the example](../examples/affinity.mojo), runnable with `pixi run start`.

`AffinityPropagation[DType.float64]` is the default specialization; Float32 is
also supported. Rows are samples. Constructor keywords and defaults are:
`damping=0.5`, `max_iter=200`, `convergence_iter=15`, `copy=True`,
`preference=Preference()` (automatic median), `affinity="euclidean"`,
`verbose=False`, `random_state=None`.

```mojo
from moxie.cluster import AffinityPropagation
from moxie.cluster._affinity_propagation import Preference

var model = AffinityPropagation(random_state=5, preference=Preference(-10))
model.fit(X)
var labels = model.predict(X)
```

`affinity="euclidean"` computes negative squared Euclidean similarities.
`affinity="precomputed"` accepts any finite, nonempty square dense similarity
matrix; symmetry is not required. `predict` is available only after Euclidean
fitting. `fit_predict(X)` fits and returns owned Int64 labels. No weighting,
transform, score or incremental methods are supplied by this estimator.

`Preference()` means the median of all similarities, including the diagonal.
`Preference(value)` accepts a scalar Float64/Int; `Preference(values)` accepts
an owned copy of a Float64 NuMojo vector of length n (or one for broadcasting).
Import this typed configuration from `_affinity_propagation`. Scalar preferences
must be finite. Nonfinite entries in vector preferences retain sklearn
message propagation and nonconvergence behavior; they do not produce fabricated
finite preferences. Array preference validation follows the reference's shape
semantics for a scalar or vector; Python arbitrary broadcastable ranks are
not part of the typed vector API.

For deterministic runs use `random_state=seed` (0 through 2**32-1). Each fit
resets the integer-seeded stream. To advance a shared stream, call
`model.fit(X, state)` or `model.fit_predict(X, state)` with a mutable
`moxie._random.RandomState`. This explicit stream overload takes precedence over
the configured seed. The default `None` requires that overload with a retained
application context, optionally initialized once with `entropy_state()`.
Calling the seedless overload without a seed raises a descriptive error;
there is no hidden process-global RNG.

## Ownership and learned state

`fit` and `set_params` mutate and return a borrow of the same estimator for fluent chaining. Normal fitting replaces
learned state. Validation failures leave prior fitted state intact. Learned
attributes are `cluster_centers_indices_`, `labels_`, `affinity_matrix_`,
`n_iter_`, `n_features_in_`, and optional `cluster_centers_`. Centers are
independent snapshots of training rows; the optional field is absent after
precomputed fitting. Access requires `.value()` after checking its presence.
Fields hold empty buffers before fitting; `predict` raises `NotFittedError`.

Precomputed `copy=True` owns an independent similarity snapshot. For
`copy=False`, use `model.fit_inplace(X)` or `model.fit_inplace(X, state)`, which
requires a mutable input and retains an explicit NuMojo view. Diagonal
preferences and perturbations modify the caller's array and the fitted
`affinity_matrix_` aliases it. This exception to snapshot ownership is the
reference's explicit memory-saving mode. `fit(X)` rejects that combination
rather than mutate an immutable borrow. For Euclidean input, `copy=False`
does not mutate features. Positive/negative strided inputs use logical row order.

`get_params(deep=True)` returns an independent `AffinityParams` record.
`set_params(params)` replaces configuration only; `set_params(damping=0.8, ...)`
updates named supplied options. `params.copy_input` stores the public `copy`
parameter because a Mojo `Copyable` record already has a `copy()` method.
To reset a seed to None use the full record; use `Preference()` to restore
median preferences. `moxie.base.clone(model)` copies constructor configuration
into an unfitted estimator. A normal `.copy()` copies learned state as well.

## Functional result and diagnostics

`affinity_propagation(S, ...)` accepts the same options except `affinity` and
adds `return_n_iter=False`. It requires mutable S to express both copy modes
with one typed signature, though `copy=True` leaves S untouched. The overload
`affinity_propagation(S, state, ...)` consumes a mutable RNG stream.
It returns an `AffinityResult` containing `cluster_centers_indices`, `labels`,
and optional `n_iter` (present only when requested).

Mojo has no Python warning-category channel. Inspect `converged_` and
`equal_similarities_` on the estimator (`converged` and `equal_similarities` on
the result). These replace convergence and equal-similarity warnings. With
`verbose=True`, status is also printed. Nonconvergence can yield valid-sized
but degenerate exemplars; no exemplars means empty centers and all labels -1,
including subsequent predictions. Equal similarities with equal preferences
return one or n exemplars at iteration zero, following the reference's strict
preference comparison. They are flagged separately as arbitrary results.

Dtype is compile-time Float32/Float64. Labels/indices are Int64; responsibility
and availability messages are Float64 in both specializations. Python dynamic
dictionaries, feature names, metadata routing, serialization and Python warning
classes are separate binding work. Dense input only; see [scope.md](scope.md).

## Numerical behavior and cost

The engine preserves perturbation traversal, damping, convergence-window tests,
first-maximum ties, exemplar refinement and sorted label numbering. It uses
Mojo 1.0.0 / MAX 26.5.0, current NuMojo and `--fp-mode=contract=off`.
Dense matmul reduction order can change an exactly or nearly tied Euclidean
exemplar by roundoff. Tests require exact labels and exemplar order for
precomputed similarities. Euclidean comparisons allow a different tied exemplar
only with the identical partition and a within-cluster similarity-score gap
bounded by 2e-14 (Float64) or 2e-6 (Float32), scaled by score magnitude.
This is a numerical adaptation, not a guarantee of identical integer labels
on all ill-conditioned or tied feature data. Exemplar IDs are always sorted.

Time is O(n² * iterations + n² * features); message storage is O(n²) and
convergence history O(n * convergence_iter). Float64 messages use three n×n
buffers. Euclidean similarity construction additionally materializes Float64
features, transpose and Gram matrix. Median selection allocates a flattened
array and a NuMojo sorted copy. Copies of strided NuMojo arrays can retain the
size of their backing allocation. Concurrent mutation is unsupported.

Run `pixi run build`, `pixi run test`, and `pixi run parity`. The differential
suite executes original functions and estimator methods from the pinned local
sklearn source, using dense host plumbing for validation and nearest-center
queries. It does not replace the target with an installed sklearn wheel.

For a repeatable short performance baseline after validation:

```sh
PIXI_CACHE_DIR=$PWD/.pixi/pixi-cache pixi run python tools/benchmark.py \
  --processes 2 --values 3 --warmups 1 --min-time 0.02 \
  --affinity 0 -o benchmarks/affinity.json
```

Use an available CPU instead of 0 if necessary. pyperf measures complete native
fitting including allocations, with input construction and process startup
excluded. Raw results and hardware/tool/source metadata stay in `benchmarks/`.
These measurements are a baseline, not a claim of competitive performance.
