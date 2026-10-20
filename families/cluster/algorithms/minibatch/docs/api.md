# MiniBatchKMeans

Import `MiniBatchKMeans` from `moxie.cluster`. It accepts finite dense NuMojo
Float32/Float64 matrices and optional same-dtype weight vectors. Constructor
names and defaults match scikit-learn:

```text
MiniBatchKMeans[dtype=DType.float64](n_clusters=8, *, init="k-means++",
    max_iter=100, batch_size=1024, verbose=0, compute_labels=True,
    random_state=None, tol=0.0, max_no_improvement=10, init_size=None,
    n_init="auto", reassignment_ratio=0.01)
```

`init` accepts k-means++, random, or an independently copied center array via
`KMeansInit[dtype](centers)` from `moxie.cluster._kmeans`. `n_init="auto"` selects
one initialization for k-means++/arrays and three for random. An explicit array
always uses one initialization. Initialization draws subsamples with replacement;
`fit` chooses the best initialization on a separately sampled validation set.
`init_size` defaults to three times the effective batch size, increases to three
times the cluster count when too small, and is capped at the input row count.

`fit(X, sample_weight=weights.view())` draws batches with probabilities proportional
to weights and updates with unit weights. `partial_fit` consumes the entire supplied
batch and uses its actual weights. It retains centers and cumulative cluster
weights across calls; subsequent batches may have fewer rows than clusters.
The first batch must contain at least `n_clusters` rows. Low-count centers are
reassigned to uniformly selected batch rows, with reset learning rates, following
the reference's reassignment schedule and half-batch cap. Tied counts use the
locked NumPy 2.5.3 AVX2 argsort order, independently of host CPU; other NumPy
versions/backends may order ties differently.

`fit` stops on variance-scaled squared center movement (`tol`), lack of improvement
in exponentially smoothed batch inertia (`max_no_improvement`), or the iteration
budget. `tol=0` disables movement stopping; `max_no_improvement=None` disables
inertia stopping. The first step does not participate in convergence checks.
`partial_fit` performs exactly one update and does not apply these stopping rules.

Learned state includes `cluster_centers_`, Int64 `labels_`, `n_features_in_`, and
`n_steps_`. With `compute_labels=True`, labels and inertia describe the complete
input to the most recent fit or partial fit. With it disabled, `fit` reports
smoothed inertia multiplied by total sample weight; `partial_fit` leaves labels
and inertia untouched. Fresh models use an empty labels vector for an absent
`labels_`; `inertia_` is `Optional[Float64]` and `n_iter_` is `Optional[Int]`.
`n_iter_` is set only by `fit`, to the ceiling of processed batches divided by
input size. `partial_fit` increments `n_steps_` and preserves any existing
`n_iter_`. For a one-step fit with `compute_labels=False`, the reference cannot
compute smoothed inertia; this implementation raises a descriptive error.

`predict`, `transform`, and `score` provide nearest-center labels, Euclidean
distances, and negative weighted inertia. `fit_predict` requires
`compute_labels=True`; `fit_transform` returns distances after fitting.
`fit`, `partial_fit`, and `set_params` mutate and return a borrow of the same estimator for fluent chaining. `get_params()`
returns `MiniBatchKMeansParams`; both typed-record and named `set_params` are
supported. Nested optionals distinguish an omitted parameter update from setting
`init_size`, `max_no_improvement`, or `random_state` to None. `clone` copies
configuration into an unfitted model with independent initial-center storage.

An integer seed resets each `fit`; repeated `partial_fit` calls retain their
own stream. Explicit `random_state=mutable_rng` overloads borrow and advance the
provided stream; use the same stream for subsequent calls. Default-stream calls
use the shared process RNG. Builds disable floating-point contraction.

## Scope and validation

Dense finite inputs only; no sparse conversion, callable initialization,
scalar/broadcast weights, Python bindings, metadata routing, or dataframe output.
Unsupported initialization strings raise. Python warning categories are not
implemented: small `init_size` and explicit-array `n_init` use the reference
fallbacks without emitting Python warnings. `verbose` prints progress and stop
reasons, with native formatting. Fitted buffers own their storage and do not
retain aliases to caller data. Failed operations are not transactional.

The existing KMeans gaps remain documented in its [API](../../kmeans/docs/api.md):
callable initialization, `copy_x=False` allocation behavior, and large-offset
Float32 prediction differences. MiniBatchKMeans shares the same assignment and
distance kernels, so the numerical caveat applies here too. Native reduction
orders may differ in low bits from NumPy/BLAS; no universal bitwise numerical
parity or performance equivalence is claimed.

Float32 k-means++ candidate potentials can also be nearly tied. Native scalar
and BLAS reduction orders may choose different candidates with the same RNG
state, especially when selecting many centers. The MiniBatch differential suite
records a 40-center regression: the first differing candidate scores agree
within eight Float32 epsilons, but their ordering differs. This is a numerical
initialization limitation, not a claim of identical fitted trajectories on such
inputs; batch-update parity is tested from shared explicit centers separately.

Run `pixi run build`, `pixi run test`, and `pixi run parity` at the family root.
The differential suite uses unedited Python method bodies and compiled Cython
kernels from local sklearn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
The test harness supplies dense validation plumbing; Python decorators and sparse
paths are outside this comparison. Incremental batch states and fit prefixes
compare centers, cumulative counts, labels, complete MT19937 state, step counts,
reassignment clocks, smoothed inertia, and stopping decisions. Labels, RNG and
integer state compare exactly. Unit/integer-weight counts compare exactly;
fractional-weight counts allow four dtype epsilons of relative error.
Centers/inertia allow Float32 rtol 4e-5 and atol
2e-6, Float64 rtol 2e-12 and atol 2e-13 for reduction order. Raw reports are stored
in ignored `builds/minibatch-parity.json`.

## Transform feature names

`get_feature_names_out(input_features=None)` returns an owned `List[String]`
with one name per fitted center, in the same order as `transform` columns.
Names are `minibatchkmeans0`, `minibatchkmeans1`, and so on. Optional input
names must have `n_features_in_` entries; their values do not change the output
names. Dense arrays do not record `feature_names_in_`.

The method rejects unfitted estimators and clones. Parameter changes alone keep
the names of the current fitted centers; a successful refit replaces them.
The native list replaces sklearn's NumPy object array.

Names are also available after `partial_fit`, including with `compute_labels=False`.
