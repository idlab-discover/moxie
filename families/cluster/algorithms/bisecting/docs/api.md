# Dense BisectingKMeans

Import `BisectingKMeans` from `moxie.cluster`. Inputs are finite NuMojo
Float32/Float64 matrices `(samples, features)` and optional same-dtype weight
vectors. Centers preserve the input dtype; labels are Int64.

```mojo
from moxie.cluster import BisectingKMeans

var model = BisectingKMeans(4, random_state=42,
                           bisecting_strategy="biggest_inertia")
model.fit(x)
var labels = model.predict(new_x)
```

A runnable example is [examples/bisecting.mojo](../../../examples/bisecting.mojo).

Constructor defaults follow sklearn: `n_clusters=8`, `init="random"`,
`n_init=1`, `random_state=None`, `max_iter=300`, `verbose=0`, `tol=1e-4`,
`copy_x=True`, `algorithm="lloyd"`, `bisecting_strategy="biggest_inertia"`.
`init="k-means++"`, `algorithm="elkan"`, and
`bisecting_strategy="largest_cluster"` are also supported. `n_init` must be a
positive integer and applies independently to every split.

Each split reuses weighted initialization and Lloyd/Elkan. The first restart
wins unless another inertia is less than its value times `(1 - 1e-6)`.
Splits use the raw `tol`, matching the reference implementation. Largest-cluster
selection counts rows, including zero-weight rows; biggest-inertia selection
uses weighted squared error. Ties choose the first leaf in left-to-right tree
order. Final label numbering follows that same order. The one-cluster case
uses the unweighted mean, including when weights are supplied, as in sklearn.

`predict` descends the fitted binary tree, comparing the two child centers at
each branch. This can differ from the nearest final center. `transform` returns
Euclidean distances to all final centers. `score` returns negative weighted
inertia using the nearest final centers, independently of tree prediction.
Learned attributes are `cluster_centers_`, `labels_`, `inertia_`, and
`n_features_in_`. There is no sklearn `n_iter_` for this estimator.

`fit`, `fit_predict`, `fit_transform`, and `score` accept optional
`sample_weight`; ignored `y` is supported. Fit and parameter updates mutate and return a borrow of the same estimator. `get_params` returns `BisectingKMeansParams`; `set_params` accepts
that record or named updates. `clone` produces an unfitted configuration copy.
Successful refits replace the tree and learned arrays; failed fits preserve
previous fitted state. Inference checks fitted state and feature counts.

Integer seeds reset on each fit. `None` leases the shared process RNG.
`fit(..., random_state=mutable_rng)` borrows and advances an explicit stream,
overriding the constructor setting for that call. All splits and restarts
consume that stream in order. Reset a seed to None through the parameter record
or a present `Optional[Optional[Int]]` containing None.

## Scope and numerical limits

Sparse inputs, callable initialization, scalar weights, runtime dtype conversion,
Python metadata routing, `feature_names_in_`, and pandas output are unavailable
(`get_feature_names_out` is supported; see below).
Invalid initialization strings raise. Random initialization requires two positive
sampling weights in each selected leaf. Negative/all-zero sampling weights raise.
Inputs are borrowed immutably: `copy_x=False` still uses an owned centered
working matrix, preserving caller values exactly. Verbose progress uses native
text rather than sklearn's exact messages.

The [KMeans numerical limitations](../../kmeans/docs/api.md#scope-and-numerical-limits)
apply to every split. Float32 reductions and nearly tied k-means++ potentials
can alter a split and all subsequent tree decisions. Large coordinate offsets
can erase norm/dot prediction distinctions. Use Float64 or center/rescale such
inputs. This implementation does not resolve those broader kernel gaps or
claim full sklearn parity or performance equivalence.

The private tree stores nodes in split-creation order, retaining centered
centers, child indices, selection scores, cluster weights and split inertia.
Training membership vectors are released after splitting/final labeling.
This is an internal validation interface, not a public persistence format.

## Validation

Run `pixi run test` and `pixi run parity` from the family root. Native contracts
cover both dtypes, configuration, cloning, repeat fits, RNG modes, weighted
single-cluster behavior and input errors. The differential suite executes
original sklearn Python bodies and compiled Cython at revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, with validation-only host plumbing.
It compares each split's centers, weights, inertia, selection score, tree links,
final labels, query predictions, score, distances and RNG continuation.
Ordinary tree links, labels and RNG draws compare exactly. A retained Float32
regression (three separated groups, seed 42, random initialization, three
restarts, Lloyd, `tol=0.5`) reverses the second split's sibling orientation under
both strategies. Its first assignment contains exactly equidistant points;
scalar and BLAS norm/dot rounding break those ties differently. The suite checks
the exact tied distances and an epsilon-scaled score bound, then verifies the
entire tree, labels, predictions and statistics under that specific sibling
permutation. These two cases are reported as label-roundoff gaps, not exact
sklearn label parity. RNG continuation and inertia still agree. Float32 numerical tolerances
are `rtol=4e-5, atol=4e-6`; Float64 uses `rtol=3e-12, atol=3e-13` for reduction
order. Cases include both strategies, initializers and solvers, restarts,
early stopping, duplicates, empty child predictions and zero weights.
Raw results and environment details are in `builds/bisecting-parity.json`.

## Transform feature names

`get_feature_names_out(input_features=None)` returns an owned `List[String]`
with one name per fitted center, in the same order as `transform` columns.
Names are `bisectingkmeans0`, `bisectingkmeans1`, and so on. Optional input
names must have `n_features_in_` entries; their values do not change the output
names. Dense arrays do not record `feature_names_in_`.

The method rejects unfitted estimators and clones. Parameter changes alone keep
the names of the current fitted centers; a successful refit replaces them.
The native list replaces sklearn's NumPy object array.
