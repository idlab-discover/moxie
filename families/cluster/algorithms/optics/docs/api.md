# Dense OPTICS

The public imports supply native Mojo implementations:

```mojo
from moxie.cluster import (
    OPTICS, compute_optics_graph, cluster_optics_dbscan, cluster_optics_xi,
)
```

The reference is scikit-learn `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
See [the runnable example](../../../examples/optics.mojo).

`OPTICS[dtype]` accepts finite dense Float32/Float64 feature matrices and square,
finite, nonnegative dense precomputed distances. Like sklearn's estimator, fit
converts both input dtypes to Float64. Core and reachability distances are
Float64; labels, ordering, predecessors and hierarchy are Int64. Input rows stay
in sample order except `ordering_` and the hierarchy's inclusive ordering-index
intervals. Noise labels and seed predecessors are -1; unreachable distances are
positive infinity. Equal reachability chooses the lowest unprocessed sample ID.

Constructor defaults match sklearn: `min_samples=5`, `max_eps=inf`,
`metric="minkowski"`, `p=2`, `metric_params=None`, `cluster_method="xi"`,
`eps=None`, `xi=0.05`, `predecessor_correction=True`, `min_cluster_size=None`,
`algorithm="auto"`, `leaf_size=30`, `memory=None`, `n_jobs=None`.

Supported metrics are Euclidean (`euclidean`, `l2`), Manhattan (`manhattan`,
`cityblock`, `l1`), Chebyshev, Minkowski (including feature weights), and
`precomputed`. `metric_params` is an optional neighbor-owned
`Metric[DType.float64]`; its metric must agree with `metric` and its `p` overrides
the constructor's `p`. `auto`, `brute`, `kd_tree`, and `ball_tree` use the live
neighbor implementation and its compatibility checks. `n_jobs` is a scheduling
hint; the neighbor owner currently executes serially. Other metrics and
non-None `memory` raise clear errors. Sparse input, Python callbacks, metadata routing and feature-name
containers are unavailable. There is no predict, transform, score,
sample-weight or incremental-fit capability.

Use `OPTICSSize` from `moxie.cluster._optics` to represent counts and fractions:
`OPTICSSize(3)` is an integer count; `OPTICSSize(Float64(0.1))` is a fraction.
Counts must be at least two and, when used, at most the sample count.
`min_cluster_size` is unused in DBSCAN extraction. Fractions multiply
the sample count, truncate, and round up to a minimum of two. Integer 1 is
invalid; fraction 1.0 means all samples. `min_samples` permits fraction zero;
the estimator's `min_cluster_size` excludes it, while `cluster_optics_xi` accepts
it (resolving to two), as in sklearn. A missing cluster size follows min_samples.
The estimator requires a finite `p >= 1`; `compute_optics_graph` accepts any
positive `p`, including infinity, matching sklearn's two parameter constraints.

`fit(X, y=None)` and both typed and keyword `set_params` return a mutable borrow.
`get_params` returns `OPTICSParams`; clone preserves parameters and clears all
learned state. `fit_predict` returns independent labels. Successful refit
replaces state; failed fitting preserves the previous learned arrays.
`cluster_hierarchy_` is `Optional[Labels]`: present for Xi, absent for DBSCAN,
including after switching modes. An empty Xi hierarchy has shape `(0, 2)`.
Nullable keyword updates use nested optionals, as in the other family estimators.

`compute_optics_graph` takes sklearn's explicit keyword configuration and returns
`OPTICSGraph(ordering, core_distances, reachability, predecessor)`. Its `p`
argument requires a Float64; the Python helper's `p=None` is not accepted. This helper
also promotes Float32 to Float64, an explicit difference from sklearn's direct
Float32 helper. `cluster_optics_dbscan` takes Float64 reachability/core vectors,
Int64 ordering, and eps, returning labels. `cluster_optics_xi` takes Float64
reachability, Int64 predecessor/ordering and size configuration, returning
`OPTICSXiResult(labels, clusters)`. Helpers validate shapes, permutation indices,
and nonnegative distances; infinity is allowed. Returned arrays own their data.

The graph uses one-row kNN and radius queries and a linear minimum scan per
processed sample, giving quadratic selection work and linear graph storage.
It does not allocate the full pairwise distance matrix or all radius lists.
Input and neighbor snapshots require O(n_samples * n_features) storage. Xi
extraction retains nested intervals, smaller clusters before enclosing ones,
and applies predecessor correction by original sample ID. Float64 distances
are rounded to 15 decimal places before reachability updates, following sklearn.
Xi extraction rejects `xi=1` with a descriptive error (the pinned sklearn
implementation divides by zero at this endpoint). All-infinite reachability
produces noise without sklearn's Python warning.
No performance benchmark or complete scikit-learn parity is claimed.

Run `pixi run test` and `pixi run parity`. The focused differential command is
`pixi run python algorithms/optics/tests/parity.py` after a family build.
It executes the pinned sklearn clustering and extraction bodies with independent
SciPy distance/query primitives; this checks OPTICS rather than reproducing
neighbor-backend implementation details. Graph ordering, predecessors, labels
and hierarchy must match exactly. Distance tolerance is 2e-12 absolute/relative
for Float64 arithmetic and decimal-rounding differences. Both input dtypes,
duplicates, ties, nested densities, zero/finite/infinite radii, extraction modes,
fractional sizes, metrics and query backends are exercised. Raw evidence is
written to ignored `builds/optics-parity.json`.
