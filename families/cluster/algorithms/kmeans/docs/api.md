# Dense KMeans

Import `KMeans`, `k_means`, and `kmeans_plusplus` from `moxie.cluster`.
Inputs are NuMojo Float32 or Float64 matrices `(samples, features)`. Optional
weights and squared norms are vectors with the input dtype. Outputs own their
storage; labels and selected row indices are Int64.

```mojo
from moxie.cluster import KMeans, kmeans_plusplus
import numojo as nm

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(4, 2))
    x[nm.Item(1, 0)] = 1
    x[nm.Item(2, 0)] = 10
    x[nm.Item(3, 0)] = 11
    var seeds = kmeans_plusplus(x, 2, random_state=42)
    print(seeds.indices)
    var model = KMeans(2, random_state=42, algorithm="elkan")
    model.fit(x)
    print(model.labels_, model.inertia_)
    print(model.predict(x))
    print(model.transform(x))
```

`kmeans_plusplus(X, n_clusters, *, sample_weight=None, x_squared_norms=None,
random_state=None, n_local_trials=None)` returns `KMeansPlusPlusResult` with
`centers` and `indices` fields. Centers are exact selected input rows. The first
center uses weighted sampling; later centers use distance-weighted sampling and
choose the best local trial. The default trial count is `2 + int(log(k))`;
`n_local_trials=1` gives ordinary k-means++. When all remaining distances vanish,
indices can repeat, including rows with zero weight, matching sklearn's CDF
search behavior. All-zero or negative sampling weights raise. Float32 squared
norms are validated but recomputed in Float64, as in sklearn's distance path;
Float64 supplied norms are used, without verifying that they equal the true norms.

`KMeans[dtype=DType.float64](n_clusters=8, *, init="k-means++", n_init="auto",
max_iter=300, tol=1e-4, verbose=0, random_state=None, copy_x=True,
algorithm="lloyd")` supports `"lloyd"` and `"elkan"`. Elkan with one cluster
uses Lloyd. Initialization accepts `"k-means++"`, `"random"`, or an owned snapshot
of a center array via `KMeansInit[dtype](centers)`. `n_init` accepts a positive
integer or `"auto"`: one run for k-means++/arrays and ten for random sampling.
An explicit array always performs one run. Random initialization samples without
replacement according to weights, requiring at least k positive weights.

Fits center the data, use variance-scaled tolerance, stop on unchanged labels
or squared center movement, and perform a final assignment after non-strict
convergence. Multiple runs retain the first best partition; relabelings alone
do not replace it. The existing centroid engine owns relocation and Elkan bounds.
Learned attributes are `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`, and
`n_features_in_`. `n_distinct_clusters_` additionally reports collapsed clusters.
`predict` returns nearest-center labels; `transform` returns Euclidean distances
`(samples, clusters)`; `score` returns negative weighted squared-distance sum.
`fit_predict` and `fit_transform` combine their respective operations.

Integer seeds reset on each fit/helper call. `None` leases the shared process RNG;
`seed_global_random_state` permits reproducible default-stream use. Both helpers
and `fit` also accept `random_state=mutable_rng`, borrowing and advancing it.
The explicit fit stream overrides constructor seed configuration for that call.
Use the same stream for repeated borrowed fits. All builds disable FP contraction.

`get_params()` returns `KMeansParams[dtype]`; `set_params(params)` copies it.
Named updates are also available. To reset `random_state` to None, update the
parameter record, or pass an outer present `Optional[Optional[Int]]` containing
None. `clone(model)` copies configuration, including independent explicit
initial centers, into an unfitted estimator. Failed fits preserve prior state.
Normal successful fits replace it. Inference checks fitted state and features.

`k_means` accepts the sklearn parameters and returns `KMeansResult` with `centers`,
`labels`, `inertia`, and optional `n_iter`. The latter is present only when
`return_n_iter=True`. Native records replace Python tuples; fitting mutates in
place and returns a borrow of the same estimator for fluent chaining. For an optional array argument, use `.view()` to
explicitly borrow its storage for the call or `.copy()` for an independent value.

## Scope and numerical limits

Dense finite inputs only. Sparse data, callable initialization, scalar/broadcast
weights, runtime dtype conversion, Python bindings, metadata routing and pandas
output are not implemented. Unsupported initialization strings raise. Inputs are
borrowed immutably: the implementation always materializes a centered working
matrix, including when `copy_x=False`. This documented allocation adaptation
preserves input values exactly and never retains caller data as fitted state.
Python warning classes are unavailable: inspect `n_distinct_clusters_` for the
condition that sklearn reports as `ConvergenceWarning`. Explicit-array n_init
and one-cluster Elkan fallbacks are described above; no Python warning is emitted.

The engine uses native reductions, whose last bits can differ from BLAS/NumPy.
In particular, Float32 prediction with large common coordinate offsets can lose
nearest-center distinctions in norm/dot scores, and produce different labels
and scores from sklearn's BLAS backend. Center or rescale such inputs, or use
Float64. The parity suite retains an offset regression with an explicit
floating-point error bound and reports those prediction differences separately;
it does not claim exact labels for ill-conditioned inputs. Transformation uses
the engine's direct coordinate differences, avoiding norm/dot cancellation.
Finite inputs whose initialization potentials overflow raise a clear error.
No performance equivalence or full sklearn parity is claimed.

Float32 k-means++ candidate potentials can also be nearly tied. Native scalar
and BLAS reduction orders may choose different candidates with the same RNG
state, especially when selecting many centers. The MiniBatch differential suite
records a 40-center regression: the first differing candidate scores agree
within eight Float32 epsilons, but their ordering differs. This is a numerical
initialization limitation, not a claim of identical fitted trajectories on such
inputs; batch-update parity is tested from shared explicit centers separately.

## Validation

Run `pixi run build`, `pixi run test`, and `pixi run parity` at the family root.
Native tests cover lifecycle, weighted fitting, deterministic integer/default/
borrowed RNG, both dtypes, strided input, duplicate centers, zero weights,
parameter validation, helper return branches and two independent consumers.
Differential tests execute original local sklearn Python bodies with its original
compiled Cython kernels at revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Validation-only Python plumbing supplies dense input checks and disables thread
pool management. This is not validation of sklearn's Python decorators/bindings.

Selected indices, RNG continuation, ordinary labels and iteration counts compare
exactly. Centers/inertia/distances allow Float32 rtol up to 4e-5 and Float64
2e-12 for reduction order; absolute center tolerances are 3e-6 and 2e-13.
Offset prediction and near-zero/offset norm-based reference distances have explicit
machine-epsilon error bounds. Reports reside in `builds/kmeans-parity.json`.

## Transform feature names

`get_feature_names_out(input_features=None)` returns an owned `List[String]`
with one name per fitted center, in the same order as `transform` columns.
Names are `kmeans0`, `kmeans1`, and so on. Optional input
names must have `n_features_in_` entries; their values do not change the output
names. Dense arrays do not record `feature_names_in_`.

The method rejects unfitted estimators and clones. Parameter changes alone keep
the names of the current fitted centers; a successful refit replaces them.
The native list replaces sklearn's NumPy object array.
