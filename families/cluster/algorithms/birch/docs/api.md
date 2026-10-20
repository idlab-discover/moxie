# Dense Birch

Import `Birch` from `moxie.cluster`. It builds a clustering-feature tree and
optionally applies Ward agglomerative clustering to the leaf subcluster centers.
The implementation is pure Mojo. Float32 and Float64 NuMojo matrices are
supported, including valid strided views. Inputs must be finite and nonempty.
The estimator dtype is a compile-time parameter, defaulting to Float64.

```mojo
import numojo as nm
from moxie.cluster import Birch

var batch = nm.zeros[DType.float64](nm.Shape(4, 2))
batch[nm.Item(2, 0)] = 8
batch[nm.Item(3, 0)] = 9
var model = Birch(threshold=0.5, n_clusters=None)
_ = model.partial_fit(batch)
_ = model.set_params(n_clusters=Optional[Int](2)).partial_fit()
print(model.predict(batch))
print(model.transform(batch))
```

## Configuration and methods

Constructor defaults match sklearn:
`Birch(*, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True)`.
Threshold must be positive and finite; branching factor must be at least two.
`n_clusters` is `Optional[Int]`: None keeps the leaf subclusters, and a positive
integer requests Ward clustering. If fewer subclusters exist than requested,
they retain distinct labels. `n_clusters_warning_` reports that condition in
place of sklearn's `ConvergenceWarning`. The one-subcluster, `n_clusters=1`
case raises because the global agglomerative step requires at least two samples,
as in the reference.

`get_params(deep=True)` returns an independent `BirchParams` record.
`set_params(record)` replaces configuration, and a keyword overload updates
individual values. To clear the optional count, use
`set_params(n_clusters=Optional[Int](None))`. `clone(model)` copies configuration
into an unfitted estimator. A normal object copy includes independent fitted
arrays and tree statistics.

`fit(X, y=None)` builds a fresh tree. `partial_fit(X, y=None)` extends it in
sample order, with the same feature count. `partial_fit()` or
`partial_fit(None)` only reruns global clustering and does not change `labels_`.
Targets are ignored. Fit and parameter mutations return a mutable borrow of the
same estimator for chaining; generic incremental trait calls are supported.

`predict(X)` returns the global label of the nearest leaf subcluster. Ties use
the first subcluster. `transform(X)` returns Euclidean distances to all leaf
subcluster centers, with the input dtype and shape `(samples, subclusters)`.
The number of transform columns can exceed `n_clusters`. `fit_predict` and
`fit_transform` combine the corresponding operations. `get_feature_names_out`
returns an owned string list `birch0`, `birch1`, etc., in transform-column order.
Optional input names must have `n_features_in_` entries.

## Learned state and ownership

`subcluster_centers_` has shape `(subclusters, features)` and the input dtype.
`subcluster_labels_` is an Int64 vector of length `subclusters`. `labels_` is an
optional Int64 vector. When computed by `partial_fit`, it labels only the latest
batch. With `compute_labels=False`, an existing `labels_` remains unchanged;
a fresh estimator has None. This preserves the reference's conditional label
updates while replacing Python attribute absence with an explicit optional.
`fit_predict` raises when no labels have ever been computed.

`n_features_in_` records the fitted feature count. Dense matrices do not record
`feature_names_in_`. Inference and feature-name queries reject unfitted models.
Validation and numerical failures during data fitting preserve the previous
model. Partial fitting makes a temporary tree copy before updating; it retains
CF statistics, not training samples. Prediction currently copies the subcluster
centers into the shared brute-force neighbor engine.

The native tree uses integer node handles rather than Python object references.
`root_` and `dummy_leaf_` are optional handles. `get_node(handle)` returns an owned
`CFNode[dtype]` snapshot with `subclusters_`, `is_leaf`, `threshold`,
`branching_factor`, `prev_leaf_`, and `next_leaf_`. Each `CFSubcluster[dtype]`
exposes `n_samples_`, `linear_sum_`, `squared_sum_`, `centroid_`, `sq_norm_`,
optional `child_`, and `radius()`. Resolve child/leaf handles through the model.
Handles belong to that fitted model and must not be reused after refitting.
Changing a snapshot does not mutate the model. Splits reuse a node slot and
append the second child; no discarded tree nodes accumulate.

Existing nodes retain the threshold and capacity with which they were created.
A parameter change affects newly created nodes during subsequent splits, as in
sklearn. Refit to apply changed tree parameters to all nodes.

## Scope and numerical behavior

Estimator-valued `n_clusters`, sparse inputs, Python metadata routing,
`set_output`, dataframe feature-name tracking, and Python bindings are not
implemented. Unsupported estimator objects are excluded by the typed
constructor. Birch has no score or sample-weight capability.

CF sums and cached squared norms use the input dtype, preserving the reference's
radius-based acceptance rule. Float32 threshold decisions and nearly tied
routing distances can depend on reduction order. Center data with large offsets
or use Float64 when cancellation in `SS/N - ||LS/N||²` matters. Statistics that
overflow raise a clear error. Transform uses the shared direct-distance kernel;
small differences from sklearn's norm/dot formula are expected.

Run `pixi run test` and `pixi run parity` at the family root. Native tests cover
lifecycle, cloning, tree balance, capacity, radius, linked leaves, sample counts,
parameter changes, optional labels, and generic incremental calls. Differential
tests execute the original sklearn CF-tree and estimator bodies and the existing
agglomerative oracle, with dense validation and pairwise-argmin adapters.
They compare centers, tree structure and statistics, label order, predictions,
transforms, partial fits, global-only finalization, duplicates, and strided
inputs. Reports are written to `builds/birch-parity.json`. Float32 tolerance is
3e-5 and Float64 tolerance is 2e-12 for the bounded numerical test data.
Discrete outputs are compared exactly except for reported sibling-order cases.
For Float64, a one-ulp difference between the two orientations of a farthest
pair can reverse a split. This can change internal sibling order, leaf-center
order, and numeric labels. The suite requires a recorded farthest-pair reversal,
then verifies the same parent-child structure and CF statistics, and the same
predictions and global clustering under the induced permutation. Exact label
numbering is not promised in those cases. Transforms are checked against direct
Euclidean distances at those tolerances. The sklearn norm/dot comparison uses
squared distances with an additional rounding bound of
`32 * dtype_epsilon * (features + 2) * (||x||² + ||center||²)`.
This accounts for cancellation near zero before taking square roots.
This does not establish full sklearn compatibility or performance equivalence.
