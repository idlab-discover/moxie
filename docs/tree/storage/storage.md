# Using the private storage engine

Import `Tree`, `new_tree`, `copy_tree`, and `validate_tree` from
`moxie.tree._tree`. These are private engine APIs, not public estimators.
`Tree()` is stateless; pass the shared `TreeData` explicitly to each method.
The [manual example](../../../tests/tree/storage/manual_tree.mojo) builds a two-leaf regression
tree and predicts 10 and 20. Run it through `pixi run tree-test`.

`new_tree(n_features, n_classes, n_categories)` creates an empty tree with
independent metadata. Use one class per output for regression, or the number
of sorted classes for classification. Category count -1 means numeric; encoded
categorical counts range from zero to 2**24. Unknown categories must already be
encoded as NaN. Bitset nodes support at most 256 categories; hash nodes use the
seed in word zero. Native callers receive clear errors for invalid codes instead
of Cython's unchecked integer casts.

Call `Tree.add_node(tree, parent, is_left, node)` in creation order; root parent
is `TREE_UNDEFINED` (-2). The method returns a consecutive node ID, resets child
links and canonicalizes leaf fields. Attach children only to unoccupied links of
internal nodes. Values live in `tree.value` with shape
`(capacity, n_outputs, max_n_classes)`; unused class slots must stay zero.
Builders populate values and publish `max_depth` (root edge depth zero).
`compute_node_depths` returns actual topology depths with root depth one.

`resize(tree, -1)` starts at capacity three then doubles. Explicit capacities
can grow or shrink; newly allocated values are zero. Shrinking can leave child
IDs dangling, so callers must repair topology before inspection. Traversal
validates that every non-root node has one parent and child IDs follow parent
IDs, supporting depth-first and best-first creation orders. Empty/incomplete
trees raise descriptive Mojo errors, rather than dereferencing invalid memory.

Features are rank-two NuMojo Float32 arrays, including offset views and signed
strides. `apply` returns independent Int64 `(n_samples,)` IDs. `predict` always
returns independent Float64 `(n_samples, n_outputs, max_n_classes)` values,
including for one output. This fixed rank is the family adaptation of sklearn's
single-output rank-two engine result; estimator owners squeeze axes and decode
classes. Prediction neither renormalizes values nor clips them: criteria and
builders enforce class proportions and monotonic bounds before publication.

`compute_feature_importances(tree, normalize=True)` accumulates weighted
impurity decreases by feature, divides by root weight, then divides by the sum
if it is positive. A pure tree with positive weight returns zeros; a zero-weight
root produces NaNs as in the pinned private sklearn engine. Negative decreases
in manually supplied inconsistent statistics are preserved.

Borrow inputs during computation. Use `copy_tree` for an independent snapshot
of nodes, metadata and values. Explicit views share mutation; resizing replaces
the value allocation and invalidates assumptions about internal borrows. Do not
mutate a tree concurrently with inspection. State cloning for unfitted
estimators belongs to their wrappers, not this fitted-state copy operation.

Build and tests assemble the dependency closure from current local sources
through Moxie's shared assembler. To record a small performance baseline after
parity, run:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pixi run python tools/tree_storage_benchmark.py \
  --processes 2 --values 3 --warmups 1 --min-time 0.02 \
  --inherit-environ OMP_NUM_THREADS,OPENBLAS_NUM_THREADS -o benchmarks/storage.json
```

The benchmark includes validation and output allocation, excludes startup and
tree/input construction, and records process RSS with GNU time. Source hashes,
compiler flags, hardware and raw results remain under ignored `benchmarks/`.
These short runs establish reproducibility, not a throughput guarantee.
