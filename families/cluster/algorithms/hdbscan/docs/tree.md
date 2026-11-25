# HDBSCAN tree extraction

The private `condense_tree(tree, min_cluster_size)` consumes a complete
`MergeTree` with sorted Float64 merge distances. It borrows the linkage arrays,
including strided views, and returns owned, breadth-first edges. Each edge
records the parent, child, branch size and inverse distance at which the branch
splits or the point falls out. Leaves keep their sample indices. Condensed
cluster IDs start at the sample count and place each parent before its children.

`tree_to_labels(tree, min_cluster_size=10, ...)` selects clusters by excess of
mass (`"eom"`) or `"leaf"`. It supports `allow_single_cluster`,
`cluster_selection_epsilon`, and the EOM-only optional `max_cluster_size`.
It returns a native `FlatClustering` record with owned Int64 labels and
Float64 membership probabilities, both in input sample order. Selected clusters
are numbered by ascending condensed ID. Noise has label −1 and probability 0.
A complete linkage is required. A partial tree, missing merge distances, NaN
heights, decreasing heights, nonfinite epsilon, or other invalid options raise Mojo `ValueError` errors.
Zero and infinite heights follow the reference's inverse-distance rules.
If single-cluster mode selects only a nonroot cluster and a leftover point
passes the root threshold, the pinned reference raises `KeyError`. The native
extractor reports this case as a Mojo `Error` with a `KeyError` prefix.

Pruning traverses each discarded subtree once. Condensed cluster statistics and
selection use linear passes over topologically numbered IDs, with linear scratch.
Irregular tree traversals and ordered stability accumulation use native Lists.
Numax `sum` operates over dense tensor axes and would require
materializing ragged edges into a rectangular tensor or changing accumulation
order. NuMojo supplies the strided linkage reads and owned outputs.

## Focused validation

From the repository root with the locked Mojo 1.0.0 and MAX 26.5.0 environment:

```sh
python -B -c 'import sys; sys.path.insert(0, "families/cluster/tools"); import check; check.assemble(check.ROOT, check.ROOT / "DEPENDENCIES.json"); check.check_integration(); check.compile_one(check.ROOT, check.ROOT / "algorithms/hdbscan/tests/tree_driver.mojo", "hdbscan_tree_driver"); check.compile_one(check.ROOT, check.ROOT / "algorithms/hdbscan/tests/hierarchy_driver.mojo", "hdbscan_hierarchy_driver")'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B families/cluster/algorithms/hdbscan/tests/tree_parity.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B families/cluster/algorithms/hdbscan/tests/hierarchy_parity.py
```

The parity runner compiles the pinned sklearn `_tree.pyx` oracle. Its private
build explicitly extracts singleton values at two epsilon traversal reads for
NumPy 2.5 compatibility. It compares the complete condensed edge sequence,
exact labels, and probabilities. It also
passes Float32 and Float64 strided precomputed inputs through native linkage and
tree extraction, then compares both steps against pinned sklearn. Float64
values use a relative tolerance of 2e-12 for decimal output from the drivers.
