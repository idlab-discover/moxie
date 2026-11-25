# Histogram gradient boosting tree growth

`grow_tree` in `moxie.ensemble._hist_gradient_boosting.grower` fits one
tree from mapper-compatible UInt8 bins and Float32 gradient and Hessian spans.
Pass an empty Hessian span for unit Hessians. Missing values use the final bin.
The grower detects which features saw missing values. If a split feature has
none in training, it sends a missing prediction to the larger child, with ties
going right.

Nodes grow in best-first order under the leaf, depth, sample, Hessian, and gain
limits. For each split, the smaller child is accumulated in partition order
and the other child reuses the parent's histogram storage through subtraction.
The right child is accumulated directly when sizes are equal. Monotonic bounds
propagate from the unshrunk child values. Shrinkage is applied to the leaves
after growth.

Pass normalized interaction groups as feature-index lists. An empty outer list
allows all features. With overlapping groups, each split retains the groups
that contain every split feature on its path. The resulting union is scanned
in ascending feature order. This establishes a deterministic tie rule and can
choose a different feature when scikit-learn's integer set iteration has a
different order. The private grower accepts a leaf cap of one for an explicit
stump.

The returned training result holds the tree, a single partition of original
row IDs, and the leaf node IDs. Each leaf's half-open `[start, stop)` range into
the partition gives the rows for loss-specific leaf updates. After those
updates, use `take_tree()` to keep the predictor and release the partition.
`predict` takes a rank-2 matrix in the same byte encoding and returns a rank-1
Float64 array in input row order. It supports strided matrices.

For inference, call `make_raw_tree(grown.tree, mapper)` from the private
`predictor` module after any loss-specific leaf updates. Pass the same fitted
mapper that produced the training bins. The result owns a snapshot of the
tree's leaf values and raw split metadata. Its `predict` accepts strided
Float32 and Float64 NuMojo matrices and returns a rank-1 Float64 array.
Traversal reads only the features used on each row's path and allocates
only the result. Numax `take` and `gather` would require indexed TensorLike
inputs and intermediate arrays to advance the tree one level at a time.

Numeric splits compare against the original Float64 thresholds. A split in
the final nonmissing bin puts all non-NaN values on one side, including
infinities. Categorical features take encoded integer IDs in `[0, 254]`.
Unknown IDs, NaNs and other values outside that encoding follow the split's
missing-value branch. The predictor keeps the categorical bitset even if
the sole known category splits from missing values. In that case the
pinned scikit-learn tree can drop the bitset when building its predictor
and send the known category to the wrong child.

Root totals reuse the family's `numpy_sum` reduction with one temporary
Float64 vector of at most 256 bins. Numax `stats.sum` would require converting
the bin records to a `TensorLike` and can reassociate the additions. Keeping
the reference reduction order protects near-tied gains. The remaining
statistics reuse the [histogram](hgb-histograms.md) and
[splitter](hgb-splitting.md) operations.

Run the focused check from the Moxie root in the locked toolchain.

```sh
pixi run ensemble-prepare-oracle
PYTHONDONTWRITEBYTECODE=1 python families/ensemble/tools/check.py hgb-grower
```

The driver compares tree structure, splits, gains, leaf values, row
partitions, and binned and raw predictions bit for bit against scikit-learn
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Native contracts cover
striding, independent leaf snapshots, invalid feature counts, and routing
the sole known category separately from missing and unknown categories.
