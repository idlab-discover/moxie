# Histogram gradient boosting split selection

`find_node_split` in
`moxie.ensemble._hist_gradient_boosting.splitting` takes
[node histograms](hgb-histograms.md), the node's gradient and Hessian totals,
sample count, and current bounded value. Pass the mapper's nonmissing bin
counts and categorical flags, the training matrix's missing-value flags, and
per-feature monotonic directions. The final histogram bin is reserved for
missing values. Empty Hessians in the histogram builder represent unit
Hessians for this splitter. Pass explicit Hessians for other losses or
weighted derivatives.

The splitter searches the supplied feature IDs in order. Pass an empty
span to exclude every feature. The grower computes this span from the node's
interaction constraints and any feature subsampling. For numeric features,
the splitter scans thresholds first with missing values on the right, then
with them on the left, keeping the first strict maximum gain. The final
nonmissing bin can isolate missing rows. The children's bounded Newton
values must respect the feature's monotonic direction, and both children
must meet the sample and Hessian floors. The gain must strictly exceed the
configured minimum.

For categorical features, the splitter orders bins by
`sum_gradients / (sum_hessians + 10)` and scans from both ends. Bins with normalized
Hessian support below 10 always go right. The missing bin participates in
this search. Equal category scores use descending bin codes, matching the
tested pinned scikit-learn build's tie order. This explicit order is portable
across platforms. The bitset indexes encoded bins. Translate it through the
bin mapper's categories for raw-value prediction.

The search returns `Optional[_Split]`. `None` indicates that no candidate
passed the constraints. `partition_rows` takes a mutable span of row IDs and
a separate mutable scratch span of at least the same length. It preserves
relative row order within each child and returns the boundary in the
original span. The node's row IDs are updated after all routes are validated
against the expected child counts. When the selected feature saw no
missing values in training, the grower must route missing predictions to
the larger child, with ties going right.

The scan keeps counts, sums, constraints, and the running best candidate
together, with only one per-feature category list of at most 256 entries.
Numax `stats.cumsum` would materialize separate cumulative arrays and still
require the candidate loop. Numax `core.sorting.argsort` takes a numeric
`TensorLike` and adds host and tensor allocations. It has no category-code
tie key. The splitter sorts the small native record list with
`std.builtin.sort.sort`.

Run from the Moxie root in the locked toolchain.

```sh
pixi run ensemble-prepare-oracle
pixi run python families/ensemble/tools/check.py hgb-splitting
```

The check compiles native tests and a differential driver. It compares split
choices, categorical masks, Float64 gains, child statistics and values,
and row order against scikit-learn
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. The native checks compose the
splitter with mapped, strided data and sibling subtraction.
