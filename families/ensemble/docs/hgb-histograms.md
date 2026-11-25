# Histogram gradient boosting node histograms

`moxie.ensemble._hist_gradient_boosting.histogram.compute_histograms` builds
per-feature bins for one node. Pass the UInt8 matrix from the
[`_BinMapper`](hgb-binning.md), its total bin count including the missing bin,
and a span of row indices into that matrix. Gradients and Hessians are borrowed
Float32 spans indexed by the same row IDs. Each bin stores a native Int count
and Float64 sums. Entries accumulate in the supplied partition order, including
for a full-size partition. Use ascending row IDs at the root to reproduce
scikit-learn's root accumulation. An empty Hessian span represents constant
Hessians and leaves their sums at zero. The splitter can use the count and the
loss's constant in that case.

Pass `allowed_features` to compute bins for a subset of original feature IDs.
An empty selection computes no features. The result borrows each feature's
bin list through `feature(index)`, which returns an empty list for excluded
features. Missing and categorical bin codes use the same accumulation as
numeric codes. The matrix can be column-major, row-major, or strided.

After a partition, build a histogram for the smaller child, then call
`parent^.subtract_sibling(smaller)` to get the other child's histogram.
Subtraction consumes the parent allocation and computes only the sibling's
available features. Build the sibling with every feature needed by the other
child, then apply each child's own feature selection while splitting.
The caller provides a genuine subset of parent rows from the same feature
matrix and loss iteration. The subtraction checks layout, feature presence
and per-bin counts. Floating-point cancellation follows the reference
subtraction.

The native serial loop borrows stride-aware NuMojo storage and fuses counts,
gradient sums and Hessian sums without gathering sample-sized buffers.
Numax's `stats.bincount` takes `TensorLike` plus optional tensor weights,
copies inputs to the host, and accumulates in the weight dtype. Composing it
here would require feature gathers, Float64 weight tensors, and independent
passes for counts and each sum. Numax's `stats.histogram` likewise returns one
count or weight sum per bin and adds bin-edge classification to already
encoded input. NuMojo supplies storage and stride handling.

Run from the Moxie root in the locked toolchain:

```sh
pixi run ensemble-prepare-oracle
pixi run python families/ensemble/tools/check.py hgb-histograms
```

The focused check compiles the native contracts and parity driver. It
compares counts and Float64 representations against `HistogramBuilder` in
source-built scikit-learn at `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
It covers root and nonroot partitions, constant and variable Hessians, feature
subsets, and parent-minus-sibling subtraction. Native contracts also exercise
mapper output, reversed views, partition order and invalid inputs.
