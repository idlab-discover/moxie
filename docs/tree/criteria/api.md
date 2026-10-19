# Weighted criteria (implementation under validation)

Import `Gini`, `Entropy`, `MSE`, `MAE`, or `Poisson` from
`moxie.tree._criterion`. These private engines implement the family's exact
`Criterion` trait and accept its `CriterionConfig(n_outputs, n_classes,
n_samples)`. They are not estimators and export no names through `moxie.tree`.

Names selected by estimator owners are: `gini` → Gini, `entropy`/`log_loss` →
Entropy, `squared_error` → MSE, `absolute_error` → MAE, `poisson` → Poisson.
The pinned estimator maps deprecated `friedman_mse` to MSE; there is no separate
Friedman criterion in this revision. Warnings and string dispatch belong to
estimator wrappers.

`initialize(data, samples, start, end, weighted_n_samples)` binds explicit
reference-counted, read-only views of y, weights and samples. Unit weights are
represented by `None`. Float64 targets have contiguous output axes; sample and
weight vectors may be strided. Mutate the permutation only between reset cycles,
and preserve the node's sample multiset. Keep targets/weights fixed throughout
an initialized node. Each new build needs a separate criterion instance.

`reset` empties the left child; `update(new_pos)` moves forward; `reverse_reset`
empties the right child. The MAE reverse operation raises, matching the pinned
source. Queries on uninitialized state, invalid dimensions/encoded labels,
nonfinite targets/weights, empty nodes and nonpositive node weights raise Mojo
`Error`. Empty/nonpositive child impurity queries raise rather than perform
Cython's unchecked zero division. The splitter should reject those candidates.

The destination must not overlap retained targets, weights or sample indices.
`node_value` writes a caller-owned, contiguous Float64 matrix with shape
`(n_outputs, max_n_classes)`. Unused class slots are zero. Values are weighted
class proportions, means (MSE/Poisson), or medians (MAE). Other methods expose
node/child impurity, exact and proxy improvement, midpoint values, clipping and
monotonic checks. Bounds apply only to single-output regression or binary
classification; classification checks use class-zero probability, so estimator
or splitter code must reverse the user's positive-class sign as specified by
the family contract.

Missing features are handled through sample ordering by the partitioner; these
criteria never inspect X values. There is no missing-target mode. Categorical
ordering/routing belongs to splitters, and unsupported estimator/criterion
combinations must be checked there or in estimator wrappers.

MAE precomputes prefix/suffix absolute errors and medians on every reset,
following the reference three-way sort and weighted Fenwick sweep. Scratch is
O(n_samples); preprocessing is O(n_outputs * n_node_samples * log(n_node_samples)).
Child impurity queries then read the precomputed errors. Reset must follow any
change to the shared sample permutation, including random partitioning. These
are algorithmic bounds, not measured performance claims. Finite signed MAE
weights use the pinned Fenwick draw/search order. Nonpositive-weight prefixes
and suffixes skip median search because the splitter cannot accept them; this
avoids reference out-of-bounds reads without changing valid child queries.

The pinned sklearn private MAE `node_impurity` reads stale/uninitialized slot
zero for nonzero `start`. This implementation returns the actual node impurity.
The differential oracle obtains that quantity from the original
`children_impurity` at reset (the whole node is the right child); the unchanged
bug reproducer separately verifies the defect. Both public builders carry child
impurities from their parent's split and query node impurity only at the root,
so this deliberate private correction does not alter public estimator parity.

Build: `pixi run tree-build`; native contracts: `pixi run tree-test`; differential check:
`pixi run tree-parity`. Run the compiled example with `builds/tree/criteria/weighted_split` after
building. Validation uses local Cython source bodies; no Python participates in
native criterion calculations.
