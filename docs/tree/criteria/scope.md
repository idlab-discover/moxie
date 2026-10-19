# Dense-only criteria scope

Implemented, with validation still incomplete: private Gini, entropy/log_loss,
squared-error, absolute-error and Poisson criteria on dense Float64 targets and
weights, with Float32 features and Int64 sample permutations from the family.
Weighted updates, multioutput statistics, values, improvement and monotonic
checks are native Mojo. MAE uses reference-order three-way sorting and Fenwick
prefix/suffix precomputation, refreshed after sample permutation changes.
Signed MAE weights and missing-sample permutations are supported. No estimator or fit/predict wrapper is supplied here.

Sparse branches in this criterion module: none. Inherited tree exclusions are
CSR/CSC/COO inputs, sparse partitioners and sparse decision_path outputs;
see ../../docs/scope.md and the pinned sklearn `_partitioner.pxd`, `_tree.pxd`
and `_classes.py`. Sparse data is not densified. No default criterion requires
a sparse setting, and dense callers need no special constructor selection.

Build and native contracts pass. The differential suite passes all **1,028**
comparisons across five criterion
families, one to three outputs, varying/zero/signed weights, permutations and
node offsets. MAE node impurity is checked through the reference whole-right-child
query at reset; see api.md for the deliberate correction of its private stale-slot
bug. `tests/tree/criteria/repro_mae_offset.py` preserves a separate regression of the unchanged
oracle defect. End-to-end MAE checks live in the root `tests/tree/mae_parity.py`.

Native negative checks cover uninitialized access, invalid update position,
reversed bounds, malformed sample indices and MAE reverse_reset. Sparse APIs
are absent at the typed boundary. Retained-view lifetime, strided indices, independent instances and missing
feature permutations also pass native checks. Strided targets/weights, broader
invalid/extreme inputs and consumer integration still need coverage;
performance baselines have not been measured. This document declares scope,
not completion or full scikit-learn parity.
