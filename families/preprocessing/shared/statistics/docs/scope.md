# Accepted dense scope

This project implements private `moxie.preprocessing._moments` version 1:
Float32/64 logical-batch moments with Float64 accumulation, optional variance,
unweighted/vector/scalar weights, count projections, constant detection,
per-feature minimum/maximum/max-absolute statistics, and exact robust quantiles.
It implements no public estimator or transform. Tested layouts include contiguous,
offset, positive/negative strided inputs/weights and tiny views of large buffers.

CSR/CSC/COO inputs, sparse reductions, sparse weight containers, sparse output,
and densification adapters are deferred by `session-plan/SPARSITY-SCOPE.md`.
Reference sparse branches are in `sklearn/preprocessing/_data.py` and
`sklearn/utils/sparsefuncs.py`; none enter this module. All entry points require
native dense NDArrays by type. The inherited `require_dense_output(True)` check
raises. No sparse default needs overriding in this engine; a future estimator
with sparse defaults must document its explicit dense selection.

The numerical dependency accepts Float32/64 storage, not a generic
integer/sparse array ABI. Integer unweighted counts are projected elementwise
as Int64 without creating an unsupported integer array. The foundation and family
contracts follow their module ownership. Categorical vocabularies and public scaler
algorithms remain their separate owners' responsibilities.

Executed acceptance commands: `pixi run build`, `pixi run test`,
`pixi run parity`, `pixi run memory`. Tests cover dense-only type signatures,
sparse-output rejection, state validation/independence, numerical branches,
live-source assembly and both isolated consumers. Memory checks cover borrowed
fitting and exact column quantiles; estimator transform/output APIs are not
implemented here and need their own downstream tests.

The separate input-precision contract additionally covers dense Float32/64
NaN-aware mean and population stddev for both axes, signed/gapped/column-major
views, empty and all-missing reductions, IEEE overflow/underflow, and repeated
mean correction passes required by functional scale. It returns one scalar
record per requested row/column without heap allocation. Sparse settings and
public scaler implementations remain excluded as above. Original finite-input
validation belongs to the estimator; correction reductions accept generated Inf.

## Known unweighted overflow limitation

The current logical-batch unweighted sum uses serial accumulation. For the
finite Float64 column `[1e308, 0, 1e308, -1e308, -1e308, 0, 0, 0]`, it produces
an infinite mean while the reference produces zero. With variance enabled, the
native variance is NaN while the reference variance is infinite. The bounded
parity suite does not certify this case. The scaler consumer's
`../../algorithms/scalers/tests/unweighted_reduction.py` records the diagnostic;
public scaler acceptance must remain failing until this is corrected.
