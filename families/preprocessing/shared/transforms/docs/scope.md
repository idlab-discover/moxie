# Dense-only scope

This project owns the private `moxie.preprocessing._dense_transforms` engine.
It exports no public estimators. Supported modes: Float32/64 dense rank-2 affine
forward/inverse, clipping, thresholding, L1/L2/max normalization on axes 0 and 1;
logical copying, explicit in-place, caller-output, borrowed sink tiles and
multipass feature-normalization state. Float64 affine parameters with Float32
input preserve stage rounding. Positive/negative strides and offsets are read
without materialization. Writability, overlap, shape and option errors are tested.
See [transforms.md](transforms.md) for native API adaptations and constraints.

Deferred: all CSR/CSC/COO inputs, sparse normalization, sparse clipping/scaling,
and sparse binarization branches in sklearn `preprocessing/_data.py::normalize`
and `binarize`. No implicit densification and no sparse backend. The inherited
family requires explicit dense output selections for future encoders; this engine
has no sparse default setting or public estimator constructor to override.
Python object adapters, metadata routing, reflection, serialization and public
estimator validation/lifecycle belong to consumers or bindings. Norm kernels do
not implement weighted moments; the statistics owner handles weights/NaN policy.

Required commands: `pixi run build`, `pixi run test`, `pixi run parity`, and
`pixi run memory`. Native tests cover unsupported rank/options/readonly storage,
shape mismatch, overlap and malformed parameters. Sparse input is unavailable
through the accepted NDArray type. The two isolated consumers verify the same
declared module owner and consumed source hash while using different operations/dtypes. Parity
executes AST-extracted functions from the pinned local sklearn dense paths,
with validation plumbing isolated as tooling, and staged NumPy affine formulas.
Numerical and memory evidence applies to this private dense engine only.
