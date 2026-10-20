# Dense MLP engine scope

This project implements private parameter preparation helpers only (see api.md).
It is not yet a forward/backpropagation or optimization engine. There are no
package-level engine exports.
The required dense scope is Float32/Float64 layer initialization and packing,
forward/backpropagation, identity/logistic/tanh/relu hidden activations,
identity/exp/logistic/softmax outputs, squared-error/Poisson/categorical/binary
losses, weighted objectives, coefficient-only L2, SGD schedules and momentum,
Adam, and full-batch L-BFGS. Both MLP wrappers will consume this engine.

Sparse input/target storage, CSR/CSC conversions and sparse multiplication are
deferred by the session scope. Reference: sklearn neural_network
_multilayer_perceptron.py, _fit and _forward_pass; dense NDArray is the required
selection. There is no sparse-required constructor default for this private API.
The family's numerical exclusions and storage policy remain applicable.

Executed commands: `PIXI_CACHE_DIR="$PWD/.pixi/cache" pixi run build`,
`pixi run test`, and `pixi run parity` (with the same local cache setting).
Build and parameter contracts pass; parameter parity passes 32 oracle cases.
The overall parity task deliberately fails because forward/backpropagation and
optimizer comparisons are missing. Native tests reject unsupported hidden
activation, malformed packed rank/length, nonfinite values and Float32 overflow
before mutation; they cover both dtypes, strided tensors and independent copies.
Finite differences, solver tests and performance evidence remain unfinished.
