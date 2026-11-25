# Rollup validation

Run from the Moxie root with its compatible locked Mojo 1.0.0 / MAX 26.5.0 environment:

```sh
pixi run neural-network-check
pixi run python families/neural_network/tools/integration.py mlp
pixi run python families/neural_network/tools/integration.py mlp-incremental
pixi run python families/neural_network/shared/mlp_engine/tools/check.py parity
```

The combined check precompiles the full assembled Moxie namespace and compiles
RBM/MLP helper tests and examples. Native tests cover family capabilities, finite
input and shape validation, strides, ownership, refit, clone, errors and incremental
eligibility. Two isolated fixture consumers verify shared sources and reject
missing files and conflicting ownership. The `mlp` target compiles only the
public ordinary-fit MLP tests and their directly affected compositions.
The `mlp-incremental` target compiles the public incremental trait and error
contracts and tests 160 successive estimator snapshots, including updates
after ordinary fit. It checks the next draw of explicit RNG streams,
class domains and missing classes, and runs the public comparison below.

Numerical comparisons run original method bodies from local scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. They cover six RBM kernel workloads
with complete RNG state, six public RBM lifecycle workloads, three Float64
incremental workloads, 32 MLP initialization/packing cases, 40 MLP
forward/gradient fixtures, 76 L-BFGS fixtures and 200 stochastic training
snapshots, plus 72 public estimator fits and 160 incremental sequence snapshots.
The public comparison checks parameters, probabilities, predictions, weighted
scores, loss history, fitted metadata and warning text. It uses absolute and relative tolerances of 2e-5
for Float32 L-BFGS, 3e-6 for Float32 stochastic training and 2e-11 for Float64
to allow for rounding in the native reductions and accumulated solver steps.
See [scope](scope.md) for exclusions.

The isolated public consumer receives only copied assembled dependencies and
probe files, and must produce byte-identical lifecycle output. Ownership checks compare module and license bytes with current
core owners. The standalone source is not a runtime or build dependency.

The focused MLP check validates forward propagation, batch losses and gradients,
and the L-BFGS, SGD and Adam trainers. Public MLP validation-set early stopping,
warm-start refitting and Float32 RBM incremental promotion remain open.
