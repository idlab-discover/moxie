# Rollup validation

Run from the Moxie root with its compatible locked Mojo 1.0.0 / MAX 26.5.0 environment:

```sh
pixi run neural-network-check
pixi run python families/neural_network/shared/mlp_engine/tools/check.py parity
```

The combined check precompiles the full assembled Moxie namespace and compiles
RBM/MLP helper tests and examples. Native tests cover family capabilities, finite
input and shape validation, strides, ownership, refit, clone, errors and incremental
eligibility. Two isolated fixture consumers verify shared sources and reject
missing files and conflicting ownership. The public probe also verifies that
both MLP declarations raise instead of producing models.

Numerical comparisons run original method bodies from local scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. They cover six RBM kernel workloads
with complete RNG state, six public RBM lifecycle workloads, three Float64
incremental workloads, 32 MLP initialization/packing cases and 40 MLP
forward/gradient fixtures. See [scope](scope.md) for tolerances and exclusions.

The isolated public consumer receives only copied assembled dependencies and
probe files, and must produce byte-identical lifecycle output. Ownership checks compare module and license bytes with current
core owners. The standalone source is not a runtime or build dependency.

The focused MLP check validates forward propagation, batch losses and gradients.
Estimator-level MLP solver and lifecycle validation remains open, as does
Float32 RBM incremental promotion.
