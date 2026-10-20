# Public neural-network API

Import `BernoulliRBM` from `moxie.neural_network` for the implemented dense subset.
Its [API](../algorithms/rbm/docs/api.md) records defaults, learned attributes,
RNG semantics and limitations. The default dtype is Float64; Float32 is a
compile-time specialization. Fit and typed set_params support fluent mutation.

`MLPClassifier` and `MLPRegressor` remain raising declarations with the recorded
sklearn constructor signatures. They do not train or predict. Family capability
traits remain exported. Private initialization/packing helpers are documented
[here](../shared/mlp_engine/docs/api.md).

Core `DEPENDENCIES.json` selects the actual RBM implementation and all family
modules in one namespace. `pixi run neural-network-build` compiles that namespace;
`pixi run neural-network-check` validates the supported [scope](scope.md).
