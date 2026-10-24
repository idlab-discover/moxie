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

## Composition

BernoulliRBM is a `Pipeline`/`make_pipeline` transformer (fit through
`fit_transform`, predict through `transform`, as in sklearn), a `FeatureUnion`
branch (`bernoullirbm__bernoullirbm0` names) and supports `clone` with typed
nested `set_params`; see `algorithms/rbm/tests/test_composition.mojo`.
Cross-validation and search over such pipelines need a typed
model_selection adapter that does not ship yet (model_selection tracking
PR #3 part 1). The MLP declarations implement no estimator trait and cannot be
composed.
