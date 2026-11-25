# Public neural-network API

Import `BernoulliRBM` from `moxie.neural_network` for the implemented dense subset.
Its [API](../algorithms/rbm/docs/api.md) records defaults, learned attributes,
RNG semantics and limitations. The default dtype is Float64; Float32 is a
compile-time specialization. Fit and typed set_params support fluent mutation.

`MLPClassifier` and `MLPRegressor` implement [dense ordinary supervised
fit](mlp.md) and support both feature dtypes, optional sample weights and
three solvers. SGD and Adam support incremental training through `partial_fit`.
`MLPParams` is their typed configuration record.

Core `DEPENDENCIES.json` selects the actual RBM implementation and all family
modules in one namespace. `pixi run neural-network-build` compiles that namespace;
`pixi run neural-network-check` validates the supported [scope](scope.md).

## Composition

BernoulliRBM is a `Pipeline`/`make_pipeline` transformer (fit through
`fit_transform`, predict through `transform`, as in sklearn), a `FeatureUnion`
branch (`bernoullirbm__bernoullirbm0` names) and supports `clone` with typed
nested `set_params`; see `algorithms/rbm/tests/test_composition.mojo`.
The MLPs compose through the supervised predictor traits.
`tests/test_mlp_composition.mojo` covers a StandardScaler classifier pipeline
and weighted regression cross-validation through the dense adapter.
