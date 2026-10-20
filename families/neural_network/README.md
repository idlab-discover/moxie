# Neural networks in Moxie

`moxie.neural_network` exports the implemented dense `BernoulliRBM` subset,
`MLPClassifierAPI`, `MLPRegressorAPI` and `RBMTransformer` traits. `MLPClassifier`
and `MLPRegressor` preserve their public declarations but raise
`NotImplementedError`; neither trains a model. Private `_mlp_engine` supplies
initialization and parameter packing, not forward/backpropagation or optimization.

From the Moxie root:

```sh
pixi run neural-network-check
pixi run neural-network-build
pixi run neural-network-test
pixi run neural-network-parity
pixi run neural-network-independence
```

The combined check compiles the assembled public namespace, runs native contracts,
compares the supported numerical subset against local sklearn, and executes an
isolated public consumer. It does not establish full sklearn parity. Historical
child `parity` tasks still fail deliberately on unfinished full acceptance gates.

See [scope](docs/scope.md), [RBM API](algorithms/rbm/docs/api.md),
[MLP helpers](shared/mlp_engine/docs/api.md), and [provenance](provenance/rollup.json).
Tests, examples, internal engines and license notices live here. Core
`DEPENDENCIES.json` supplies ownership and current external sources. Compatible
child Pixi locks and the original scaffold baseline are retained.
