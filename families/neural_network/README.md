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
isolated public consumer. Its assembly lives in `families/neural_network/builds/deps`
and consumes the current root manifest without replacing the root build's inputs.
It does not establish full sklearn parity. Historical
child `parity` tasks still fail deliberately on unfinished full acceptance gates.

Side-by-side notebooks against the official gallery live in
`notebooks/neural_network/`: `plot_rbm_logistic_classification` (Float64
numerical parity of the RBM pipeline, Float32 distribution checks) and
`plot_mlp_alpha` (documents the raising MLP declarations). Run them with
`pixi run -e notebooks notebooks notebooks/neural_network`.

See [scope](docs/scope.md), [RBM API](algorithms/rbm/docs/api.md),
[MLP helpers](shared/mlp_engine/docs/api.md), and [provenance](provenance/rollup.json).
Tests, examples, internal engines and license notices live here. Core
`DEPENDENCIES.json` supplies ownership and current external sources. Compatible
child Pixi locks and the original scaffold baseline are retained.
