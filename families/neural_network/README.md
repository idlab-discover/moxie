# Neural networks in Moxie

`moxie.neural_network` exports dense `BernoulliRBM`, `MLPClassifier` and
`MLPRegressor`. The MLP estimators support ordinary dense Float32 and Float64
training with L-BFGS, SGD and Adam and incremental training with SGD and Adam.
See the [MLP API](docs/mlp.md) for fitted state, constructor options and
composition examples.

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
The [focused MLP parity check](shared/mlp_engine/docs/scope.md) validates the
private engine in isolation.

Side-by-side notebooks against the official gallery live in
`notebooks/neural_network/`: `plot_rbm_logistic_classification` (Float64
numerical parity of the RBM pipeline, Float32 distribution checks) and
`plot_mlp_alpha` (bounded dense MLP parity and full-budget diagnostics).
The [MLP training-curves notebook](../../notebooks/neural_network/plot_mlp_training_curves.ipynb)
compares full loss curves, stopping epochs and predictions for the seven
gallery strategies on Iris, Digits, Circles and Moons.

```sh
pixi run -e notebooks notebooks notebooks/neural_network/plot_mlp_training_curves.ipynb
```

See [scope](docs/scope.md), [RBM API](algorithms/rbm/docs/api.md),
[MLP engine](shared/mlp_engine/docs/api.md), and [provenance](provenance/rollup.json).
Tests, examples, internal engines and license notices live here. Core
`DEPENDENCIES.json` supplies ownership and current external sources. Compatible
child Pixi locks and the original scaffold baseline are retained.
