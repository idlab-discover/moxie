# Semi-supervised owner

This directory owns the dense semi-supervised estimators, affinity and probability
engines, adapters, tests and examples. Follow [Moxie rules](../../AGENTS.md).
Before changing estimator behavior, read [the API guide](docs/api-surface.md),
[scope](docs/scope.md) and [adapter contracts](docs/plumbing.md).

Use the current Moxie GaussianNB and shared numerical owners. Preserve the
recorded sklearn oracle revision and the compatible Pixi lock. Sparse input,
KNN graphs and Python metadata routing remain deferred.

Run `pixi run semi-supervised-test`, `pixi run semi-supervised-parity` and
`pixi run semi-supervised-independence` from the Moxie root after changes.
