# Cluster development

Cluster is owned by Moxie. Follow [Moxie rules](../../AGENTS.md) and
[family development](../../docs/family-development.md). This directory owns
its algorithms, centroid engine, contracts, tests, examples and license assets.
Read [scope](docs/scope.md), [interfaces](../../docs/interfaces.md) and
[plumbing](docs/plumbing.md) before changing an estimator or engine.

Use the local scikit-learn checkout at the revision recorded in the family docs.
Preserve dense Float32/Float64 behavior, explicit unsupported capabilities and
pure Mojo runtime execution. Neighbor queries belong to `../neighbors`;
global numerical and RNG contracts belong to Moxie's shared owners.

From Moxie, run `pixi run cluster-test` and `pixi run cluster-parity`.
The family runner builds the complete Moxie namespace and two isolated consumers.
For centroid kernel changes also run the engine's tools/check.py test and parity
through Moxie's Pixi environment. Keep `--fp-mode=contract=off`.
Preserve toolchain locks, source provenance and third-party notices.
