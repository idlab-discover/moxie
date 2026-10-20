# Neural-network ownership

Follow [Moxie contributor rules](../../AGENTS.md) and
[family development](../../docs/family-development.md). This directory owns the
neural-network family, its RBM implementation and private MLP parameter helpers.
The standalone repository is historical provenance, not a dependency.

Before changing estimator behavior, read [scope](docs/scope.md),
[RBM API](algorithms/rbm/docs/api.md), and the current
[shared interfaces](../../docs/interfaces.md). Before numerical changes, read
[the numerical policy](../../shared/numerics/docs/numerics.md).
Use the live core dense storage and RNG owners. Python is validation tooling only.

Run `pixi run neural-network-check` from the Moxie root before committing.
This checks the supported subset; full MLP training and Float32 RBM incremental
promotion remain unfinished. Preserve those explicit exclusions.
Historical family guidance, including the changes present before integration,
is retained in [provenance](provenance/source-AGENTS.md).
