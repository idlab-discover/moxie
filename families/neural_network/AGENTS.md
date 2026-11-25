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

Validate the affected estimator and its direct compositions before committing.
Use the focused commands in [validation](docs/validation.md).
See [scope](docs/scope.md) for current coverage.
Historical family guidance, including the changes present before integration,
is retained in [provenance](provenance/source-AGENTS.md).
