> Historical declaration inventory. Working assembled exports and supported
> behavior are documented in [estimators.md](estimators.md); declaration coverage
> is not implementation coverage.

# Public API scaffold

Public names and callable parameter names/defaults follow the local scikit-learn
checkout. Newly declared constructors and operations raise `NotImplementedError`
through Mojo `Error`; they do not perform numerical work. Existing implementations
and family traits remain intact.

Mojo adaptations: heterogeneous inputs use `Some[Movable]`, accepting native
values without Python. Opaque configuration, learned attributes and deferred
results use the local `APIValue` marker. Defaults are preserved as native literals or equivalent native values; the exact
reference signature is recorded on every declaration. Python `**kwargs` becomes an optional named native value
parameter. Fitting and parameter updates mutate in place, without fluent returns.
Properties are exposed as opaque fields. Inherited public methods are flattened
onto each struct; dynamic metadata-routing methods are outside this source-only
surface. The historical inventory declarations implement neither numerical behavior nor
Python reflection. Current numerical implementations are documented separately.

## Compile the API

Use the existing Pixi environment and assembled dependency tree. Copy
`builds/deps` to ignored `builds/api-surface`, then overlay `src/moxie` onto
`builds/api-surface/moxie` so current family declarations take precedence.
From the family root, compile without running tests:

```sh
pixi run mojo precompile builds/api-surface/moxie -I builds/api-surface -o builds/api-surface.mojoc
```

Dependency manifests name live source paths; builds read current `../..` HEAD.

This historical inventory does not describe the live registry. The assembled
`VALID_METRICS` and tree `valid_metrics` are implemented; `VALID_METRICS_SPARSE`
is excluded. See [BallTree](balltree.md).
