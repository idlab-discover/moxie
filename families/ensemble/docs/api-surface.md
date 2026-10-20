# Public API implementation status

RandomForestClassifier/Regressor and ExtraTreesClassifier/Regressor are
implemented in `_forest`; see [forests.md](forests.md). The remaining public
names and callable parameter names/defaults follow the local scikit-learn
checkout as declarations. Their constructors and operations raise `NotImplementedError`
through Mojo `Error`; they do not perform numerical work. Existing implementations
and family traits remain intact. `BaseEnsemble` is implemented in
`moxie.ensemble._forest_engine` (see forest_engine.md), not declared here.

For the remaining declarations, Mojo adaptations are as follows. Heterogeneous inputs use `Some[Movable]`, accepting native
values without Python. Opaque configuration, learned attributes and deferred
results use the local `APIValue` marker. Defaults are preserved as native literals or equivalent native values; the exact
reference signature is recorded on every declaration. Python `**kwargs` becomes an optional named native value
parameter. `fit` and `set_params` mutate in place and return a mutable borrow of
the estimator, following Moxie's fluent convention.
Properties are exposed as opaque fields. Inherited public methods are flattened
onto each struct; dynamic metadata-routing methods are outside this source-only
surface. These declarations do not implement numerical behavior or Python reflection.

## Compile the API

From Moxie's root, run `pixi run ensemble-build`. This assembles the root
`DEPENDENCIES.json` and precompiles the complete public namespace, including
all ensemble declarations and concrete estimators. Sources belong to Moxie;
`EXPORTS.json` in this family is checked against the root manifest.
