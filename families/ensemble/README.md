# Ensemble

Moxie owns `moxie.ensemble` in `src/moxie/ensemble`. This directory contains
its tests, examples, documentation, license assets and focused validation tools.
The standalone ensemble checkout is no longer a build dependency.

Implemented: RandomForestClassifier, RandomForestRegressor,
ExtraTreesClassifier, ExtraTreesRegressor, BaseEnsemble and the shared dense
forest/sampling engine. Other estimator declarations raise explicitly.
See [forests](docs/forests.md), [scope](docs/scope.md), and
[API status](docs/api-surface.md) for supported inputs and limitations.

Run from the Moxie root, using its locked toolchain:

```sh
pixi run ensemble-build
pixi run ensemble-test
pixi run ensemble-start
pixi run ensemble-prepare-oracle
pixi run ensemble-parity
```

The tests assemble the full public namespace and two isolated consumers using
Moxie's current dependency manifest. They check module ownership, shared arrays,
clone lifecycle, missing-source failures and export/module collisions.
The oracle is built from local scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b` into this family's ignored builds directory.
Sampling, engine and public-estimator comparisons retain their existing exact
comparisons and documented floating-point tolerances. They do not establish
full sklearn parity.

Examples are in [examples](examples). After building, run
`families/ensemble/builds/forest_engine_example` or `plumbing_example` there.
[Provenance](PROVENANCE.json) records the original source and transferred modules.
The original compatible Pixi manifest and lock are retained in `toolchain/` as
provenance; Moxie's root Pixi environment runs all current commands.
