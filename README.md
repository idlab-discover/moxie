# moxie

Pure Mojo estimator contracts and dense scikit-learn-style baseline estimators.
`DummyRegressor` implements mean, median, quantile and constant strategies;
`DummyClassifier` implements prior, most_frequent, uniform, stratified and
constant strategies. Both support sample weights, cloning and experimental
native model persistence.

```sh
pixi install --locked
pixi run build
pixi run test
pixi run parity
pixi run start
```

`start` runs the [Dummy example](examples/dummy.mojo). `scaffold-start` runs the
original contract demonstration. The locked environment uses Mojo 1.0.0 and
MAX 26.5.0. Python and scikit-learn are validation/build tools only.

## Imports and dependencies

```mojo
from moxie.dummy import DummyClassifier, DummyRegressor
from moxie.dummy_serialization import DummyClassifierSchema, DummyRegressorSchema
from moxie.serialization import save_model, load_model
```

The estimator and schema types are also exported from `moxie`. Implementation
sources live in `src/moxie`; shared random state stays in `shared/random`.
Builds assemble one namespace with the current local sources declared in
`DEPENDENCIES.json`, including NuMojo, mojo-linalg (used by NuMojo), and the
optional Snappy codec. Keep those checkouts at the declared paths. Missing
sources and module ownership conflicts fail explicitly. The `moxie-dummy`
checkout is not required to build or use Moxie.

After `pixi run build`, compile a consumer from this repository root:

```sh
pixi run mojo build -I builds/deps path/to/consumer.mojo -o builds/consumer
builds/consumer
```

Generated source, license assets and a receipt of the consumed source bytes
are in `builds/deps`. Do not edit that generated directory.

## Shared owners

Children consume current local sources; see [development policy](docs/family-development.md).

| Component | Contract |
| --- | --- |
| Namespace assembly | [assembly](shared/assembly/docs/assembly.md) |
| Dense storage/reductions | [numerics](shared/numerics/docs/numerics.md) |
| RNG/normalization | [random](shared/random/docs/random.md) |
| Classification labels/groups | [labels](shared/labels/docs/labels.md) |
| Covariance solves | [linalg](shared/linalg/docs/linalg.md) |
| Incomplete gamma | [special](shared/special/docs/special.md) |
| Nullable categorical features | [categorical](docs/categorical.md) |
| Model persistence | [serialization](docs/serialization.md) |

## Scope and validation

- [Dummy native API](docs/dummy/api.md) and [dense scope](docs/dummy/scope.md)
- [Categorical feature columns](docs/categorical.md)
- [Model serialization](docs/serialization.md)
- [Estimator interfaces and differences from Python](docs/interfaces.md)
- [Family repositories](docs/families.md) and [contribution rules](AGENTS.md)

`test` builds and runs shared contracts, container/codec checks and the migrated
Dummy estimator, dtype and persistence tests. `parity` compares Dummy with the
local `../scikit-learn` checkout at revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Sparse data, Python bindings and
complete scikit-learn compatibility remain outside the implemented scope.

Children consume current Moxie HEAD; see [development policy](docs/family-development.md).
`scaffold-v1.1` remains a historical provenance tag. Build outputs, research notes and raw validation evidence
stay in ignored local directories.
