# moxie

Pure Mojo estimator contracts and dense scikit-learn-style baseline estimators.
`DummyRegressor` implements mean, median, quantile and constant strategies;
`DummyClassifier` implements prior, most_frequent, uniform, stratified and
constant strategies. Both support sample weights, cloning and experimental
native model persistence. Dense `ColumnTransformer`, `make_column_transformer`,
`make_column_selector`, and `TransformedTargetRegressor` are also implemented.
Typed supervised `Pipeline` and dense `FeatureUnion` are integrated; see the
[Pipeline API](docs/pipeline/sequential.md) and [FeatureUnion API](docs/pipeline/feature-union.md).
Their `make_pipeline` and `make_union` factories remain explicit scaffolds.
All 29 preprocessing public names have dense implementations; see the
[Preprocessing API](families/preprocessing/docs/api.md) for native scope and limitations.

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

`FrozenEstimator` preserves fitted state and conditionally forwards supported
methods; see the [Frozen API and coverage](docs/frozen/api.md).

KernelRidge is integrated with dense named, precomputed and native callable kernels,
weighted multioutput fits and native persistence. See [KernelRidge](families/kernel_ridge/README.md).
Its contracts and reference comparisons run in the root test and parity tasks.

Dense covariance is integrated with all 16 public exports. See [Covariance](families/covariance/README.md)
for APIs, validation commands and numerical limitations.

## Imports and dependencies

```mojo
from moxie.dummy import DummyClassifier, DummyRegressor
from moxie.dummy_serialization import DummyClassifierSchema, DummyRegressorSchema
from moxie.serialization import save_model, load_model
from moxie.compose import ColumnTransformer, TransformedTargetRegressor
from moxie.pipeline import Pipeline, FeatureUnion
from moxie.preprocessing import StandardScaler, OneHotEncoder
from moxie.frozen import FrozenEstimator
from moxie.kernel_ridge import KernelRidge
from moxie.covariance import EmpiricalCovariance, MinCovDet, GraphicalLassoCV
```

The Dummy estimator and schema types are also exported from `moxie`. Implementation
sources live in `src/moxie`; shared random state stays in `shared/random`.
Builds assemble one namespace with the current local sources declared in
`DEPENDENCIES.json`, including NuMojo, mojo-linalg (used by NuMojo), and the
optional Snappy codec. Keep those checkouts at the declared paths. Missing
sources and module ownership conflicts fail explicitly. The `moxie-dummy`,
`moxie-compose`, `moxie-pipeline`, `moxie-frozen`, and `moxie-preprocessing`
checkouts, as well as `moxie-kernel_ridge` and `moxie-covariance`, are not required to build or use Moxie.
Integrated composition tests consume the LinearRegression subset of
`../moxie-linear_model`, and its existing NNLS dependency at
`../.worktrees/mojo-optimize-linear_model`. These remain separately owned live
source dependencies; this integration does not import their entire families.

Covariance uses the tested `../.worktrees/mojo-linalg-covariance` numerical checkout
and the declared `../moxie-model_selection` subset for graphical cross-validation.

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

- [Preprocessing API and validation](families/preprocessing/README.md)
- [Covariance API and numerical limits](families/covariance/README.md)
- [Dummy native API](docs/dummy/api.md) and [dense scope](docs/dummy/scope.md)
- [Compose API, scope and validation](docs/compose/scope.md)
- [Categorical feature columns](docs/categorical.md)
- [Model serialization](docs/serialization.md)
- [Estimator interfaces and differences from Python](docs/interfaces.md)
- [Family repositories](docs/families.md) and [contribution rules](AGENTS.md)

`test` builds and runs shared contracts, container/codec checks and the migrated
Dummy estimator, dtype and persistence tests, plus compose contracts and real
Pipeline/Dummy and transformed-target consumers, plus preprocessing contracts,
API signatures and isolated consumers. Frozen ownership, forwarding, and nine
negative compilation checks run in `test`; 61 Frozen reference comparisons run
in `parity`. All nine preprocessing parity groups
run as part of `parity`. Family engines, tests and standalone tooling live in
`families/preprocessing`; public modules live in `src/moxie/preprocessing`.
Parity checks use the local `../scikit-learn` checkout at revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Sparse data, Python bindings and
complete scikit-learn compatibility remain outside the implemented scope.

Children consume current Moxie HEAD; see [development policy](docs/family-development.md).
`scaffold-v1.1` remains a historical provenance tag. Build outputs, research notes and raw validation evidence
stay in ignored local directories.

Dense [kernel approximation](docs/kernel_approximation/README.md) is integrated,
including RBFSampler, SkewedChi2Sampler, AdditiveChi2Sampler, PolynomialCountSketch
and Nystroem. The retired `moxie-kernel_approximation` checkout is not required.
