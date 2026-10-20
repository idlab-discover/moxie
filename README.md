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
Linear models are owned by Moxie under `src/moxie/linear_model`, including
LinearRegression, Ridge/RidgeClassifier, ElasticNet/Lasso and BayesianRidge.
See [supported scope and commands](families/linear_model/README.md).
Their existing external linalg and optimizer worktrees remain declared live
numerical dependencies; the standalone family checkout is not required.

Covariance uses the tested `../.worktrees/mojo-linalg-covariance` numerical checkout
and the integrated `families/model_selection` subset for graphical cross-validation.

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

Dense [discriminant analysis](docs/discriminant_analysis/README.md) is integrated.
Import `LinearDiscriminantAnalysis` and `QuadraticDiscriminantAnalysis` from
`moxie.discriminant_analysis`; the retired sibling checkout is not required.

[Isotonic regression](docs/isotonic/README.md) is owned by Moxie, with Float32/Float64 List inputs.
The retired isotonic sibling is not a build dependency.

[Random projection](docs/random_projection/README.md) is owned by Moxie, including Gaussian and sparse
projections with Float32/Float64 dense and CSR storage. The retired sibling
is not a build dependency.

[Naive Bayes](docs/naive_bayes/README.md) is owned by Moxie core, with all five dense
classifiers, List/NDArray inputs and typed labels. The retired sibling is not a
build dependency.

[Multiclass classifiers](docs/multiclass/README.md) are owned by Moxie core, including OvR, OvO,
and OutputCode with dense numeric/string labels and their documented optional
capabilities. The retired multiclass sibling is not a build dependency.

[Multioutput estimators](docs/multioutput/README.md) are owned by Moxie core, including independent
classification/regression and both chain types. The retired sibling is not a build
dependency.

[Calibration](docs/calibration/README.md) is owned by Moxie core, including all three calibration
methods, curves and native SVG displays. Its retired sibling is not a build
dependency.

[Gaussian processes](docs/gaussian_process/README.md) now live in `src/moxie/gaussian_process/`,
including dense regression and kernels. Classification remains unimplemented.
The retired sibling is not a build dependency; the dedicated GP linalg worktree
and external optimizer remain explicit numerical dependencies.

Dense decision trees and extra trees are integrated in `src/moxie/tree`, with
shared engines, prepared forest inputs and native text/DOT/SVG/Treelite exports.
See [tree documentation](docs/tree/README.md).

Dense feature selection is integrated under `moxie.feature_selection`. See the
[feature-selection guide](docs/feature_selection/README.md) for all 19 exports,
live dependencies and complete family validation commands.

[Cross decomposition](docs/cross_decomposition/README.md) is owned by Moxie core,
including PLSRegression, PLSCanonical, CCA and PLSSVD with dense Float32/Float64
inputs and current fluent mutation contracts. Its retired sibling is not a
build dependency.

[Mixture models](docs/mixture/README.md) are owned by Moxie core. GaussianMixture and
BayesianGaussianMixture share their EM lifecycle and Gaussian kernels.
The dedicated mixture linalg worktree remains an explicit numerical dependency.

[Model selection](families/model_selection/README.md) is owned by Moxie, with
splitters, typed search, CV, curves and native SVG displays.

[Neighbors](families/neighbors/README.md) is owned by Moxie, including its dense estimators, NCA,
query engine and spatial trees. Sparse graph APIs remain deferred.

[Imputation](families/impute/README.md) is owned by Moxie, including dense simple, integer,
string and KNN imputers and numeric/string indicators. IterativeImputer remains
deferred. The standalone checkout is not a build dependency.

Dense [ensemble estimators](families/ensemble/README.md) are integrated in
`moxie.ensemble`. Run `pixi run ensemble-test` and `pixi run ensemble-parity`
for their focused checks.

[Decomposition](families/decomposition/README.md) is owned by Moxie, including its SVD and threshold
sparse-coding engines. Dictionary learning, SparsePCA and regression coding
modes remain unfinished. The standalone checkout is not a build dependency.

Dense [SVM estimators](families/svm/README.md) and their native LIBSVM/LIBLINEAR
engines are integrated. Run `pixi run svm-test` and `pixi run svm-parity` for
focused validation.

[Manifold](families/manifold/README.md) is owned by Moxie, including ClassicalMDS,
trustworthiness and dense exchange helpers. Other manifold algorithms remain
explicit unsupported declarations. The standalone checkout is not a build dependency.

[Feature extraction](families/feature_extraction/README.md) is owned by Moxie,
including dense dictionary vectorization, image patches and dense graphs.
Run `pixi run feature-extraction-test` and `pixi run feature-extraction-parity`.
Text, TF-IDF, hashing and sparse output remain deferred; complex and extended
dictionary dtypes remain unfinished.
