# Family repositories

Sibling `moxie-<module>` repositories consume current Moxie HEAD under the
[family policy](family-development.md). `scaffold-v1.1` records their origin.
Public imports stay under `moxie.<module>`, matching the scikit-learn module
structure. Family APIs live in their own current docs. Dense Dummy
estimators, Compose, Pipeline, Frozen, KernelRidge, dense Covariance, and dense Preprocessing have since been integrated into this
repository; see the [Dummy API](dummy/api.md), [Compose scope](compose/scope.md),
[Pipeline API](pipeline/sequential.md),
[Preprocessing API](../families/preprocessing/docs/api.md),
[Frozen API](frozen/api.md), and [KernelRidge API](../families/kernel_ridge/docs/family.md). Their original sibling
checkouts are not build dependencies. Other families continue to develop
independently. Each repository contains its own Pixi environment, Git history, family contracts,
compile checks, and API guide. `SCAFFOLD_BASE` records its exact foundation.

| Repository | Reference module |
| --- | --- |
| `moxie-linear_model` | `sklearn/linear_model` |
| `moxie-svm` | `sklearn/svm` |
| `moxie-neighbors` | `sklearn/neighbors` |
| `moxie-tree` | `sklearn/tree` |
| `moxie-ensemble` | `sklearn/ensemble` |
| `moxie-naive_bayes` | `sklearn/naive_bayes.py` |
| Integrated: `src/moxie/discriminant_analysis.mojo` | `sklearn/discriminant_analysis.py` |
| `moxie-gaussian_process` | `sklearn/gaussian_process` |
| `moxie-neural_network` | `sklearn/neural_network` |
| Integrated: `families/kernel_ridge` | `sklearn/kernel_ridge.py` |
| `moxie-dummy` | `sklearn/dummy.py` |
| `moxie-semi_supervised` | `sklearn/semi_supervised` |
| Integrated: `src/moxie/multiclass.mojo` | `sklearn/multiclass.py` |
| Integrated: `src/moxie/multioutput.mojo` | `sklearn/multioutput.py` |
| Integrated: `src/moxie/calibration.mojo` | `sklearn/calibration.py` |
| `moxie-cluster` | `sklearn/cluster` |
| `moxie-mixture` | `sklearn/mixture` |
| Integrated: `families/covariance` | `sklearn/covariance` |
| `moxie-decomposition` | `sklearn/decomposition` |
| `moxie-cross_decomposition` | `sklearn/cross_decomposition` |
| `moxie-manifold` | `sklearn/manifold` |
| `moxie-preprocessing` | `sklearn/preprocessing` |
| `moxie-impute` | `sklearn/impute` |
| `moxie-feature_selection` | `sklearn/feature_selection` |
| `moxie-feature_extraction` | `sklearn/feature_extraction` |
| Integrated: `src/moxie/kernel_approximation.mojo` | `sklearn/kernel_approximation.py` |
| Integrated: `src/moxie/random_projection.mojo` | `sklearn/random_projection.py` |
| Integrated: `src/moxie/isotonic.mojo` | `sklearn/isotonic.py` |
| `moxie-pipeline` | `sklearn/pipeline.py` |
| `moxie-compose` | `sklearn/compose` |
| `moxie-model_selection` | `sklearn/model_selection` |
| `moxie-frozen` | `sklearn/frozen` |

## Porting scope

The table covers estimator-bearing families, including unsupervised learning,
feature preparation and estimator composition. `metrics`, `datasets`,
`inspection`, `utils`, `exceptions`, `callback`, configuration, and experimental
enablement are shared/supporting work, not duplicate algorithm-family repos.
Third-party `externals` and build internals are reference infrastructure rather
than Mojo algorithms to translate literally. Current shared owners are indexed in
[README](../README.md); check their contracts before requesting new helpers.

Derive concrete algorithms inside the corresponding family, keeping the
reference class names and module locations. Family traits express shared
capabilities, not a frozen list of every constructor argument. Before starting
an algorithm, inspect its reference constructors, inherited methods, tests and
math documentation; add its exact typed contract and parity tests first.

## Local comparison sources

Family repositories keep a revision record and links to the local scikit-learn
source and user guide under `docs/private/reference/`. Specialized checkouts
belong under `docs/private/<algorithm>/sources/` with a URL, revision and license
record. These are local research inputs, never runtime dependencies or tracked
vendor copies. Deriving a repository with Git does not copy ignored resources;
prepare them again at the new algorithm location.

See [Covariance](../families/covariance/README.md) for its API and remaining numerical differences.

[Kernel approximation](kernel_approximation/README.md) is owned by Moxie.

[Discriminant analysis](discriminant_analysis/README.md) is owned by Moxie.
Its retired sibling is not a build dependency.

[Isotonic regression](isotonic/README.md) is owned by Moxie, with Float32/Float64 List inputs.
The retired isotonic sibling is not a build dependency.

[Random projection](random_projection/README.md) is owned by Moxie, including Gaussian and sparse
projections with Float32/Float64 dense and CSR storage. The retired sibling
is not a build dependency.

[Multioutput estimators](multioutput/README.md) are owned by Moxie core, including independent
classification/regression and both chain types. The retired sibling is not a build
dependency.

[Calibration](calibration/README.md) is owned by Moxie core, including all three calibration
methods, curves and native SVG displays. Its retired sibling is not a build
dependency.
