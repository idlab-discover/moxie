# Common family requirements

Track unresolved needs shared across families. This file does not restrict
family edits or define implemented support; follow [family development](docs/family-development.md)
and the current owner contracts linked below. Check those contracts before
adding a request, merge overlapping requests, and update or remove resolved ones.
Git history retains completed requests and their original evidence.

Each entry records the requesting families, the missing contract, and its shared
owner or reason for common ownership. Link current source or reproducible checks;
keep raw research and measurements in ignored directories. A request alone does
not expand an estimator's authorized scope.

## Open requests

| Families | Missing contract | Ownership and current evidence |
| --- | --- | --- |
| naive_bayes, other classifiers | A common label-domain trait with associated target type, explicit-class construction, classification validation, encode/decode, class count/equality and accuracy comparison. | [Shared labels](shared/labels/docs/labels.md) supplies encodings but no common classifier-domain trait. The current family adapters are `ClassDomain`, `NumericDomain` and `StringDomain` in [the naive Bayes implementation](src/moxie/_naive_bayes_dense.mojo). |
| naive_bayes | Boolean targets in shared `encode_column` and `type_of_target`, preserving `[False, True]` classes. | [Shared labels](shared/labels/docs/labels.md) currently accepts Int64, Float32, Float64 and String targets. Boolean categorical feature storage does not supply this target contract. |
| naive_bayes, numerical consumers | A shared accuracy policy and validation for scalar Float64 exp/log. | [Naive Bayes](docs/naive_bayes/family.md) and [special functions](shared/special/docs/special.md) use Numax `core.libm`; the [shared numerical policy](shared/numerics/docs/numerics.md) does not yet define scalar exp/log accuracy. Preserve the family numerical evidence when defining the common contract. |
| compose, dummy, pipeline, model_selection | Typed per-method metadata requests and routing, including sample_weight, aliases, request/reject/unset states, introspection and transform_input semantics. | Moxie owns request semantics; each composition owner implements forwarding. [Compose scope](docs/compose/scope.md) and [Pipeline](docs/pipeline/sequential.md) retain this gap. Direct weighted estimator methods do not establish routing support. |
| kernel_ridge, model_selection | A shared negative mean squared error scorer for learning curves. | [Model selection](families/model_selection/docs/plumbing.md) provides `EstimatorScore` for estimator-default scoring. The [KernelRidge gallery](notebooks/kernel_ridge/plot_kernel_ridge_regression.ipynb) uses a local `TypedScorer` for negative MSE and needs to extract the Float64 arm of KernelRidge's prediction variant. |
| compose, text consumers | Native Python-compatible regex search with explicit syntax errors for `make_column_selector(pattern=)`. | Reusable parsing/matching belongs in shared infrastructure. [Compose](docs/compose/scope.md) currently rejects regex selection; substring matching is not equivalent. |
| compose, preprocessing, impute, feature_extraction | Heterogeneous selection and composition for raw integer, Boolean and string features, with typed inputs at each consumer. | [ColumnTable](docs/compose/named-columns.md) routes string, Boolean, Int64, Float32 and Float64 categorical columns through the [shared categorical contract](docs/categorical.md). The raw numeric block still has one floating dtype. Compose owns further table storage and routing, and each consumer owns its typed input support. |
| covariance, general linear-algebra consumers | Evaluate partial-pivoted LU with solve/inverse and signed log determinant for Float32/Float64, both triangles, valid strides and exact-zero-pivot errors. Compare rounding and inverse construction before adoption; do not assume LAPACK parity. | Numerical owner: mojo-linalg and shared adapters. [Shared covariance solves](shared/linalg/docs/linalg.md) are SPD-only. [Covariance parity](families/covariance/docs/parity.md) records remaining near-singular determinant/inverse differences and reproducible audits. |
| mixture, pipeline | Support an unsupervised final estimator after a transformer prefix. Forward `fit`, `predict`, `predict_proba`, `score_samples`, `score` and `fit_predict` according to the final estimator's capabilities. | Pipeline owns the final-step contract. [Sequential Pipeline](docs/pipeline/sequential.md) lists this gap. The [mixture selection notebook](notebooks/mixture/plot_gmm_selection.ipynb) records the failed scaler and mixture composition. Tracked in #21 part 1 and #24 part 1. |
| mixture, model_selection | Support target-free dense search with estimator-default and callable scoring, and a grid that carries integer and string parameters without string conversions. | Model selection owns the adapter and grid value contracts. [Dense search](families/model_selection/algorithms/search/docs/search.md) documents the current supervised adapter and user-defined tagged grid values. The [mixture selection notebook](notebooks/mixture/plot_gmm_selection.ipynb) uses an ignored target and string conversions for component count and covariance type. Tracked in #3 part 3 and #24 part 2. |

## Deferred scope

| Families | Requirement | Scope and ownership |
| --- | --- | --- |
| dummy, sparse composition consumers | Shared CSR/CSC/COO contracts for implicit zeros, validation, shape access, weighted sparse-target counts and sparse predictions; explicit uniform-strategy/output-type behavior. | [Dummy scope](docs/dummy/scope.md) remains dense-only. Existing family-specific sparse support does not authorize these branches. Storage/operations belong to numerical owners, estimator behavior to families, and forwarding/concatenation to composition owners. No implicit densification. |
