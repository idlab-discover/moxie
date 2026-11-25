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
| kernel_ridge, model_selection | Configuration-dependent pairwise input tags and dense train/test Gram matrix slicing. | The [estimator interfaces](docs/interfaces.md) do not define input tags. [Model selection](families/model_selection/docs/plumbing.md) exposes the input row selection seam. Its pairwise policy needs train-by-train and test-by-train blocks for precomputed kernels. |
| random_projection, kernel_approximation | A shared fit warning contract that preserves warning category and message. | The [estimator interfaces](docs/interfaces.md) have no warning contract. [Nystroem](docs/kernel_approximation/nystroem/api.md) records messages locally. Random projection needs to report `DataDimensionalityWarning` for explicit expansion. |
| compose, text consumers | Native Python-compatible regex search with explicit syntax errors for `make_column_selector(pattern=)`. | Reusable parsing/matching belongs in shared infrastructure. [Compose](docs/compose/scope.md) currently rejects regex selection; substring matching is not equivalent. |
| compose, preprocessing, feature_extraction | Heterogeneous column selection and composition preserving integer, Boolean and native string dtypes without lossy float conversion. | [ColumnTable](docs/compose/named-columns.md) supports a Float32/Float64 matrix alongside string categorical columns and routes those columns to encoders. Extending table storage and dtype-based dispatch to the remaining column kinds belongs to Compose. [Categorical columns](docs/categorical.md) already provide typed categorical storage. |
| covariance, general linear-algebra consumers | Evaluate partial-pivoted LU with solve/inverse and signed log determinant for Float32/Float64, both triangles, valid strides and exact-zero-pivot errors. Compare rounding and inverse construction before adoption; do not assume LAPACK parity. | Numerical owner: mojo-linalg and shared adapters. [Shared covariance solves](shared/linalg/docs/linalg.md) are SPD-only. [Covariance parity](families/covariance/docs/parity.md) records remaining near-singular determinant/inverse differences and reproducible audits. |

## Deferred scope

| Families | Requirement | Scope and ownership |
| --- | --- | --- |
| dummy, sparse composition consumers | Shared CSR/CSC/COO contracts for implicit zeros, validation, shape access, weighted sparse-target counts and sparse predictions; explicit uniform-strategy/output-type behavior. | [Dummy scope](docs/dummy/scope.md) remains dense-only. Existing family-specific sparse support does not authorize these branches. Storage/operations belong to numerical owners, estimator behavior to families, and forwarding/concatenation to composition owners. No implicit densification. |
