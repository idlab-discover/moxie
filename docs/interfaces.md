# Estimator interfaces

## Reference and scope

The reference is the local scikit-learn 1.10.dev0 checkout at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Current Mojo
contracts follow that reference; they do not establish complete scikit-learn
compatibility or a universal array ABI.
Dense Covariance, Dummy, Compose, Pipeline, Preprocessing, and Frozen estimators are implemented; see the
[Covariance API](../families/covariance/docs/api-surface.md), [Dummy API](dummy/api.md), [Compose scope](compose/scope.md),
[Pipeline API](pipeline/sequential.md),
[Preprocessing API](../families/preprocessing/docs/api.md), and
[Frozen API](frozen/api.md).

The primary source contracts are `sklearn/base.py`,
`doc/developers/develop.rst`, `doc/glossary.rst`,
`sklearn/utils/validation.py`, and `sklearn/utils/estimator_checks.py`.
Family docs identify more specific sources. Consult the local checkout at the
recorded revision when adding algorithms; upstream main may have changed.

## Shared contract

| Capability | Mojo trait | Required behavior |
| --- | --- | --- |
| Configuration and clone | `BaseEstimator` | Constructor parameters only; clone is unfitted |
| Supervised fit | `SupervisedEstimator` | `fit(X, y)` mutates learned state |
| Unsupervised fit | `UnsupervisedEstimator` | `fit(X, y=None)` accepts ignored target |
| Prediction | `Predictor` | `predict(X)` preserves row order |
| Classification | `ClassifierMixin` | `score(X, y, sample_weight=None)` is accuracy |
| Regression | `RegressorMixin` | Default score semantics are R2 |
| Probability | `ProbabilityPredictor`, `LogProbabilityPredictor` | Columns follow `classes_`; rows normalize |
| Decision values | `DecisionFunction` | Shape/sign follow the individual estimator |
| Transformation | `TransformerMixin`, `FitTransformer`, `InverseTransformer` | Independently opt into each capability |
| Clustering | `ClusterMixin` | `fit_predict`; unseen-data prediction is separate |
| Density | `DensityMixin` | `score_samples`; `score` aggregation follows the estimator |
| Outliers | `OutlierMixin` | +1 inliers, -1 outliers; zero decision boundary |
| Weighted fit | `WeightedSupervisedEstimator` | Explicit opt-in overload |
| Incremental learning | `IncrementalClassifier`, `IncrementalRegressor`, `IncrementalTransformer` | Preserve state according to estimator rules |

Input, target, prediction, probability, and other result types are associated
Mojo types. They may represent dense or sparse arrays, multiple outputs,
non-numeric labels, or structured results. The baseline deliberately does not
choose their storage, dtype, ownership layout, or numerical backend. Concrete
estimators must document shapes and validate their supported inputs. Traits
check signatures; behavioral parity still requires algorithm-specific tests.

Constructor names/defaults and learned public names such as `classes_`, `coef_`,
`intercept_`, and `n_features_in_` follow scikit-learn. Algorithms expose these
attributes themselves; traits cannot enforce runtime fields. `FittedState` is
an optional composition helper for ordinary nonempty tabular inputs. It records
feature counts only after successful validation, checks inference dimensions,
and can reset. Estimators with different validation rules need their own checks.

Density scoring is estimator-specific: `KernelDensity.score` sums sample log
likelihoods, while mixture models average them. Their common trait must not
change that distinction.

Normal fit replaces learned state. `partial_fit` and `warm_start` are separate,
explicit capabilities. Inference must reject unfitted state unless the original
estimator is stateless. A parameter copy contains no fitted buffers. Clone must
recursively clone nested estimators instead of copying their learned state.
Explicit reference exceptions such as `FrozenEstimator` preserve fitted state
through their own `__sklearn_clone__` implementation.

## Explicit Mojo differences

- `fit`, `set_params`, and `partial_fit` mutate in place and return a mutable
  borrow of the original estimator. Concrete and shared-trait calls support
  `model.set_params(params).fit(X, y).predict(X)`. Standalone mutation calls
  remain valid; write `_ = model.fit(X, y)` to explicitly discard the borrow.
  `fit_transform` and `fit_predict` still return transformed data and predictions.
  Clone remains a separate operation with each estimator's existing semantics.
- `get_params(deep=True)` returns an estimator-specific typed `Params` record
  or mapping. `set_params(params)` accepts that same type. Concrete estimators
  should also provide named keyword overloads for familiar partial updates.
  Dynamic heterogeneous dictionaries, arbitrary `**kwargs`, and nested
  `component__parameter` string routing are not implemented. A meta-estimator
  must define how its typed configuration represents nested updates and honor
  deep traversal; the base trait alone does not supply it.
- Errors currently use Mojo `Error` with descriptive `NotFittedError` or
  `ValueError` prefixes, not Python exception classes.
- Optional capabilities are compile-time traits. A concrete algorithm must
  additionally enforce configuration-dependent availability (SVC probability,
  LOF novelty, solver-dependent transforms, etc.).
- The weighted-fit trait requires an explicit `sample_weight` value on its
  overload. Score accepts optional weights. Incremental weight support and
  estimator-specific extra fit arguments are added by the concrete algorithm.
- Python reflection, rich repr, pandas output, metadata routing,
  full sklearn tags, and Python bindings are outside this scaffold. They must
  be designed and tested before claiming compatibility with those facilities.

## Fluent implementation convention

On Mojo 1.0.0, use an explicit mutable origin for the receiver and immutable
references for the mutation inputs in both the trait and its implementations:

```mojo
def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
    ref[origin] self,
    ref[x_origin] X: Self.Input,
    ref[y_origin] y: Self.Target,
) raises -> ref[self] Self:
    self._fit(X, y)
    return self
```

Use the same convention for `set_params(params)`, unsupervised `fit`, weighted
`fit`, and all supported `partial_fit` overloads. Give required inputs separate
immutable origins; keep optional inputs as ordinary `Optional` arguments so
callers can omit them. Keep keyword weights, defaults, and validation unchanged.
The result borrows `self`; do not return a copy or transfer it with `^`. Callers need a mutable estimator. The
borrow adds no model copy or allocation; runtime cost has not been benchmarked.
A raising mutation yields no result and preserves the estimator's existing
failure-state rules; a chain is not a transaction and does not undo earlier calls.

Explicit immutable input references avoid a Mojo 1.0.0 conversion-thunk failure
when an associated type is register-passable (such as `Int`). The simpler
`X: Self.Input` spelling works for some array estimators but is insufficient for
all existing generic composition consumers. Preserve these input conventions
when implementing shared traits. Named immutable origins allow generic
composition to forward borrowed inputs. A fixed `ImmutAnyOrigin` is insufficient
for some constrained associated types, and inferred references on omitted
optional arguments can crash this compiler.

This is a source migration: a `None`-returning implementation no longer satisfies
these traits. Families must migrate every matching overload together with their
shared Moxie source, including test estimators and generic consumers. Keep
family-owned methods in their owner; no alternative family contract is needed.
For KernelRidge, migrate its typed and mapping `set_params`, all `fit` overloads
(including weighted and feature-name variants), and use the same borrowed
receiver/result on its request setters. Update standalone calls to discard
results, then run its Float32/Float64 direct and generic chaining tests and
existing parity checks against the assembled current Moxie sources.

## Validation and extension

Run `pixi run build`, `pixi run test`, and `pixi run start`. Tests cover generic
trait dispatch, parameter ownership, unfitted clone behavior, fitted-state
checks, repeated fitting, mismatched targets, and changed feature counts.
`tests/test_fluent.mojo` covers scalar and array inputs, concrete and generic
chaining, weighted/incremental updates, and composition. Compile-failure probes
reject mutation through immutable receivers and escaping local borrows.
Family tests compile concrete probes for their additional required methods;
probes raise instead of fabricating algorithm results.

Keep generic contracts small. Require a capability only when every estimator
represented by that trait actually provides it. Add optional traits for real
variation rather than methods that raise on most algorithms.

Mojo language references: [traits](https://mojolang.org/docs/manual/traits/)
and [reference lifetimes](https://mojolang.org/docs/manual/values/lifetimes/).

Experimental native persistence is an opt-in `ModelSchema` adapter, separate
from estimator traits; see [serialization](serialization.md). It does not imply
Python pickle compatibility or automatic persistence for existing families.

Dense [kernel approximation](kernel_approximation/README.md) is implemented with
its documented native adaptations and exclusions.

Dense [discriminant analysis](discriminant_analysis/README.md) provides LDA and
QDA with current fluent fit/set-params contracts, native covariance-estimator
adapters, and the documented Python-specific exclusions.

[Isotonic regression](isotonic/README.md) is owned by Moxie, with Float32/Float64 List inputs.
The retired isotonic sibling is not a build dependency.

[Random projection](random_projection/README.md) is owned by Moxie, including Gaussian and sparse
projections with Float32/Float64 dense and CSR storage. The retired sibling
is not a build dependency.

[Naive Bayes](naive_bayes/README.md) is owned by Moxie core, with all five dense
classifiers, List/NDArray inputs and typed labels. The retired sibling is not a
build dependency.

[Multiclass classifiers](multiclass/README.md) are owned by Moxie core, including OvR, OvO,
and OutputCode with dense numeric/string labels and their documented optional
capabilities. The retired multiclass sibling is not a build dependency.

[Multioutput estimators](multioutput/README.md) are owned by Moxie core, including independent
classification/regression and both chain types. The retired sibling is not a build
dependency.

[Calibration](calibration/README.md) is owned by Moxie core, including all three calibration
methods, curves and native SVG displays. Its retired sibling is not a build
dependency.

[Gaussian processes](gaussian_process/README.md) now live in `src/moxie/gaussian_process/`,
including dense regression and kernels. Classification remains unimplemented.
The retired sibling is not a build dependency; the dedicated GP linalg worktree
and external optimizer remain explicit numerical dependencies.

[Cross decomposition](cross_decomposition/README.md) is owned by Moxie core,
including PLSRegression, PLSCanonical, CCA and PLSSVD with dense Float32/Float64
inputs and current fluent mutation contracts. Its retired sibling is not a
build dependency.

[Mixture models](mixture/README.md) are owned by Moxie core. GaussianMixture and
BayesianGaussianMixture share their EM lifecycle and Gaussian kernels.
The dedicated mixture linalg worktree remains an explicit numerical dependency.

[Model selection](../families/model_selection/README.md) is owned by Moxie.
Its typed adapters cover dense serial CV, search and incremental learning curves;
threshold tuning and successive halving remain unsupported.

[Linear models](../families/linear_model/README.md) are owned by Moxie, with the
supported dense LinearRegression, Ridge/RidgeClassifier, ElasticNet/Lasso and
BayesianRidge contracts. The standalone checkout is not a build dependency;
remaining dense algorithms and sparse support are explicitly unfinished.

[Neighbors](../families/neighbors/README.md) is owned by Moxie, including its dense estimators, NCA,
query engine and spatial trees. Sparse graph APIs remain deferred.

[Imputation](../families/impute/README.md) is owned by Moxie, including dense simple, integer,
string and KNN imputers and numeric/string indicators. IterativeImputer remains
deferred. The standalone checkout is not a build dependency.

[Ensemble](../families/ensemble/README.md) is owned by Moxie, including the four
dense RandomForest/ExtraTrees estimators and shared forest engine. Other ensemble
estimators remain explicit declarations; sparse input and decision paths are deferred.
The retired ensemble checkout is not a build dependency.

[Decomposition](../families/decomposition/docs/api-surface.md) is owned by Moxie, including its SVD and threshold
sparse-coding engines. Dictionary learning, SparsePCA and regression coding
modes remain unfinished. The standalone checkout is not a build dependency.

[Clustering](../families/cluster/README.md) is owned by Moxie, including the documented
dense affinity, centroid, density, linkage and Birch implementations. Spectral
clustering, biclustering, HDBSCAN and sparse inputs remain deferred.

[SVM estimators](../families/svm/README.md) are owned by Moxie, including all seven
dense estimators, l1_min_c and both native engines. Sparse input, Python runtime
integration and persistence remain outside the supported contract. The retired
standalone checkout is not a build dependency.

[Neural networks](../families/neural_network/README.md) are owned by Moxie, with a
dense BernoulliRBM subset and private MLP parameter helpers. MLP estimators remain
raising declarations. See the family scope for explicit deferrals.

[Semi-supervised learning](../families/semi_supervised/README.md) is owned by Moxie,
including dense LabelPropagation, LabelSpreading and SelfTrainingClassifier.
Sparse inputs and KNN graphs remain deferred. The retired standalone checkout
is not a build dependency.

[Manifold](../families/manifold/README.md) is owned by Moxie, including ClassicalMDS,
trustworthiness and dense exchange helpers. Other manifold algorithms remain
explicit unsupported declarations. The standalone checkout is not a build dependency.

[Feature extraction](../families/feature_extraction/README.md) is owned by Moxie,
including dense dictionary vectorization, image patches and dense graphs.
Text, TF-IDF, hashing and sparse output remain deferred; complex and extended
dictionary dtypes remain unfinished. The standalone checkout is not a build dependency.
