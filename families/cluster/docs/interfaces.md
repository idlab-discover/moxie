# Estimator interfaces

The current shared contract is [Moxie interfaces](../../../docs/interfaces.md).
The family-specific adaptations below remain part of the cluster contract.

## Reference and scope

The reference is the local scikit-learn 1.10.dev0 checkout at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. This baseline freezes shared Mojo
contracts, not a claim of complete scikit-learn compatibility or an array ABI.
Dense KMeans and AffinityPropagation are integrated into the family build.
The [AffinityPropagation API](../algorithms/affinity/docs/api.md) documents
its concrete native interface and limitations. The [KMeans API](../algorithms/kmeans/docs/api.md) documents initialization,
weighted Lloyd/Elkan fitting and native adaptations. Dense `MiniBatchKMeans` is also implemented; see its
[API](../algorithms/minibatch/docs/api.md) for incremental state and RNG semantics.
Dense `BisectingKMeans` is also implemented; see its
[API](../algorithms/bisecting/docs/api.md) for tree and split semantics.
Dense `DBSCAN` and `dbscan` are implemented; see the [API](../algorithms/dbscan/docs/api.md)
for signed weights, radius backends and precomputed distances. Dense `MeanShift`, `mean_shift`, `estimate_bandwidth`, and `get_bin_seeds` are also
implemented; see the [API](../algorithms/mean_shift/docs/api.md). Other declarations remain deferred.

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

- `fit`, `set_params`, and `partial_fit` mutate and return an origin-preserving
  borrow of the same estimator, following current Moxie HEAD. Fluent chains
  such as `model.set_params(params).fit(X).predict(X)` are supported. Use
  `_ = model.fit(X)` to discard the returned borrow. This supersedes the
  baseline's former None-return adaptation; fitted copies still require an
  explicit copy and cloning still returns an unfitted estimator.
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
- Python reflection, serialization, rich repr, pandas output, metadata routing,
  full sklearn tags, and Python bindings are outside this scaffold. They must
  be designed and tested before claiming compatibility with those facilities.

## Validation and extension

Run `pixi run build`, `pixi run test`, and `pixi run start`. Tests cover generic
trait dispatch, parameter ownership, unfitted clone behavior, fitted-state
checks, repeated fitting, mismatched targets, and changed feature counts.
Family tests compile concrete probes for their additional required methods;
probes raise instead of fabricating algorithm results.

Keep generic contracts small. Require a capability only when every estimator
represented by that trait actually provides it. Add optional traits for real
variation rather than methods that raise on most algorithms.

Mojo language references: [traits](https://mojolang.org/docs/manual/traits/)
and [reference lifetimes](https://mojolang.org/docs/manual/values/lifetimes/).

Dense `OPTICS` and its graph, DBSCAN-extraction and Xi-extraction helpers are
implemented in `algorithms/optics` and exported through the family and both
isolated consumers. See the [OPTICS API](../algorithms/optics/docs/api.md) for
metrics, typed sizes/results, Float64 promotion and unsupported capabilities.

Dense agglomerative clustering and feature agglomeration are implemented in
`algorithms/agglomerative`, with the `linkage_tree` and `ward_tree` helpers.
See the [agglomerative API](../algorithms/agglomerative/docs/api.md) for native
configuration, tree outputs, pooling, numerical contracts and scope limits.

Dense `Birch` is implemented in `algorithms/birch` and exported by the family
and both isolated consumers. It reuses agglomerative Ward clustering, the
centroid distance kernel, and neighbor queries. See its [API](../algorithms/birch/docs/api.md)
for CF-tree handles, optional labels, incremental behavior, and supported
`n_clusters` configurations. `pixi run parity` writes its comparisons to
`builds/birch-parity.json`.
