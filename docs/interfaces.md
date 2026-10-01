# Estimator interfaces

## Reference and scope

The reference is the local scikit-learn 1.10.dev0 checkout at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Current Mojo
contracts follow that reference; they do not establish complete scikit-learn
compatibility or a universal array ABI.
Dense Dummy estimators are implemented; see their [native API](dummy/api.md).

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

- `fit`, `set_params`, and `partial_fit` mutate in place and return nothing.
  Use `model.fit(X, y)` followed by `model.predict(X)`. The reference-returning
  trait conformance attempted on Mojo 1.0.0 was rejected by the compiler;
  fluent chaining is not promised by this baseline.
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

Experimental native persistence is an opt-in `ModelSchema` adapter, separate
from estimator traits; see [serialization](serialization.md). It does not imply
Python pickle compatibility or automatic persistence for existing families.
