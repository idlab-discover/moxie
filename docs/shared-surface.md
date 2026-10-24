# Cross-cutting public API

scikit-learn users import several modules outside the estimator families:
`sklearn.base`, `sklearn.metrics`, `sklearn.utils`, `sklearn.datasets`,
`sklearn.inspection` and `sklearn.exceptions`. This page records where their
Moxie counterparts live and which ones do not exist yet, against the reference
revision in [interfaces](interfaces.md). Names missing from these tables are
not implemented. Estimator families document their own modules.

## `sklearn.base`

| Reference | Moxie | Status |
| --- | --- | --- |
| `clone(estimator, *, safe=True)` | `moxie.base.clone[E: BaseEstimator](estimator)` | Adapted: typed; there is no `safe=False` deep-copy fallback for non-estimators |
| `BaseEstimator` | `moxie.base.BaseEstimator` trait | Adapted: typed `Params` for `get_params`/`set_params` (see [interfaces](interfaces.md)); no repr, HTML repr, `_get_param_names` or metadata routing |
| `ClassifierMixin`, `RegressorMixin` | traits of the same names | Adapted: `score` is accuracy / R² and takes optional typed weights |
| `ClusterMixin`, `DensityMixin`, `OutlierMixin`, `TransformerMixin` | traits of the same names | Adapted: `fit_transform` is the separate `FitTransformer` trait; `OutlierMixin` has no default `fit_predict`; no `set_output` |
| `OneToOneFeatureMixin`, `ClassNamePrefixFeaturesOutMixin` | family `get_feature_names_out` methods, `moxie._feature_metadata.FeatureNamesOut` | Adapted: each estimator implements the naming rule itself |
| `MetaEstimatorMixin`, `MultiOutputMixin` | none | Deferred: they only set Python tags, which are outside the scaffold |
| `BiclusterMixin` | none | Deferred with the cluster family's biclustering estimators |
| `is_classifier`, `is_regressor`, `is_clusterer`, `is_outlier_detector` | none; use `conforms_to(E, ClassifierMixin)` and similar | Deferred: `tests/test_estimator_kinds.mojo` checks that concrete estimators declare the mixins sklearn's predicates expect. `Pipeline` and `SelfTrainingClassifier` do not declare them yet |

## `sklearn.utils` and `sklearn.utils.validation`

| Reference | Moxie | Status |
| --- | --- | --- |
| `check_random_state` | `moxie._random.check_random_state` (None, Int, borrowed `RandomState`) | Adapted: private module path; NumPy legacy `RandomState` stream, see [random](../shared/random/docs/random.md) |
| `check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all)` | `moxie.utils.check_is_fitted(estimator)` | Adapted: uses `__sklearn_is_fitted__`; no `attributes`, `msg` or `all_or_any`; message prefix `NotFittedError:` |
| `shuffle`, `resample` | none; `RandomState.permutation`/`shuffle` give the same index stream for one array | Deferred |
| `compute_class_weight`, `compute_sample_weight` | family-private helpers (ensemble, linear_model) | Deferred as public functions |
| `check_array`, `check_X_y`, `column_or_1d`, `check_consistent_length`, `type_of_target`, `unique_labels` | family validation helpers; `moxie._labels.type_of_target` | Deferred as public functions; each estimator validates its typed inputs |
| `Bunch`, tags, `all_estimators`, `estimator_html_repr`, `metadata_routing`, `murmurhash3_32`, `gen_batches`, `gen_even_slices`, other helpers | none | Deferred |

## `sklearn.metrics`

There is no `moxie.metrics` module. Scoring currently lives in three places:

- Estimator `score` methods: accuracy for `ClassifierMixin`, R² for
  `RegressorMixin`, and estimator-specific scores (density, mixture, PCA, ...).
- `moxie.model_selection` typed scorers: callers write a `TypedScorer` or
  `SearchScorer` struct; there are no string scorer names, `get_scorer`,
  `make_scorer` or `check_scoring`. See the
  [model selection plumbing](../families/model_selection/docs/plumbing.md).
- Family-private helpers, such as accuracy and R² inside estimators, distance
  kernels in neighbors, kernel ridge and Gaussian processes, and
  `moxie.calibration.calibration_curve` (sklearn's `calibration` module).

All 86 names exported by `sklearn.metrics` are deferred: classification,
regression, ranking and clustering metrics, `pairwise_distances`,
`pairwise_kernels` and the named kernels, `DistanceMetric`, scorer factories
and the display objects. A drop-in user can compute only estimator-defined
scores natively.

## `sklearn.datasets`

There is no `moxie.datasets` module. Bundled loaders (`load_iris`,
`load_digits`, ...), generators (`make_classification`, `make_blobs`,
`make_moons`, ...), svmlight I/O and network fetchers are deferred. Tests and
notebooks load data in Python and share it as NumPy files.

## `sklearn.inspection`

There is no `moxie.inspection` module. `permutation_importance`,
`partial_dependence`, `PartialDependenceDisplay` and `DecisionBoundaryDisplay`
are deferred. Forest and tree impurity importances remain estimator attributes.

## `sklearn.exceptions`

Errors are Mojo `Error` values whose message starts with the Python class
name, for example `NotFittedError:` or `ValueError:` (see
[interfaces](interfaces.md)). There are no exception or warning classes.
Warnings such as `ConvergenceWarning` are emitted by each family, as stderr
messages or recorded warning lists; their family docs describe them.
