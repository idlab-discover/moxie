# moxie.discriminant_analysis API

Pure Mojo ports of `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` and
`QuadraticDiscriminantAnalysis` (reference: scikit-learn
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`), for dense `numojo.NDArray` input.

```mojo
from numojo import Shape, zeros
from moxie.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

var lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
lda.fit(X, y)                  # X: NDArray[DType.float64] (n, p); y: List[Int]
var labels = lda.predict(X)    # List[Int] in classes_ vocabulary
var proba = lda.predict_proba(X)
var Z = lda.transform(X)       # (n, min(rank, n_components))

var qda = QuadraticDiscriminantAnalysis[DType.float32, String](reg_param=0.1)
```

See [the example](../../examples/discriminant_analysis.mojo) for a runnable example.

## Type parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `dtype` | `DType.float64` | Input and output dtype: Float32 or Float64 (compile-time; others rejected) |
| `L` | `Int` | Class-label type satisfying `ClassLabel` (ordered, printable, copyable), e.g. `Int`, `String` |
| `C` | `NoCovarianceEstimator` | Type of the optional native `covariance_estimator` |

## Constructors (sklearn names and defaults)

`LinearDiscriminantAnalysis(solver="svd", shrinkage=None, priors=None, n_components=None,
store_covariance=False, tol=1e-4, covariance_estimator=None)`

`QuadraticDiscriminantAnalysis(*, solver="svd", shrinkage=None, priors=None, reg_param=0.0,
store_covariance=False, tol=1e-4, covariance_estimator=None)`

- `shrinkage` is a `Shrinkage` value; it converts implicitly from `None`, `"auto"`, a float or an int.
- `priors` is `Optional[List[Float64]]`. Bind the list to a variable and pass it (`priors=p^`);
  a bare list literal does not convert to `Optional` in Mojo 1.0.
- `covariance_estimator` is `Optional[C]` where `C: CovarianceEstimator`.
- Parameters are validated in `fit`, with sklearn's messages (`ValueError: The 'solver'
  parameter of ... must be ...`).

## Methods

For a scaled projection, use
`make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))`
and call `pipe.fit_transform(X, y.copy())` for training data or
`pipe.transform(X_query)` for held-out data. The same fitted pipeline exposes
`predict` and forwards `n_features_in_()` from the scaler. See
[supervised pipeline transforms](../pipeline/sequential.md#supervised-predictors-with-transforms).

| Method | LDA | QDA | Result |
| --- | --- | --- | --- |
| `fit(X, y)` | yes | yes | mutates in place, returns a mutable borrow of self |
| `predict(X)` | yes | yes | `List[L]` |
| `decision_function(X)` | yes | yes | `(n,)` for two classes, else `(n, n_classes)` |
| `predict_proba(X)`, `predict_log_proba(X)` | yes | yes | `(n, n_classes)`; columns follow `classes_` |
| `score(X, y, sample_weight=None)` | yes | yes | accuracy (`Float64`) |
| `transform(X)`, `fit_transform(X, y)` | yes (not `lsqr`) | no | `(n, min(rank, n_components))` |
| `transform_table(X)`, `fit_transform_table(X, y)` | yes (not `lsqr`) | no | `ColumnTable[dtype]` with component names |
| `get_feature_names_out(input_features=None)` | yes | no | `lineardiscriminantanalysis0`, ... |
| `get_n_features_in()` | yes | yes | `n_features_in_`; `NotFittedError` before fit |
| `get_classes()` | integer labels | integer labels | owned `List[Int]` in fitted order, raises `NotFittedError` before fit |
| `get_params` | yes | yes | typed params record |
| `set_params` | yes | yes | mutable borrow of self |
| `__sklearn_clone__`, `__sklearn_is_fitted__` | yes | yes | unfitted copy, fitted-state Boolean |

LDA conforms to the family trait `LinearDiscriminantTransformer`; QDA to
`DiscriminantClassifier` (both re-exported from this module). Both also declare the
shared `InputFeatureCount` contract and LDA the shared `FeatureNamesOut` contract, so
`Pipeline.n_features_in_()` and `Pipeline.get_feature_names_out()` work with them
(tests/discriminant_analysis/composition.mojo covers pipelines, clone with typed nested
`set_params`, and `cross_val_score`). QDA has no
`transform`; calling it is a compile error.

`LinearDiscriminantAnalysis.copy()` preserves fitted state in an independent
copy. Use `clone` to copy constructor configuration into an unfitted
estimator. With the default `List[Int]` labels, both estimators implement
`NumericClassifierClasses`. Pipeline and FrozenEstimator forward that
capability for [calibration](../calibration/family.md).
The default label type also works with the multiclass
[`BinaryAdapter`](../multiclass/family.md#binary-adapter-for-numeric-list-labels).

Weighted scoring rejects NaN, infinity, all-zero weights and weights whose sum
is zero, as in scikit-learn. Signed weights with a nonzero sum are accepted.

## Named dense inputs

Pass a [`ColumnTable`](../compose/named-columns.md) backed by a single Float32
or Float64 numeric matrix to `fit`, `decision_function`, `predict`,
`predict_proba`, `predict_log_proba`, or `score`. LDA also accepts it in
`transform` and `fit_transform`. These are direct estimator overloads.
The overloads reject tables with individually added columns.

```mojo
from moxie.compose import ColumnTable
from moxie.discriminant_analysis import LinearDiscriminantAnalysis

var names: List[String] = ["sepal_length", "sepal_width"]
var table = ColumnTable(X, names)
var lda = LinearDiscriminantAnalysis()
var projected = lda.fit_transform(table, y.copy())
var components = lda.get_feature_names_out()
```

The estimator stores an owned `Optional[List[String]]` in `feature_names_in_`.
Named inference checks the fitted names and order when present. LDA also checks
an explicit `input_features` against them. Numeric inference arrays are
positional. Switching between named and positional inputs is silent. A positional refit, failed fit, or unfitted clone has
`feature_names_in_ = None`.

### Labeled projections

Use `fit_transform_table(X, y)` and `transform_table(X)` to return a native
`ColumnTable[dtype]` from LDA. Both methods accept a dense NDArray or a
numeric `ColumnTable[dtype]` and preserve the projection's dtype and row order.
The table uses positional rows and names its columns
`lineardiscriminantanalysis0`, `lineardiscriminantanalysis1`, and so on.

```mojo
var training = lda.fit_transform_table(table, y.copy())
var validation = lda.transform_table(X_query)
var components = validation.schema().names
```

The explicit methods give labeled output a static Mojo return type.
When an SVD projection has fewer directions than the configured component
count, the table names only the columns present. A zero-width projection
retains all rows and has an empty column schema. `get_feature_names_out()`
reports the fitted component limit.

## Score sample-weight requests

LDA and QDA expose the shared [sample-weight request contract](../interfaces.md#sample-weight-requests).
A parent can read `get_metadata_routing().score` before fit and resolve
weights from its validated metadata. For example,
`model.set_score_request(sample_weight=String("validation_weight"))` lets the parent
resolve that key and pass the resulting array as `sample_weight` to `score`.
Use `True` for the standard key, `False` to decline weights, or `None` to
restore the unset request. Omitting the argument keeps the previous request.

The typed request is separate from constructor parameters. An unfitted clone
retains the request. Direct `score` calls accept an explicit
weight array independently of the request. The fit slot is unset because
neither estimator accepts fit weights. With either estimator as the final step
of a two-stage supervised [Pipeline](../pipeline/sequential.md#scoring), call
`pipe.score(X, y, metadata={"validation_weight": weights.copy()})` after
setting the alias on `pipe.estimator`.

## Native model persistence

Use the experimental [Moxie archive format](../serialization.md) to save fitted
or unfitted LDA and QDA models and restore them without training data.

```mojo
from moxie.discriminant_analysis_serialization import (
    LinearDiscriminantAnalysisSchema,
    QuadraticDiscriminantAnalysisSchema,
)
from moxie.serialization import load_model, save_model

save_model[LinearDiscriminantAnalysisSchema[]](lda, "lda.moxie")
var restored_lda = load_model[LinearDiscriminantAnalysisSchema[]]("lda.moxie")
save_model[QuadraticDiscriminantAnalysisSchema[]](qda, "qda.moxie")
var restored_qda = load_model[QuadraticDiscriminantAnalysisSchema[]]("qda.moxie")
```

Choose the same dtype and label type when saving and loading, for example
`LinearDiscriminantAnalysisSchema[DType.float32, String]`. With the default
`NoCovarianceEstimator` type, the schemas cover both dtypes, every solver
and built-in shrinkage choices. The archive
saves current configuration separately from learned state. A post-fit setting
change preserves LDA's fitted projection state and QDA's fitted class factors
on load. LDA uses version 2 when it saves a prior-renormalization warning,
including a warning retained after a failed refit. LDA version 3 and QDA
version 2 preserve fitted feature names. LDA version 3 also preserves the
warning when present. With built-in `Int` or `String` encoding and neither
names nor warnings, archives use version 1. Loaders accept earlier versions.
QDA archives preserve optional fitted
covariances independently of the current `store_covariance` setting. Loading
validates array shapes, class order, feature names and required learned fields.

For another native `ClassLabel` type, implement `DiscriminantLabelCodec` and
pass it as the third schema parameter. Use the same schema for saving and
loading. QDA uses the same convention.

```mojo
from moxie.discriminant_analysis_serialization import (
    DiscriminantLabelCodec,
    LinearDiscriminantAnalysisSchema,
)
from moxie.serialization import BinaryReader, BinaryWriter

struct Int32LabelCodec(DiscriminantLabelCodec):
    comptime Label = Int32
    comptime TypeId = "example.int32-label.v1"

    @staticmethod
    def encode(mut writer: BinaryWriter, label: Int32):
        writer.scalar[DType.int32](label)

    @staticmethod
    def decode(mut reader: BinaryReader[_]) raises -> Int32:
        return reader.scalar[DType.int32]()

comptime Int32LDASchema = LinearDiscriminantAnalysisSchema[
    DType.float64, Int32, Int32LabelCodec,
]
```

Give the codec a stable, versioned `TypeId` that identifies its label type
and wire encoding. The codec owns validation of individual label values.
Each label has its own length-prefixed record, and the decoder must consume
the full record. The archive stores the codec ID in LDA version 4 and QDA
version 3. Loading requires a matching codec selected by the caller at
compile time, including for unfitted models.

To persist a native `covariance_estimator`, implement
`DiscriminantCovarianceCodec` and pass it as the `CovarianceCodec` schema
parameter. Here is a schema using a caller-provided codec.

```mojo
comptime CustomLDASchema = LinearDiscriminantAnalysisSchema[
    CovarianceCodec=MyCovarianceCodec,
]
```

The codec's associated `Estimator` type determines the model's covariance
type parameter. The codec writes and validates the constructor configuration
with `BinaryWriter` and `BinaryReader`. LDA and QDA use temporary clones for
covariance fitting, so only the current configured estimator is passed to the
codec. Loading restores the discriminant model's learned state without
refitting, even if the covariance configuration changed after training.

Give the codec a stable, versioned `TypeId`. LDA version 5 and QDA version 4
require a matching caller-selected codec even when the covariance parameter is
`None`. When present, the configuration is stored in a length-prefixed record
that the codec must consume in full. These versions also support named features
and custom label codecs.

For a complete codec and round trips for both estimators, see the
[covariance archive contract](../../tests/discriminant_analysis/covariance_serialization.mojo).

LDA version 6 and QDA version 5 preserve an explicit score-weight request
alongside the earlier named input, label codec and covariance codec state.
Older archives restore an unset score request. Saving with an unset request
uses the previous archive layout.

## Cross-validation and shrinkage search

For default five-fold stratified cross-validation with Float64 features and
`List[Int]` labels, use the built-in model-selection adapter. Grid search
uses the same folds and accuracy scorer. Pass a typed parameter applier to
map the grid's candidate values to LDA's configuration.

```mojo
from moxie.discriminant_analysis import LinearDiscriminantAnalysis, Shrinkage
from moxie.model_selection import DenseSupervisedAdapter, GridSearchCV, cross_val_score

comptime LDA = LinearDiscriminantAnalysis[]
comptime Adapter = DenseSupervisedAdapter[LDA, List[Int]]

def apply_shrinkage(mut estimator: LDA, params: Dict[String, Float64]) raises:
    for key in params.keys():
        if key != "shrinkage":
            raise Error("ValueError: unsupported LDA search parameter " + key)
        _ = estimator.set_params(shrinkage=Shrinkage(params[key]))

# Inside a function with X and y:
var model = LDA(solver="lsqr")
var scores = cross_val_score[Adapter](model, X, y, Adapter())
var grid = Dict[String, List[Float64]]()
grid["shrinkage"] = [0.0, 0.25, 1.0]
var search = GridSearchCV[Adapter](model, grid, Adapter(apply_shrinkage))
_ = search.fit(X, y)
```

For Float32 features, use `LinearDiscriminantAnalysis[DType.float32]` and
specify `DType.float32` as the adapter's third type parameter. See the
[model-selection search contract](../../families/model_selection/algorithms/search/docs/search.md).

## Learned attributes

LDA: `coef_` `(1, p)` binary or `(K, p)`, `intercept_` `(1,)` or `(K,)`, `means_`, `priors_`,
`classes_` (sorted), `n_features_in_`, and `Optional` attributes that exist only where
sklearn defines them: `covariance_` (lsqr/eigen always; svd with `store_covariance=True`),
`explained_variance_ratio_` and `scalings_` (svd/eigen), `xbar_` (svd).

QDA: `means_` `(K, p)`, `priors_`, `rotations_` (List of `(p, n_k)`), `scalings_` (List of `(n_k,)`),
`classes_`, `n_features_in_`, `covariance_` (`Optional[List]`, with `store_covariance=True`).

## Solvers and numerics

- `svd`: two economy SVDs (within-class whitening, then between-class centroids); tall inputs
  are reduced by a Householder R factor first, which preserves singular values and right
  singular vectors. `tol` truncates singular values.
- `lsqr` uses minimum-norm least squares on the class-mean system. With empirical
  covariance (`shrinkage=None` or `0`, with no custom covariance estimator), it drops
  singular values at or below `n_features * eps * s_max`, using the Float64 working
  precision. This accounts for Jacobi SVD roundoff in null directions that can exceed
  scikit-learn's `eps * s_max` cutoff. Shrunk and custom covariance estimates use
  `eps * s_max`.
- `eigen`: generalized symmetric-definite eigenproblem `Sb v = lambda Sw v` by Cholesky reduction
  (eigenvectors satisfy `V^T Sw V = I`, like `scipy.linalg.eigh(Sb, Sw)`); a singular `Sw`
  raises `LinAlgError`, as in sklearn.
- QDA `svd`/`eigen` factor each class (SVD of the centered class data, or eigh of its covariance).
- Shrinkage: `'auto'` is Ledoit-Wolf on standardized features (sklearn `_cov`), a float applies
  `shrunk_covariance`.
- All statistics and factorizations run in Float64 and are rounded to `dtype` at the end.
  sklearn computes some Float32 paths in single precision, so Float32 results agree to about
  1e-5 relative and are usually closer to the exact value.
- `log`/`exp` (intercepts, QDA log-determinants, log posteriors, probabilities) use Numax
  `core.libm` (fdlibm, within one ulp, like NumPy's libm); Mojo 1.0's `std.math.log` Float64 approximation
  was off by up to ~1e-9 and moved QDA `predict_proba` by ~8e-10.
- Eigenvector/singular-vector signs (`scalings_`, `rotations_`, `transform` columns) are not
  specified, as in LAPACK.

## Mojo adaptations (each covered by tests)

- `fit`/`set_params` return a mutable borrow of the same estimator, following current
  Moxie contracts. Concrete and generic chains are supported. A failed `fit` leaves the estimator
  **unfitted**, not partially updated (tests/lifecycle.mojo).
- `covariance_estimator` is a native trait instead of a duck-typed object:
  ```mojo
  trait CovarianceEstimator(Copyable, Movable, Deinitable):
      def fit(mut self, X: NDArray[DType.float64]) raises: ...
      def covariance(self) raises -> NDArray[DType.float64]: ...
  ```
  `fit` receives a Float64 copy of the samples. LDA estimates within-class covariance and,
  for `eigen`, total covariance. QDA estimates covariance for each class. Both fit private
  configurations (sklearn mutates the object you pass). Using a type
  without `fit`/`covariance` (sklearn: `ValueError` at fit) fails at compile time
  (tests/compile_fail/covariance_without_trait.mojo).

  Use the public `CovarianceEstimatorAdapter` to pass an estimator from `moxie.covariance`.

  ```mojo
  from moxie.covariance import OAS
  from moxie.discriminant_analysis import CovarianceEstimatorAdapter, LinearDiscriminantAnalysis

  var covariance = CovarianceEstimatorAdapter(OAS(store_precision=False))
  var lda = LinearDiscriminantAnalysis[DType.float64, Int, type_of(covariance)](
      solver="lsqr", covariance_estimator=covariance^
  )
  ```

  The adapter supports LDA's `lsqr` and `eigen` and QDA's `eigen` paths. Use Float64
  covariance estimators to preserve the discriminant-analysis working precision, including
  with Float32 classifier input. With a Float32 covariance estimator, the adapter casts the
  samples to Float32 for its fit and widens its fitted covariance back to Float64.
  Native adapters clone the wrapped estimator's configuration for fitting and cloning.
  Custom implementations of the discriminant-analysis trait can override
  `clone_configuration()` to discard any fitted state. Its default copies the value.
- `transform` rejects a current `solver='lsqr'`, as in sklearn. Changing between `svd` and
  `eigen` keeps the projection learned at fit time, including whether to subtract `xbar_`.
  sklearn selects centering from the current `solver`. A model fitted with `lsqr` needs a
  new fit with `svd` or `eigen` before it can transform.
- LDA records a `moxie.warnings.FitWarning` in `fit_warnings` when it
  renormalizes priors whose Float64 sum differs from 1 by more than `1e-5`.
  The record carries scikit-learn's `UserWarning` category and message.
  Each successful fit replaces the list. A failed fit preserves the last
  successful warning list while resetting the fitted model. An unfitted
  clone starts with an empty list.
- `priors` whose length differs from the number of classes raise `ValueError` (sklearn fails
  later with an unrelated broadcasting error or silently misbehaves).
- `predict_log_proba` of LDA rounds probabilities to the output dtype before taking
  logarithms; exact zeros are replaced by that dtype's smallest normal, like sklearn.
- Errors are Mojo `Error`s whose text starts with the sklearn exception name
  (`ValueError:`, `NotImplementedError:`, `NotFittedError:`, `LinAlgError:`).
- `get_params` returns a typed `LinearDiscriminantAnalysisParams` / `QuadraticDiscriminantAnalysisParams`
  record. `set_params` takes that record, or keyword `Optional` values (a nullable parameter
  such as `n_components` takes `Optional[Optional[Int]]`, e.g. `n_components=Optional[Int](1)`).
- Python reflection, pickling, pandas inputs, transform output configuration
  and array-API dispatch remain open.
