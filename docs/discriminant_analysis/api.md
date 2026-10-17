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

| Method | LDA | QDA | Result |
| --- | --- | --- | --- |
| `fit(X, y)` | yes | yes | mutates in place, returns a mutable borrow of self |
| `predict(X)` | yes | yes | `List[L]` |
| `decision_function(X)` | yes | yes | `(n,)` for two classes, else `(n, n_classes)` |
| `predict_proba(X)`, `predict_log_proba(X)` | yes | yes | `(n, n_classes)`; columns follow `classes_` |
| `score(X, y, sample_weight=None)` | yes | yes | accuracy (`Float64`) |
| `transform(X)`, `fit_transform(X, y)` | yes (not `lsqr`) | no | `(n, min(rank, n_components))` |
| `get_feature_names_out(input_features=None)` | yes | no | `lineardiscriminantanalysis0`, ... |
| `get_params` | yes | yes | typed params record |
| `set_params` | yes | yes | mutable borrow of self |
| `__sklearn_clone__`, `__sklearn_is_fitted__` | yes | yes | unfitted copy, fitted-state Boolean |

LDA conforms to the family trait `LinearDiscriminantTransformer`; QDA to
`DiscriminantClassifier` (both re-exported from this module). QDA has no
`transform`; calling it is a compile error.

Weighted scoring rejects NaN, infinity, all-zero weights and weights whose sum
is zero, as in scikit-learn. Signed weights with a nonzero sum are accepted.

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
- `lsqr`: minimum-norm least squares on the class-mean system with the gelsd cutoff
  `eps * s_max`, like `scipy.linalg.lstsq`.
- `eigen`: generalized symmetric-definite eigenproblem `Sb v = lambda Sw v` by Cholesky reduction
  (eigenvectors satisfy `V^T Sw V = I`, like `scipy.linalg.eigh(Sb, Sw)`); a singular `Sw`
  raises `LinAlgError`, as in sklearn.
- QDA `svd`/`eigen` factor each class (SVD of the centered class data, or eigh of its covariance).
- Shrinkage: `'auto'` is Ledoit-Wolf on standardized features (sklearn `_cov`), a float applies
  `shrunk_covariance`.
- All statistics and factorizations run in Float64 and are rounded to `dtype` at the end.
  sklearn computes some Float32 paths in single precision, so Float32 results agree to about
  1e-5 relative and are usually closer to the exact value.
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
  `fit` receives a Float64 copy of the (per-class) samples. The estimator fits a private copy,
  so the configured object is never mutated (sklearn mutates the object you pass). Using a type
  without `fit`/`covariance` (sklearn: `ValueError` at fit) fails at compile time
  (tests/compile_fail/covariance_without_trait.mojo).
- Python warnings do not exist: LDA priors that do not sum to 1 are renormalized silently
  (sklearn also emits `UserWarning`).
- `priors` whose length differs from the number of classes raise `ValueError` (sklearn fails
  later with an unrelated broadcasting error or silently misbehaves).
- `predict_log_proba` of LDA rounds probabilities to the output dtype before taking
  logarithms; exact zeros are replaced by that dtype's smallest normal, like sklearn.
- Errors are Mojo `Error`s whose text starts with the sklearn exception name
  (`ValueError:`, `NotImplementedError:`, `NotFittedError:`, `LinAlgError:`).
- `get_params` returns a typed `LinearDiscriminantAnalysisParams` / `QuadraticDiscriminantAnalysisParams`
  record. `set_params` takes that record, or keyword `Optional` values (a nullable parameter
  such as `n_components` takes `Optional[Optional[Int]]`, e.g. `n_components=Optional[Int](1)`).
- Python reflection, pickling, pandas feature names (`feature_names_in_`), metadata routing
  and array-API dispatch are outside this port.
