# Scope: discriminant_analysis.estimators (dense only)

## 1. Supported dense estimators, methods and tested settings

- `LinearDiscriminantAnalysis`: solvers `svd`, `lsqr`, `eigen`; `shrinkage` None / `'auto'`
  (standardized Ledoit-Wolf) / float in [0, 1]; `priors` (validated, renormalized);
  `n_components`; `tol`; `store_covariance`; native `covariance_estimator`; binary and
  multiclass; `fit`, `predict`, `predict_proba`, `predict_log_proba`, `decision_function`,
  `score` (with `sample_weight`), `transform`, `fit_transform`, `get_feature_names_out`,
  `get_params`/`set_params`/clone.
- `QuadraticDiscriminantAnalysis`: solvers `svd` (with `reg_param`) and `eigen` (with
  `shrinkage` or `covariance_estimator`); `priors`; `store_covariance`; `tol` rank check;
  all prediction/decision/score methods.
- Input: dense `numojo.NDArray[DType.float32 | DType.float64]` of shape (n, p), any strides.
  Labels: any `ClassLabel` type (tested with `Int` and `String`).
- Tested in parity (604 cases, pinned sklearn source): toy data from the sklearn test
  suite, 1-D data, 2/3/4 classes, string labels, Gaussian blobs (seeds 0-2), features
  scaled by 1..1e4, wide data (p > n), duplicated collinear features, saturated binary
  probabilities, every shrinkage/solver combination, five covariance estimators, priors,
  n_components, tol and all fit-time error branches, in Float64 and Float32.

## 2. Deferred exports/branches

- Sparse input: none. sklearn's `discriminant_analysis` accepts only dense arrays
  (`validate_data` without `accept_sparse`), so no branch is deferred.
- Array-API (non-NumPy namespace/device) dispatch of the `svd` solver: not applicable to
  native Mojo arrays.
- Python-only facilities: pickling, pandas `feature_names_in_`, metadata routing, `set_output`.
- Inherited restrictions: mojo-linalg factorizations are scalar Float64 kernels
  (see performance.md); no GPU path.

## 3. Unsupported default settings / required explicit selections

None. All sklearn defaults (`solver='svd'`, `shrinkage=None`, ...) are supported.
Differences in behavior are listed in api.md ("Mojo adaptations").

## 4. Executed checks

```sh
pixi run build    # verified assembly + compile of all modules, tests, example
pixi run test     # contracts (13), numerics (11), lifecycle (7), composition, example,
                  # 4 compile-time negative capability checks
pixi run parity   # 604 differential cases vs pinned sklearn (60 expected-error cases)
```

Negative capability checks: a covariance estimator type without `fit`/`covariance` is
rejected at compile time; `QuadraticDiscriminantAnalysis.transform` does not exist;
incremental `partial_fit` is absent, a missing `predict` cannot satisfy the
shared trait, and `transform` with `solver='lsqr'` raises `NotImplementedError`.

Parity tolerances (norm-wise, relative to each array's largest magnitude, floor 1):
Float64 1e-8 (both sides double precision; differences come from Jacobi vs LAPACK
factorizations and summation order); Float32 5e-4 (sklearn evaluates several Float32
paths in single precision, the port in Float64). Float32 log-probabilities are compared
through `exp`, because sklearn's single-precision `1 - expit(d)` and FLT_MIN clamping make
its own log values inaccurate for tiny probabilities. Signs of singular/eigenvector columns
are aligned before comparison; only columns with separated eigenvalues are compared.

## Source ownership

The implementation, tests, example and documentation belong to Moxie.
Root validation includes this family. Builds use current Moxie assembly and
live local dependency sources, with no historical framework recovery.
See [numerical ownership](numerics.md) for reused kernels and family-specific
behavior. Inherited framework copies and standalone project manifests are not
part of the integrated source tree.
