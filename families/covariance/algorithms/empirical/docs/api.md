# Empirical covariance API

```mojo
from moxie.covariance import EmpiricalCovariance, ShrunkCovariance
from moxie.covariance import empirical_covariance, shrunk_covariance, log_likelihood

var model = EmpiricalCovariance()            # [dtype=DType.float64]
model.fit(X)                                 # X: NDArray (n_samples, n_features)
var S = model.covariance_()                  # (p, p)
var P = model.get_precision()                # pinvh(covariance_) if not stored
var d2 = model.mahalanobis(X_test)           # squared distances, (n,)
var ll = model.score(X_test)                 # mean Gaussian log-likelihood
var err = model.error_norm(C, norm="spectral", scaling=True, squared=True)
```

| sklearn | Mojo |
| --- | --- |
| `EmpiricalCovariance(*, store_precision=True, assume_centered=False)` | same keywords; `EmpiricalCovariance[dtype]` with dtype Float32/Float64 |
| `ShrunkCovariance(*, store_precision=True, assume_centered=False, shrinkage=0.1)` | same |
| `fit(X, y=None)` returns self | mutates in place, returns a mutable borrow of self |
| attributes `location_`, `covariance_`, `precision_`, `n_features_in_` | accessor methods `location_()`, `covariance_()`, `precision_()` (Optional; None when store_precision=False), `n_features_in_()`; raise NotFittedError before fit |
| `get_params()` / `set_params(**kw)` | typed `EmpiricalCovarianceParams` / `ShrunkCovarianceParams` record, plus keyword overloads |
| `clone(est)` | `moxie.base.clone(est)` returns an unfitted copy |
| `empirical_covariance(X, *, assume_centered=False)` | same; 1-D X is one sample |
| `shrunk_covariance(emp_cov, shrinkage=0.1)` | same, including `(..., p, p)` batches |
| `log_likelihood(emp_cov, precision)` | same, returns Float64 |

Behavior follows the pinned scikit-learn: MLE divisor n; precision is
`scipy.linalg.pinvh` (cutoff n*eps(dtype)*max|w|, zero and singular
covariances allowed); `fast_logdet` returns -inf for nonpositive determinants;
`score` centers X_test by `location_`; `mahalanobis` returns squared distances.

`log_likelihood` also accepts broadcastable rank-2 operands, as in the reference
NumPy path: `(1, p)`, `(p, 1)` or `(1, 1)` empirical inputs may be combined
with `(p, p)` precision. A `(1, 1)` precision broadcasts across any rank-2
empirical input; the normalization still uses the precision dimension.
Incompatible dimensions raise `ValueError`. Vectors and higher-rank operands
are unsupported. Reductions retain the native Float64 accumulation policy.

## Explicit Mojo differences

- Array results keep the estimator dtype. For Float32 input sklearn's centered
  branch (np.cov) produces Float64 covariance/precision; the port computes that
  branch in Float64 (precision with the Float64 cutoff, score and distances
  with the Float64 precision) and rounds stored arrays to Float32.
- `get_precision()` recomputes the pseudoinverse whenever no precision is
  stored, even if `store_precision` was switched on after fitting (sklearn
  would return None there).
- Squared Mahalanobis distances are the direct quadratic form; sklearn takes
  `cdist(...)**2`, which differs only by rounding.
- `fast_logdet` uses symmetric LDLᵀ or pivoted Householder QR for nonsymmetric
  precision. Arbitrary singular/near-singular determinant rounding can still
  differ from sklearn's LU, including finite versus `-inf` scores.
- Singleton inputs emit `UserWarning: ...` to stderr. Python warning objects
  and filters are not provided. Errors are Mojo `Error`s with `ValueError:`, `NotFittedError:`,
  `InvalidParameterError:`, `NotImplementedError:` or `LinAlgError:` prefixes.
- Python-only facilities (metadata routing, feature names, array API
  namespaces, repr) are not provided.

## Shared numerics for other children

`moxie.covariance._empirical_covariance` also exports `CovarianceModel[dtype]`
(learned state + inherited sklearn methods, composed by every covariance
estimator), `EmpiricalCovarianceEngine[dtype]` (the family `CovarianceEngine`),
`pinvh`, `fast_logdet`, `squared_mahalanobis`, `error_norm_value`,
`column_means`, `center`, `centered_columns`, `gram` and `validate_samples`.

Mahalanobis distances return NaN for negative quadratic forms, matching the
NumPy/scipy reference path when precision is indefinite.

Fits publish location before numerical covariance work. After a failed numerical
refit, location can be new while covariance/precision are from the previous fit.
Invalid-input rejection leaves prior estimates intact.
