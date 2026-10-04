# Robust covariance API

```mojo
from moxie.covariance import MinCovDet, EllipticEnvelope, fast_mcd

var mcd = MinCovDet(random_state=0)          # [dtype=DType.float64]
mcd.fit(X)
var raw = mcd.raw_covariance_()              # FastMCD estimate (not consistency-scaled)
var robust = mcd.covariance_()               # reweighted, consistency-corrected
var inliers = mcd.support_()                 # NDArray[DType.bool] (n,)

var detector = EllipticEnvelope(contamination=0.1, random_state=0)
var training_labels = detector.fit_predict(X) # fits, then labels X (+1 / -1)
var labels = detector.predict(X_test)        # NDArray[DType.int64], +1 / -1
var margin = detector.decision_function(X_test)
var result = fast_mcd(X, None, 0)            # FastMCDResult: location, covariance, support, dist
```

| sklearn | Mojo |
| --- | --- |
| `MinCovDet(*, store_precision=True, assume_centered=False, support_fraction=None, random_state=None)` | same keywords; `random_state: Optional[Int]` |
| `random_state=RandomState(...)` instance | `fit(X, mut random_state)` / `fast_mcd(X, support_fraction, mut random_state)` share a caller stream |
| `raw_location_`, `raw_covariance_`, `raw_support_`, `location_`, `covariance_`, `precision_`, `support_`, `dist_` | accessor methods (snapshots) |
| `EllipticEnvelope(*, ..., contamination=0.1, random_state=None)` | same; `offset_()` |
| `predict`, `decision_function`, `score_samples`, `score(X, y, sample_weight=None)` | same; labels are Int64, score is (weighted) accuracy |
| `fit_predict(X, y=None, **kwargs)` | fits and returns Int64 labels `(n_samples,)`; ignores typed optional y; an explicit `fit_predict(X, mut random_state)` overload forwards the caller stream to fit; no dynamic kwargs |
| `fast_mcd(X, support_fraction=None, cov_computation_method=..., random_state=None)` | returns `FastMCDResult[dtype]`; the covariance method is fixed to the empirical MLE |

Behavior follows the pinned sklearn revision, including its consistency
correction (#32117): the raw FastMCD distances are divided by
alpha / chi2.cdf(chi2.ppf(alpha, p), p + 2) with alpha = h / n before the
0.975 chi-squared reweighting mask; the reweighted covariance is multiplied by
the factor for alpha = 0.975. `raw_covariance_` stays unscaled.

`score` accepts optional Float64 weights of shape `(n_samples,)`, including
strided vectors. Weights must be finite and contain at least one nonzero value.
Negative weights are accepted, as in scikit-learn. All-zero weights raise an
error with a `ValueError` prefix; nonzero weights that sum to zero raise an
error with a `ZeroDivisionError` prefix. Labels remain a 1D Int64 array.

## Explicit Mojo differences

- Exact ties in distances (argpartition) and determinants (argsort) are
  resolved toward the lower sample/trial index; NumPy leaves them to its
  unstable, platform-dependent sorts. Ties are systematic in the univariate
  branch (both endpoints of the shortest half); Float32 input reproduces
  sklearn's Float32 rounding in that branch.
- `random_state=None` uses an entropy-seeded native state (not NumPy's global
  RandomState); pass an Int for reproducibility.
- Rank-deficient data and determinant increases emit category-prefixed warnings
  on stderr; Python warning objects and filters are not provided.
- Arrays keep the estimator dtype; computation is Float64 (sklearn's np.cov path).
- `cov_computation_method` callables are not supported.

## Public correction and reweighting

Both estimators expose `correct_covariance(data)` and
`reweight_covariance(data)` for the training data, in its original row order.
Correction returns a corrected raw covariance and divides the current `dist_`
by the consistency factor computed from the current `support_`. It leaves the
installed covariance and precision unchanged; repeated calls rescale distances
again. Like scikit-learn, this method does not inspect `data`.

Reweighting returns a tuple `(location, covariance, support)` of independent
snapshots. The returned covariance is **unscaled**; the installed `covariance_`
is consistency-scaled. Reweighting replaces location, support, covariance,
precision and distances, preserving all raw attributes. It uses the current
`assume_centered` and `store_precision` settings. EllipticEnvelope's `offset_`
is unchanged by either method.

Native validation explicitly rejects unfitted calls and, for reweighting,
nonfinite or nonmatrix input, changed feature/sample counts and empty selected
support. These descriptive errors replace Python's incidental attribute/index
errors. Validation failures preserve learned state. Dtypes follow the existing
native estimator rule (including tuple arrays); inputs may be strided.

## Covariance error norms

Both estimators expose `error_norm(comp_cov, norm="frobenius", scaling=True,
squared=True) -> Float64`. `norm="spectral"` selects the largest singular value
of the covariance difference; otherwise the Frobenius norm is used. Scaling
divides the squared norm by the feature count, before the optional square root.
Float32/Float64 comparisons may use positive or negative strides.

As in the empirical native API, comparison arrays must have exactly the fitted
covariance shape; NumPy broadcasting is not supported. Invalid shapes raise
`ValueError`, unknown norms raise `NotImplementedError`, and unfitted calls
raise a descriptive `NotFittedError` (rather than sklearn's incidental missing
attribute error). These names are prefixes on Mojo `Error`.

The Frobenius path propagates NaN and infinity, matching scikit-learn.

## Parameter updates

`set_params` accepts a typed record from `get_params()` or keyword partial
updates. Omitted keywords preserve constructor settings; supplied values do not
clear fitted state. Clones contain the updated configuration and no learned
state. Validation remains deferred until fit, and successful refits replace
learned state. Both methods return a mutable borrow of the original estimator, supporting
chained updates and fitting.

Both estimators accept `support_fraction=None` and `random_state=None` to
explicitly reset nullable parameters. For example:

```mojo
model.set_params(store_precision=False, support_fraction=None, random_state=7)
```

Fit publishes raw location/covariance/support/distances before correction and
reweighting. Failed numerical refits can retain those partial updates with the
previous covariance/precision and (for EllipticEnvelope) offset.
