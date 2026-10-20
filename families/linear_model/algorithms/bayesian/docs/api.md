# BayesianRidge

Import `BayesianRidge` and `BayesianRidgeParams` from `moxie.linear_model`.
The implementation lives in `moxie.linear_model._bayes`. It fits dense
Float32/Float64 arrays without Python or SciPy at runtime.

```mojo
from moxie.linear_model import BayesianRidge

var model = BayesianRidge[DType.float64](compute_score=True)
_ = model.fit(X, y)
var prediction = model.predict(Z)
var posterior = model.predict_std(Z)
var mean = posterior[0].copy()
var deviation = posterior[1].copy()
```

`X` has shape `(samples, features)` and `y` has shape `(samples,)` or
`(samples, 1)`. Predictions always have shape `(samples,)`. Fit accepts
nonnegative finite scalar or vector sample weights with positive total mass.
Weights affect centering, residuals, initial target variance and precision
updates. Inputs may be strided; fitting preserves caller arrays, including
when `copy_X=False`. Fitting again replaces learned state only after success.

The keyword constructor follows scikit-learn's defaults:

| Parameter | Default |
| --- | --- |
| `max_iter` | `300` |
| `tol` | `1e-3` |
| `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2` | `1e-6` |
| `alpha_init`, `lambda_init` | `None` |
| `compute_score` | `False` |
| `fit_intercept`, `copy_X` | `True` |
| `verbose` | `False` |

`get_params()` returns a typed configuration record. `set_params(record)`
replaces configuration; keyword `set_params` updates supplied fields. For the
optional initial precisions, keyword updates use `Optional[Optional[Float64]]`;
the record provides a simpler way to reset them to `None`. `clone(model)`
copies configuration into an unfitted estimator. Fluent `fit` and `set_params`
return a borrow of the estimator. `score` uses the family's weighted R2.

Learned attributes are `coef_` `(features,)`, `intercept_` `(1,)`, `alpha_`,
`lambda_`, `sigma_` `(features, features)`, `n_iter_`, `n_features_in_`,
`X_offset_`, `X_scale_`, and `scores_`. `scores_` is empty when disabled,
otherwise it contains `n_iter_ + 1` evidence values. Float64 intermediates
feed same-dtype array attributes; `alpha_` and `lambda_` are Float64 scalars.
The length-one intercept array follows the family interface adaptation.
The final evidence value retains the pinned reference's use of the last loop
coefficient norm with the final recomputed residual.

`predict(X)` returns the mean array. `predict(X, return_std=flag)` returns a
`PredictionResult` Variant containing the mean array when false, or a
`Tuple[NDArray, NDArray]` of mean and standard deviation when true.
`predict_std(X)` returns that tuple directly, for statically typed consumers
such as an iterative imputer. Both uncertainty calls use centered query rows
and include observation noise. The covariance retains the prior variance in
unobserved feature directions, including when there are more features than rows.

The solver uses the adopted mojo-linalg SVD. Wide matrices are padded with
zero rows to obtain the complete right singular basis. Covariance storage and
the wide factorization require quadratic space in the feature count. This
implementation makes no performance claim.

Invalid inputs raise descriptive Mojo `Error`s. Unfitted prediction raises a
`NotFittedError` message. Nonpositive/nonfinite precision updates and singular
initial posteriors raise `LinAlgError` messages rather than storing invalid
learned state. Evidence scoring with a zero initial precision raises a
`ValueError`, as in the reference. A valid zero noise-precision initialization
is supported with scoring disabled. With zero weight precision, singular
initial systems are rejected. Negative weights, sparse inputs, multiple
regression targets, feature names, metadata routing and persistence are not
supported. ARDRegression remains unimplemented.

# Validation

Run `pixi run build`, `pixi run test` and `pixi run parity` at the family root.
The family gates include this estimator's native contracts and differential
checks. The oracle executes `_bayes.py` and `_base.py` from the pinned local
scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Generated inputs, native output and comparison metadata live in
`algorithms/bayesian/builds/parity/`.

The suite checks tall, wide, square and single-feature inputs, rank deficiency,
conditioning, constant targets/features, single samples, scalar/vector weights,
custom priors, early stopping, iteration limits, coefficients, intercepts,
precisions, covariance, evidence history and predictive means/deviations.
Native contracts cover both dtypes, ownership, strides, cloning, repeat fits,
parameter changes, fluent calls and invalid-input state preservation.

Float64 comparisons use `rtol=atol=1e-7` for Jacobi versus LAPACK rounding.
Float32 array comparisons use `3e-4` because the native intermediate arithmetic
is Float64, while sklearn uses Float32 reductions and factorizations. Learned
Float32 precisions use `rtol=3e-3`: strong shrinkage in the wide fixture amplifies
rounding in the small coefficient norm and produced a 0.19% difference in
`lambda_`. Prediction and covariance checks retain their stricter tolerance.
An independent precision-times-covariance identity checks full covariance with
a bound scaled by the measured matrix condition number. Degenerate weighted
cases that fail in the oracle must also fail natively.
