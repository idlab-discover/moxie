# Dense LogisticRegression

Import `LogisticRegression`, `LogisticRegressionParams` and `ClassWeight`
from `moxie.linear_model`. The dense `lbfgs`, `newton-cg` and
`newton-cholesky` solvers fit binary and multinomial problems with an L2
penalty or an unpenalized objective. The defaults are `solver="lbfgs"` and
L2. Set `C` to infinity or pass `None` as the penalty to remove
regularization. Other solvers and L1 or elastic-net fitting are open work.

```mojo
from moxie.linear_model import ClassWeight, LogisticRegression

var model = LogisticRegression(C=0.5, class_weight=ClassWeight.balanced_weights())
var probabilities = model.fit(X, y).predict_proba(X)
```

## Shapes and configuration

- `LogisticRegression[dtype, label_dtype]` defaults to Float64 features and
  Int64 labels. Features and array sample weights use the same Float32 or
  Float64 dtype. A Float64 scalar fit weight expands to that dtype.
- `X` has shape `(n_samples, n_features)`. Numeric integer-valued `y` has
  shape `(n_samples,)` or `(n_samples, 1)`. Labels must fit the signed Int
  domain used by the family's label encoding and `ClassWeight`. Prediction
  preserves the label dtype and returns a vector. Training requires at least
  two classes. Class weights multiply sample weights. Balanced class weights
  use the sample-weighted class frequencies.
- `classes_` contains the sorted labels. Binary `coef_` has shape
  `(1, n_features)`, `intercept_` has one entry and `decision_function`
  returns a vector for the second class. The multinomial equivalents have one
  coefficient row and intercept per class, with a decision matrix of shape
  `(n_samples, n_classes)`. The probability and log-probability matrices
  always have one column per class. Rounded zero probabilities give negative
  infinity in `predict_log_proba`, as in scikit-learn.
- `n_features_in_` stores the fitted feature count. `n_iter_` is an Int32
  array of length one. `score` returns Float64 accuracy and accepts optional
  vector weights.
- `get_params` returns the typed record. Keyword `set_params` updates supplied
  fields. Use the record to reset optional fields to `None`. An explicit
  legacy penalty uses `String("l2")`. `fit` and `set_params` return mutable
  borrows for fluent calls. `clone` is unfitted. A fitted `copy()` owns its
  own state.
- A normal fit replaces learned state. With `warm_start=True`, fit initializes
  from the previous coefficient matrix when its dimensions agree with the
  current problem. An incompatible fit or invalid input leaves the previously
  fitted state available.

## Numerical and runtime behavior

The optimizer uses Float64 arithmetic in both feature dtypes. The returned
coefficient, intercept, decision and probability arrays retain the feature
dtype. The objective is the sample-weighted mean negative log likelihood plus
`||coef||² / (2 * C * sum(effective_weights))`. Intercepts are unpenalized.
The weighted mean requires a finite, positive sum. Balanced weights also
require nonzero input mass in each class.

The family reuses `mojo_optimize.LBFGSB` without bounds and
`_linear_base.linear_decision` for inference. The objective streams a compact
NuMojo snapshot of `X` and uses one row of class scratch. NuMojo
`matmul_2darray` would require materializing sample-by-class raw scores and
pointwise gradients for the two matrix products. Numax `softmax` and tensor
`logsumexp` work on MAX tensor layouts and only cover the probability or
reduction step. Numax's scalar `sigmoid` computes `exp(-score)`, which
overflows for large negative scores before their probability underflows.
The streamed objective uses Mojo `exp` and `log1p` with nonpositive exponents
and accumulates the gradient directly.

Select `solver="newton-cholesky"` to use analytic Newton steps with Armijo
backtracking. This solver assembles the lower triangle of a reduced dense
Hessian and requires quadratic memory in the number of free coefficients.
Streaming the per-row curvature avoids constructing a weighted design matrix
per class pair for NuMojo's rank-two matrix multiplication.
For multinomial fits it fixes one reference class per unpenalized column
during optimization and returns centered coefficients for those columns.
It uses the linear-model linalg owner's pivoted symmetric LDLᵀ solve,
analogous to sklearn's symmetric solve. Numax's Array Cholesky fixes the
matrix order at compile time and floors pivots. Its Tensor Cholesky requires
compile-time dimensions and assumes positive definiteness. The family uses
the existing runtime-sized symmetric solve to handle signed weights and
surface factorization failures. A singular system, a tiny pivot relative to
the other pivots, a non-descent step or failed backtracking triggers L-BFGS
within the remaining budget. The pivot check differs from SciPy's reciprocal
condition estimate, so borderline ill-conditioned systems can switch solvers
at different iterations.

Select `solver="newton-cg"` to solve each Newton step with truncated
conjugate gradient. It stops when the maximum absolute gradient is at most
`tol`. It streams signed binary curvature or the multinomial softmax
covariance through a Hessian-vector product. It needs linear scratch space
in the coefficient count and class count beyond the compact input snapshot.
The product fuses the two design-matrix projections with row curvature. It
avoids the projected and weighted sample-by-class arrays that separate
NuMojo `matmul_2darray` calls would allocate. Numax `linalg.dot` takes
fixed-size arrays or MAX tensors with compile-time lengths. This solver
uses the optimizer dependency's contiguous NDArray dot and strong-Wolfe
line search. That search uses bracketing and bisection, so accepted step
lengths and iteration counts can differ from sklearn's SciPy line search.
With unpenalized multinomial fits, a common offset across classes leaves
the probabilities and predictions invariant. The pinned Float32 solver
can drift along this direction. Moxie's Float64 arithmetic keeps a centered
cold start close to centered, so raw coefficients and decision values may
differ by that common offset.

The model retains the optimizer's accepted iterate on convergence stalls and
iteration limits. An optimization attempt with no finite accepted iterate
raises. Mojo exposes `n_iter_` and errors. Python's convergence and
deprecation warnings, metadata routing, callback hooks and verbose logging
are unavailable. `n_jobs` has no effect. `random_state` and
`intercept_scaling` are accepted configuration parameters and have no effect
with these solvers.

## Focused validation

Run the focused commands from the repository root with the locked environment active.

```sh
python families/linear_model/tools/check.py logistic-test
python families/linear_model/tools/check.py logistic-parity-only
python tools/linear_model_checks.py logistic-composition
```

The first command assembles the family namespace and builds both the native
contracts and the differential driver. The second executes the estimator and
linear-loss Python sources and the Newton-CG optimizer at scikit-learn
revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`, using the locked
support wheel and SciPy. The last compiles the StandardScaler pipeline
through the integrated namespace and checks class order, decision scores,
probabilities, accuracy, nested clone and nested parameter updates.

The differential suite checks binary and multinomial fits in both dtypes
for all three solvers. Cases cover weighting, unpenalized and warm-start
fits, zero and one iteration budgets, line-search backtracking, singular
and nearly singular Hessians, stationary and negative-curvature Newton-CG
starts, and invalid inputs. The native contracts compare the streamed
Hessian-vector product against central differences of the analytic gradient
with signed and zero weights.

The Float64 and widened Float32 oracles use absolute bounds of 2e-6 for
coefficients, intercepts and probabilities and 5e-6 for decision and
log-probability values. The Float32 sklearn objective rounds intermediate
values in the feature dtype and can stop earlier due to its rounded loss.
Its bounds are 9e-4 and 3e-3 respectively. For unpenalized multinomial
Newton-CG, the Float32 reference comparison uses class-centered coefficients
and decision values. The widened comparison checks both arrays directly.
Pinned unpenalized multinomial Newton-Cholesky pads a Float32 step with
Float64 zeros and raises a buffer dtype error. The native Float32 fit
succeeds, and those two cases compare against the pinned solver on widened
inputs. The widened binary log-probability comparison rounds the sigmoid
to Float32 before computing its complement, since rounding near one is
amplified by the log. Classification outputs are exact. Iteration counts
are exact for zero- and one-iteration budgets, the stationary start, the
backtracking case, and the singular (two-iteration budget) and nearly
singular fallback cases. The other cases check iteration count shape and
range.
