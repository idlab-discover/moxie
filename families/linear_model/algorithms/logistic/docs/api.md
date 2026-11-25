# Dense LogisticRegression

Import `LogisticRegression`, `LogisticRegressionParams` and `ClassWeight`
from `moxie.linear_model`. The dense `lbfgs`, `newton-cg` and
`newton-cholesky` solvers fit binary and multinomial problems with an L2
penalty or an unpenalized objective. The defaults are `solver="lbfgs"` and
L2. Dense `sag` and `saga` also support both class modes. Use
`solver="saga", l1_ratio=1.0` for L1 or `0 < l1_ratio < 1` for elastic
net. Set `C` to infinity or pass `None` as the penalty to remove
regularization for any of these five solvers. For binary classification, `solver="liblinear"`
supports primal L2, dual L2 and primal L1 with finite `C` and positive
`tol`. Select L1 with `l1_ratio=1.0`. Use `newton-cd` for binary or
`newton-cd-gram` for binary and multinomial fits with L2, L1, elastic net
or no penalty. LogisticRegressionCV remains open.

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
  current problem. LIBLINEAR starts each fit from scratch. An incompatible
  fit or invalid input leaves the previously fitted state available.

## Numerical and runtime behavior

The solvers use Float64 arithmetic in both feature dtypes. The returned
coefficient, intercept, decision and probability arrays retain the feature
dtype. For the three smooth solvers, the objective is the sample-weighted
mean negative log likelihood plus `||coef||² / (2 * C * sum(effective_weights))`.
Intercepts are unpenalized.
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
The streamed objective uses Numax scalar Float64 `exp` with nonpositive
exponents and Mojo `log1p`. It accumulates the gradient directly. Probability
inference uses Numax scalar Float64 `exp` and `log`.

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

Select `solver="newton-cd"` for binary proximal Newton iterations.
The solver clips negative pointwise curvature to zero in the quadratic
subproblem and keeps signed weights in the logistic objective. It uses
the family's weighted centering and rescaling and the same cyclic residual
engine as ElasticNet, with at least one inner sweep. The working design
and its column-major copy take linear space in the number of input
elements. Numax's fixed-size dot and NuMojo's dense matrix product do not
supply the in-place residual updates and Gap Safe screening used by this
engine. The outer iteration uses composite Armijo backtracking and checks
the minimum-norm subgradient and Newton decrement. Failed backtracking
uses L-BFGS for smooth objectives within the remaining budget. With L1,
it raises Mojo `Error`, corresponding to the pinned solver's raised
`ConvergenceWarning`. The solver accepts strided features through the
same compact snapshot as the other solvers.

Select `solver="newton-cd-gram"` to solve the quadratic Newton subproblem
with cyclic coordinate descent on the dense Hessian. It eliminates unpenalized
intercepts with a Schur complement and recovers them after the feature step.
Like the pinned reference, it applies Gap Safe screening for L1 and uses the
row count as the squared norm of the multinomial dual-gap target. For pure L1
and unpenalized multinomial fits, it shifts the warm start to the reference
class. Unpenalized fits return class-centered coefficient columns. L1 fits
preserve the zeros from coordinate descent. Backtracking uses the same
composite line search as `newton-cd`. When the quadratic step or line search
fails, smooth objectives can use the remaining L-BFGS budget. With L1, the
fit raises Mojo `Error`.

The Gram solver needs quadratic space in the number of free coefficients.
It reuses the streamed Hessian, the linear-model linalg owner's
`symmetric_pinv` on the intercept block, and NuMojo's rank-two matrix
multiplication for the intercept map. The coordinate sweeps maintain the
Gram-vector product with in-place row updates. Numax's fixed-size Array
`axpy` requires compile-time lengths and its Tensor overload requires MAX
tensor storage. NuMojo array expressions for the row updates would allocate
a product-sized temporary at each coordinate.

SAG and SAGA sample rows with replacement and update the average of the
gradients for rows seen so far. SAGA adds a correction for the current
gradient and soft-thresholds the feature coefficients for L1 and elastic net.
The step size uses the maximum squared row norm and divides the penalty
strength by the row count. Each sample gradient uses its effective weight
directly, including zero and signed weights. The intercept is unpenalized.
The solver uses Moxie's dense `row_squared_norms` and NuMojo's vector
maximum for the step size. Numax's dot requires compile-time dimensions,
and independent Numax activations and NDArray products would require
separating row prediction, gradient memory updates and the proximal step.
The solver reuses the logistic row projection and shifted exponentials,
then applies each update immediately in the dense coefficient layout.
It keeps one gradient value per sample and output, plus coefficient-sized
gradient sums and convergence state.

An integer `random_state` repeats the pinned sampler's row sequence on
every fit. `None` draws a seed from Moxie's process-wide stream. Warm starts
reuse coefficients and intercepts and rebuild the gradient memory. As in
scikit-learn, convergence compares feature coefficients after each epoch
and `max_iter=0` returns the initialization with `n_iter_=[1]`. Moxie
applies the dense updates eagerly in Float64, so low-bit differences and
iteration counts can vary from sklearn's delayed updates. The row-norm
vector retains the feature dtype. An unusable step size or nonfinite
iterate raises a Mojo `Error` with a `ValueError` prefix. When the product
of the step size and scaled L2 penalty is one, the error uses the
reference's `ZeroDivisionError` prefix.

LIBLINEAR uses the existing `moxie.svm._liblinear` engine. Class weights are
applied once to sample weights, and the engine drops nonpositive effective
weights before optimization. At least one positive effective weight is
required. Public `classes_` still records both original classes if one loses
all its samples. The engine minimizes the sum of weighted logistic losses
scaled by `C`, plus the L1 norm or half the squared L2 norm. When
`fit_intercept=True`, the regularizer includes the synthetic bias
coefficient and `intercept_scaling` sets its feature value.
An integer `random_state` starts a fresh MT19937 stream for each fit. With
`None`, the solver draws its seed from Moxie's process-wide stream.

For Float32 input, the wrapper widens the logical matrix to Float64 before
calling the engine and casts the fitted arrays back to Float32. The pinned
reference retains Float64 learned arrays and inference outputs for LIBLINEAR.
Moxie keeps the same feature-dtype interface across solvers. With intercept
disabled, Moxie returns a length-one zero intercept array where scikit-learn
returns a scalar.

The model retains the optimizer's accepted iterate on convergence stalls and
iteration limits. An optimization attempt with no finite accepted iterate
raises. Mojo exposes `n_iter_` and errors. Python's convergence and
deprecation warnings, metadata routing, callback hooks and verbose logging
are unavailable. `n_jobs` has no effect. `random_state` and
`intercept_scaling` are accepted configuration parameters and have no effect
with L-BFGS or either Newton solver.

## Focused validation

Run the focused commands from the repository root with the locked environment active.

```sh
python families/linear_model/tools/check.py logistic-test
python families/linear_model/tools/check.py logistic-parity-only
python tools/linear_model_checks.py logistic-composition
```

The first command assembles the family namespace and builds both the native
contracts and the differential driver. The second executes the estimator and
linear-loss Python sources, the Newton-CG optimizer, the LIBLINEAR wrapper,
the LIBLINEAR C++ sources, the SAG/SAGA Cython engine and its dataset
sampler, and the two Newton-CD solvers and coordinate-descent Cython engine at
scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
It uses the locked support wheel, SciPy and the SVM owner's C++ oracle
adapter. The last command compiles the StandardScaler pipeline through the
integrated namespace and checks class
order, decision scores, probabilities, accuracy, nested clone and nested
parameter updates.

The differential suite checks binary and multinomial fits in both dtypes
for the three smooth solvers. Cases cover weighting, unpenalized and warm-start
fits, zero and one iteration budgets, line-search backtracking, singular
and nearly singular Hessians, stationary and negative-curvature Newton-CG
starts, and invalid inputs. The native contracts compare the streamed
Hessian-vector product against central differences of the analytic gradient
with signed and zero weights.

The Float64 parity cases also call the pinned probability methods with
identical decision scores or rounded probabilities to isolate inference
from optimizer differences. Probability uses a relative tolerance of
`2e-15` and an absolute tolerance of `2e-16` to allow rounding when
subtracting a sigmoid from one. Log probability uses `2e-15` for both.

The LIBLINEAR matrix covers primal L1, primal L2, dual L2, signed and class
weights, loss of one class after weight filtering, scaled and disabled
bias, warm-start configuration, and limited iterations. It compares the
native typed results against the Float64 reference with absolute bounds of
`2e-6` for coefficients, intercepts and probabilities and `5e-6` for
decisions and log probabilities. These bounds account for rounding back to
Float32 and accumulation differences.

The SAG/SAGA cases cover fixed-seed short budgets, convergence, warm refits,
L1 zeros, elastic net, both forms of disabling the penalty, and sample and
class weighting. Float64 fits compare iteration counts exactly and use an
absolute bound of `5e-10` for coefficients, intercepts, probabilities and
decisions. The log-probability bound is `3e-9` to account
for differences in logarithm evaluation. Native Float32 uses bounds of
`5e-6` for fitted arrays and probabilities and `2e-5` for decisions and
log probabilities, allowing for rounding during gradient accumulation and
convergence. Widened Float32 uses `3e-6` across those outputs to account
for rounding public arrays back to Float32. Native contracts also cover
seed changes, strided input, zero weights, clone and fit failure after a
successful fit.

Newton-CD cases cover L2, L1 and elastic net, no penalty, zero and one
iteration, signed and zero sample weights with balanced classes, warm
refit, a zero column, and multiclass rejection. The suite checks L1 zero
patterns exactly and exercises the raised error after L1 backtracking
fails. The Float32 cases use `tol=1e-6` because the pinned native Float32
line search can stall below that precision. Newton-CD-Gram also covers
multinomial penalties, balanced and signed weights, an intercept-only
Hessian, and attempts to enable dual mode. Both binary and multinomial
comparisons check L1 or elastic-net zero patterns where the published
coefficient convention preserves them.

For L-BFGS and the Newton solvers, the Float64 and widened Float32 oracles use
absolute bounds of 2e-6 for coefficients, intercepts and probabilities and
5e-6 for decision and log-probability values. The Float32 sklearn objective rounds intermediate
values in the feature dtype and can stop earlier due to its rounded loss.
Its bounds are 9e-4 and 3e-3 respectively. For unpenalized multinomial
Newton-CG, the Float32 reference comparison uses class-centered coefficients
and decision values. The widened comparison checks both arrays directly.
The pinned unpenalized multinomial Newton-Cholesky and Newton-CD-Gram
solvers pad Float32 steps with Float64 zeros. Cases where this raises a
buffer dtype error compare against the pinned solver on widened inputs.
The widened binary log-probability comparison rounds the sigmoid to Float32 before computing its complement, since rounding near one is
amplified by the log. Classification outputs are exact. Iteration counts
are exact for zero- and one-iteration budgets, the stationary start, the
backtracking case, and the singular (two-iteration budget) and nearly
singular fallback cases. The other cases check iteration count shape and
range.
