# Dense LogisticRegression

Import `LogisticRegression`, `LogisticRegressionParams` and `ClassWeight`
from `moxie.linear_model`. The dense `lbfgs` solver fits binary and
multinomial problems with an L2 penalty or an unpenalized objective. The
default is L2. Set `C` to infinity or pass `None` as the penalty to remove
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

The model retains the optimizer's accepted iterate on convergence stalls and
iteration limits. An optimization attempt with no finite accepted iterate
raises. Mojo exposes `n_iter_` and errors. Python's convergence and
deprecation warnings, metadata routing, callback hooks and verbose logging
are unavailable. `n_jobs` has no effect. `random_state` and
`intercept_scaling` are accepted configuration parameters and have no effect
with `lbfgs`.

## Focused validation

Run the focused commands from the repository root with the locked environment active.

```sh
python families/linear_model/tools/check.py logistic-test
python families/linear_model/tools/check.py logistic-parity-only
python tools/linear_model_checks.py logistic-composition
```

The first command assembles the family namespace and builds both the native
contracts and the differential driver. The second executes the estimator and
linear-loss Python sources at scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, using the locked support wheel and
SciPy. The last compiles the StandardScaler pipeline through the integrated
namespace and checks class order, decision scores, probabilities, accuracy,
nested clone and nested parameter updates.

The differential suite checks binary and multinomial fits in both dtypes,
including weighting, unpenalized and warm-start fits, zero and one
iteration budgets, and invalid inputs. Float64 and the widened Float32 oracle
use absolute bounds of 2e-6 for coefficients, intercepts and probabilities and
5e-6 for decision and log-probability values. The Float32 sklearn objective
rounds intermediate values in the feature dtype and can stop earlier due to
its rounded loss. Its bounds are 9e-4 and 3e-3 respectively. Classification
outputs and iteration counts for zero and one iteration budgets match exactly.
For other iteration budgets the suite checks the count's shape and valid range.
