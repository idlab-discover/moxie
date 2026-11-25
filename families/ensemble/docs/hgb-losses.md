# Histogram gradient boosting losses

The private `moxie.ensemble._hist_gradient_boosting.losses` module implements
half squared error, absolute error, half Poisson deviance, half Gamma deviance,
and pinball loss for the histogram booster. It uses Float64 targets, weights,
raw predictions, intercepts, and scalar scores. Supported vectors can be
strided. `fit_intercept` validates targets and weights. Supply the validated
vectors in training row order to subsequent derivative and leaf operations.
Weights are finite and nonnegative, matching the private bin mapper contract.
Gamma requires strictly positive targets. Poisson accepts zero targets and
requires at least one positive target.

`gradient_hessian` refills caller-owned Float32 buffers after computing weighted
derivatives in Float64. It leaves the Hessian buffer empty when all Hessians
are implicitly one, the [grower's](hgb-grower.md) unit-Hessian representation.
Poisson and Gamma use the log link. Their scores drop terms that depend only
on targets, as in scikit-learn's loss-based early stopping.

Absolute error and pinball use approximate Hessians during split search.
`update_leaves` replaces each such leaf with the median or quantile of its
residuals, scaled by the tree's learning rate. Call it before snapshotting the
predictor or adding the tree to the raw predictions. Unweighted pinball uses
linear interpolation. Weighted pinball and weighted absolute error select the
first value at the cumulative weight rank. At the midpoint of two central
values, explicit unit weights select the lower value and the unweighted median
averages the pair. An all-zero weight vector yields NaN
for a weighted percentile. Weighted means raise when their denominator is zero.

Intercept means and objective scores reuse the ensemble's Float64 pairwise
reduction. Numax `stats.quantile` and `stats.percentile` accept TensorLike
inputs and require a host copy for selection. Neither accepts sample weights.
Using the existing paired observations and standard library sort keeps both
location methods on the same leaf-sized workspace. Order among equal
observations can affect rounding at a weighted cutoff because scikit-learn
uses NumPy's unstable argsort. The log link uses LLVM Float64 scalar
`llvm.log` and `llvm.exp` to match the C math functions in the reference
derivatives, following the shared RNG owner's precedent.

With the locked Mojo toolchain active, run the scoped check from the Moxie root.

```sh
PYTHONDONTWRITEBYTECODE=1 python families/ensemble/tools/check.py hgb-losses
```

The regression driver composes the mapper, grower, residual leaf updates, and
raw predictor. It compares the intercept, Float32 derivatives, training loss
at every stage, and raw and linked query predictions with the source-built
scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`. The check
requires bitwise equality of the Float32 derivatives and uses absolute and
relative tolerances of 1e-12 for Float64 comparisons to account for
platform library rounding. Test cases include nonuniform and zero weights,
missing features, quantile interpolation and subgradients at ties.

## Classification

`_ClassificationLoss` implements binary and multinomial log loss. Pass a
Float64 vector of integral class codes in fitted class order and an optional
Float64 weight vector. `fit_intercept` validates finite nonnegative weights
and returns clipped, linked class priors as a vector with one element per
tree in a stage. A single class uses the binomial path with class code 0,
following the pinned classifier. Use the same validated weights when
computing derivatives or training loss.

Raw predictions use a rank-two Float64 NuMojo array with one column for
binomial loss or one column per class for multinomial loss. This makes the
internal stage layout uniform. The binomial link returns two probability
columns, including for a single encoded class, matching the pinned loss.
Multinomial probabilities use the same class order as the raw predictions.
The Float32 gradient and diagonal Hessian buffers store one contiguous span
of training rows per tree. Refill both buffers once per stage before
growing any trees for that stage. Newton values need no residual leaf update
for these two losses.

Multinomial derivatives and training loss reuse a row-sized workspace for
shifted exponentials. Numax `special.softmax` expects TileTensor inputs and
produces normalized probabilities. Here the grower consumes Float32
derivatives and the objective needs the unnormalized sum in class order,
so the native loop avoids a full probability matrix and preserves the reference
reduction order on strided NuMojo inputs. The binomial implementation uses
the same scalar exponential and logarithm as the regression links. The
low-logit loss branch compensates for rounding when adding the exponential
to one. The pinned `std.math.log1p` exceeded the loss tolerance in the
weighted binary parity case at raw logit -2.

```sh
PYTHONDONTWRITEBYTECODE=1 python families/ensemble/tools/check.py hgb-classification
```

The classification check compares intercepts, losses, probabilities, and
Float32 derivatives against the pinned loss objects. It also grows native
trees and compares staged raw predictions, probabilities, and training
objectives against the pinned public classifier. The Float64 checks use
absolute and relative tolerances of 1e-12 for platform math and reduction
rounding. Native contracts cover strided raw predictions and invalid inputs.
