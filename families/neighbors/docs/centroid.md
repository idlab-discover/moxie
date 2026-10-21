# NearestCentroid

See the runnable [centroid example](../examples/centroid.mojo).

`NearestCentroid[dtype](metric="euclidean", shrink_threshold=None, priors="uniform")`
implements dense centroid and shrunken-centroid classification. Manhattan uses
feature-wise medians; Euclidean uses class means. `priors` accepts `"uniform"`,
`"empirical"` or, through a constructor overload, a Float64 NuMojo vector in
sorted class order. Finite nonnegative vectors with positive total are required;
nonunit totals are normalized using sklearn's closeness threshold. Warning
emission is not implemented. Malformed/nonfinite/all-zero priors explicitly
raise, including cases for which sklearn may instead produce invalid state.

This estimator implements `ClassifierMixin`. Targets use `NearestCentroid[dtype, Y]`, where Y defaults to NDArray[dtype]
and may be Int64/Float32/Float64 NDArray or StringTarget. Rank one and single-column
matrices are accepted. Predictions and `classes_` preserve label storage and dtype;
class encoding comes from the shared label owner. Learned
`centroids_`, `deviations_`, `within_class_std_dev_` and `class_prior_` are Float64;
`n_features_in_` is the feature count. No sparse input or feature-name tracking
is supplied.

`fit`, `predict` and weighted accuracy `score` are supported for both metrics.
Uniform priors use nearest centroids without variance scaling; nonuniform priors
use squared, variance-scaled distances and a log-prior correction, following the
pinned reference. This means `predict` need not equal probability argmax for
uniform priors. `decision_function`, `predict_proba` and `predict_log_proba`
require Euclidean distance; Manhattan calls raise an `AttributeError`-prefixed
Mojo error. Binary decisions have shape `(n_samples,)`; multiclass decisions
and both probability arrays have shape `(n_samples, n_classes)`.

The constructor accepts only positive finite shrink thresholds or `None`.
Constant training data raises. Zero within-class standard deviations and
undefined deviations remain visible, as in sklearn. A dataset with one sample
per class may fit and predict with uniform priors but cannot provide standardized
scores because its pooled variance is undefined. Shrinkage that creates nonfinite
centroids also causes inference to raise rather than fabricating results.

`get_params()` returns `NearestCentroidParams`; its `priors` string is
`"explicit"` for array priors, with the owned vector in `prior_probabilities`.
`set_params(params)` copies configuration and preserves fitted state; refit for
fit-time changes. Cloning and constructor/get/set parameter vectors own independent
copies. Fit publishes learned state only after success and retains no input alias.
Python fluent returns, partial keyword parameter updates, warning objects and
reflection are not supplied.

Training uses dtype-preserving NuMojo means, shared pooled-variance reductions
and the pinned NuMojo median on compact Float64 vectors. Queries are made
contiguous before dtype conversion to preserve arbitrary strides. Centroid parity
tolerances are rtol=atol=3e-5 for Float32 and 2e-11
for Float64 learned state on the tested data. Float64 decisions, probabilities
and log-probability outputs allow 2e-9; they use Numax `core.libm` exp/log
(fdlibm, within one ulp), so this bound is conservative. The 432-case parity matrix executes the pinned
sklearn fit/prediction and discriminant mixin bodies with NumPy/SciPy plumbing;
it checks learned arrays, labels, probabilities, shrinkage and degenerate cases.
It is not the full sklearn estimator-check suite or a performance claim.

Run `pixi run test` for native contracts and isolated consumers, and
`pixi run parity` for the pinned-source comparisons.
