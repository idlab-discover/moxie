# Gaussian process classification

`GaussianProcessClassifier` fits binary Laplace posteriors for binary and
multiclass classification.
It accepts a nonempty, finite rank-two NuMojo `NDArray` of Float64 features by
default. Use `GaussianProcessClassifier[DType.float32]` for Float32 features.
Targets are rank-one or single-column Int64 arrays by default. The second
dtype parameter selects Float32 or Float64 integral-valued categorical labels. Predictions
are rank one with the original label dtype and probability columns are in
sorted class order.

```mojo
from moxie.gaussian_process import GaussianProcessClassifier
from numojo import arange, Shape, zeros

def main() raises:
    var X = arange[DType.float64](0.0, 4.0).reshape(Shape(4, 1))
    var y = zeros[DType.int64](Shape(4))
    y.unsafe_set(2, 1)
    y.unsafe_set(3, 1)
    var model = GaussianProcessClassifier()
    var probabilities = model.fit(X, y).predict_proba(X)
    print(probabilities)
```

The default kernel is a fixed unit `ConstantKernel * RBF`. An explicit kernel
with free parameters uses bounded L-BFGS-B by default. `optimizer=None` uses the
configured hyperparameters. `n_restarts_optimizer` draws from the seeded
stream or shared default stream. Restarts follow the pinned sklearn
classifier's exponentiation of sampled log bounds before the optimizer call.
`max_iter_predict` limits Newton iterations. `warm_start` reuses the previous
mode within a fit and between likelihood evaluations of a fitted component.
A new public fit always creates fresh components.

- For two classes, `predict` uses the sign of the posterior latent mean. An
  exact zero selects the first class.
- For two classes, `predict_proba` integrates the latent Gaussian with
  sklearn's five-term error-function approximation and returns Float64
  `(n_queries, 2)`. At zero latent variance, it evaluates the finite limit
  of that approximation.
- For two classes, `latent_mean_and_variance` returns `LatentMoments` with
  two Float64 `(n_queries,)` arrays.
- `log_marginal_likelihood` returns a `LikelihoodResult` with a cached value
  when theta is omitted, or a value and optional analytic gradient at supplied
  log theta. `clone_kernel=False` updates the learned kernel's theta. Cached
  evidence and posterior factors retain the fit values.
- `score` computes accuracy, optionally with a Float64 weight vector.
- `classes_()`, `n_classes_()`, `kernel_()` and
  `log_marginal_likelihood_value_()` read fitted metadata. `base_estimator_()`
  returns an independent snapshot. Its `estimators_` list has one member for
  a binary fit, one per sorted class for OvR, or one per sorted class pair
  for OvO.
  `n_features_in_` records the number of columns seen at fit.

Both `copy_X_train` settings take an independent training snapshot, and
Float32 input is widened before kernel computation. `get_params` returns the
typed configuration and `set_params` supports named partial updates. For
nullable settings, pass a typed inner `Optional` to set or clear the value.
For example, `model.set_params(optimizer=Optional[String]())` disables
optimization. `clone(model)` copies only configuration, while `model.copy()`
copies fitted state. Inference requires a successful fit.

## Multiclass fits

`multi_class="one_vs_rest"` (the default) fits one binary classifier per
class. `predict` selects the largest positive-class probability, breaking
ties by sorted class order. `predict_proba` normalizes those probabilities
by row and returns Float64 `(n_queries, n_classes)`.

`multi_class="one_vs_one"` fits one classifier per sorted class pair.
`predict` counts binary votes and uses summed positive-class probabilities
to break ties as sklearn does. Calling `predict_proba` raises for a multiclass
OvO fit. Both strategies use a single binary component when there are exactly
two classes. `n_jobs` is accepted and component fits run serially. A value
of zero raises for multiclass fits.

Multiclass `kernel_()` returns a `CompoundKernel` with the component kernels
in fit order. The log marginal likelihood is the mean of the component
values. Supply a shared log-theta vector or concatenate a vector per
component. For OvO with four or more classes, the concatenation has one block
per pair. The pinned sklearn implementation checks for one block per class,
which cannot represent all fitted pairs. Multiclass likelihood gradients and
latent moments raise. Changing `multi_class` through `set_params` takes
effect on the next successful fit.

## Structured inputs

Use `StructuredGaussianProcessClassifier[MyKernel, Sample]` with an explicit
`KernelContract` whose `Input` is `List[Sample]` and whose
`requires_vector_input()` returns false. For example, a sequence kernel can
take `List[String]`, with each string representing one sample. `Sample`
must be copyable. The optional third type parameter selects the label dtype.
Training and query lists must contain at least one sample. The kernel defines
the semantics and validity of each sample.

The same kernel can be used with `StructuredGaussianProcessRegressor[MyKernel]`.
Return covariance with shape `(n_X, n_Y)`, a diagonal of shape `(n_X,)`, and
log-theta derivatives of shape `(n_X, n_X, n_dims)` when requested. The
classifier passes the stored training list as `X` and the query list as
explicit `Y` for cross-covariance. It copies retained samples using their
native copy operations. OvO selects the two classes' samples in their
original input order. Binary, OvR, and OvO fits use the same Float64 Laplace
engine and optimizer as the dense classifier.

For structured fits, `kernel_()` returns a typed `List[MyKernel]` in the
same order as `base_estimator_().estimators_`, with one element in the binary
case. This preserves the fitted kernel type across binary and multiclass
fits. `n_features_in_` is zero because the input consists of structured
samples. Targets, class ordering, output shapes and fit lifecycle follow the
dense classifier contract.

The shared sequence-kernel probe in
[structured parity](../../../tests/gaussian_process/regression/structured_parity.py)
compares fixed and optimized binary models and three-class OvR and OvO
models with pinned scikit-learn. The OvO oracle uses integer row IDs to
carry sequences through scikit-learn's numeric matrix validation and decodes
the IDs inside the same kernel. The probe checks class order, predictions,
probabilities, latent moments, fitted theta and log marginal likelihood.

## Numerical implementation

Algorithm 3.1 factors `I + sqrt(W) K sqrt(W)` and uses triangular solves.
Algorithm 3.2 reuses the factor for latent variance. Algorithm 5.1 contracts
kernel derivatives against shared terms, without a new dense matrix per
hyperparameter. The factorization and triangular solves use the family's
existing `CholeskyAlgebra` over mojo-linalg. Numax `linalg.cholesky` and
`linalg.matvec` operate on TensorLike layouts with compile-time extents for
these paths. The family adapters already accept runtime-shaped NuMojo arrays,
and the products use the authorized NuMojo rank-two `@` path with column
right-hand sides. Numax `core.libm.erf` supplies the high-accuracy scalar erf
because the weighted mixture amplifies approximation error. The likelihood
uses the same Float64 exp and log as the kernel engine, with a correction for
rounding in `log(1 + z)`. The loops
apply diagonal scaling and fuse reductions using Float64.

The implementation follows [GPML chapter 3](https://gaussianprocess.org/gpml/chapters/RW3.pdf)
and [chapter 5](https://gaussianprocess.org/gpml/chapters/RW5.pdf). The
classification parity script executes pinned sklearn's GPC, kernel and
multiclass source with installed support modules and SciPy. It exercises Float32/Float64,
binary, OvR and OvO fits, fixed and optimized theta, seeded restarts, and a
one-step Newton limit with warm-start enabled and disabled. Tolerances account
for Float32 kernel arithmetic and stopping decisions in the reference optimizer.

Run `python tools/gaussian_process_checks.py classification` in the locked
environment to assemble live sources, compile the classification and
structured-input probes, and run their contracts and pinned parity.
