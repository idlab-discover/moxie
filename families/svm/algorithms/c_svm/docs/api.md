# Dense kernel SVM estimators

Import `SVC`, `SVR`, `NuSVC`, `NuSVR`, and `OneClassSVM` from `moxie.svm`.
The five estimators call the native family LIBSVM engine. Constructors retain
scikit-learn's parameter names and defaults, including `gamma="scale"`,
`max_iter=-1`, and the pinned reference's `probability="deprecated"` default.
`probability=True` enables fitted classifier calibration and probability methods.
Disabled probability methods raise `AttributeError`-prefixed Mojo errors.

Matrices use `numojo.NDArray[float64]` or Float32 inputs converted to Float64.
Regressor targets are Float64 vectors. Classification accepts `List[Label]`,
with `Label=Float64` by default, or numeric NDArrays for Float64 labels.
`SVC[String]` and `NuSVC[String]` retain sorted string class labels.
Sample weights are optional Float64 vectors. Pass `weights.copy()` when
converting an existing owned NDArray to the optional argument.

`fit` and `set_params` return a mutable borrow for chaining. `get_params` returns
the estimator-specific typed record; `set_params` accepts that record or named
partial updates. `__sklearn_clone__` copies constructor configuration into an
unfitted model. Successful fitting replaces all learned state. Failed fitting
leaves the prior fitted model intact. Learned public arrays own their storage;
mutating them cannot change the internal prediction buffers.

Kernel names are `linear`, `poly`, `rbf`, `sigmoid`, `precomputed`, and `callable`.
`Gamma` accepts numeric values and `scale`/`auto` strings. Precomputed training
matrices must be square, with prediction columns equal to the training count.
`coef_()` returns the linear-kernel coefficient matrix and raises for other
kernels. The method call is the native adaptation of sklearn's conditional
property. `support_`, `n_support_`, and classifier `n_iter_` use checked Int32 metadata.
Classifiers expose array `n_iter_`; regressors and OneClassSVM expose an integer.
`shape_fit_` is a native `Shape`. Binary classifier decisions are rank one and
positive for `classes_[1]`; multiclass OVO columns follow lexicographic pairs.
OVR confidence adjustment, voting, and `break_ties` follow sklearn's conventions.
OneClassSVM predictions are Int64 +1/-1, with
`decision_function(X) = score_samples(X) - offset_[0]`.

Native callable kernels implement `KernelProvider.evaluate(X, train) -> Dense`.
Set `kernel="callable"`, then call `fit_with_kernel(X, y, provider)` for supervised
estimators, or `fit_with_kernel(X, provider)` for OneClassSVM. Pass the provider
again to `predict(X, provider)`, `decision_function(X, provider)`, or classifier
probability methods. Supervised scores accept `score(X, y, provider)` and
optional sample weights. OneClassSVM also accepts `score_samples(X, provider)`
and `fit_predict(X, provider)`. These provider overloads accept Float32 and
Float64 inputs; provider evaluation receives Float64 matrices.
The estimator retains an independent training snapshot; each result must be a
finite matrix of the exact cross-kernel shape. The provider remains caller-owned,
so the caller must use equivalent provider configuration for fit and inference.
Arbitrary Python callables and Python object lifetime are binding work.

Classifiers accept an integer `random_state`; -1 represents the native None
configuration backed by an estimator-owned entropy stream. All five estimators
also offer `fit_with_random_state` with a mutable shared `RandomState`, consuming
one native seed per fitting call. Integer seeds reset on each fit.

Sparse matrices, Python bindings, metadata routing, reflection, and persistence
remain outside the accepted dense implementation scope. This native API does not
claim Python estimator-check compatibility.

Run the family build and test commands for assembled public imports, native
contracts, and isolated consumers. Run the public kernel parity command for
numerical comparisons against the pinned local scikit-learn checkout.

Nonpositive sample weights remove individual training rows. Removing every
positive-weight row of an entire classification class raises a clear error.
The pinned reference can retain an incompatible public class list when a
multiclass fit loses a whole class; reproducing that inconsistent model state is
not supported. Positive class weights and partial row masking remain supported.
