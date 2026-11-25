# Dense kernel SVM estimators

Import `SVC`, `SVR`, `NuSVC`, `NuSVR`, and `OneClassSVM` from `moxie.svm`.
The five estimators call the native family LIBSVM engine. Constructors retain
scikit-learn's parameter names and defaults, including `gamma="scale"`,
`max_iter=-1`, and the pinned reference's `probability="deprecated"` default.
`probability=True` enables fitted classifier calibration and probability methods.
Disabled probability methods raise `AttributeError`-prefixed Mojo errors; enabling
`probability` after a fit without it raises a `NotFittedError`-prefixed error, as in sklearn.

Matrices use `numojo.NDArray[float64]` or Float32 inputs converted to Float64.
Noncontiguous inputs requiring conversion are packed in logical row order
before casting.
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

Read `fit_warnings` for the last successful fit's native `FitWarning` records.
For SVC and NuSVC, explicitly passing either boolean to `probability` records
the pinned reference's `FutureWarning`. The default sentinel leaves it silent.
A capped solver records `ConvergenceWarning` after any probability warning.
Each record has a `category` and `message`. An unfitted clone has an empty list,
and a failed fit preserves the previous list.

Kernel names are `linear`, `poly`, `rbf`, `sigmoid`, `precomputed`, and `callable`.
`Gamma` accepts numeric values and `scale`/`auto` strings. Precomputed training
matrices must be square, with prediction columns equal to the training count.
`coef_()` returns the linear-kernel coefficient matrix. Like sklearn's property, it
raises an `AttributeError`-prefixed error whenever the current `kernel` parameter is not
`linear` (fitted or not). The method call is the native adaptation of sklearn's conditional
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

Sparse matrices, Python bindings, and reflection
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

## Sample-weight requests

All kernel estimators expose the shared [sample-weight request API](../../../../../docs/interfaces.md#sample-weight-requests).
SVC, NuSVC, SVR, and NuSVR expose separate fit and score requests. OneClassSVM
exposes a fit request through the same setter and returns a single
`SampleWeightRequest` from `get_metadata_routing()`.

```mojo
var model = SVC()
_ = model.set_fit_request(sample_weight="training_weight").set_score_request(
    sample_weight=False
)
var requests = model.get_metadata_routing()
```

`True` requests `sample_weight`, `False` declines it, and `None` resets the
request to unset. An ASCII identifier is accepted as an alias. Omitting the
keyword retains the current request. The getter returns an owned copy, and an
unfitted clone preserves the configured requests. Native setters are always
available. Direct calls to fit and score accept explicit weights independently
of requests. Composition parents own automatic forwarding.

## Feature names

Pass `feature_names=names` with a `List[String]` in column order when fitting
a dense kernel estimator. The fluent fit stores an owned copy in
`feature_names_in_: Optional[List[String]]`. A failed fit preserves the
previous names. A successful unnamed fit clears them. Copies preserve names
and unfitted clones start without them.

```mojo
var names: List[String] = ["length", "width"]
var model = SVC(kernel="rbf")
var predictions = model.fit(X, y, feature_names=names).predict(
    X_query, feature_names=names,
)
```

Named prediction, decision values, scoring, probabilities, and one-class
`fit_predict` and `score_samples` use the same keyword on their supported
methods. The list length must match the number of columns, and named queries
must use the fitted names in the same order when names were recorded.
Named and unnamed calls can be mixed without warnings. For precomputed
kernels, the names identify Gram matrix columns in training-row order.
Callable-kernel fit and provider-based inference use the array directly,
matching the reference callable path.

## Native model files

Use the experimental [model-file API](../../../../../docs/serialization.md)
with `moxie.svm.kernel_serialization`. The schemas are `SVCSchema[Label]`,
`NuSVCSchema[Label]`, `SVRSchema`, `NuSVRSchema`, and
`OneClassSVMSchema`. Match the classifier schema's label type to the
estimator. Supported archive labels are String, Int, Int32, Int64, Float32,
and Float64. Use `SVCSchema[]` or `NuSVCSchema[]` for default Float64 labels.

```mojo
from moxie.serialization import load_model, save_model
from moxie.svm import SVC
from moxie.svm.kernel_serialization import SVCSchema

var model = SVC[String](kernel="rbf", probability=True, random_state=12)
_ = model.fit(X, y)
save_model[SVCSchema[String]](model, "classifier.moxie")
var restored = load_model[SVCSchema[String]]("classifier.moxie")
var probabilities = restored.predict_proba(X_query)
```

Fitted and unfitted archives preserve current constructor parameters and the
estimator-owned random stream. Fitted models restore the resolved kernel,
learned state, and probability calibration without refitting. The loader
rebuilds the public copies of engine arrays from canonical fitted state.
Changes to those copies are local to the in-memory estimator.
Version 4 preserves fitted feature names. Versions 1 through 3 load with names
unset. Versions 3 and 4 preserve sample-weight requests. Versions 1 and 2 load
with unset requests. Versions 2 through 4 restore diagnostics from the original
fit's options, even after changes to `max_iter` or `probability`. Version 1 loads
with an empty warning history.

The archive contains the retained training snapshot, including the Gram
matrix for a precomputed fit. Query columns for a precomputed fit follow
the original training order. For callable kernels, supply the same native
`KernelProvider` again when predicting. The archive stores its fitted
training data, while the caller supplies the provider implementation.

After assembling the live SVM dependencies, compile and run
`families/svm/tests/test_kernel_serialization.mojo` against
`families/svm/builds/deps`, using `families/svm` as the working directory
when running the binary. Compile `families/svm/tests/public_kernel_driver.mojo`
to `families/svm/builds/public_kernel_driver` with
`--fp-mode=contract=off`, then run
`python families/svm/tests/public_kernel_parity.py --round-trip` with the
pinned sklearn oracle environment. It compares the loaded estimators'
public arrays and outputs with exact metadata and label checks and
Float64 tolerances of `rtol=3e-5` and `atol=3e-6` for solver results. Warning
categories, messages, and order are compared exactly.
