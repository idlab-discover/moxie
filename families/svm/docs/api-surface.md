# Public API status

The family exports SVC, NuSVC, SVR, NuSVR, OneClassSVM, LinearSVC, LinearSVR
and l1_min_c as native implementations. It also exports their typed parameter
records and capability traits. The [kernel API](../algorithms/c_svm/docs/api.md)
and [linear API](../algorithms/linear/docs/api.md) describe signatures, supported
settings, learned arrays and native adaptations.

Fit and parameter setters return a mutable borrow. Cloning copies constructor
configuration into an unfitted estimator. Kernel `coef_()` is a computed method;
native callable kernels use a `KernelProvider` passed explicitly during fitting
and inference. Python reflection, metadata routing and serialization are outside
this native interface. See [scope.md](scope.md) for exclusions and edge limitations.

## Multiclass composition

Wrap a numeric-label SVC in the multiclass family's
[`BinaryAdapter`](../../../docs/multiclass/family.md#binary-adapter-for-numeric-list-labels)
to use it in `OneVsRestClassifier` or `OneVsOneClassifier`.

```mojo
from moxie.multiclass import BinaryAdapter, OneVsOneClassifier
from moxie.svm import SVC

var model = OneVsOneClassifier(BinaryAdapter(SVC(kernel="linear")))
_ = model.fit(X, y)
```

`X` is a Float64 matrix and `y` is an Int64 vector with the default wrapper
parameters. The adapter translates each binary target to SVC's native List
labels. With `kernel="precomputed"`, OvO selects a square submatrix for each
class pair during fit. Prediction takes one column per original training sample
in training order.

`SVC.copy()` copies fitted state, including the engine buffers and any
retained training snapshot. `clone(svc)` copies constructor settings into an
unfitted estimator. The
[tie-breaking notebook](../../../notebooks/svm/plot_svm_tie_breaking.ipynb)
compares OvR and OvO class order and predictions exactly and Float64 decision
values within the family solver tolerance against scikit-learn.

## Compile and validate

Run `pixi run build`, `pixi run test` and `pixi run parity` from the family root.
Explicit manifests compose the family, engines and algorithms into one verified
namespace. Two isolated consumers compile and execute real estimators against
live Moxie sources. Parity checks compare numerical results with the pinned local
scikit-learn source; compilation alone does not establish parity.
