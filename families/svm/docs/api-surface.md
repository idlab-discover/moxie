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

## Compile and validate

Run `pixi run build`, `pixi run test` and `pixi run parity` from the family root.
Explicit manifests compose the family, engines and algorithms into one verified
namespace. Two isolated consumers compile and execute real estimators against
live Moxie sources. Parity checks compare numerical results with the pinned local
scikit-learn source; compilation alone does not establish parity.
