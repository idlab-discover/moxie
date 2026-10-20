# Dense SVM implementation scope

The public dense family implements `SVC`, `NuSVC`, `SVR`, `NuSVR`,
`OneClassSVM`, `LinearSVC`, `LinearSVR`, and `l1_min_c` in pure Mojo.
The assembled `moxie.svm` package imports actual implementations. No public
estimator is a raising declaration or test double.

Kernel estimators support linear, polynomial, RBF, sigmoid, precomputed, and
native callable kernels; numeric, auto, and scale gamma; sample and class
weights where applicable; shrinking and cache limits; capped fitting; OVO/OVR
presentation and tie handling; and seeded classifier probability calibration.
The LIBSVM owner implements standard and nu SMO, support packing, prediction,
and probability coupling. See [kernel API](../algorithms/c_svm/docs/api.md)
and [engine scope](../shared/libsvm/docs/scope.md).

The linear estimators retain their implemented solver modes, intercept behavior,
class/weight handling, and `l1_min_c` bounds. Their restrictions are in the
[linear scope](../algorithms/linear/docs/scope.md).

## Native adaptations and exclusions

- Inputs are dense Float64 or Float32 matrices. Sparse storage, dispatch,
  sparse coefficient conversions, and the sparse `l1_min_c` branch are deferred.
- Kernel `coef_()` is a method, with an error for nonlinear kernels. Native
  callable kernels use an explicit `KernelProvider` for fit and inference.
- Typed parameter records replace dynamic Python dictionaries. Native errors
  use descriptive prefixes; convergence diagnostics print a warning message.
  Python warning filters and deprecation-warning machinery are not reproduced.
- Classifiers reject weights that remove every positive-weight sample from a
  class. This avoids upstream's inconsistent original-label/compacted-model
  behavior when only some classes disappear. Individual nonpositive-weight
  samples are masked normally.
- Learned arrays own separate storage. Mutating public arrays does not mutate
  the prediction model. Integer random seeds reset per fit; None uses a native
  estimator-owned stream, rather than Python's process-global stream.
- Python bindings, feature-name metadata, reflection, metadata routing, pandas
  output, Python callable objects, and persistence remain separate work.

These are explicit limits, not claims of full Python sklearn compatibility.

## Verification and migration

`pixi run build` assembles live Moxie sources and compiles the public package.
`pixi run test` runs contracts, real kernel lifecycle tests, and two isolated
consumers. `pixi run parity` builds and compares both engines and public estimator
groups against the pinned local sklearn source. Numerical tolerances and fixture
coverage are recorded with each reproducible suite; they are not universal
accuracy or performance guarantees.

See [migration inputs](migration.md) for module ownership and required checks.
Moxie owns implementation and validation under `families/svm`; the retired
standalone checkout is not a source dependency.
