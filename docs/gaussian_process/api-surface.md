# Gaussian-process API coverage

The reference is local scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. The root assembly exports the actual
regression and kernel implementations. Removed source-only placeholders are not
implementations and must not be used as evidence of parity.

| API | Current implementation | Remaining differences |
| --- | --- | --- |
| GaussianProcessRegressor | Dense Float32/Float64, single/multiple targets, prior/posterior, uncertainty, sampling, LML/gradient, optimization, scoring, clone, named partial parameter updates | Structured/custom kernels, training aliasing for copy_X_train=False, nested parameter routing |
| GaussianProcessClassifier | Not implemented | Binary Laplace posterior, latent moments, optimization, multiclass through the multiclass owner, probability and score contracts |
| Standard kernels and operators | Dense covariance/gradients, theta/bounds, nested parameters, composition, metadata, precomputed PairwiseKernel and explicit-Y pairwise gradients | Native custom/structured kernels, callable PairwiseKernel |
| Kernel base classes (`Kernel`, `KernelOperator`, `NormalizedKernelMixin`, `StationaryKernelMixin`, `GenericKernelMixin`) | `Kernel` is the type-erased value every kernel converts to; `k1()`/`k2()` replace `KernelOperator` attributes | No user-defined kernel extension point: the abstract base and mixins are not exported, so custom kernels cannot be written |
| Per-hyperparameter properties | `hyperparameters()` returns every `Hyperparameter` in sklearn order | `hyperparameter_<name>` properties (e.g. `RBF().hyperparameter_length_scale`) are not provided; select by `.name` instead |
| Python integration | Not implemented | Reflection, metadata routing, pandas feature names, pickling and Python bindings |

See [regression details](regression/api.md) and
[kernel details](kernels/api.md) for typed Mojo adaptations,
precision and sampling contracts. Full scikit-learn parity is not claimed.

Run `pixi run test` for the root package and native contracts, and `pixi run parity`
for comparisons against the pinned local source. Passing these suites covers their
listed fixtures, not every possible input or omitted API.
