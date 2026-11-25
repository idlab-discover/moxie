# Gaussian-process API coverage

The reference is local scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. The root assembly exports the actual
regression, classification and kernel implementations. Removed source-only placeholders are not
implementations and must not be used as evidence of parity.

| API | Current implementation | Remaining differences |
| --- | --- | --- |
| GaussianProcessRegressor | Dense Float32/Float64, single/multiple targets, prior/posterior, uncertainty, sampling, LML/gradient, optimization, scoring, clone, named partial parameter updates, and numeric nested kernel routing through `GaussianProcessSearchAdapter` | Training aliasing for copy_X_train=False, heterogeneous parameter search |
| DenseGaussianProcessRegressor | Typed native vector kernels with dense input validation, GP regression and a KernelRidge adapter | Native persistence and built-in search adapter |
| StructuredGaussianProcessRegressor | Native typed input batches and kernels, prior/posterior, uncertainty, sampling, LML/gradient, L-BFGS-B, scoring, clone and parameter updates | Dense search adapter, native persistence, custom optimizers |
| GaussianProcessClassifier | Dense binary Laplace posterior, multiclass OvR/OvO, binary latent moments and LML gradients, optimization and restarts, binary and OvR probabilities, weighted accuracy, and clone | Callable optimizers, training aliasing |
| StructuredGaussianProcessClassifier | Typed kernels on native sample lists, binary and OvR/OvO fits, binary latent moments and LML gradients, optimization, probabilities, weighted accuracy and clone | Typed list of fitted kernels, numeric labels and serial component fits |
| Standard kernels and operators | Dense covariance/gradients, theta/bounds, nested parameters, composition with native custom leaves, metadata, precomputed PairwiseKernel and explicit-Y pairwise gradients | Callable PairwiseKernel |
| Kernel base classes (`Kernel`, `KernelOperator`, `NormalizedKernelMixin`, `StationaryKernelMixin`, `GenericKernelMixin`) | `Kernel` is the type-erased value for built-in kernels and native custom vector leaves. `KernelContract` defines native typed kernels for vector and structured inputs. `k1()`/`k2()` replace `KernelOperator` attributes | Python base classes and mixins |
| Per-hyperparameter properties | Named leaf methods such as `RBF().hyperparameter_length_scale()`; `hyperparameter(name)` supports composed names | Mojo methods for sklearn properties |
| Python integration | Not implemented | Reflection, metadata routing, pandas feature names, pickling and Python bindings |

See [regression details](regression/api.md) and
[kernel details](kernels/api.md) for typed Mojo adaptations,
precision and sampling contracts. Full scikit-learn parity is not claimed.

Run `pixi run test` for the root package and native contracts, and `pixi run parity`
for comparisons against the pinned local source. Passing these suites covers their
listed fixtures, not every possible input or omitted API.
