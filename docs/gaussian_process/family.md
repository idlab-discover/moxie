# gaussian_process family interfaces

This repository implements dense regression and kernels alongside shared capability
contracts. Classification remains unimplemented. See [API coverage](api-surface.md).
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](../interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `GaussianProcessClassifierAPI` | `ClassifierMixin, ProbabilityPredictor` | Inherited only |
| `GaussianProcessRegressorAPI` | `RegressorMixin` | Inherited only |
| `GaussianProcessPosterior` | `InputEstimator` | `sample_y`, `log_marginal_likelihood` |
| `GaussianUncertainty` | `InputEstimator` | `predict` |

## Behavioral distinctions

GPR supports prior prediction before fit, unlike ordinary predictors. Uncertainty is an opt-in overload (explicit return_std, including False for covariance-only); both flags true must fail. The result type must model mean-only, mean/std and mean/cov results. sample_y is GPR-only; GPC has separate posterior facilities. kernel_ is learned state.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../scikit-learn/sklearn/gaussian_process`.
- Guide and mathematical references: `../scikit-learn/doc/modules/gaussian_process.rst`.
- Shared contracts: `../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`GaussianProcessRegressor`, `GaussianProcessClassifier`, `Hyperparameter`, `Kernel`, `NormalizedKernelMixin`, `StationaryKernelMixin`, `GenericKernelMixin`, `CompoundKernel`, `KernelOperator`, `Sum`, `Product`, `Exponentiation`, `ConstantKernel`, `WhiteKernel`, `RBF`, `Matern`, `RationalQuadratic`, `ExpSineSquared`, `DotProduct`, `PairwiseKernel`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

Read [shared plumbing](plumbing.md) for exact private array/extension contracts,
the kernel/posterior ownership boundary and verified dependency assembly.
[Dense scope](scope.md) defines the tested capabilities and exclusions.
