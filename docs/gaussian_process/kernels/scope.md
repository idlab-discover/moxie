# Dense Gaussian-process kernel engine scope

Implemented (dense Float32/Float64 input, Float64 arithmetic): ConstantKernel,
WhiteKernel, RBF (isotropic/anisotropic), Matern (nu 0.5/1.5/2.5/inf and general
nu via native Bessel K_nu), RationalQuadratic, ExpSineSquared, DotProduct,
PairwiseKernel (dense string metrics and precomputed matrices), Sum, Product, Exponentiation,
CompoundKernel; hyperparameter metadata, log theta/bounds, set/clone with theta,
nested get/set params, equality, sklearn repr, diag, stationarity, gradients;
Cholesky algebra adapter; L-BFGS-B optimizer adapter and restart policy.

Not supported: callable PairwiseKernel metrics, non-vector
(structured/GenericKernelMixin object) inputs beyond dense arrays, custom Python
kernels. Sparse input (CSR/CSC/COO) is deferred by the session's dense-only
policy (`../session-plan/SPARSITY-SCOPE.md`); nothing is densified implicitly.

Validation: `pixi run test` (native contracts incl. central-difference gradient
checks, algebra/optimizer tests, example), `pixi run parity` (42 kernels x 5
datasets against the pinned sklearn kernels.py: K, K(X,Y), diag, theta, bounds,
repr, hyperparameters, params, clone_with_theta, gradients; tolerances 1e-12
rel / 1e-13 abs Float64, 2e-6 Float32, 2e-5 for forward-difference gradients),
The native benchmark driver is retained under `tests/gaussian_process/kernels/`.

`pixi run parity` also checks precomputed matrices and explicit-Y pairwise
gradients, including fixed gamma, rectangular shapes, composition, Float32,
invalid dimensions, empty inputs and nonfinite values.
