# Scope

Implemented: scalar CPU regularized lower incomplete gamma for Float32/Float64,
with domain validation and explicit convergence errors. This is a shared private
numerical operation, not a public estimator or general SciPy special module.

Not supplied: upper/log/inverse gamma APIs, complex numbers, derivatives,
SIMD lanes, GPU kernels, array broadcasting, or family integration. Existing
NuMojo operation exclusions remain applicable.
