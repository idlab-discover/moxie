# Shared linear algebra scope

Implemented: validated symmetric positive-definite covariance factorization,
reusable vector/matrix solves and inverse-via-solve on NuMojo Float32/Float64.
The factorization/solve kernels remain owned by mojo-linalg. See
[linalg.md](linalg.md) for exact input, ownership, error and numerical contracts.

Not supplied: covariance estimation, general matrix inversion, indefinite or
rank-deficient solves, pseudoinverses, jitter, implicit dtype promotion,
condition estimates, GPU execution, or family integration. Numax toolchain and
storage integration remain a separate adoption task; the current fallback is
pinned and explicitly justified for this runtime-sized CPU contract.
