# Dense-only Fourier scope

Supported public exports: `RBFSampler`, `SkewedChi2Sampler`. Dense float32/float64
NuMojo matrices, seeded and explicit mutable-context fit/fit_transform,
transform, clone, typed parameters, and native feature names are implemented.
RBF includes numeric gamma (including zero) and `scale`; skewed supports any
finite skewedness with the strict transform domain. Tests cover singleton and
rectangular shapes, finite-input checks, noncontiguous views, repeat fits,
parameter changes, ownership, input dimensions and shared RNG draw consumption.
See [API and language adaptations](api.md).

Deferred branch: RBFSampler CSR fit/transform and sparse `gamma="scale"`
variance, present in sklearn/kernel_approximation.py at reference revision
4358542a1bd39af2e7e1cbace8b9fed067473c5b. SkewedChi2Sampler has no sparse branch.
Under the [dense-only scope](../README.md), sparse matrices are unavailable typed inputs;
there is no implicit densification. Foundation sparse storage is deferred and
is absent from the assembled namespace. Neither estimator requires a sparse
setting by default. All five estimators share `moxie.kernel_approximation`.

Default random_state=None requires an application-owned context passed to
fit_with_rng; ordinary fit rejects a missing context. This is the foundation's
explicit adaptation of NumPy's global singleton, not a sparse exclusion.

Required checks: `pixi run build`, `pixi run test`, `pixi run parity`.