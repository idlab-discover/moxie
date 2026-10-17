# Dense-only Nystroem scope

Supported: `Nystroem` on dense Float32/Float64 NuMojo matrices (including
strided views) with every named kernel, typed callable kernels, precomputed
kernels, typed kernel_params, seeded and explicit-context fit, clone and
native feature names. See [API and language adaptations](api.md).

Deferred branch: CSR input for fit/transform (sparse basis indexing and sparse
pairwise kernels) of sklearn/kernel_approximation.py at revision
4358542a1bd39af2e7e1cbace8b9fed067473c5b; under the [dense-only scope](../README.md)
there is no sparse input or implicit densification. Array API/device dispatch
and joblib `n_jobs` contexts are binding work.

Builds consume live Moxie and mojo-linalg sources. Scheduling and output dtype
adaptations, numerical acceptance tests and limits are documented in the API.
