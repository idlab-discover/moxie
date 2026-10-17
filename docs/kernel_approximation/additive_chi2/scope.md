# Dense-only additive chi-squared scope

Supported public export: `AdditiveChi2Sampler`, dense Float32/Float64 NuMojo
matrices (including strided views), stateless transform, fit-time parameter and
domain validation, feature-count checks after fit, typed parameters, clone and
native feature names. See [API and language adaptations](api.md).

Deferred branch: the CSR `_transform_sparse` path (sparse input and sparse
`hstack` output) of sklearn/kernel_approximation.py at reference revision
4358542a1bd39af2e7e1cbace8b9fed067473c5b. Under the [dense-only scope](../README.md),
sparse matrices are unavailable typed inputs; there is no implicit
densification and no sparse output. The default configuration needs no sparse
selection: dense input always produces the dense map.
