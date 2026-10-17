# Dense-only polynomial count sketch scope

Supported public export: `PolynomialCountSketch` on dense Float32/Float64 NuMojo
matrices (including strided views) with Float64 output, seeded and
explicit-context fit, typed parameters, clone and native feature names. See
[API and language adaptations](api.md).

Deferred branch: the CSC sparse fit/transform path (`sp.hstack` augmentation
and sparse column sketches) of sklearn/kernel_approximation.py at reference
revision 4358542a1bd39af2e7e1cbace8b9fed067473c5b. Under the [dense-only scope](../README.md) sparse input is unavailable; there is no
implicit densification. Dense input never needs a sparse selection.
