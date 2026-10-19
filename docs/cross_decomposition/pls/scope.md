# Scope and validation (dense only)

Implemented: PLSRegression, PLSCanonical (nipals and svd), CCA, with every
native branch of the reference `_PLS.fit/transform/inverse_transform/predict/
fit_transform/score`: modes A/B, both deflation modes, scale=True/False,
copy semantics, 1-D and 2-D targets, max_iter/tol diagnostics, constant
residual early stop, Float32/Float64 inputs. Sparse input is excluded by
the dense-only porting scope (the reference has no sparse branch either);
it is never densified. Python-only surfaces (pandas set_output,
feature_names_in_, routing, pickling) are binding work.

The migrated suite contains 15 estimator contracts, six engine tests and
124 parity cases against the recorded
`_pls.py` executed from the reference checkout: tall/wide/tiny/two-sample,
ill-conditioned, constant columns/targets, 1-D y, max_iter=2, full-rank,
copy=False paths, Float32 and 11 invalid-input/parameter error cases.
Tolerance 1e-10 scaled (observed <= 1.1e-13); ill-conditioned cases 1e-5
(observed <= 1.3e-8, reference self-sensitivity ~1e-6). 12 cases whose
reference changes by more than its tolerance under a 1e-13 input
perturbation (degenerate canonical correlations, exact sign ties, rank below
k) are compared by shapes, finiteness and per-component objective values.

Current Moxie commands and report locations are documented in
[the family guide](../README.md). The native benchmark program is retained
under `tests/cross_decomposition/pls/benchmark.mojo`; performance has not
been remeasured during integration.
