# Scope and validation (dense only)

Implemented: PLSSVD fit/transform (single and paired)/fit_transform, scale on
and off, copy semantics, 1-D and 2-D y, Float32/Float64. Sparse input is
excluded (the dense-only porting scope; the reference has no sparse
branch). Python-only surfaces (pandas output, feature_names_in_) are binding
work.

The migrated suite contains five estimator contracts and 93 parity cases
against the recorded `_pls.py`, all datasets of the PLS child, k in
{1, 2, rank}, scale on/off, Float32, copy=False end state, 5 error cases.
Tolerance 1e-10 scaled (observed <= 4.7e-14), 1e-5 for tiny/ill-conditioned
cases. Components beyond the rank of X.T @ y are checked for
orthonormality only; exact sign-tie or perturbation-sensitive references are
compared by score spectra.

Current Moxie commands and report locations are documented in
[the family guide](../README.md). The native benchmark program is retained
under `tests/cross_decomposition/plssvd/benchmark.mojo`; performance has not
been remeasured during integration.
