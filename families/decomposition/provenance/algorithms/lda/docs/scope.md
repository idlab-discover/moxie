# Dense LDA scope

`LatentDirichletAllocation` is planned, but is not implemented or exported.
There are currently no supported estimator methods or validated settings.
The required scope is Float32/Float64 dense nonnegative counts, batch and online
fit, partial_fit, normalized/unnormalized transform, fit_transform, score,
perplexity, priors, convergence, and seeded/shared RNG lifecycle behavior.

CSR/CSC/COO inputs and sparse count branches are deferred by the workspace's
dense-only policy. See scikit-learn `sklearn/decomposition/_lda.py`, especially
`_update_doc_distribution` and `_check_non_neg_array`. Inputs will not be silently
densified. No sparse-specific constructor setting is needed for dense LDA.
The inherited scaffold accepts only dense NuMojo arrays; Python bindings,
metadata routing, pandas adapters and serialization remain separate work.

Dense implementation is blocked by the shared RNG owner's missing legacy
RandomState Gamma sampler. Both component initialization and training E-steps
require Gamma(100, 0.01), including the default batch setting. Replacing those
draws with uniform or normal values would change the algorithm.

`pixi run reproduce` tests a working seeded RNG import and the missing Gamma API.
It is a dependency diagnosis, not a sparse rejection test or LDA parity check.
Required commands `pixi run build`, `pixi run test`, and `pixi run parity` fail
closed until implementation and real acceptance checks replace the diagnostic
gates. Dense numerical validation and sparse negative-capability tests remain
unexecuted because no estimator exists.
