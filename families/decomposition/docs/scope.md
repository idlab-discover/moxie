# Current dense scope

Implemented for Float32/Float64 runtime matrices, each with build/test/parity:
PCA (full, covariance_eigh, dense-exact arpack, randomized, auto), IncrementalPCA,
TruncatedSVD (randomized, dense-exact arpack), FactorAnalysis (lapack and
randomized SVD, varimax/quartimax), FastICA/fastica, NMF/MiniBatchNMF/
non_negative_factorization (cd and mu, all beta losses), LatentDirichletAllocation
(batch/online, sequential E-step; see [lda.md](lda.md)), KernelPCA
(named/precomputed kernels, dense/dense-exact arpack/randomized eigenspaces,
learned inverse maps; see [kernel_pca.md](kernel_pca.md)), SparseCoder/sparse_encode
(threshold only; see [sparse_coding.md](sparse_coding.md)), and randomized_svd (both
drivers) from `shared/svd_engine`. See the per-estimator guides linked from the
README for adaptations and tolerances.

"arpack" is served by an exact truncated factorization: on dense input the
leading singular triplets are what ARPACK converges to, so every `tol` is
satisfied; no Lanczos iteration is performed.

Not yet implemented (raising declarations): DictionaryLearning, MiniBatchDictionaryLearning, dict_learning,
dict_learning_online, SparsePCA and
MiniBatchSparsePCA (integration with the current Moxie linear-model LARS/Gram-coordinate/OMP solver contracts is unfinished).

SparseCoder/sparse_encode regression modes (lasso_lars/lasso_cd/lars/omp) are
unavailable; their reference defaults therefore require an explicit threshold
selection. These are unfinished dense modes, not sparse-container deferrals.

Historical performance measurements for PCA/SVD/FA/ICA/NMF: the native kernels are scalar and certify every
factorization; on 200x10 to 1000x20 inputs fits are 20-100x slower than
sklearn with LAPACK (`benchmarks/family.json`, single thread). No speed claim
is made; optimization is separate work.

Sparse inputs and Python-only facilities (callable `fun`, warnings, metadata
routing, pandas output) remain deferred.

Sparse inputs and outputs are unsupported; there is no implicit densification.
Dense coefficient sparsity in dictionary learning and SparsePCA remains unfinished.
The original scaffold documentation is available in the standalone Git history.

## Current family validation

The current assembled family passed `pixi run test`: package precompilation,
all nine algorithm-group contract programs and examples, and two isolated
consumers exercising the implemented APIs. Differential checks were then run
without rebuilding those binaries using:

```sh
pixi run python -c 'import sys; sys.path.insert(0,"tools"); import family; family.parity()'
```

All 1,806 cases passed: PCA 349, IncrementalPCA 318, TruncatedSVD 358,
FactorAnalysis 186, FastICA 180, NMF/MiniBatchNMF 156, LDA 60, KernelPCA 55,
and threshold sparse coding 144. `pixi run parity` runs these same checks after
building. Tolerances and coverage limits are in the individual guides. This is
partial dense coverage; dictionary learning and SparsePCA remain unfinished.

Persistence is not implemented: no decomposition estimator defines a
`ModelSchema` for the shared `save_model`/`load_model` (see
`docs/serialization.md`), so fitted models cannot be saved and reloaded yet.
