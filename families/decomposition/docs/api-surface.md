# Public API surface

The reference is local scikit-learn
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Public imports come from the single
namespace assembled by Moxie's root `DEPENDENCIES.json` and `tools/family.py`.
The family and engine export manifests name each module's owner; importing a
reserved declaration does not mean its algorithm is implemented.

| Public API | Implemented modes and native adaptations |
| --- | --- |
| PCA | [Dense solvers, whitening, covariance and scores](pca.md) |
| IncrementalPCA | [Dense fit/partial_fit and inference](incremental_pca.md) |
| TruncatedSVD | [Randomized and dense-exact arpack](truncated_svd.md) |
| FactorAnalysis | [EM, rotations and scores](factor_analysis.md) |
| FastICA, fastica | [Whitening and parallel/deflation ICA](fastica.md) |
| NMF, MiniBatchNMF, non_negative_factorization | [Dense CD/MU and online updates](nmf.md) |
| LatentDirichletAllocation | [Batch/online variational Bayes and scores](lda.md) |
| KernelPCA | [Named/precomputed kernels and learned inverse maps](kernel_pca.md) |
| SparseCoder, sparse_encode | [Threshold coding only](sparse_coding.md); reference default regression solvers unavailable |
| DictionaryLearning, MiniBatchDictionaryLearning, dict_learning, dict_learning_online | Raising declarations; dense fitting solvers and dictionary optimization remain unfinished |
| SparsePCA, MiniBatchSparsePCA | Raising declarations; dense dictionary-fitting prerequisites remain unfinished |

The family also exports randomized_svd from its shared SVD engine and typed
parameter/result records for implemented estimators. Capability traits remain
in `_base`; numerical storage and generic estimator contracts remain external
foundation-owned dependencies.

Implemented estimators use typed NuMojo inputs and parameter records, not the
historical opaque APIValue declarations. Fit and set_params mutate in place;
learned array fields are generally Optional owned arrays. See each guide for
exact dtypes, defaults, validation, clone and unsupported settings. Python
reflection, metadata routing and pandas output are not implemented.

The remaining `_api.mojo` declarations retain recorded reference signatures and
raise descriptive NotImplementedError messages. Historical scaffold locks and
the historical `provenance/algorithms/lda` diagnostic child are preserved as provenance;
the current LDA implementation is the root `_lda.mojo` family module.

Use `pixi run test` for package compilation, native contracts, examples and two
isolated consumers. Use `pixi run parity` for differential checks. Current tested
scope and sparse exclusions are in [scope.md](scope.md).
