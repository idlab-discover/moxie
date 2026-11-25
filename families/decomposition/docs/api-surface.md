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
| SparseCoder, sparse_encode | [Threshold and Lasso CD coding](sparse_coding.md); reference defaults require LARS/OMP |
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

Every implemented estimator declares the shared `FeatureNamesOut` and
`InputFeatureCount` traits (`moxie._feature_metadata`), so it works as a
`moxie.pipeline` `Pipeline` transformer and as a `FeatureUnion`/`make_union`
branch, with sklearn's `<step>__<name>` output names and `n_features_in_()`
delegation; `get_n_features_in()` raises an `AttributeError`-prefixed error
before fit, like sklearn's absent attribute. Call
`moxie.model_selection.cross_val_score(estimator, X)` to use the default score
on PCA, FactorAnalysis or LatentDirichletAllocation. Search uses a typed adapter.
`tests/test_composition.mojo` covers PCA regression/classification
pipelines, a PCA + TruncatedSVD union, clone with nested `set_params`, and the
PCA/FactorAnalysis log-likelihood `cross_val_score` and
`GridSearchCV(PCA(), {"n_components": ...})` of
`plot_pca_vs_fa_model_selection.py`. The test also covers a transformer-only
`make_pipeline(StandardScaler(), PCA())` and three-step principal-component
regression. In Mojo the three steps use a nested pipeline, written as
`make_pipeline(StandardScaler(), make_pipeline(PCA(), LinearRegression()))`.

To check targetless CV against the pinned reference, build
`tests/unsupervised_cv/driver.mojo` with `-O3 --fp-mode=contract=off`, the
assembled dependency directory, and `-I tests/support`. Set
`MOXIE_UNSUPERVISED_CV_DRIVER` to the binary path and run
`python tests/unsupervised_cv/parity.py`. The checks cover default, integer
and explicit folds, the estimator's own score, and fitted-state isolation.

The remaining `_api.mojo` declarations retain recorded reference signatures and
raise descriptive NotImplementedError messages. Historical scaffold locks and
the historical `provenance/algorithms/lda` diagnostic child are preserved as provenance;
the current LDA implementation is the root `_lda.mojo` family module.

Use `pixi run test` for package compilation, native contracts, examples and two
isolated consumers. Use `pixi run parity` for differential checks. Current tested
scope and sparse exclusions are in [scope.md](scope.md).
