# Decomposition in Moxie

Pure Mojo `sklearn.decomposition` for dense Float32/Float64 NuMojo arrays.
Derived from moxie `scaffold-v1.1`; reference: local scikit-learn
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

| Estimator | Status | Guide |
| --- | --- | --- |
| `PCA` | full, covariance_eigh, arpack (dense-exact), randomized, auto; whiten, MLE/fraction ranks, scores | [pca.md](docs/pca.md) |
| `IncrementalPCA` | fit, partial_fit, batches, whiten | [incremental_pca.md](docs/incremental_pca.md) |
| `TruncatedSVD` | randomized, arpack (dense-exact) | [truncated_svd.md](docs/truncated_svd.md) |
| `FactorAnalysis` | lapack/randomized SVD EM, varimax/quartimax, scores | [factor_analysis.md](docs/factor_analysis.md) |
| `FastICA`, `fastica` | parallel/deflation, logcosh/exp/cube, all whiten modes | [fastica.md](docs/fastica.md) |
| `NMF`, `MiniBatchNMF`, `non_negative_factorization` | cd and mu solvers, all beta losses, inits, online updates | [nmf.md](docs/nmf.md) |
| `SparseCoder`, `sparse_encode` | threshold only; regression solver modes remain unavailable | [sparse_coding.md](docs/sparse_coding.md) |
| `KernelPCA` | centered named/precomputed kernels, dense/randomized eigenspaces, learned inverse maps | [kernel_pca.md](docs/kernel_pca.md) |
| `LatentDirichletAllocation` | batch/online variational Bayes, partial_fit, transform, score/perplexity | [lda.md](docs/lda.md) |
| `randomized_svd` | gesdd (Jacobi) and gesvd (Golub-Kahan) drivers | [svd_engine](shared/svd_engine/docs/api.md) |

The remaining exports (dictionary learning, SparsePCA and their functions)
are raising API declarations; see [scope](docs/scope.md).

```sh
pixi install --locked
pixi run build     # assemble current Moxie dependencies and compile
pixi run test      # native contracts, examples, two isolated consumers
pixi run parity    # differential checks against the pinned sklearn source
pixi run start     # PCA example
```

Set `MOXIE_ESTIMATORS=pca,fastica` to build/check a subset. A baseline
timing run is `pixi run python tools/benchmark_family.py -o benchmarks/family.json`.
Compile-time training shapes can additionally use the opt-in Numax backend
(`pixi run -e numax numax-start`, Mojo 1.1).

- [Family interfaces and reference sources](docs/family.md)
- [Dense plumbing, ownership and dependency assembly](docs/plumbing.md)
- [Supported dense scope and sparse exclusions](docs/scope.md)
- [Shared estimator contract and Mojo differences](docs/interfaces.md)
- [Contribution rules](AGENTS.md)

From Moxie, use `pixi run decomposition-build`, `decomposition-test`,
`decomposition-parity`, or `decomposition-start`. The standalone checkout is not
a build dependency. Internal engine checks use
`pixi run python families/decomposition/shared/svd_engine/tools/check.py test`
and `parity`. Historical scaffold inputs are retained under [provenance](provenance/README.md).

See [validation](docs/validation.md) for rollup commands, outcomes and independence evidence.
