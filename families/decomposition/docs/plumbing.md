# Dense decomposition plumbing

Current integration includes [KernelPCA](kernel_pca.md), using the current Moxie
KernelRidge numerical owner, [LDA](lda.md), and [threshold sparse coding](sparse_coding.md)
from its family-owned shared engine. The root family environment builds these
modules without creating separate child environments.

Earlier integration: [dense IncrementalPCA](incremental_pca.md) shares PCA
inference helpers and has a bounded-batch Numax SVD adapter.
[dense PCA](pca.md) lives in `src/moxie/decomposition/_pca.mojo`
and consumes the existing shared SVD engine plus current Numax/mojo-linalg owners.
An opt-in `backends/numax/moxie_numax` module supplies static factorizations through
`_pca_backend.PCABackend`; the estimator and public NDArray contract are shared.
Its separately pinned Mojo 1.1/MAX 26.6 environment applies the recorded NuMojo
syntax patch only to a generated dependency copy.
The organizational child folders do not impose separate public namespaces.
Moxie's root `DEPENDENCIES.json` assembles the current family API through `tools/family.py`.
The Numax-first policy in `../session-plan/NUMERICS.md` supersedes the older
operation restrictions below; PCA-specific compatibility findings are in pca.md.


The historical scaffold owns shared validation, ownership boundaries and
extension contracts. The current family assembly also integrates the estimators
listed in [scope.md](scope.md).
The unchanged public capability traits remain in `moxie.decomposition._base`:
`DecompositionTransformer`, `InvertibleDecomposition`, `IncrementalDecomposition`.
The original scaffold exported only these traits; private `_BasePCA` and
`_BaseNMF` in sklearn are not additional public estimators.

## Storage and exact interfaces

Use the accepted `numojo.NDArray[DType.float32]` or `NDArray[DType.float64]`,
`Shape` and `Item` directly. Dtype is static and shared across an engine call;
there is no implicit promotion. Input X has shape `(n_samples, n_features)`.
Components/dictionary H has shape `(n_components, n_features)` and coefficients
W have shape `(n_samples, n_components)`. Means, singular values and variances
are rank-1 arrays. KernelPCA's eigenvectors are `(n_training_samples, k)` and
are not feature-space components. LDA components are `(topics, vocabulary)`
Dirichlet parameters, not orthonormal loadings or normalized topic probabilities.

The scaffold's `moxie.decomposition._contracts` exports these private helpers:

| Interface | Behavior |
| --- | --- |
| `validate_matrix[dtype](x, *, min_samples: Int = 1, nonnegative: Bool = False) raises` | Float32/64, rank 2, at least min_samples rows, positive feature count, finite values; optional nonnegativity; no copy or mutation |
| `owned_matrix[dtype](x) raises -> NDArray[dtype]` | Same checks with min_samples=0, then independent `.copy()` |
| `validate_projection[dtype](x, components) raises` | Nonempty X and finite `(k,p)` components with matching p; k=0 allowed; performs no projection |
| `validate_mean[dtype](mean, n_features: Int) raises` | Positive feature count, finite rank-1 mean of exactly that length |
| `SVDResult[dtype](U, singular_values, Vt)` | Movable/copyable owned record `(n,k)`, `(k)`, `(k,p)` |
| `EigenResult[dtype](eigenvalues, eigenvectors)` | Movable/copyable owned record `(k)`, `(p,k)` |

Result record construction is structural, not a numerical certificate. The
engine must validate shapes, descending singular/eigenvalue order, nonnegative
singular values, orthogonality, residuals and finite values before publishing a
result. Eigenvalues can be negative; PSD clipping is estimator-specific.
Repeated eigenvalues require subspace comparisons, not elementwise equality.
Owned arrays passed with `^` move; `.copy()` of a record copies its arrays.
No fitted status is implied by constructing a record.

Borrow computation inputs. Explicit `.view()` aliases and retains the original
buffer lifetime. `.copy()` is independent even for strided inputs, but can copy
the full backing allocation rather than just logical elements. Valid positive
or negative strides and nonzero offsets are accepted; strides count elements.
Shape/stride metadata must satisfy NuMojo's own invariants. Arbitrary layout
construction stays in shared storage code and tests. Transpose, slicing,
reshape and astype can allocate; never assume they are views. Fitted snapshots
must not retain mutable caller aliases. Concurrent mutation is unsupported.
Validation costs O(n*p) time and O(1) auxiliary storage; snapshot cost follows
the backing allocation. There is no speed or zero-copy claim.

`FittedState` remains owned by `moxie.utils.validation`. Normal successful fits
replace learned buffers and feature metadata. Clone copies constructor settings
into an unfitted object; copying result records is a different operation.
Estimator owners define failed-fit, warm-start and partial-fit behavior against
sklearn; the helpers do not impose a universal transactional fit policy.
Shape validation does not enforce rank bounds, solver-specific minima or NMF's
nonzero initialization requirement. Those checks belong to the caller.

## Numerical boundaries and extension points

`DecompositionEngine` is a native trait with associated `Input: Movable`,
`Params: Copyable`, `State: Movable`, `Output: Movable` and signature
`run(mut self, x: Self.Input, params: Self.Params, mut state: Self.State)
raises -> Self.Output`. It is an internal typed dispatch seam, not a public
estimator API. Engines provide separate concrete implementations and typed
parameter/state records; fitting, transformation and incremental capability
traits remain independent. A raising compile probe checks this seam only.

| Engine owner | Input/output and state contract |
| --- | --- |
| SVD/eigen/range | Dense matrix; SVDResult or EigenResult; range basis `(n,l)` where l is requested workspace rank. Solver settings live in Params; borrowed mutable RandomState is carried by State. Full, truncated, randomized and eigen methods remain distinct calls/implementations. |
| Sparse coding/dictionary | Dense X and dictionary `(k,p)` plus optional Gram `(k,k)`, covariance `(k,n)`, initial code `(n,k)` in a typed input record; output dense code `(n,k)`. State owns reusable workspace and mutable RNG where used. Dictionary update state owns accumulated sufficient statistics and batch counters. |
| NMF | Dense nonnegative X, optional W `(n,k)` and H `(k,p)`; output owned W, H, iteration/convergence metadata. State owns solver workspace, RNG and mini-batch accumulators. Update-H and transform-only behavior stay explicit. |
| Variational topics | Dense nonnegative counts X; output document-topic matrix `(n,k)` and iteration/convergence metadata; State owns topic parameters `(k,p)`, expected-log caches, RNG and online batch counters. Bound/perplexity and normalized transform are distinct operations. |

New Params preserve each sklearn constructor/function name, default and meaning,
including tagged alternatives such as integer/fraction/"mle" component counts.
Each owner specifies these records and tests before implementation; this scaffold
freezes data orientation and ownership, not invented estimator defaults.
Intermediate workspaces may be specialized structures but numeric arrays stay
NuMojo-owned. Common numerical changes require their single owner's session,
not local copies of `_dense`, RNG, buffers or solvers.

Centering is columnwise `mean_(p)`; PCA/IncrementalPCA center, TruncatedSVD does
not. The accepted baseline uses explicit-dtype NuMojo mean and broadcast
subtraction. Projection is `X @ components_.T - mean_ @ components_.T` when
centering is required. A kernel may fuse this only with correctness/cost evidence.
PCA whitening divides by `max(sqrt(explained_variance_), eps(dtype))` in transform;
inverse uses `Z @ (sqrt(variance)[:,None] * components_) + mean_`. Without
whitening, inverse is `Z @ components_ + mean_`. Do not apply these formulas to
NMF, coding or LDA transforms: those are inference engines. PCA covariance,
precision and scoring remain its owner's work. SparsePCA uses regularized
least squares, not an orthogonal projection.

SVD sign orientation belongs once in the SVD engine: sklearn `svd_flip` uses
first maximal-absolute loading in each U column by default, or Vt row for
`u_based_decision=False`; flip paired U/Vt signs together. PCA uses Vt decisions,
dictionary initialization uses U decisions, randomized_svd preserves its
public flip_sign/transpose semantics. No universal sign normalization is
applied to NMF/topics, and ICA has its own conventions. Implement sign handling
and centering kernels in their engine sessions; no hidden engine is supplied here.

Shared arithmetic comes directly from `moxie/shared/numerics`: `sum_axis2d`
(Float64 accumulation, same-dtype output), `row_squared_norms`, and its approved
NuMojo operation subset. General NuMojo axis sum, dot, variance/ddof, solve,
inverse, and RNG remain excluded. Use `moxie._random.RandomState`, matching
integer reset versus borrowed mutable stream semantics. A None-like random
state uses the application-owned context documented by that owner. QR/SVD/eigen
are not available from `_dense`; their dedicated future engine must supply
validated implementations. No new operation-specific exception is taken here.

## Ownership and dependency graph

Paths below are future session destinations, not child projects created by this
scaffold. All rows consume `decomposition.scaffold`; additional dependencies
are listed explicitly. The exports column is the future public API inventory,
verified against sklearn/decomposition/__init__.py.

| Owner path / task suffix | Public exports | Additional dependencies |
| --- | --- | --- |
| `shared/svd_engine/` / svd_engine | randomized_svd | none |
| `algorithms/lda/` / lda | LatentDirichletAllocation | none |
| `algorithms/pca/` / pca | PCA | decomposition.svd_engine |
| `algorithms/truncated_svd/` / truncated_svd | TruncatedSVD | decomposition.svd_engine |
| `algorithms/kernel_pca/` / kernel_pca | KernelPCA | decomposition.svd_engine, preprocessing.simple, kernel_ridge.regression |
| `algorithms/nmf/` / nmf | NMF, MiniBatchNMF, non_negative_factorization | decomposition.svd_engine |
| `algorithms/ica/` / ica | FastICA, fastica | decomposition.svd_engine |
| `algorithms/factor_analysis/` / factor_analysis | FactorAnalysis | decomposition.svd_engine |
| `algorithms/incremental_pca/` / incremental_pca | IncrementalPCA | decomposition.pca |
| `shared/sparse_coding/` / sparse_coding | SparseCoder, sparse_encode | linear_model.lars, linear_model.elastic_net, linear_model.omp |
| `algorithms/dictionary/` / dictionary | DictionaryLearning, MiniBatchDictionaryLearning, dict_learning, dict_learning_online | decomposition.sparse_coding |
| `algorithms/sparse_pca/` / sparse_pca | SparsePCA, MiniBatchSparsePCA | decomposition.dictionary |

Native modules follow sklearn `_pca`, `_truncated_svd`, `_kernel_pca`, `_nmf`,
`_fastica`, `_factor_analysis`, `_incremental_pca`, `_lda`, `_sparse_pca`.
The dictionary owner owns `_dict_learning`; sparse coding uses `_sparse_coding`
and re-exports the public SparseCoder/sparse_encode names from decomposition.
This split avoids two owners for sklearn's monolithic `_dict_learning` module.
The SVD owner uses decomposition `_svd_engine`; randomized_svd is re-exported
from decomposition as in sklearn (whose implementation lives in utils.extmath).
Shared helpers stay private; initializers export only approved public names.
NMF batch/incremental models share one update engine; dictionary batch/online
and SparsePCA share the coding/dictionary engine, and IncrementalPCA reuses PCA
projection contracts plus the transitive SVD owner.

## Dependency assembly and native compatibility

Moxie's `DEPENDENCIES.json` owns the assembled public namespace. This family
contributes `src/moxie/decomposition`, the shared SVD engine and sparse coding.
Generic contracts, numerical storage, RNG and KernelRidge kernels resolve to
current Moxie owners. Mojo-linalg uses Moxie's existing compatible owner;
no standalone family checkout is consumed. Source hashes in provenance are
historical evidence, not build gates.

The default toolchain remains Mojo 1.0.0/MAX 26.5.0 with
`-O3 --fp-mode=contract=off`. The optional Numax backend retains its separate
Mojo 1.1/MAX 26.6 Pixi lock and generated-copy NuMojo compatibility patch.
`DEPENDENCIES.json` in this directory is the minimal manifest for that backend.

From Moxie run `pixi run decomposition-test` and `pixi run decomposition-parity`.
The native gate compiles the assembled public namespace, all nine estimator
groups and examples, and two isolated consumers. Engine checks additionally
cover both SVD drivers and randomized orchestration. Per-estimator guides
record tolerances and unsupported settings.
