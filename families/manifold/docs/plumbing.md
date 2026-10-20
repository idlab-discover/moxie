# Manifold plumbing

This directory owns manifold contracts, validation, ClassicalMDS and trustworthiness.
Existing `EmbeddingEstimator` and `OutOfSampleEmbedding` remain unchanged.
Public imports follow sklearn.manifold; `_plumbing` is a private native exchange
module. Its records are not new sklearn public exports.

## Ownership and future contributions

| Owner / child directory | Public exports / responsibility | Dependencies |
| --- | --- | --- |
| manifold.scaffold (this project) | `_base`, `_plumbing`, dense ownership and exchange validation | foundation.readiness, numerics, random, assembly |
| `algorithms/mds/` | MDS, ClassicalMDS, smacof; stress/disparities and classical scaling | scaffold, isotonic.regression, decomposition.svd_engine |
| `algorithms/spectral/` | SpectralEmbedding, spectral_embedding; affinities, Laplacians, connectivity diagnostics | scaffold, neighbors.nearest, decomposition.svd_engine |
| `algorithms/lle/` | LocallyLinearEmbedding, locally_linear_embedding; local reconstruction | scaffold, neighbors.nearest, decomposition.svd_engine |
| `algorithms/tsne/` | TSNE, trustworthiness; exact probabilities, KL optimization and rank diagnostics | scaffold, neighbors.nearest, decomposition.pca |
| `algorithms/isomap/` (deferred) | Isomap; graph shortest paths | scaffold, neighbors.nearest, decomposition.kernel_pca |

The child directories in this table describe planned organization; no algorithm child
projects currently exist. Implemented algorithms live in `src/moxie/manifold`. Use `_mds`, `_classical_mds`,
`_spectral_embedding`, `_locally_linear`, `_t_sne` module boundaries as in sklearn.
The shared private adapter module is a deliberate native adaptation: it fixes
storage and ownership without combining unrelated optimizers. No stress, KL,
reconstruction, shortest-path, affinity-building or spectral engine lives here.

## Exact dense exchange contract

All numeric matrices use `numojo.NDArray[dtype]`, dtype float32 or float64 as a
compile-time parameter. `Shape` and `Item` are NuMojo metadata. Matrices have
sample rows and feature/component columns. Rank, shape, finite values and
semantics are validated separately; there is no implicit dtype promotion.
Inputs are borrowed. Both contiguous and valid strided NuMojo arrays (element
strides and offset) are accepted; `.unsafe_get` includes the upstream offset.
No family code constructs arbitrary layouts. Tests construct views to verify
that validation and snapshots honor the foundation contract.

Constructors below deep-copy their arrays with `.copy()`. This preserves the
upstream layout and may copy the entire backing allocation for a small view.
Returned records move. Their fields are mutable native arrays; consumers must
revalidate after direct mutation and must not assume a permanently checked type.
Explicit `.view()` aliases mutation and retains storage lifetime. Public fitted
state and clone configuration must not accidentally retain mutable input aliases.
Concurrent mutation is outside the contract. No generic container is introduced.

| Import from `moxie.manifold._plumbing` | Signature and invariant |
| --- | --- |
| `validate_matrix[dtype]` | `(x: NDArray[dtype]) raises`; finite nonempty rank two |
| `validate_pairwise[dtype]` | `(x, *, nonnegative=False, zero_diagonal=False, symmetric=True, tolerance: Float64=0) raises`; square matrix, optional strict nonnegativity, absolute diagonal/symmetry tolerance; tolerance finite and nonnegative |
| `DenseAffinity[dtype]` | `(x: NDArray[dtype]) raises`; owns `matrix`, nonnegative symmetric square; diagonal preserved |
| `DenseNeighborhood[dtype]` | `(indices: NDArray[int64], distances: NDArray[dtype], n_reference: Int) raises`; owns matching nonempty `(n_query, k)` arrays, finite nonnegative distances, indices in `[0,n_reference)` |
| `EmbeddingResult[dtype]` | `(coordinates: NDArray[dtype], n_samples: Int, n_components: Int) raises`; owns finite `embedding` exactly `(n_samples,n_components)` |
| `DenseSpectralProblem[dtype]` | `(operator: NDArray[dtype], n_components: Int, skip: Int=0) raises`; owns finite symmetric signed square operator, positive component count, nonnegative skip, sum at most number of rows |
| `DenseNeighborQuery` | associated `dtype: DType`; `query(self, x: NDArray[Self.dtype], n_neighbors: Int) raises -> DenseNeighborhood[Self.dtype]` |
| `DenseSpectralSolver` | associated `dtype: DType`; `solve(self, problem: DenseSpectralProblem[Self.dtype]) raises -> EmbeddingResult[Self.dtype]` |

Neighborhood rows preserve query order; columns preserve the provider's ordering
and ties. The adapter neither sorts, removes self-neighbors, deduplicates, nor
squares distances. Distances are ordinary metric distances, not affinities.
The caller explicitly chooses whether queries include training samples and asks
the neighbor owner for the required self-exclusion semantics. Duplicate indices
are representable; algorithm-specific restrictions belong to the child.
No sparse graph is constructed from the table.

Pairwise values retain their input meaning. MDS dissimilarities, squared t-SNE
distances, affinities and signed reconstruction operators are not interchangeable.
`validate_pairwise` is a building block, not sklearn's `check_symmetric`: it does
not warn, repair asymmetry, normalize diagonals, or symmetrize. Children reproduce
reference repair/warning behavior before entering a strict adapter. Zero-distance
nonmetric MDS missing-pair semantics belong to MDS, not this validator.

The spectral seam accepts an already-constructed symmetric operator, requests the
smallest algebraic eigenvectors in ascending eigenvalue order, discards `skip`,
and returns the next `n_components` columns in `EmbeddingResult`. Columns are
orthonormal before any algorithm-specific normalization. The decomposition
adapter owns solver selection, residual/convergence checks, degeneracy and sign
conventions; its configuration is separate from the problem. The caller owns
Laplacian construction, degree normalization, isolated vertices and dropping the
trivial mode. This seam does not implement an eigensolver or silently translate
an unsupported solver name. Classical MDS's largest-eigenvalue ordering and
scaling belong to its child and decomposition integration, not this smallest-mode
seam. Sparse spectral operators are deferred.

`EmbeddingResult` has no generic loss field: MDS stress and iteration count,
LLE reconstruction error, t-SNE KL divergence and iteration count, and classical
MDS eigenvalues are distinct child-owned result records/learned attributes.
Public helpers keep their sklearn return structures via typed child records
where dynamic tuple arity requires an adaptation. `fit_transform` returns the
coordinate array; the internal exchange record does not change that API.
Only LLE (and deferred Isomap) opts into out-of-sample transformation. MDS,
ClassicalMDS, TSNE and SpectralEmbedding must not gain a fabricated `transform`.

## Shared numerical and native obligations

Consume `../../../shared/numerics/docs/numerics.md` directly, including its operation
matrix and measured copy costs. Reuse `_dense.sum_axis2d` and `row_squared_norms`.
NuMojo finite elementwise multiply, column subtraction, rank-two matmul and
explicit-output-dtype mean are approved within that matrix. General axis sum,
`dot`, variance/ddof, unpivoted solve/inverse/LU and NuMojo random are excluded.
No new operation-specific numerical exception is introduced here; validation
loops only inspect entries and allocate no arithmetic result.

Seeded streams come from `moxie._random.RandomState`; integer seeds reset where
sklearn resets, mutable states advance, and native None uses an application-owned
entropy context. Preserve the foundation's compatibility constraints and draw
consumption. Use Mojo 1.0.0, MAX 26.5.0, the live NuMojo source checkout, and compilation
with `-O3 --fp-mode=contract=off`. Pixi locks the toolchain; assembly receipts
record the source hashes consumed by each build.
Python is build/validation tooling only. Additional numerical libraries need a
user decision; unsupported dense numerical capabilities need an owner task,
not a local fallback or a claim of completed parity.

## Dependency assembly and extension rules

`tools/check.py` invokes the current `../../../shared/assembly` launcher.
The assembler reads live dependency sources on every build and materializes
`builds/deps/`, the compiler's single moxie import root. Foundation modules come
from the current Moxie checkout, including working edits. The original Git history retains inherited source provenance. Explicit export manifests define public names; ambiguous
module ownership and symbol collisions fail.

The `DEPENDENCIES.lock.json` filenames remain for consumer compatibility, but
contain live-source manifests, not source locks. They declare paths, modules,
license assets and exports. `NUMERICAL-DEPENDENCIES.json` declares the live
decomposition engine. Builds neither recover historical sources nor use cached
source snapshots. Missing dependencies fail; generated receipts record consumed
hashes. Keep toolchain locks and the scikit-learn correctness reference revision.

`tests/consumers/a` and `b` are isolated test fixtures, not algorithm children.
Both independently assemble contributions from the two fixture projects into
the same namespace and pass data across them. Both call the same live `_dense`,
`_random` and manifold helpers, in float32/float64. Separate output files and
consumer-local caches prove isolation. Tests reject missing sources, duplicate
module ownership, duplicate exports, and import of an unimplemented estimator.
Inherited contract tests also remain in the build/test gate. Extension probes
raise explicitly; they never fabricate an embedding.

Run `pixi run build` and `pixi run test`; each fixture can also run those tasks
from its own directory. Algorithm sessions add parity against the pinned sklearn
source, mathematical references and performance evidence. This scaffold makes
no algorithm parity or performance claim.
