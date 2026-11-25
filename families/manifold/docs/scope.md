# Dense-only scope

This project implements dense contracts, validation, owned snapshots,
dense TSNE, Isomap, MDS, LLE, spectral embedding, ClassicalMDS and trustworthiness.
See [dense t-SNE](tsne.md),
[dense Isomap](isomap.md), [dense MDS and SMACOF](mds.md), [dense LLE](lle.md),
[dense spectral embedding](spectral.md), and [algorithm support](algorithms.md) for settings,
native API differences and parity coverage. Tests cover Float32 and Float64,
finite nonempty rank-two inputs, contiguous and offset or negative-stride
views, square pairwise matrices, dense fixed-k neighbor tables, signed
symmetric spectral operators and coordinate shape checks.

See [plumbing.md](plumbing.md) for ownership and dependencies.

Sparse containers, conversions, arithmetic, graph-returning APIs and solvers
are deferred by the accepted dense-only policy.
This includes spectral nearest-neighbor and precomputed-nearest-neighbor
affinities, sparse spectral solvers, sparse t-SNE inputs and probability
graphs, and Isomap on sparse neighbor graphs. The shared numerics owner does not
provide dense factorizations. ClassicalMDS consumes the live decomposition engine.

Reference: sklearn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`,
`sklearn/manifold/_spectral_embedding.py` (`_get_affinity_matrix`,
`_spectral_embedding`), `_locally_linear.py` (`_locally_linear_embedding`),
`_t_sne.py` (`_fit`, `_joint_probabilities_nn`), `_isomap.py`, and the manifold
guide. LLE's dense native implementation may form its dense reconstruction
operator directly where sklearn constructs a sparse intermediate; a child must
validate the mathematical equivalence, not call that intermediate a sparse API.

Dense TSNE supports the default `method="barnes_hut"` and `method="exact"`.
Select `affinity="rbf"`, `affinity="precomputed"`, or `affinity="callable"`
with a native callback, and use `eigen_solver="dense"` for SpectralEmbedding.
The default affinity selects
nearest neighbors, and the default solver selects ARPACK. Both require an
explicit dense setting in this implementation. LLE uses a dense operator
and eigensolve for both `"dense"` and `"auto"`.

Validation commands: `pixi run build`, `pixi run test`. These compile all owned
modules and two independent assembled consumers; native tests check ownership,
shape/rank/nonfinite/negative/asymmetry failures and extension signatures.
Negative capability checks reject unsupported spectral settings and missing
required trait methods. Assembly tests reject missing source files and module/export
collisions. Each algorithm guide records its focused scikit-learn parity command.
Python reflection, metadata routing, serialization and plotting remain binding
work under the inherited contract.
