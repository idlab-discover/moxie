# Dense-only scope

This project implements dense contracts, validation, owned snapshots,
ClassicalMDS and trustworthiness. See [algorithm support](algorithms.md) for
implemented settings, native API differences and parity coverage. The supported
and tested settings are float32/float64, finite nonempty rank-two inputs,
contiguous and offset/negative-stride views, square pairwise matrices, dense
fixed-k neighbor tables, signed symmetric spectral operators and coordinate
shape checks. Public estimator traits and the baseline remain unchanged.

Remaining dense work: MDS and smacof; SpectralEmbedding and
spectral_embedding; all four LLE reconstruction methods with dense assembly;
TSNE exact optimization. These remain unimplemented.
See [plumbing.md](plumbing.md) for the ownership/dependency map.

Sparse containers, conversions, arithmetic, graph-returning APIs and solvers
are deferred by the accepted dense-only policy.
Isomap is wholly deferred. Spectral nearest-neighbor and
precomputed-nearest-neighbor affinities, sparse spectral solver branches and
t-SNE Barnes-Hut probability graphs are deferred. No sparse input is densified.
No `_sparse` module is required or provided. The shared numerics owner does not
provide dense factorizations. ClassicalMDS consumes the live decomposition engine.

Reference: sklearn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`,
`sklearn/manifold/_spectral_embedding.py` (`_get_affinity_matrix`,
`_spectral_embedding`), `_locally_linear.py` (`_locally_linear_embedding`),
`_t_sne.py` (`_fit`, `_joint_probabilities_nn`), `_isomap.py`, and the manifold
guide. LLE's dense native implementation may form its dense reconstruction
operator directly where sklearn constructs a sparse intermediate; a child must
validate the mathematical equivalence, not call that intermediate a sparse API.

Unsupported defaults in future dense children must be explicit: TSNE defaults
to `method="barnes_hut"`, so select `method="exact"`; spectral defaults to
`affinity="nearest_neighbors"`, so select dense `"rbf"`, `"precomputed"` or a
native dense callable. LLE selects `eigen_solver="dense"` for dense operator
assembly. Spectral's default `eigen_solver=None` resolves to ARPACK: the child
must expose and document supported dense solver selection and reject excluded
solver settings; this scaffold supplies no solver under any name. Never silently
change defaults or treat missing dense branches as sparse exclusions.

Validation commands: `pixi run build`, `pixi run test`. These compile all owned
modules and two independent assembled consumers; native tests check ownership,
shape/rank/nonfinite/negative/asymmetry failures and extension signatures.
Negative capability checks reject deferred estimator construction and missing
required trait methods. Assembly tests reject missing source files and module/export
collisions. The `pixi run parity` task checks implemented algorithms against
the local reference; compilation and raising probes are not estimator parity.
Python reflection, metadata routing, serialization and plotting remain binding
work under the inherited contract.
