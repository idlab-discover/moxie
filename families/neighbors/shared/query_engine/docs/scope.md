# Dense query engine scope and current status

This is an **incomplete private engine**, not a public NearestNeighbors estimator.
It owns `moxie.neighbors._distance` and `_query_engine` and consumes the family's
unchanged `_contracts` through verified foundation assembly. Tree construction,
estimator wrappers, classification and regression belong to other projects.

Implemented native paths use NuMojo Float32/Float64 dense arrays: explicit metric
parameters, owned fit snapshots, streamed top-k selection, ragged radius results,
deterministic ties, self exclusion for `None`, dense precomputed distances, and
inverse-distance weighting with zero-distance dominance. Optional distance fields
implement `return_distance=False`. Supported formulas in the current source are
Euclidean/squared Euclidean, Manhattan, Chebyshev, Minkowski and legacy weighted
Minkowski, cosine, correlation, Haversine, nan-Euclidean, Canberra, Bray-Curtis,
Hamming, Jaccard, Dice, Rogers-Tanimoto, Russell-Rao, Sokal-Michener,
Sokal-Sneath, Yule, standardized Euclidean with explicit or self-query-inferred V, and Mahalanobis with
explicit or self-query-inferred VI. Presence of a formula is not a full
numerical parity claim.

Aliases l1/cityblock, l2 and infinity are normalized. Tests exercise Minkowski
p in {0.5,1,1.5,2,3,infinity}, with and without feature weights. Native lifecycle
tests pass for both dtypes. The historical Float64 Minkowski p=0.5 precision failure was reproduced and
fixed with std.math sqrt/cbrt and multiplication compositions for common powers.
All 108 dtype/metric parity configurations (96 explicit-parameter, 12 inferred
Mahalanobis/standardized-Euclidean) pass at the unchanged
2e-12 Float64 / 3e-6 Float32 tolerances. Arbitrary powers use the compiler
llvm.pow.f64 intrinsic; additional powers 0.1, 0.7, 1.1, 2.3, 7.1 and 31.4 pass
the same tolerances. Required dense scope is not complete.

Mahalanobis without VI follows sklearn `_precompute_metric_params`: self
queries (`X=None`) use `VI = inv(cov(X.T)).T` from the fitted data, while explicit
queries raise ValueError, even for identical data. The covariance is computed in
Float64 for both input dtypes, like `np.cov`, through accepted NuMojo centering
and rank-2 products; the Gram triangle is mirrored before the SPD inverse from
`moxie._linalg` (moxie/shared/linalg). Singular covariance and one-feature data
raise on the self query, not at fit, matching sklearn's LinAlgError timing. A
single fitted row evaluates no pairs. Covariance-form `V` for Mahalanobis raises
at fit; sklearn's brute path does not accept it either (trees do).

Standardized Euclidean without V infers sample variance (ddof=1) only for self
queries; explicit queries require V. Shared sum_axis2d reductions, subtraction and
multiplication compute the variance with Float64 accumulation. Constant columns
produce undefined pair distances. Undefined built-in distances sort last in top-k
and never belong to a radius result; custom callback NaNs still raise. Finite
missing-value sentinels are supported by Metric(missing_values=...) for
nan_euclidean; the default sentinel is NaN. Input infinities remain invalid.
Bray-Curtis follows sklearn's native sum(abs(x)+abs(y)) denominator and zero-vector
rule. NuMojo's general inverse/solve/LU and variance remain excluded. Extreme
numerical domains and backend-dependent tie behavior are not exhaustively certified.

Deferred by the session's dense-only scope: CSR/CSC/COO storage, sparse feature
input, sparse precomputed distances, sparse graph-returning APIs and
VALID_METRICS_SPARSE. No implicit densification or alternate dense graph output
is provided. References: scikit-learn `neighbors/_base.py`, `neighbors/_graph.py`,
`neighbors/__init__.py`, and `metrics/pairwise.py` at revision
4358542a1bd39af2e7e1cbace8b9fed067473c5b. Inherited storage and family contracts
also provide only their accepted dense subset.

Executed commands: `pixi run build`, `pixi run test`, `pixi run parity`.
Build compiles both owned modules, assembled dependencies, native tests and the
example. Test checks lifecycle, result layout, precomputed, self/duplicate/tie,
radius options and optional outputs, plus negative compilation of the excluded
sparse export. Parity executes pinned local sklearn top-k/radius reducer bodies
and the pinned `_precompute_metric_params` body for inferred Mahalanobis (three
fixtures, covariance condition below 12, unchanged 2e-12/3e-6 tolerances), and
checks metric formulas against SciPy; it does not import a wheel's sklearn
implementation. The former fractional-power failure is resolved on its fixture. A performance baseline and the
complete dense correctness campaign remain unfinished; no speed claim is made.
