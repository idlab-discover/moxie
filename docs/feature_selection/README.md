# Feature selection

Moxie owns all 19 reference exports under `moxie.feature_selection`, along with
native scorer, parameter, result and adapter types. See [API coverage](api-surface.md),
[usage and limits](scope.md), and [extension contracts](plumbing.md).

Sources live in `src/moxie/feature_selection`, tests in `tests/feature_selection`,
and examples in `examples/feature_selection.mojo` and `examples/feature_selection/`.
The previous family checkout is retired; consumers must remove its module and
export contributions and use Moxie's current dependency manifests.

Run from Moxie:

```sh
pixi run feature-selection-build
pixi run feature-selection-test
pixi run feature-selection-parity
```

Moxie's root commands also include these checks. Compilation uses a shared lock,
a hard 8 GiB per-process address-space limit and a ten-minute timeout. The
resource-limit and namespace-collision checks are part of the native test command.

## External sources

Trees come from `src/moxie/tree`. Linear models remain in `../moxie-linear_model`,
including its coordinate engine. PCA, IncrementalPCA and the SVD engine remain
in `../moxie-decomposition`. CV and splitters remain in `../moxie-model_selection`.
Nearest-neighbor queries and their transitive spatial trees remain in
`../moxie-neighbors`. Shared dense arrays, labels, random state and linear algebra
retain their current owners. The root assembly uses Moxie's existing
`../.worktrees/mojo-linalg-covariance`; isolated consumers retain `../mojo-linalg`.
Paths in each `DEPENDENCIES.json` are authoritative and consume live sources.

## Reference and limits

The reference is scikit-learn commit
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. License notices are in
`LICENSES/feature-selection-scikit-learn.txt`. Each parity runner retains its
original extraction/loading method, exact comparisons and numerical tolerances.
Variance and univariate checks execute extracted reference bodies with independent
validation helpers; they do not exercise the full sklearn validation pipeline.
SelectFromModel and RFE checks combine extracted reference logic with independent
helpers, including analytic orthogonal Lasso/ElasticNet solutions. Incremental,
CV and classifier runners load recorded reference classes in version-checked
Python environments. Each runner records its precise comparison scope.

Sparse inputs, Python reflection, metadata-routing APIs, container output and
warning transport remain deferred. Numerical stress cases and broader custom
adapter compatibility remain incomplete. Mutual information remains serial and
preserves the documented random traversal and input-ownership behavior.
