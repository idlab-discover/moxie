# Linear models

Moxie owns the dense Float32/Float64 implementations of `LinearRegression`,
`Ridge`, `RidgeClassifier`, `ridge_regression`, `ElasticNet`, `Lasso`, and
`BayesianRidge`. Import them from `moxie.linear_model`. Implementation modules
live in `../../src/moxie/linear_model`; this directory owns their tests,
examples, documentation, and internal-engine validation.

From the Moxie root:

```sh
pixi run linear-model-build
pixi run linear-model-test
pixi run linear-model-parity
pixi run linear-model-start
```

The build assembles Moxie's complete public namespace in `builds/deps`.
Compile applications with `-I builds/deps`. Focused family and child checks
also create isolated namespaces from their live dependency manifests. Root
`build`, `test`, and `parity` include the corresponding linear-model checks.

The family is partial. ElasticNet/Lasso support cyclic residual solving;
random selection and Gram precomputation raise explicit errors. Ridge supports
auto, svd, cholesky, lsqr, sparse_cg on dense inputs, and positive lbfgs;
sag/saga remain unimplemented. BayesianRidge supports posterior covariance
and predictive uncertainty; ARDRegression is unimplemented. See the
[coverage table](docs/coverage.md), [scope](docs/scope.md), and
[public API](docs/api-surface.md) for the supported contract.

The standalone checkout is not a dependency. Its source revision and module
relocations are recorded in [PROVENANCE.json](PROVENANCE.json). Historical
scaffold declarations remain in that repository's Git history and are not
exported by Moxie. The original compatible Pixi locks, scikit-learn license,
and third-party notices are retained here. The oracle remains the local
scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

The existing external `mojo-linalg-linear_model` worktree supplies private
`moxie._linear_linalg` modules. Its namespace is separate from Moxie's other
linalg dependency because the worktrees have different solver implementations.
The existing `mojo-optimize-linear_model` worktree supplies NNLS and L-BFGS-B.
Both remain live external dependencies, as declared in `DEPENDENCIES.json`
at the Moxie root. This rollup does not transfer ownership of those libraries.
