# Covariance

Moxie implements all 16 public `sklearn.covariance` exports for dense Float32
and Float64 NuMojo arrays. The eight estimators include empirical, fixed and
adaptive shrinkage, robust covariance/outlier detection, and graphical lasso
with cross-validation. See [public API](docs/api-surface.md),
[scope](docs/scope.md), and [numerical limits and validation](docs/parity.md).

Implementation lives in `../../src/moxie/covariance`; this directory owns the
examples, contracts and differential tests. Moxie's root manifest owns the
assembled namespace. The original `moxie-covariance` checkout is not required.
`ORIGIN.json` preserves source and reference provenance.

From the Moxie root:

```sh
pixi run covariance-test
pixi run covariance-parity
pixi run covariance-audit
pixi run covariance-numerical-audit
```

Root `build`, `test` and `parity` include this family. Compile applications with
`-I builds/deps` after the root build. The [ownership example](examples/ownership.mojo)
shows independent learned snapshots. Validation uses the pinned local sklearn
checkout and the root Python environment; neither is a library runtime.

External numerical ownership remains in `../.worktrees/mojo-linalg-covariance`,
including incomplete gamma functions and its tested logarithm implementation.
GraphicalLassoCV consumes `KFold`, row selection and `cross_val_score` from
Moxie's own `families/model_selection`. These dependencies are explicit in
Moxie's manifest; they are not copied into covariance.
