# Covariance public API

The root build assembles all sixteen public scikit-learn covariance exports
under `moxie.covariance`. Use `builds/deps` as the Mojo import path; the tracked
source directories are contributions to that package, not separate installed
packages. `EXPORTS.json` lists the public names and their concrete owners.

```mojo
from moxie.covariance import MinCovDet, GraphicalLassoCV

var robust = MinCovDet(random_state=7, support_fraction=0.8)
robust.set_params(assume_centered=True)  # keeps the seed and support fraction
robust.set_params(support_fraction=None)  # restores automatic support selection
var config = robust.get_params()
robust.set_params(config)

var cv = GraphicalLassoCV()
cv.set_params(tol=1e-5, cv=3)
cv.set_params(cv=None, n_jobs=None)  # default folds and serial execution
```

All eight estimators support typed parameter records and keyword partial
updates. `MinCovDet` and `EllipticEnvelope` accept explicit `None` for
`support_fraction` and `random_state`; `GraphicalLasso` for `covariance`;
`GraphicalLassoCV` for `cv` and `n_jobs`. Omitting a keyword preserves its value.
Typed `Optional` arguments are also accepted for these nullable parameters.
`GraphicalLassoCV.set_params(alphas=...)` accepts an integer grid size, an explicit
`List[Float64]`, or an `AlphaGrid`; see its [API](../algorithms/graphical/docs/api.md).

`get_params(deep=True)` returns a typed record rather than a Python dictionary.
Passing that record to `set_params` replaces the complete configuration. Records
own their mutable configuration, including alpha lists. Keyword and record
updates leave fitted arrays, robust masks, offsets and CV results intact.
Parameter validation happens at fit; a successful refit replaces learned state.
`clone(model)` from `moxie.base` copies configuration into an unfitted estimator.

Mojo adaptations remain explicit: `fit` and `set_params` mutate in place and
return a mutable borrow of the estimator; learned attributes are accessed through methods such as
`covariance_()`; unknown keywords fail at compile time. Arbitrary heterogeneous
`**kwargs`, nested string parameter routing, metadata routing, Python reflection
and bindings are outside the native API. Estimator dtypes are static parameters.

See [feature parity](parity.md) for algorithm-specific limitations and validation
commands. Graphical solvers and cross-validation support `mode="cd"` and
`mode="lars"`. Export coverage does not
establish complete scikit-learn behavioral parity.
