# Dense covariance scope

The family implements all sixteen public covariance exports in one assembled
`moxie.covariance` namespace:

| Algorithm project | Public exports |
| --- | --- |
| `algorithms/empirical` | EmpiricalCovariance, ShrunkCovariance, empirical_covariance, shrunk_covariance, log_likelihood |
| `algorithms/shrinkage` | LedoitWolf, OAS, ledoit_wolf, ledoit_wolf_shrinkage, oas |
| `algorithms/robust` | MinCovDet, EllipticEnvelope, fast_mcd |
| `algorithms/graphical` | GraphicalLasso, GraphicalLassoCV, graphical_lasso |

Inputs are dense Float32/Float64 NuMojo arrays. Graphical lasso estimates sparse
coefficients in dense matrices; this does not imply sparse input support.
CSR/CSC/COO input, conversions and sparse dependencies remain deferred; no
automatic densification is performed. Batched `shrunk_covariance` is supported.

Graphical coordinate descent, LARS and cross-validation are implemented. Other
numerical and language-driven differences are listed in [feature parity](parity.md)
and the linked algorithm API guides. The method-by-method audit and remaining
roll-up requirements are recorded there; full behavioral parity is not claimed.

Run `pixi run build` or `pixi run test` at the root. Tests cover every algorithm's
native contracts, all public imports, parameter updates, shared ownership and
both isolated plumbing consumers. `pixi run params-parity` compares keyword
updates and refits with local scikit-learn. The algorithm projects also provide
separate numerical parity tasks; see [validation commands](parity.md).
