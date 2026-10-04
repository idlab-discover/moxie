# Covariance shared plumbing

This family assembles concrete dense estimators alongside shared contracts.
The reference is scikit-learn `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
`BaseGraphicalLasso` is private. See [scope](scope.md) for public exports and
Moxie's family-development policy for family-wide ownership.

## Ownership and dependencies

| Child owner | Exports and shared responsibility | Dependencies |
| --- | --- | --- |
| `algorithms/empirical/` (`covariance.empirical`) | EmpiricalCovariance, ShrunkCovariance, empirical_covariance, shrunk_covariance, log_likelihood; centering, covariance installation, precision policy, squared Mahalanobis, likelihood, error norms | covariance.scaffold |
| `algorithms/shrinkage/` (`covariance.shrinkage`) | LedoitWolf, OAS, ledoit_wolf, ledoit_wolf_shrinkage, oas; common shrinkage statistics | scaffold, empirical |
| `algorithms/graphical/` (`covariance.graphical`) | GraphicalLasso, graphical_lasso; coordinate and LARS solver orchestration | family contracts, empirical; local Gram coordinate and least-angle engines |
| `algorithms/robust/` (`covariance.robust`) | MinCovDet, EllipticEnvelope, fast_mcd; raw/corrected/reweighted covariance state | scaffold, empirical; foundation.random for actual RNG use |
| `algorithms/graphical/` (`covariance.graphical`) | GraphicalLassoCV; shared validation and graphical path orchestration in `_graph_lasso_cv`, facade `_graph_lasso`, solver `_graph_lasso_solver` | scaffold, empirical, model_selection splitters/validation |

The family root owns `moxie.covariance._base`, `_state`, `_engines` and their
contracts. Algorithm implementations reuse these contracts and do not duplicate storage or
validation. Changes follow Moxie contributor rules. `moxie.base` and `moxie.utils.validation` belong to the foundation;
`moxie._dense` to `moxie/shared/numerics`; `moxie._random` to
`moxie/shared/random`; dependency assembly to `moxie/shared/assembly`.
Family numerical helpers remain local; genuinely global numerical changes belong
to their shared owners.

Use sklearn's `_empirical_covariance`, `_shrunk_covariance`, `_graph_lasso`,
`_robust_covariance`, `_elliptic_envelope` module boundaries. Contributions from
multiple children to one reference module must use distinct private native
modules and one explicitly owned facade (for example empirical's fixed shrinkage
helper and shrinkage's adaptive helpers). Never create two owners for the same
`_shrunk_covariance` or `_graph_lasso` module. Public exports remain under
`moxie.covariance`; no import-order overrides or copied source forks.

## Exact native data contract

Arrays are `numojo.NDArray[dtype]`, with `Shape` and `Item` metadata; dtype is
Float32 or Float64 at compile time. Boolean support masks use `NDArray[DType.bool]`.
Samples are rows `(n,p)`; location is `(p,)`; covariance and precision are full
`(p,p)` matrices (both triangles); distances are squared values `(n,)` in sample
order. Dense precision with many zero entries is still dense storage.
Inputs are borrowed and may have valid positive/negative element strides and
nonzero offsets. There is no hidden contiguous-input requirement. Constructors
must not manually fabricate invalid metadata. `.view()` aliases and retains
lifetime; `.copy()` is independent. A strided copy can copy the entire backing
allocation. No concurrent mutation or overlapping in-place operations are approved.

`_state.validate_matrix(x, min_samples=1)` requires rank 2, nonempty features,
minimum positive sample count and finite entries. `validate_square(x, features)`
adds exact square dimensions; `validate_location(x, features)` requires a finite
vector. They validate structure, not symmetry or positive definiteness.
The empirical helper's reference 1-D input and nonfinite behavior must be handled
at its own API boundary; this estimator-state validator must not silently narrow
that helper. MinCovDet requires at least two samples.

`CovarianceState[dtype]` owns `fitted: FittedState` and optional `location_`,
`covariance_`, `precision_`. New/reset state has none of these learned buffers.
`replace(location, covariance, precision=None)` checks all supplied values and
copies before committing; failed validation preserves the prior state. Optional
precision absence means absent cache, not an identity matrix. `check_input(x)`
rejects unfitted state, invalid input and a changed feature count. The helper
never computes precision; the empirical owner provides that behavior. Fields are
internal composition storage: algorithms expose sklearn learned attributes and
must preserve shape/ownership invariants. `CovarianceModel.update_location`
publishes fit progress before numerical covariance installation; a failed refit
can retain the previous covariance. Inference validates the resulting feature
counts before indexing, and requires an installed covariance.
Copying state preserves fitted snapshots independently; estimator clone instead
copies constructor configuration and constructs a fresh empty state. Parameters
are never stored in `CovarianceState`. Normal successful fit replaces state.

`CenteredData[dtype]` owns `values: NDArray[dtype]` `(n,p)` and `location`
`(p,)`. `ShrinkageStatistics[dtype]` owns `empirical` `(p,p)`, `n_samples: Int`,
`trace`, `squared_frobenius`, `fourth_moment_sum` (Float64 scalars). For centered
Z, these are tr(S), sum(S_ij²), and sum_i(sum_j Z_ij²)², where S=ZᵀZ/n.
No hidden normalization is applied to the fourth moment. `block_size` only
controls workspace and must not alter the formula. This widened internal engine
contract is distinct from the public Float32 shrinkage path, which preserves
reference intermediate dtypes and can overflow before these widened moments.

`RobustCovarianceState[dtype]` owns `estimate: CovarianceState[dtype]`,
`raw_location_ (p,)`, `raw_covariance_ (p,p)`, `raw_support_ (n,)`,
`support_ (n,)`, and `dist_ (n,)`. Masks are Boolean; raw and reweighted snapshots
must not alias. The robust owner validates common n/p and support selection when
constructing it. These fieldwise result records are transport types, not fitted
estimators or automatic shape validators. EllipticEnvelope consumes this same
state and adds its scalar `offset_`, rather than fitting another covariance.

## Numerical extension points

`_engines.DenseCovarianceKernel` declares the single associated `dtype`.
`CovarianceEngine` and `ShrinkageEngine` inherit it so combined generic dispatch
uses one associated type (independent same-named declarations are incompatible
on the pinned compiler). `CovarianceEngine` has static raising methods:

| Method | Contract |
| --- | --- |
| `center(x, assume_centered) -> CenteredData[dtype]` | mean location and independently owned centered values; true uses zero location and an owned unchanged sample matrix |
| `empirical(x, assume_centered) -> NDArray[dtype]` | MLE divisor n, not n-1; one-feature output remains `(1,1)` |
| `precision(covariance) -> NDArray[dtype]` | symmetric pseudoinverse with sklearn-compatible rank cutoff, including singular covariance |
| `logdet(precision) -> Float64` | fast_logdet semantics: nonpositive determinant sign returns negative infinity |
| `mahalanobis(x, location, precision) -> NDArray[dtype]` | `(x-location)ᵀ P (x-location)` per row; negative forms yield NaN as in sklearn |
| `likelihood(empirical, precision) -> Float64` | `(-sum(S*P)+logdet(P)-p*log(2*pi))/2` |
| `error_norm(covariance, comparison, norm="frobenius", scaling=True, squared=True) -> Float64` | squared Frobenius or squared spectral norm of difference; divide by p when scaling, square-root after scaling when not squared |
| `shrink(empirical, shrinkage=0.1) -> NDArray[dtype]` | `(1-alpha)S + alpha*tr(S)/p*I`, alpha in [0,1]; supports `(...,p,p)` batches |

`ShrinkageEngine` has associated `dtype` and static methods
`statistics(centered, block_size=1000) -> ShrinkageStatistics[dtype]`,
`ledoit_wolf(statistics) -> Float64` and `oas(statistics) -> Float64`.
The empirical and shrinkage sessions supply concrete engines respectively.
No engine method has a default algorithm or returns a placeholder. Raising
implementations exist only in tests, including generic trait dispatch.

The internal same-dtype engine is a native adaptation, not a promise to override
public dtype promotion. The public empirical NumPy branch returns Float64 when
centering even for Float32 input, while assume_centered can retain input dtype.
Its owner must dispatch/cast into the appropriate engine and test those results;
it must not claim same-dtype parity for that branch. Dynamic option-dependent
returns use typed records/optional fields. Fit/set_params return a mutable borrow of the original estimator, typed
Params replace dynamic kwargs, and errors have descriptive Mojo Error prefixes,
as in the inherited interfaces. Python metadata routing, pandas feature names,
reflection, serialization and bindings remain separate work.

Reuse NuMojo creation, deep copy/view, elementwise multiply, column subtraction,
mean with explicit returned_dtype and rank-2 matmul as approved baselines.
Use `moxie._dense.sum_axis2d` for axis reductions and `row_squared_norms` for
squared norms. General `numojo.sum(axis)`, `dot`, variance/ddof, solve, inverse,
unpivoted LU and random sampling are excluded. Global sum is not a widened or
stable accumulation guarantee. Rank-2 matmul acceptance does not approve batched
products. Shared reductions widen to Float64 then cast to input dtype. No new
numerical operation exception is introduced by validation loops here.

## Estimator behavior and validation

Empirical defaults are store_precision=True and assume_centered=False;
ShrunkCovariance adds shrinkage=0.1, LedoitWolf adds block_size=1000, OAS retains
the empirical defaults. MinCovDet adds support_fraction=None, random_state=None;
EllipticEnvelope adds contamination=0.1. Graphical settings (including cd/lars,
alpha=0, precomputed covariance and CV paths) remain dense numerical obligations.
Constructor contracts follow the local reference; see the algorithm API guides.

Empirical score centers test observations by the training location and uses
mean Gaussian log-likelihood, not a per-sample API. `store_precision=False`
recomputes pseudoinverse on demand. MinCovDet inherits this scoring and exposes
raw state, correction and reweighting. EllipticEnvelope instead scores weighted
label accuracy; score_samples is negative squared Mahalanobis, decision subtracts
offset, and labels are +1 at decision >=0 and -1 below. It must not be forced
into a Gaussian score overload by trait inheritance. Existing separate traits
permit that distinction. None of these estimators gains transform, partial_fit,
probabilities or weighted fit merely by sharing covariance storage.

Algorithm suites test numerical rank deficiency, constant columns, singleton
inputs, shrinkage endpoints, blocked statistics, robust raw/corrected/reweighted
states and deterministic RNG, graphical coordinate descent, CV folds, precision cache
modes, invalid parameters, fit/clone ownership and exact public output shapes.
Plumbing checks use exact equality on representable ownership fixtures. Numerical
parity tolerances and limitations live with the algorithm test suites.

## Integrated assembly

Moxie's root `DEPENDENCIES.json` owns all covariance modules and exports.
Implementation is in `src/moxie/covariance` relative to Moxie; algorithm folders
here retain tests and API documentation. Shared numerics, random state and
assembly keep their existing owners. External numerical and model-selection
sources stay explicit dependencies. No sibling covariance source is consumed.

The root build and family driver compile against this manifest. Both isolated
consumers add their own probe modules and check source identity, missing sources,
and ownership/export collisions. Update their explicit manifests alongside the
root manifest when changing dependency ownership. Generated trees and receipts
remain under ignored builds directories.
