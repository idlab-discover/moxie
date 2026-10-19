# Cross decomposition

Moxie owns four dense estimators under `moxie.cross_decomposition`:
`PLSRegression`, `PLSCanonical`, `CCA`, and `PLSSVD`. Import them from that
public namespace. The implementation, contracts, tests, and examples live in
Moxie; the former family checkout is not a dependency.

```mojo
from moxie.cross_decomposition import PLSRegression

var model = PLSRegression(n_components=2)
var predictions = model.fit(X, y).predict(X)
```

Inputs are NuMojo Float32 or Float64 arrays. Computation and learned arrays use
Float64. X is `(samples, features)` and y is either `(samples,)` or
`(samples, targets)`. Predictions preserve the target rank. Single-block
transform returns a matrix; paired transform and fit-transform return
`PairedScores`. See [PLS/CCA API](pls/api.md) and [PLSSVD API](plssvd/api.md).

`fit`, typed and keyword `set_params`, and `fit_inplace` return a mutable borrow
of the estimator. Clone copies configuration into an unfitted estimator;
ordinary copies own independent learned arrays. Refit replaces learned state.
The estimators satisfy current Moxie supervised and transformer contracts;
the three PLS variants also satisfy regression contracts. Tests exercise real
`FrozenEstimator` composition. PLSSVD has no predict, score, or inverse transform.

`fit` borrows inputs and always works on copies. Use `fit_inplace` to honor
`copy=False`: Float64 inputs receive the centered/scaled workspaces, plus
residual deflation for the three PLS variants. Float32 conversion allocates,
so Float32 inputs remain unchanged. Overlapping mutable X/y storage and
concurrent input mutation are unsupported.

PLSRegression uses mode A and regression deflation. PLSCanonical uses mode A
and canonical deflation, with NIPALS or SVD. CCA uses mode B and canonical
deflation. PLSSVD takes one SVD of centered/scaled cross covariance. All four
share centering/scaling and the PLS engine. Python warnings are available as
native diagnostics. Sparse inputs, pandas output, Python metadata routing,
feature-name input metadata, pickling, and persistence adapters are excluded.

Examples cover [PLS/CCA](../../examples/cross_decomposition.mojo),
[PLSSVD](../../examples/cross_decomposition_svd.mojo), and
[owned blocks](../../examples/cross_decomposition_blocks.mojo).

## Validation

From Moxie, run:

```sh
pixi run python tools/cross_decomposition_checks.py build
pixi run python tools/cross_decomposition_checks.py test
pixi run python tools/cross_decomposition_checks.py parity
pixi run build
pixi run test
pixi run parity
```

The focused commands compile the complete migrated family. Root commands include
these sources and checks. Tests retain PLS estimator and engine contracts,
PLSSVD contracts, family block validation, Float32/Float64, owned learned state,
cloning/refit, and copy semantics. Two isolated assemblies exercise all public
estimators and check Moxie ownership. Negative compilation rejects PLSSVD
regression and inverse-transform methods; assembly tests reject conflicting
module and export ownership.

The parity suites retain 124 PLS/CCA and 93 PLSSVD cases against scikit-learn
revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Reports under
`builds/cross_decomposition/{pls,plssvd}/parity/` record current results and
runner hashes. The suites retain conditioning-aware tolerances, reference
perturbation checks, component objectives, score spectra, and null-space
orthonormality comparisons. Performance has not been
remeasured during integration.

## Numerical dependencies

See [plumbing](plumbing.md). NuMojo remains at the external `../../NuMojo`
checkout. The assembled `mojo_linalg` SVD comes from
`../.worktrees/mojo-linalg-covariance`, already used by Moxie. Its SVD source
matches the family's `../mojo-linalg` source at integration. This migration
preserves the SVD implementation, signs, cutoff rules, reductions and stopping
semantics. It does not consume or integrate decomposition's SVD engine.
Mojo 1.0.0 and MAX 26.5.0 remain locked. Scikit-learn is a validation dependency,
not an estimator runtime dependency. License notices are in `LICENSES/`.
