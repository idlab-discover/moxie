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
`FrozenEstimator` composition and, in `tests/cross_decomposition/composition.mojo`,
`make_pipeline(StandardScaler(), PLSRegression|PLSCanonical|CCA)`, cloned
pipelines with typed nested `set_params`, and PLS regression with
`cross_val_score` and `GridSearchCV`. PLSSVD has no predict, score, or inverse
transform.

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
feature-name input metadata, and pickling are excluded.

Examples cover [PLS/CCA](../../examples/cross_decomposition.mojo),
[PLSSVD](../../examples/cross_decomposition_svd.mojo), and
[owned blocks](../../examples/cross_decomposition_blocks.mojo).

## Native persistence

Use the experimental [Moxie model container](../serialization.md) with
`PLSRegressionSchema`, `PLSCanonicalSchema`, `CCASchema`, or `PLSSVDSchema`
from `moxie.cross_decomposition`. Choose the estimator's input dtype on the
schema. Use empty brackets such as `PLSRegressionSchema[]` to select the
Float64 default. Learned arrays are stored as Float64 for both input dtypes.

```mojo
from moxie.cross_decomposition import PLSRegression, PLSRegressionSchema
from moxie.serialization import load_model, save_model

var model = PLSRegression[DType.float32](n_components=2)
_ = model.fit(X, y)
save_model[PLSRegressionSchema[DType.float32]](model, "regression.moxie")
var restored = load_model[PLSRegressionSchema[DType.float32]]("regression.moxie")
var predictions = restored.predict(X)
```

The schemas round-trip both fitted and unfitted estimators. Loading reconstructs
independent learned arrays without fitting or requiring the original inputs.
Parameters changed after fit are saved separately from the learned dimensions,
projection state, and prediction rank. Learned arrays are saved in logical row
order, so strided arrays load as contiguous arrays.

## Cross-validation and search

Use the shared [dense adapter](../../families/model_selection/algorithms/search/docs/search.md)
with `PLSRegression` to get its default R² scorer and five unshuffled KFold
splits. Cross-validation only needs the adapter type and an instance. For a
nonempty search grid, give the adapter a noncapturing parameter applier.
`n_components` is an integer, so this grid uses `Int` as its value type and
applies each candidate through the estimator's named `set_params` overload.

```mojo
from moxie.cross_decomposition import PLSRegression
from moxie.model_selection import DenseSupervisedAdapter, GridSearchCV, cross_val_score
from numojo import NDArray

comptime Regression = PLSRegression[]
comptime Adapter = DenseSupervisedAdapter[Regression, NDArray[DType.float64], DType.float64, Int]

def set_components(mut estimator: Regression, params: Dict[String, Int]) raises:
    for name in params.keys():
        if name != "n_components":
            raise Error("Unexpected PLS parameter: " + name)
        _ = estimator.set_params(n_components=params[name])

# With dense Float64 X and y:
var model = Regression(n_components=2)
var scores = cross_val_score[Adapter](model, X, y, Adapter())
var grid = Dict[String, List[Int]]()
grid["n_components"] = [1, 2, 3]
var search = GridSearchCV[Adapter](model, grid, Adapter(set_components))
_ = search.fit(X, y)
var best_components = search.best_params_()["n_components"]
```

For Float32 inputs, select the Float32 estimator, target and input dtype on the
adapter and pass `weight_dtype=DType.float64` to match PLS's score weight type.
The default unweighted calls accept rank-one and rank-two targets and preserve
their rank through search and refit.

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
module and export ownership. The serialization contract checks on-disk
round trips, exact learned and inference values, and malformed archives.

The parity suites retain 124 PLS/CCA and 93 PLSSVD cases against scikit-learn
revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Native results are evaluated
after serializing and restoring each fitted model. Reports under
`builds/cross_decomposition/{pls,plssvd}/parity/` record current results and
runner hashes. The suites retain conditioning-aware tolerances, reference
perturbation checks, component objectives, score spectra, and null-space
orthonormality comparisons. Performance has not been
remeasured during integration.

To validate only the PLS composition, assemble live sources and run the
following with the locked environment. The composition binary checks the search
against independent native fits. The Python runner also compares candidate
ordering, fold scores, the refit selection and predictions with the pinned PLS,
search and splitter sources.

```sh
python shared/assembly/tools/assemble_family.py --consumer . --lock DEPENDENCIES.json --output builds/deps
mkdir -p builds/tests/cross_decomposition
mojo build -O3 --fp-mode=contract=off -I builds/deps tests/cross_decomposition/composition.mojo \
  -o builds/tests/cross_decomposition/composition
python tests/cross_decomposition/composition_parity.py builds/tests/cross_decomposition/composition
```

The [multilabel classification notebook](../../notebooks/cross_decomposition/plot_multilabel.ipynb)
replays the scikit-learn gallery example with native CCA, PCA and
`OneVsRestClassifier(BinaryAdapter(SVC(kernel="linear")))`. It compares the
signed projections, label indicators, decision values and fitted boundaries on
both gallery datasets. Run this notebook on its own with the pinned
scikit-learn checkout beside Moxie.

```sh
pixi run -e notebooks notebooks notebooks/cross_decomposition/plot_multilabel.ipynb
```

## Numerical dependencies

See [plumbing](plumbing.md). NuMojo remains at the external `../../NuMojo`
checkout. The assembled `mojo_linalg` SVD comes from
`../.worktrees/mojo-linalg-covariance`, already used by Moxie. Its SVD source
matches the family's `../mojo-linalg` source at integration. This migration
preserves the SVD implementation, signs, cutoff rules, reductions and stopping
semantics. It does not consume or integrate decomposition's SVD engine.
Mojo 1.0.0 and MAX 26.5.0 remain locked. Scikit-learn is a validation dependency,
not an estimator runtime dependency. License notices are in `LICENSES/`.
