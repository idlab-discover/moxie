# Dense IterativeImputer

`IterativeImputer[DType.float64]()` fits a Bayesian ridge model per feature
and round. `transform` replays the saved sequence in order on new missing
cells. For floating inputs, it preserves dtype and observed values. Output
rows follow input order.
Pass `DType.float32` for Float32 inputs. Input arrays must be nonempty rank-two
NuMojo arrays and can include strided views. The output owns its storage.

The Float64 specialization also accepts signed and unsigned integer NDArrays
through its concrete `fit`, `fit_named`, `fit_transform` and `transform`
overloads. Values are promoted in logical row order before missing-value
masking and initial imputation. Finite markers use Float64 equality after
rounding, including for integers beyond exact Float64 precision. Results use
Float64 storage. Trait-based composition uses the Float64 input type, so cast
integer data before passing it through a generic pipeline.

```mojo
from moxie.impute import IterativeImputer

var imputer = IterativeImputer(random_state=17, add_indicator=True)
var filled_train = imputer.fit_transform(X_train)
var filled_test = imputer.transform(X_test)
```

The default marker is NaN. Pass `MissingValue[dtype](value)` for a finite
marker, which requires finite input. The initial `SimpleImputer` supports
`"mean"`, `"median"`, `"most_frequent"` and `"constant"`. Use `fill_value`
with the constant strategy. Columns entirely missing at fit are dropped by
default. With `keep_empty_features=True`, their missing cells use the initial
fill and newly observed values are preserved. `add_indicator=True` appends
the original missing mask columns selected at fit. `fit_named(X, names)`
records feature names explicitly.

Supported orders are `ascending`, `descending`, `roman`, `arabic` and
`random`. `skip_complete=True` leaves newly missing cells in previously
complete columns at their initial fill. Optional `n_nearest_features` samples
predictors without replacement, weighted by their absolute correlations on
the initially completed training matrix. `random_state` accepts a UInt32-range
integer seed or `None` for the shared default stream. The fitted
`random_state_` stores an owned snapshot of the state at the end of fit.
Seeded repeat fits continue the estimator's stream. Configuration-only clones
restart it.

Deterministic imputation stops once the largest absolute row sum of the change
falls strictly below `tol * max(abs(observed_training_values))`, or after
`max_iter` rounds. A zero iteration limit or one retained feature uses the
initial fill. Bounds apply to regression predictions. Pass scalar or
original-feature-order vector bounds using `IterativeBound(Float64(...))` or
`IterativeBound(bounds^)` for a `List[Float64]`. Omitting either argument uses
its corresponding infinity. Pass `verbose` for diagnostics on fit.

`IterativeImputer[dtype, Regressor](estimator)` accepts a copyable native
supervised predictor whose inputs, rank-one targets and rank-one predictions
use matching NDArrays. Named regressor values can be passed with `^` or
`.copy()`. The constructor and typed parameter updates retain an unfitted
configuration clone. Each `imputation_sequence_` step owns a separate fitted
regressor and stores the retained-space target and predictor indices.
`get_params` returns a typed record, and `set_params` accepts the record or
named updates and clears learned state. Use the record to clear nullable
settings, or supply an outer `Optional` containing the new optional value.
Failed fits preserve the previous fitted estimator.

A dense k-nearest-neighbors regressor can serve as the deterministic
regressor. Choose `n_neighbors` no greater than the number of observed
training rows for any feature the imputer fits.

```mojo
from moxie.impute import IterativeImputer
from moxie.neighbors import KNeighborsRegressor

comptime Regressor = KNeighborsRegressor[DType.float64]
var imputer = IterativeImputer[DType.float64, Regressor](
    Regressor(n_neighbors=10), random_state=0, max_iter=40, tol=1e-2,
)
```

The polynomial Nystroem + Ridge gallery regressor uses the regular typed
pipeline. Pass its type to the imputer and move in the configured pipeline.
Both children support fitted copies, so copying the imputer also copies
each round's fitted basis and coefficients.

```mojo
from moxie.impute import IterativeImputer
from moxie.kernel_approximation import Nystroem
from moxie.linear_model import Ridge
from moxie.pipeline import make_pipeline

var regressor = make_pipeline(
    Nystroem(kernel="polynomial", degree=2.0, random_state=0),
    Ridge(alpha=1e4),
)
var imputer = IterativeImputer[DType.float64, type_of(regressor)](
    regressor^, random_state=0, max_iter=40, tol=1e-1,
)
```

Set `sample_posterior=True` to draw from the predictive Gaussian truncated to
the fitted feature bounds. Each missing row with a positive deviation and a
mean inside the bounds consumes one uniform draw. A nonpositive deviation
uses the mean, and a mean outside the bounds uses the nearest bound. Posterior
fitting runs all `max_iter` rounds when more than one feature remains.
Repeated transforms continue the stream, even with an explicit seed. Copying
a fitted seeded imputer captures an independent copy of its current stream.
With `random_state=None`, fits and transforms draw from the process default
stream.

BayesianRidge supplies uncertainty by default. A custom posterior regressor
implements `IterativePosteriorRegressor` from `moxie.impute` with
`predict_std(X)` returning `Tuple[Prediction, Prediction]`
containing the mean and standard deviation. Each result must be finite and
rank one with one entry per query row. Deterministic regressors still need
only their ordinary `predict(X)` method.

Python metadata routing, warning objects and dataframe output are unavailable.
Verbose diagnostics use Mojo console output. Import directly from
`moxie.impute` without an experimental opt-in.

## Numerical implementation and validation

The implementation reuses the live BayesianRidge SVD owner, SimpleImputer,
MissingIndicator and shared RandomState. Numax `stats.corrcoef` takes a
Numax Tensor. With NuMojo storage, an adapter would require materializing and
converting the training array for that operation. Correlation uses the approved
NuMojo rank-two product over a centered Float64 workspace, followed by
normalization in the Gram buffer. The convergence loop combines the row
reduction and the snapshot update in one pass. Integer promotion uses NuMojo
`astype` for contiguous inputs and `contiguous` followed by `astype` for
strided inputs because the cast traverses physical storage.
Posterior sampling composes Moxie's existing Cephes normal CDF and quantile
with Float64 inverse-CDF interpolation. The complementary tail avoids
cancellation near probability one. For intervals contained within
`1e-8` standard deviations of the mean, the Gaussian density varies below
Float64 precision and interpolation uses the original bounds. Numax's scalar
`erfcinv` goes through `1 - p`, which loses the small tail in this sampler.
The existing normal quantile works directly on scalar probabilities.

From the repository root, with the locked toolchain active:

```sh
python families/impute/algorithms/iterative/tests/parity.py
```

Parity compares training fills, held-out replay, initial statistics, round
counts, estimator ordering, sampled predictor indices, RNG draws and feature names
against the pinned iterative, simple and Bayesian regression sources.
Integer cases compare signed and unsigned input layouts, mixed integer and
Float64 fit and replay, and missing-mask equality after Float64 rounding.
Float64 comparisons use absolute and relative tolerances of `5e-8`. Float32
uses `3e-4`. The allowances cover the reference's dtype-specific reductions
and the use of SciPy SVD versus the native BayesianRidge solver across
repeated updates. `test_iterative.mojo` covers typed configuration, feature
names, signed strides, bounds, unfitted errors, state replacement, and
independent clones and copies. The focused integration probe uses
LinearRegression as both an iterative regressor and a downstream pipeline
estimator.

The k-NN gallery configuration has a separate integration check. It compares
Float32 and Float64 training fills, held-out replay and fitted sequence
metadata with the pinned scikit-learn imputer and regressor on a small
deterministic fixture. The fixture also exercises a column that is complete at
fit and has missing cells at transform. Observed values and sequence metadata
match exactly. Fill tolerances account for reduction order in the neighbor
means and are `3e-6` for Float32 and `2e-12` for Float64. The native check
also verifies that clones start unfitted and that a fitted copy preserves its
replay after the original is refit.

```sh
mkdir -p builds/pr6-iterative-knn
python shared/assembly/tools/assemble_family.py --consumer . --lock DEPENDENCIES.json --output builds/deps
mojo build -O3 --fp-mode=contract=off -I builds/deps families/impute/algorithms/iterative/tests/integration/knn_regressor.mojo -o builds/pr6-iterative-knn/knn_regressor
python families/impute/algorithms/iterative/tests/knn_regressor_parity.py builds/pr6-iterative-knn/knn_regressor
```

The Nystroem + Ridge integration checks training fills, held-out replay,
round counts, feature order, sampled basis indices, and preservation of
observed cells. It also checks that a configuration clone starts unfitted
and that a fitted copy survives mutation and refitting of the original.
It uses a full-rank Float32 fixture and a rank-deficient Float64 fixture.
Fill tolerances are `3e-4` for Float32 and `5e-8` for Float64, accounting for
the different SVD and Ridge solvers. Severely ill-conditioned Float32 bases
can diverge in both fills and convergence rounds, as described in the
[Nystroem numerical contract](../../../../../docs/kernel_approximation/nystroem/api.md).

```sh
mkdir -p builds/pr6-nystroem
python shared/assembly/tools/assemble_family.py --consumer . --lock DEPENDENCIES.json --output builds/deps
mojo build -O3 --fp-mode=contract=off -I builds/deps families/impute/algorithms/iterative/tests/integration/nystroem_ridge.mojo -o builds/pr6-nystroem/nystroem_ridge
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python families/impute/algorithms/iterative/tests/nystroem_ridge_parity.py builds/pr6-nystroem/nystroem_ridge
```
