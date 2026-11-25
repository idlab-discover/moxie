# Dense IterativeImputer

`IterativeImputer[DType.float64]()` fits a Bayesian ridge model per feature
and round. `transform` replays the saved sequence in order on new missing
cells. It preserves the input dtype, observed values and row order.
Pass `DType.float32` for Float32 inputs. Input arrays must be nonempty rank-two
NuMojo arrays and can include strided views. The output owns its storage.

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
`random_state_` stores an owned snapshot. Seeded repeat fits continue the
estimator's stream. Configuration-only clones restart it.

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

`sample_posterior=True` currently raises on fit. Integer array promotion,
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
reduction and the snapshot update in one pass.

From the repository root, with the locked toolchain active:

```sh
python families/impute/algorithms/iterative/tests/parity.py
```

Parity compares training fills, held-out replay, initial statistics, round
counts, estimator ordering, sampled predictor indices and feature names
against the pinned iterative, simple and Bayesian regression sources.
Float64 comparisons use absolute and relative tolerances of `5e-8`. Float32
uses `3e-4`. The allowances cover the reference's dtype-specific reductions
and the use of SciPy SVD versus the native BayesianRidge solver across
repeated updates. `test_iterative.mojo` covers typed configuration, feature
names, signed strides, bounds, unfitted errors, state replacement, and
independent clones and copies. The focused integration probe uses
LinearRegression as both an iterative regressor and a downstream pipeline
estimator.
