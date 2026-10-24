# Dense MeanShift

```mojo
from moxie.cluster import MeanShift, mean_shift, estimate_bandwidth, get_bin_seeds
```

`MeanShift[dtype=DType.float64]` accepts keyword-only `bandwidth=None`,
`seeds=None`, `bin_seeding=False`, `min_bin_freq=1`, `cluster_all=True`,
`n_jobs=None`, and `max_iter=300`. Inputs and explicit seeds are finite dense
NuMojo Float32/Float64 matrices of the same compile-time dtype, with matching
feature counts. Sparse input, sample weights, transform and score are not
provided by this estimator. Euclidean radius and nearest-neighbor queries use
the live neighbors family; no DBSCAN-specific metric restriction is added.

`fit(X, y=None)` mutates and returns a borrow of the same estimator; `y` is ignored. `fit_predict` returns an owned
Int64 label vector. Learned fields are `cluster_centers_` `(n_clusters,d)`,
`labels_` `(n_samples,)`, `n_iter_`, and `n_features_in_`. Before fitting, arrays
are empty and counts are zero; `predict` raises `NotFittedError`. Successful
refits replace learned state, including feature count. Failed refits leave the
last successful learned state intact. Prediction rejects feature mismatches.
Dense arrays carry no column names, so `feature_names_in_` is not recorded.
Inputs, seeds, learned centers and returned label vectors do not share storage.
Positive/negative strided inputs are supported through NuMojo indexing.

Each seed moves to the mean of samples within the inclusive bandwidth radius.
It stops at displacement <= `0.001 * bandwidth` or the reference iteration cap.
As in sklearn, `max_iter=0` still computes one neighborhood mean; `n_iter_`
counts completed nonterminal shifts and is the maximum across all seeds,
including seeds that later lose their neighbors. Identical candidate centers
use the last seed's intensity. Candidates sort by descending intensity, then
descending lexicographic coordinates, before radius-based duplicate suppression.

Fitting assigns samples to their nearest retained center. With
`cluster_all=False`, samples farther than bandwidth receive -1. `predict(X)`
always assigns a nearest center, including such orphans. Nearest-distance ties
use the neighbor engine's deterministic lowest-index policy. Exact ties may
select differently from sklearn backend-specific ties; center order itself
follows the reference intensity/coordinate rule.

`get_params(deep=True)` returns `MeanShiftParams[dtype]`, with owned seeds;
`set_params(params)` copies that record. Named partial updates are also
supported. Nullable fields use nested optionals for updates: for example,
`set_params(bandwidth=Optional[Float64](None))` restores automatic estimation.
Both parameter-update overloads return the same estimator for fluent chaining.
Parameter changes retain fitted state until another fit. `clone(model)` creates
an unfitted estimator with independent configuration. Use floating literals
such as `bandwidth=2.0`; Mojo does not convert an integer literal through the
optional Float64 parameter automatically.

## Helpers

- `mean_shift(X, *, bandwidth=None, seeds=None, bin_seeding=False,
  min_bin_freq=1, cluster_all=True, max_iter=300, n_jobs=None)` returns an owned
  `MeanShiftResult[dtype]` with `cluster_centers` and `labels` fields, replacing
  sklearn's two-element tuple.
- `estimate_bandwidth(X, *, quantile=0.3, n_samples=None, random_state=0,
  n_jobs=None)` returns Float64. It averages the distance to neighbor
  `max(1, floor(n * quantile))`, including each sample itself. Explicit
  `n_samples` draws a full shared-RNG permutation, then selects up to that many
  rows. An overload takes `mut random_state: RandomState` and advances it;
  `random_state=None` uses Moxie's process stream. Without subsampling, no draws
  are consumed. Queries are batched in groups of 500. Quantile is in [0,1].
- `get_bin_seeds(X, bin_size, min_bin_freq=1)` rounds grid coordinates halfway
  to even and retains qualifying bins in first-encounter order. Coordinates
  pass through Float32 before scaling, as in sklearn. The returned array uses
  the input's compile-time dtype (sklearn normally returns Float32 bins even
  for Float64 input). Empty results have shape `(0,d)` instead of sklearn's
  rank-one empty array. Zero bin size and ineffective binning return a deep
  copy of X, rather than an alias; ineffective binning emits no Python warning.
  Negative/nonfinite bin size and nonpositive frequency raise a clear error.

Explicit bandwidth must be positive and finite. Automatic bandwidth can be
zero, supporting constant and repeated data. Empty seeds or no samples within
any seed's radius raise errors. `min_bin_freq` must be positive and `max_iter`
nonnegative. `n_jobs=0` is rejected; other values are accepted as execution
hints. Seed processing and the current query backend are serial; no Python
joblib context or parallel speedup is promised.

## Numerical scope and validation

Neighborhoods are gathered into an owned NuMojo matrix and averaged using
`moxie._dense.sum_axis2d`; displacement uses `row_squared_norms`. The shared
Float64 accumulation and input-dtype output can differ from NumPy Float32
reduction order. Bandwidth accumulation/return is Float64 even for Float32
input. Near a radius, convergence or center-order boundary, low-bit differences
can affect discrete outcomes. Extreme magnitudes inherit shared reduction and
neighbor-distance overflow/underflow limits; nonfinite computed centers and
bin coordinates raise. This is not bitwise parity over all finite inputs.
Memory includes a fitted query snapshot, one gathered neighborhood, and retained
candidate centers; it does not allocate a full sample-by-sample distance matrix
in the cluster implementation. No performance claim is made.

Run `pixi run build`, `pixi run test`, and `pixi run parity` at the family root.
The native contract test checks both dtypes, lifecycle, parameters, cloning,
copy isolation, strided inputs, RNG advancement, zero bandwidth and invalid
inputs. Two isolated consumers compile and execute MeanShift against live
shared sources. The differential driver executes pinned sklearn
`4358542a1bd39af2e7e1cbace8b9fed067473c5b` MeanShift/helper bodies with independent
SciPy distance queries and serial task dispatch. It checks exact labels,
predictions, shapes and iteration counts on its fixtures; center/bandwidth
tolerances are `3e-5` Float32 and `1e-12` Float64 for bounded inputs, accounting
for shared widened reductions. This does not test every compiled sklearn
neighbor backend or establish full sklearn estimator-check parity.
Reproduce only that comparison after building with
`pixi run python algorithms/mean_shift/tests/parity.py`; raw evidence is in
ignored `builds/mean_shift-parity.json`.
