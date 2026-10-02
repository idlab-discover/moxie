# KBinsDiscretizer

Conventions: [api.md](../api.md). Source: `src/moxie/preprocessing/_discretization.mojo`,
with NumPy-exact `linspace`, percentile methods and sklearn's
`_weighted_percentile` in the family module `_percentiles.mojo` (shared with
SplineTransformer) and pairwise sums/sorting in `_special_math.mojo`. Ported
sklearn and NumPy code keeps its BSD-3-Clause notices (`LICENSES/sklearn.txt`,
`LICENSES/numpy.txt`).

| Name | Native signature |
| --- | --- |
| `KBinsDiscretizer[dtype, out_dtype=dtype]` | `KBinsDiscretizer(n_bins=5, *, encode="onehot", strategy="quantile", quantile_method="averaged_inverted_cdf", subsample=200000, random_state=None)`; a second constructor takes `n_bins: List[Int]` (one count per feature) |

`dtype` is the input dtype. `out_dtype` is sklearn's `dtype` argument: the
transform output dtype, defaulting to the input dtype (sklearn `dtype=None`).
`subsample` and `random_state` are `Optional[Int]`. Inputs are rank-2
Float32/Float64 arrays in any strided view, without NaN or infinity.

```mojo
from moxie._random import RandomState
from moxie.preprocessing import KBinsDiscretizer

var est = KBinsDiscretizer(3, encode="ordinal", strategy="uniform")
var codes = est.fit_transform(X)          # NDArray[float64] of bin indices
var centers = est.inverse_transform(codes) # bin centers
var edges0 = est.bin_edges_[0]            # NDArray[float64], n_bins_[0] + 1 edges

var onehot = KBinsDiscretizer[DType.float64, DType.float32](List[Int]([2, 4]), encode="onehot-dense", strategy="kmeans")
var Z = onehot.fit_transform(X)            # NDArray[float32], 6 columns
var names = onehot.get_feature_names_out() # x0_0.0, x0_1.0, x1_0.0, ...

var weighted = KBinsDiscretizer(4, encode="ordinal")
weighted.fit(X, sample_weight=w)          # w: NDArray[float64], or a Float64 scalar
var rng = RandomState(42)
est.fit(X, rng)                           # sklearn random_state=RandomState(42): advances rng
```

## Behavior

Fitting follows sklearn operation by operation, in the reference's precision:

* `uniform`: `np.linspace(min, max, n_bins + 1)` in the input dtype. With
  weights, min and max ignore zero-weight samples.
* `quantile`, unweighted: `np.percentile` with any of the nine `quantile_method`
  values, the neighbor difference taken in the input dtype and the blend in
  Float64 (as NumPy's `_lerp`). Edges are Float64.
* `quantile`, weighted: sklearn `_weighted_percentile`, `inverted_cdf` or
  `averaged_inverted_cdf` (other methods raise, as in sklearn), in the input dtype.
* `kmeans`: a 1-D Lloyd k-means started from the uniform bin midpoints (see
  below); edges are the column extremes and the midpoints of the sorted centers.
* Quantile and k-means bins of width <= 1e-8 (compared in the edge dtype) are
  removed and `n_bins_` reduced. A constant feature gets one bin with edges
  `[-inf, inf]`; its inverse transform is NaN, as in sklearn.
* When `n_samples > subsample`, rows are drawn with replacement by NumPy's
  legacy `RandomState.choice`, with probabilities `w / w.sum()` when weights are
  given; the weights are then consumed, so any `quantile_method` is allowed.
  An integer `random_state` reseeds every fit, `None` uses the process default
  stream, and `fit(X, rng)` / `fit(X, rng, sample_weight=w)` advance a borrowed
  `RandomState` (the configured `random_state` is then ignored).

`transform` casts X to `out_dtype` (sklearn validates X with that dtype) and
finds each bin with `searchsorted(bin_edges_[j][1:-1], x, side="right")`.
`encode="ordinal"` returns the bin indices; `"onehot-dense"` returns their
one-hot expansion with OneHotEncoder's layout and `x0_0.0`-style names (sklearn
fits `OneHotEncoder(categories=[arange(n_bins_[j])])` on float codes). The
fixed integer categories are expanded directly instead of through the
categorical vocabulary engine, which would materialize an n-row code column per
feature. `inverse_transform` maps bins to `(edges[b] + edges[b + 1]) * 0.5`,
averaged in the edge dtype. Ordinal input is truncated toward zero, negative
indices wrap and out-of-range indices raise `IndexError`, as NumPy indexing
does; one-hot input selects the first maximal column and raises for all-zero
feature blocks.

Learned attributes: `bin_edges_`, `n_bins_` (Int64), `n_features_in_`,
`feature_names_in_` (set by `fit_named`).

### Local 1-D k-means

`moxie.cluster.KMeans` does not exist yet (the plumbing lists `cluster.kmeans`
as this child's dependency). `_kmeans_1d` is a family-private stand-in scoped to
sklearn's exact call `KMeans(n_bins, init=<uniform midpoints>, n_init=1).fit(column[:, None], sample_weight)`:
explicit init, a single Lloyd run, `max_iter=300`, `tol=1e-4 * np.var(column)`.
It reproduces `_kmeans_single_lloyd` with one OpenMP thread: data and init are
centered on the column mean; distances are formed as BLAS does
(`c*c - 2*(x*c)`); weighted sums accumulate in sample order; empty clusters
are relocated to the farthest samples; centers are averaged in place; and it stops
on unchanged labels or a squared center shift `<= tol`. All arithmetic is in the
input dtype. It is not a general KMeans and should be replaced by the cluster
family's estimator when that exists.

## Native adaptations

* sklearn's `dtype` argument is the struct parameter `out_dtype`
  (`tests/api_adaptations.json`); array-like `n_bins` uses the `List[Int]`
  constructor or `set_params(List[Int])`.
* `encode="onehot"` (the default) returns a sparse matrix in sklearn and raises
  at fit here: select `"onehot-dense"` or `"ordinal"`. Sparse output is deferred.
* `bin_edges_` is a `List[NDArray[float64]]` (sklearn: an object array of
  arrays). Values are sklearn's exactly; sklearn stores Float32 edges for
  Float32 input with the uniform, k-means and weighted-quantile strategies,
  which Float64 represents exactly.
* `inverse_transform[d](X: NDArray[d]) -> NDArray[d]` keeps the argument's
  dtype, as sklearn's ordinal branch does. sklearn's one-hot branch always
  returns Float64; pass a Float64 array to get it (a Float32 argument returns
  those centers rounded to Float32). The class therefore implements
  `Preprocessor`, not `InvertiblePreprocessor`.
* `random_state` accepts an integer or None; a RandomState instance is the
  `fit(X, rng)` overload. Keyword `set_params(subsample=Optional[Int](None))`
  selects `subsample=None` (the outer Optional marks "given").
* No warnings are emitted (constant features, removed bins): Moxie has no
  warning channel; inspect `n_bins_`.
* NumPy's `argsort`/`argpartition` order equal values unspecified (and, for
  `argpartition`, CPU-dependently through x86-simd-sort). Weighted quantiles sort
  ties stably, which can change the cumulative-weight rounding; k-means relocation
  takes the lowest-index sample among equally distant ones. Results can differ
  from sklearn only when such tied samples have different weights (weighted
  k-means on columns with repeated values); sklearn itself varies across CPUs there.

## Memory

Fitted state is O(total bins). Fitting reuses one column workspace for every
feature: unweighted quantile, n Float64 values; k-means, values, weights and
squared deviations in the input dtype plus two Int32 label vectors; weighted
quantile, the weight list, row order, merge scratch, sorted weights and
cumulative weights (O(n) each). Subsampling adds the drawn Int64 row indices
and, with weights, the Float64 probabilities and the `choice` CDF (O(n)); the
subsample itself is never materialized. `transform` allocates only its result;
`inverse_transform` its result plus one bin-count vector.

## Parity

`tests/parity/test_discretization.py`, 1862 checks, all exact (rtol = atol = 0):
`n_bins_`, every `bin_edges_` array, transforms (including a negative-stride
view), inverse transforms and feature names. Cases: five datasets (Gaussian,
skewed with repeated integers, a large offset, constant columns, few distinct
values), Float32 and Float64; uniform, k-means and all nine quantile methods;
weighted quantile (both methods, zero and integer weights), weighted uniform
(including negative weights), weighted k-means; onehot-dense with per-feature
`n_bins`; both output dtypes; column-major and reversed-row inputs; scalar
weights; feature names; subsampling with two seeds, with and without weights,
and a borrowed RandomState over two fits; empty-cluster relocation; refits and
ordinal inverse indexing. 25 error cases cover parameter validation, `n_bins`
arrays, weights, k-means sample counts, NaN, unfitted use, feature counts and
inverse shapes/indices. The oracle runs with `OMP_NUM_THREADS=1`; its k-means
is loaded from the wheel's compiled `sklearn.cluster`, whose sources the test
verifies are identical between sklearn 1.9.1 and the pinned revision.
