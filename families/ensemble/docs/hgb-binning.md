# Histogram gradient boosting bin mapper

The private `moxie.ensemble._hist_gradient_boosting.binning._BinMapper` fits the
byte bins used by histogram gradient boosting. It lives in the ensemble
package alongside the future histogram and grower modules. The public
`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` remain
raising declarations.

```mojo
from moxie.ensemble._hist_gradient_boosting.binning import _BinMapper
from numojo import NDArray, Shape

def main() raises:
    var X = NDArray[DType.float64](Shape(3, 1))
    X.unsafe_set(0, 1)
    X.unsafe_set(1, 3)
    X.unsafe_set(2, 9)
    var mapper = _BinMapper(n_bins=4, random_state=42)
    var binned = mapper.fit(X).transform(X)
    print(binned)
```

Both operations accept rank-2 Float32 or Float64 NuMojo arrays, including
strided views. Thresholds use Float64, and the result preserves sample and
feature order in a column-major UInt8 array. `n_bins` includes the missing
bin and ranges from 3 through 256. Continuous bins use the distinct-value
midpoints or the averaged inverted CDF, as in the pinned scikit-learn source.
The mapper reserves the last bin for NaNs and negative categorical codes.
Categorical values follow the reference mapper's lower-bound rule. Callers
encode categories first and supply their complete sorted domains.

Pass one optional domain per feature to `fit(known_categories=...)`. An absent
domain denotes a continuous feature, while a present empty domain denotes a
categorical feature with zero known categories. Domain presence supplies the
categorical mask, and fitted domains are copied. Learned thresholds, counts,
the mask and category bitsets use typed lists. Fit returns a mutable borrow
and replaces the fitted state on success. Transform borrows samples and
returns a new array. Fit and transform run serially.

Optional Float64 sample weights must be finite and nonnegative. Zero-weight
observations and NaNs are omitted from continuous threshold fitting. A
continuous column with zero usable observations has no thresholds, one
nonmissing bin and the reserved missing bin. The pinned reference raises
`IndexError` in that case. Above the subsample limit (200,000 by default),
the shared RandomState draws rows with replacement, using normalized weights
as probabilities when supplied. Set `subsample=None` to use every row.
Extreme weight ranges can make a cumulative weight depend on the ordering
of ties, since the native stable sort and NumPy's default argsort can order
weights on equal values differently.

The mapper sorts one column of observations at a time through
`std.builtin.sort`. Numax `sort` takes a `TensorLike` and delegates to the
same native sort after a host copy. Its vector percentile entry point needs
a static rank count and its scalar path does not implement sklearn's
sample-weighted CDF. Numax `searchsorted` expects `TensorLike` and copies
its sorted input to the host. NuMojo supplies stride-aware array storage
and column-major allocation. A binary search over the fitted thresholds
writes each bin directly into the result without intermediate arrays.
Sampling and normalization reuse the ensemble and shared RNG helpers.

Run the focused contract and parity checks from the Moxie root:

```sh
pixi run ensemble-prepare-oracle
pixi run python families/ensemble/tools/check.py hgb-binning
```

The focused command assembles current dependencies, checks ensemble ownership,
and compiles the native contracts and parity driver with
`-O3 --fp-mode=contract=off`. It compares bin thresholds and codes, learned
counts, the missing bin and category bitsets exactly against source-built
scikit-learn at `4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Native contracts
cover layout, striding, state replacement, invalid inputs and degenerate
columns.
