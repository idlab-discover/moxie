# Categorical feature columns

Contract version **1**, experimental. Import `CategoricalColumn`,
`StringCategories`, `NumericCategories`, and `categorical_column` from
`moxie._categorical`. This is shared native storage for feature columns, not
an estimator or a replacement for classification targets in `moxie._labels`.

The reference is scikit-learn at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, particularly
`sklearn/utils/_encode.py`, its tests, and `sklearn/preprocessing/_encoders.py`.
Scikit-learn does not expose this column class: the Mojo representation is
explicitly Moxie-specific. Category identity, inferred sorting, homogeneous
scalar domains, and explicit dictionary validation follow that reference.

## Construction and access

```mojo
from moxie._categorical import categorical_column

var cells: List[Optional[String]] = [
    Optional[String]("blue"), None, Optional[String]("amber"),
]
var color = categorical_column(cells)
# Dictionary: ["amber", "blue"]; row codes: [1, -1, 0].
var mask = color.missing_mask()  # NuMojo rank-one Boolean array
var restored = color.values()  # List[Optional[String]]
```

`categorical_column` infers a sorted dictionary from observed values:

| Input | Domain | Missing representation |
| --- | --- | --- |
| `List[Optional[String]]` | Owned UTF-8 strings | `None` |
| Rank-one `NDArray[bool/int64/float32/float64]` | Same numeric dtype | Optional rank-one Boolean `missing` mask; floating NaN also means missing |

Numeric inputs use NuMojo storage. The optional mask is passed as
`Optional[NDArray[DType.bool]]`. Masked numeric payloads are ignored. Unmasked
infinities raise `ValueError`. Strings compare without Unicode normalization;
empty strings and literal `"None"`/`"nan"` are ordinary categories. Int64 values
never pass through floating point. Numeric identity follows equality, so signed
zeros represent one category; bitwise preservation of signed zero and NaN
payloads is not promised.

For a fixed dictionary, construct `StringCategories(values)` or
`NumericCategories[dtype](values)`, then use
`CategoricalColumn[Domain].from_values(domain, nullable_values)`.
Explicit string dictionaries preserve caller order. Numeric dictionaries must
be finite, sorted, and unique. Both reject duplicates and can include unused
categories. `from_values` rejects unknown values; it never grows the dictionary.
Numeric `from_values` accepts `None` for missingness; normalize floating NaNs
through `categorical_column` first or supply `None` explicitly.

Alternatively, `CategoricalColumn(domain, codes)` accepts a rank-one Int64
NuMojo array. Codes must be `-1` (missing) or valid dictionary indices. Unknown
categories are errors, never silently converted into missing values.

| Method | Result / behavior |
| --- | --- |
| `rows()` | Number of rows |
| `domain()` | Independent dictionary copy; exposes `size()`, checked `get()`, `find()` (`-1` if absent), and `values()` |
| `codes()` | Independent compact Int64 array |
| `get(row)` | `Optional[Domain.Value]`; checked bounds |
| `values()` | Independent list of optional decoded scalars |
| `set(row, value)` | Replace one cell with an existing category or `None`; errors leave it unchanged |
| `missing_mask()` | Independent rank-one Boolean NuMojo array |
| `take(indices)` | Independent column in requested row order; repeats allowed, negative indices rejected; retains the complete dictionary |
| `.copy()` | Independent owned column |

Constructors borrow and copy their inputs. Positive/negative-strided NuMojo
vectors are read in logical order and compacted. Accessors expose no mutable
aliases. Empty and all-missing columns are valid, including an empty dictionary;
estimators must impose their own minimum-sample requirements. Do not mutate
underscore fields. Custom `CategoryDomain` implementations must provide an
immutable unique dictionary, stable equality, checked access, independent copies,
and `find` returning either a valid index or `-1`.

## Missingness and integration boundaries

Scikit-learn's encoding helpers include missing sentinels in their category
arrays. This storage contract instead excludes missing values from the dictionary
and reserves code `-1`. Differential tests apply this explicit remapping. It
must not be confused with sklearn's configurable unknown-category encoding.
Families decide whether a missing value is imputed, rejected, or included in
an estimator's learned category array.

The numeric factory normalizes NaN and explicit masked cells into one missing
tag. This version cannot preserve distinct Python `None`, NaN, and pandas.NA
sentinels in a single object column. It also excludes arbitrary Python objects,
bytes, mixed string/numeric scalars within a column, ordered-category comparison,
automatic dtype promotion, and serialization adapters.

This module does not yet provide a heterogeneous table, estimator adapters,
standalone categorical `MissingIndicator`, or changes to impute/compose.
Families can exchange columns through this single core owner and adopt it
explicitly through their dependency manifests. Include the `moxie._categorical`
module and the existing pinned NuMojo dependency; do not copy its implementation.

## Validation

`pixi run build` compiles the module, native contracts, and example.
`pixi run test` runs ownership, mutation, selection, shape, invalid-code,
empty/all-missing, Unicode, all supported dtype, NaN/infinity, and large-integer
contracts, plus exact comparisons against the pinned local sklearn helpers.
The comparison harness executes original helper bodies with NumPy namespace
plumbing; it verifies the source revision and cleanliness. Empty object inverse
encoding is unsupported by the pinned helper, so that empty-code behavior is
covered as a native storage contract. No numerical tolerance is used.

Sorting uses standard-library `sort(Span(...))`; strings use standard-library
`Dict` lookup. Numeric dictionaries, row codes and masks use NuMojo. Temporary
lists hold sorting and nullable decoding workspaces. This is a correctness
baseline; no performance or complete sklearn feature-parity claim is made.
