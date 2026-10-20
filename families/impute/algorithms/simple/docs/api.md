# Dense simple imputation (partial implementation)

Import `SimpleImputer`, `IntegerSimpleImputer`, `StringSimpleImputer`,
`StringMissingIndicator`, and `MissingIndicator` from `moxie.impute` in the generated namespace. The family root and this algorithm
both support `pixi run build`, `pixi run test`, and `pixi run parity` with the
same verified family modules. The full assigned dense scope is **not complete**:
non-string/heterogeneous categorical imputation and indicators remain
implementation work. Dense integers and strings use the
`IntegerSimpleImputer` and `StringSimpleImputer` specializations below.

`SimpleImputer[DType.float64]` (the default) and `[DType.float32]` accept NuMojo
rank-2 arrays. Constructor keywords are `missing_values=MissingValue[dtype]()`,
`strategy="mean"`, `fill_value=None`, `copy=True`, `add_indicator=False`, and
`keep_empty_features=False`. Use `MissingValue[dtype](value)` for a finite
sentinel; the default denotes NaN. Strategies mean, median, most_frequent and
constant use observed values only. Ties choose the smallest value. Empty columns
are dropped, including constant strategy; keep_empty_features fills empty columns
with zero, or with the constant fill value. Default numeric constant is zero.

Native callbacks implement `_simple.ColumnStrategy`: a default-constructible,
copyable object with `__call__(values: NDArray[float64]) raises -> Float64`.
Instantiate `SimpleImputer[dtype, YourStrategy](strategy="callable",
callback=YourStrategy(...))`. The callback receives a compact owned vector of
observed values, including zero-length vectors. Callback results control
statistics and feature removal exactly as numeric statistics do. A callback
error aborts fit without publishing partially learned state.

`MissingIndicator[dtype](features="missing-only", sparse="auto",
error_on_new=True)` returns an owned Boolean NDArray. `features="all"` includes
all original features. Numeric signed/unsigned integer indicator inputs use exact
sentinel comparison. Both sparse="auto" and sparse=False return dense Boolean
arrays. Selecting sparse=True raises because sparse output is deferred; strings
other than "auto" also raise at fit. The typed parameter record stores sparse as
`Variant[Bool, String]`, preserving the selected value through parameter updates
and cloning. Float16, Boolean data inputs and
complex numbers are not advertised.

Learned attributes are `statistics_` (Float64 vector), `indicator_` (optional
MissingIndicator), `features_` (indicator's List[Int]), `n_features_in_`, and
`feature_names_in_` (optional owned List[String]). Unfitted dimensions are zero;
methods still raise NotFittedError-prefixed Error. `fit_named(X, names)` retains
owned name metadata; ordinary fit clears it. `get_feature_names_out` validates
explicit names against fitted names and otherwise generates x0, x1, etc.

`fit` and `set_params` mutate in place. `fit_transform` returns the transformation.
`get_params` returns the typed record; `set_params(record)` replaces configuration
and clears fitted state. Named updates cover missing_values/strategy/fill_value/copy/add_indicator/
keep_empty_features/callback for SimpleImputer and missing_values/features/
sparse/error_on_new for MissingIndicator. Omitting fill_value preserves the
previous setting; `set_params(fill_value=None)` restores its default. Markers
use `MissingValue[dtype]`, and callbacks retain the estimator's compile-time
callback type. Updates clear fitted state, so refit before transformation.
The record's `copy_input` represents constructor `copy`, because Mojo's Copyable
trait already reserves `copy()`. `moxie.clone(model)` copies configuration only.

Fit and transform borrow input. Outputs always own independent storage, including
copy=False, following the family ownership contract. Copy is a request to avoid
unnecessary copies, not an in-place guarantee. Negative strides and offsets are
supported. `transform[DType.float32](X)` explicitly selects cross-dtype floating
output when the estimator was fitted as Float64 (and conversely). Stored
statistics are cast only at assignment. Callbacks see Float64 values.

Inverse requires add_indicator=True and returns Float64, following sklearn.
Only fitted indicators can restore missingness. The pinned reference infers
all-empty original columns from whole indicator columns during inverse; retained
columns which are entirely flagged can exhaust its traversal. This implementation
raises a descriptive ValueError-prefixed Error in that case, rather than reading
outside the result. Feature names likewise follow statistics validity, including
NaN callback statistics even with keep_empty_features=True.

Errors use Mojo Error with ValueError/NotFittedError prefixes. Python warnings,
reflection, metadata routing, pandas and arbitrary heterogeneous Python objects
remain binding concerns. This does not exclude native categorical imputation,
which is still required unfinished work. See [family scope](../../../docs/scope.md).

## Integer specialization

Import `IntegerSimpleImputer` from `moxie.impute`. It specializes the same
implementation as `SimpleImputer`, with signed/unsigned 8-, 16-, 32- and 64-bit
NuMojo arrays. Specify the dtype explicitly: `IntegerSimpleImputer[DType.int64]`.
Constructor defaults, typed parameters, callbacks, lifecycle, feature names,
indicators and inverse behavior follow the floating estimator above.

Because strategy is a runtime parameter, `transform` and `fit_transform` return
`Variant[NDArray[dtype], NDArray[DType.float64]]`. Constant and most_frequent
produce the integer arm; mean, median and callable produce the Float64 arm.
This explicit alias avoids changing the return type of existing floating
`SimpleImputer` callers. The implementation's third type parameter is internal;
use the public names and their documented dtype/callback parameters.

```mojo
import numojo as nm
from moxie.impute import IntegerSimpleImputer
from moxie.impute._plumbing import MissingValue

var X = nm.zeros[DType.int64](nm.Shape(3, 1))
X.unsafe_set(0, 1)
X.unsafe_set(1, 4)
var model = IntegerSimpleImputer[DType.int64](
    missing_values=MissingValue[DType.int64](0))
var output = model.fit_transform(X)
print(output[nm.NDArray[DType.float64]])  # [[1], [4], [2.5]]
```

A runnable example is in [examples/integer.mojo](../examples/integer.mojo).

For constant/most_frequent, sentinel comparison and observed values stay exact
in the input dtype. Mean/median/callable convert to Float64 before masking, as
scikit-learn does; adjacent integers beyond 2**53 may therefore compare equal.
The default NaN marker finds no missing integer entries. Integer transforms
accept the same input dtype used for fitting; cross-dtype overloads remain
limited to floating estimators. Integer `inverse_transform` takes the returned
Variant and produces Float64, with the same possible 64-bit rounding as sklearn.

Constant fills remain exact across the entire integer range. `statistics_`
retains this library's Float64 metadata representation, so its displayed constant
can round even when the actual fill is exact (sklearn uses object statistics for
constant). Most-frequent follows sklearn's Float64 statistics and their rounding.
If a modal Int64/UInt64 value rounds beyond its representable range, replacing
missing entries raises a ValueError instead of performing an undefined native
cast; NumPy may warn and produce a wrapped value in this case.

`pixi run parity` includes integer differential checks against the pinned sklearn
source. Integer outputs are compared exactly; Float64 results use rtol/atol
3e-13 for reduction-order differences on the tested cases.

## String specialization

`StringSimpleImputer` supports owned UTF-8 strings with `strategy="most_frequent"`
or `strategy="constant"`. Mode ties choose the lexicographically smallest string,
without case folding or Unicode normalization. Constant uses `fill_value`, or
`"missing_value"` when it is None. The constructor retains sklearn's default
`strategy="mean"`, so select a supported strategy explicitly; this specialization
does not parse numeric-looking strings or accept numeric/callable strategies.

`StringMatrix(cells, rows, columns)` copies a row-major
`List[Optional[String]]`. Each None is a missing tag, distinct from the empty
string. The default `missing_values=None` selects those tags (the equivalent of
NaN cells in sklearn object arrays). Set `missing_values="NA"`, for example,
to select an exact string sentinel instead; a tagged missing cell with a string
sentinel is rejected. Python None as an observed category is outside this type.
`StringMatrix.from_strings(values, rows, columns)` accepts a `List[String]`.
Checked `get(row, column)`/`set(row, column, value)`, `rows()`, `columns()` and
`values()` provide access. Construction, copying and `values()` own their data;
there is no strided or aliased string view API. Shape mismatches raise ValueError.
Zero dimensions can represent outputs, but fitting/transforming require positive
input dimensions.

```mojo
var cells: List[Optional[String]] = ["blue", None, "red"]
var X = StringMatrix(cells, 3, 1)
var model = StringSimpleImputer(strategy="most_frequent", add_indicator=True)
var output = model.fit_transform(X)
print(output.values.get(1, 0).value())  # blue
print(output.indicators)              # Boolean, shape (3, 1)
```

Import all three public types from `moxie.impute`: `StringSimpleImputer`,
`StringMatrix`, and `StringImputationResult`. See the runnable
[string example](../examples/string.mojo).

`fit_transform`/`transform` return `StringImputationResult`: `values` holds the
retained StringMatrix columns and `indicators` holds a Boolean NDArray. These are
separate typed components because a string matrix cannot contain Boolean cells.
Their logical column order matches sklearn: retained values followed by fitted
indicator columns. `shape` records `(rows, values.columns() + indicators.shape[1])`;
without indicators, the Boolean component has zero columns. No indicator is added
for missingness first encountered during transform. Both components own their
storage even with `copy=False`.

`statistics_` is `List[Optional[String]]`, with None for an empty dropped feature;
`n_features_in_`, `feature_names_in_`, feature naming, optional ignored `y`, cloning,
repeat fits and parameter resets follow the existing estimators. `indicator_`
contains the existing MissingIndicator fitted on the original Boolean mask;
its `features_` records original indices. Raw-string standalone MissingIndicator
is not added by this specialization. `get_params` returns `StringSimpleParams`
(from `_string`); named updates accept all constructor fields. Omitted nullable
parameters stay unchanged; explicit `fill_value=None` or `missing_values=None`
restores their defaults. Failed fits preserve the previous fitted state.

Empty features are dropped by both strategies unless `keep_empty_features=True`.
Constant can retain them using its string fill. Most-frequent raises on a retained
empty feature: sklearn fills it with numeric zero, which cannot be represented
as a string without changing its meaning. Choose constant or drop empty features.
`inverse_transform` raises an explicit unsupported error for string results;
the pinned sklearn inverse allocates Float64 and cannot restore arbitrary strings.

This covers strings and their missing tags, not heterogeneous categories,
category metadata, pandas categorical arrays, arbitrary Python objects, or sparse
storage. Exact differential checks compare values, missing statistics, Boolean
indicators and feature names with sklearn's object-string implementation.

## Standalone string missing indicators

`StringMissingIndicator` is the native string specialization of sklearn's
`MissingIndicator`. It accepts `StringMatrix` and returns an owned Boolean
NuMojo NDArray. Constructor parameters are `missing_values=None`,
`features="missing-only"`, `sparse="auto"`, and `error_on_new=True`.
Here `None` denotes the matrix's explicit missing tags, corresponding to NaN
in the reference object arrays. A string sentinel uses exact UTF-8 equality;
missing tags with a string sentinel raise a ValueError.

`fit`, `transform`, `fit_transform`, `fit_named`, `get_feature_names_out`,
`get_params`, `set_params`, and cloning follow the numeric indicator contract.
Learned attributes are `features_`, `n_features_in_`, and optional owned
`feature_names_in_`. Output names retain the `missingindicator_` prefix.
Both `sparse=False` and `sparse="auto"` produce dense output; `sparse=True`
raises. Named parameter updates clear fitted state. Omitting `missing_values`
preserves the marker; passing `missing_values=None` resets it to missing tags.

The contract test also consumes the estimator through shared generic traits.
The string parity runner compares standalone masks, feature names, and errors
exactly against the pinned sklearn estimator, including Unicode sentinels,
all-missing columns, complete training data, and new transform missingness.
