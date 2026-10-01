# Native named columns

`ColumnTable[dtype]` adds column names to an owned NuMojo matrix. It accepts
rank-2 Float32 or Float64 data, preserves its dtype, and copies the input data
and names. Copying the table owns another independent snapshot. `values()`
borrows the existing matrix without materializing it; `schema()` returns an
independent `ColumnSchema` snapshot. Schema metadata must retain matching name
and dtype lengths; its dtype names describe actual storage.

```mojo
from moxie.compose import ColumnTable, ColumnTransformer, ColumnEntry, make_column_selector

var names: List[String] = ["age", "income", "height"]
var table = ColumnTable(X, names)
var chosen: List[String] = ["income", "age"]
var model = ColumnTransformer(
    ColumnEntry("numeric", "passthrough", chosen),
    remainder="passthrough",
    n_jobs=2,
)
var result = model.fit_transform(table)
var output_names = model.get_feature_names_out()
```

Named keys accept one name (rank-1 component input), a name list (rank-2), or
`ColumnKey.name_slice(start, stop, step)` with inclusive endpoints (`step` must be None or 1, as in sklearn). Position
lists, masks and numeric slices also work. Duplicate input names are rejected at table construction. Components receive selected
numeric arrays. The output remains an NDArray; use `get_feature_names_out()`
for its names.

A named fit records `feature_names_in_`. At transform time, named inputs may
reorder columns or add extra columns. Missing columns raise only when a fitted
active component needs them, including an active remainder. Dropped columns
may be omitted. Array inputs retain the positional feature-count contract.
A subsequent array fit clears learned names; failed fits preserve previous
state. Supplied `get_feature_names_out(input_features)` must match fitted names.

`make_column_selector(dtype_include=..., dtype_exclude=...)` returns a named
key from a table or schema. Filters accept lists of numeric dtype names, such
as `float32`, `float64`, `floating`, and `number`; include and exclude are
combined. Since a table currently stores one homogeneous matrix, every column
has that matrix's actual dtype. Mixed-type and string columns are not yet
provided. Regex `pattern` raises an explicit unsupported-capability error.

TransformedTargetRegressor's matrix-regressor overloads accept the same table
for fit, weighted fit, predict and score. They store names and validate their
order at prediction; regression does not reorder features automatically.

No pandas/polars runtime or binding is involved. The named-column parity suite
uses pandas solely to construct equivalent inputs for the Python sklearn
oracle. Run `pixi run test` and `pixi run parity` from the Moxie root.

Table overloads supplement the existing NDArray-associated estimator traits;
generic pipeline wrappers do not automatically acquire schema routing. Native
schema validation reports descriptive ValueError messages rather than reproducing
pandas/narwhals-specific exception types and Python set formatting.
