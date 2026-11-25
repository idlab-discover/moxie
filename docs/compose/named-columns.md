# Native named columns

`ColumnTable[dtype]` adds column names to an owned NuMojo matrix. It accepts
rank-2 Float32 or Float64 data and copies the input matrix and names.
`add_categorical(name, column, position=None)` copies a
`CategoricalColumn[Domain]` into the same table. The supported domains are
`StringCategories` and `NumericCategories[dtype]` for Bool, Int64, Float32,
and Float64. The position is
zero-based and defaults to the end. The name must be unique and the row
count must match the numeric block. `add_strings(name, column, position=None)`
accepts a `List[String]` with the same row count and insertion rules. A matrix
with shape `(rows, 0)` can seed a table whose columns are all added individually.
`add_numeric(name, column, position=None)` takes a rank-one Float32, Float64,
Int64, or Bool NuMojo array with the same row count and insertion rules. It copies
the logical values from strided inputs into compact storage of the original
column dtype.

Copying a table creates an independent snapshot. `schema()` returns its own
`ColumnSchema` with actual storage kinds
(`float32`, `float64`, `int64`, `bool`, `category`, `string`).
`values()` borrows the original matrix when it represents the entire table.
It raises once any column has been added individually.
`categorical_columns[Domain](indices)` copies categorical
columns in selection order, preserving the dictionaries and missing codes.
Omit the domain to read string categorical columns.
`string_column(index)` copies the raw text at a zero-based column position.
Raw text stores UTF-8 strings directly, including empty strings. Use a
nullable string categorical column when you need an explicit missing value.

`take_rows(indices)` returns an independent table in the requested row order.
It preserves column names, storage dtypes, category dictionaries (including
unused values), and missing codes. Pass a rank-one Int64 NuMojo array such as
`IndexSplit.train` or `IndexSplit.test` from a model-selection splitter. Repeated
indices, strided indices, and an empty selection are supported. Indices are
zero-based and must be nonnegative, following `CategoricalColumn.take`.
The native table stores rows by position and carries no row labels.

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
numeric arrays, typed categorical columns, or a list of documents. The output remains an NDArray; use `get_feature_names_out()`
for its names.

A named fit records `feature_names_in_`. At transform time, named inputs may
reorder columns or add extra columns. Missing columns raise only when a fitted
active component needs them, including an active remainder. Dropped columns
may be omitted. Array inputs retain the positional feature-count contract.
A subsequent array fit clears learned names; failed fits preserve previous
state. Supplied `get_feature_names_out(input_features)` must match fitted names.

`make_column_selector(dtype_include=..., dtype_exclude=...)` returns a named
key from a table or schema. Use `["number"]` for floating-point and integer columns or
`["category"]` for categorical columns of any domain and `["string"]` for
raw text. `"floating"` matches both float dtypes. `"integer"`,
`"signedinteger"`, and `"int"` match Int64 storage. `"float32"`, `"float64"`,
and `"int64"` select exact dtypes. Use `["bool"]` for raw Boolean columns or
`["number", "bool"]` to include them in a numeric branch. Boolean categorical
columns are selected by `["category"]`. Typed selectors take lists of dtype names.
Regex `pattern` raises an unsupported-capability error.

## Mixed numeric and categorical preprocessing

```mojo
from moxie._categorical import categorical_column
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer, make_column_selector
from moxie.impute import SimpleImputer
from moxie.pipeline import make_pipeline
from moxie.preprocessing import OneHotEncoder, StandardScaler

# X is a three-row Float64 matrix with column names in names.
var table = ColumnTable(X, names)
# ratings is a three-element Float32 NDArray. The selector sees dtype "float32".
table.add_numeric("rating", ratings)
var cells: List[Optional[String]] = ["red", "blue", "red"]
table.add_categorical("color", categorical_column(cells))
var numeric_dtypes: List[String] = ["number"]
var categorical_dtypes: List[String] = ["category"]
var model = ColumnTransformer(
    ColumnEntry("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                make_column_selector(dtype_include=numeric_dtypes^)),
    ColumnEntry("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                make_column_selector(dtype_include=categorical_dtypes^)),
)
var result = model.fit_transform(table)
```

A categorical component receives `List[CategoricalColumn[Domain]]` according
to its associated input type. For example,
`OneHotEncoder[NumericKind[DType.int64]]` receives Int64 categorical columns
and preserves their identities through encoding, including values above
`2**53`. Import `NumericKind` from `moxie.preprocessing._vocabulary`.
Each branch must select a single domain. A branch that consumes a numeric
categorical domain can also select raw columns of the exact same dtype.
For example, use `ColumnEntry("product", OneHotEncoder[NumericKind[DType.int64]](sparse_output=False), product_names)`
with a name list `product_names` and raw Int64 product IDs. Compose constructs
the categorical input from the selected raw values on each call and treats
floating NaN as missing. The column retains its raw dtype in the schema, so
select it by name or raw dtype. Split different domains across entries and
exclude raw columns from the numeric pipeline's selector when they should
only be encoded. Use explicit categorical storage for a supplied dictionary
or a separate missing mask.

String-domain encoders also accept raw columns added with `add_strings`.
Use a name list or a selector such as `dtype_include=string_kinds^`, where
`string_kinds` is a `List[String]` containing `"string"`. Include
`"category"` as well to select explicit string categorical columns in the
same branch when the table's categorical columns all have the string
domain. Compose infers the category dictionary from each raw column on every
call. Empty strings are ordinary categories. Use `add_categorical` with a
nullable string column to represent missing values. A fitted string encoder
can receive either storage form for the same selected name at inference.
The table's `categorical_columns()` accessor still reads explicit categorical
storage.

A list, mask, slice, or dtype selector supplies a collection of features. The component's
target and dense result dtype must match the table dtype. Numeric components,
including `"passthrough"`, gather selected numeric columns into an array of
the table dtype in key order. The constructor matrix sets that dtype.
At this boundary, raw Boolean values become floating-point zero or one.
The conversion can round added Float64 values in a Float32 table and Int64
values outside the exact integer range of either float dtype. Route Int64
values to an Int64 categorical component when exact category identity matters. Fitted
numeric branches accept any supported numeric storage dtype for a fitted name.
A change from numeric to categorical or text storage raises. Use `"drop"` for unused
categorical columns. An NDArray selector callable requires a table backed
entirely by its constructor matrix.

Use a scalar key to route one document column to a text transformer. With a
Float64 table, for example, `ColumnEntry("words", CountVectorizer[DType.float64](), "review")`
accepts the raw column added by `table.add_strings("review", documents)`.
The vectorizer output dtype must match the table dtype. A singleton name list
or dtype selector represents a two-dimensional selection and raises for a
text transformer. Dtype selectors can identify text columns before you pass
a scalar name to the entry.

Categorical selections also support `OrdinalEncoder` and `TargetEncoder`
with the corresponding domain. For `TargetEncoder`, use a Float64 table and a Float64
target array. Pass `y.copy()` to keep the target or `y^` to transfer it when
calling `fit_transform(table, y)` or `fit(table, y)`. The target is required
when an active entry uses TargetEncoder. The training output uses out-of-fold
encodings, and later `transform` uses the encodings learned on all training
rows.

For nullable string categories, replace the categorical entry with a pipeline
that imputes missing cells before encoding:

```mojo
from moxie.impute import StringSimpleImputer

var categorical = make_pipeline(
    StringSimpleImputer(strategy="most_frequent").as_categorical(),
    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
)
var kinds: List[String] = ["category"]
var entry = ColumnEntry("cat", categorical^, make_column_selector(dtype_include=kinds^))
```

`as_categorical()` copies the imputer configuration into an unfitted transformer
that consumes and emits lists of string categorical columns. The pipeline uses
the learned imputer to fill missing cells and passes names of retained columns to
the encoder. This entry requires a Float64 table and a list-style column key.
See the [string imputer API](../../families/impute/algorithms/simple/docs/api.md#string-specialization)
for strategy and indicator behavior.

For nullable numeric categories, use the same pipeline structure with
`CategoricalNumericImputer[dtype]()` and `OneHotEncoder[NumericKind[dtype]]`.
The category dtype must match both stages and remains independent of the
Float64 table dtype. The numeric imputer fills tagged missing cells without
casting category values and can drop all-missing training columns. Feature
names and output spans reflect the retained columns.

The matrix-regressor overloads of TransformedTargetRegressor require tables
backed entirely by their constructor matrix. They store names and validate
their order at prediction.

Native composition runs in Mojo. The named and mixed column parity suites
use pandas solely for Python oracle inputs. The mixed parity case
compares a single ColumnTransformer with median imputation, scaling and one-hot
encoding, including reordered inputs, unknown categories, names and output spans.
It also checks two categorical columns imputed before encoding and an all-missing
training column dropped by the imputer. Another case checks float dtype
selection and the same numeric pipeline on a Float64 matrix column and a
negative-strided Float32 column, with Float64 output.
The integer case runs the same pipeline over an Int64 column and a Float64
column, and compares dtype selection, reordered inference, conversion beyond
Float64's exact integer range, and output metadata.
The Boolean case checks selection, owned strided input, reordered inference
and the dense pipeline over raw Boolean and Float64 inputs.
Raw category cases compare Int64 identity beyond `2**53` and Float64 NaN
handling through a dense OneHotEncoder alongside the numeric pipeline.
The Float64 case selects both a negative-strided matrix column and an added
column in the same branch.
The raw string case combines a raw column with nullable string categories
in one encoder and checks inference with the two storage forms swapped. It
compares fitting and inference on reordered, repeated row selections against
scikit-learn's `_safe_indexing`, including the resulting names and output spans.
The mixed encoders parity case also compares ordinal codes and multiclass
target encodings with exact equality, including cross-fitted training output.
The mixed text comparison checks scalar name and position routing with
CountVectorizer alongside the numeric pipeline and OneHotEncoder. Counts,
one-hot values, output names and output spans are checked exactly.

The [mixed-types notebook](../../notebooks/compose/plot_column_transformer_mixed_types.ipynb)
puts both transformer pipelines in one ColumnTransformer. The categorical
branch chains dense OneHotEncoder with SelectPercentile using a chi-squared
scorer. Pass a Float64 target array to the table preprocessor and use the
original Int64 labels when fitting the downstream LogisticRegression.

The notebook runs with seeded data by default. To compare on the gallery's
OpenML Titanic version 1 rows, run only this notebook:

```bash
MOXIE_TITANIC_OPENML=1 pixi run -e notebooks notebooks notebooks/compose/plot_column_transformer_mixed_types.ipynb
```

The OpenML download is cached under `builds/titanic-reference`. The comparison
uses the gallery's 80/20 split with seed 0 and its named route's feature groups.
It casts passenger class to a string category so dtype selectors pick the same
columns as the named route. Missing embarkation values stay missing through
the notebook's exchange with Mojo.

## Predictive pipelines on mixed tables

Use `as_table_transformer()` when a generic pipeline needs
`ColumnTable[dtype]` as its associated input. The method moves a
`ColumnTransformer` and its fitted state, if present, into a
`TableColumnTransformer`. Its dense output becomes the input of the next
stage. The example reuses the numeric and categorical entries above.

```mojo
from moxie.linear_model import Ridge

var pipe = make_pipeline(model^.as_table_transformer(), Ridge())
_ = pipe.fit(table, y)
var predictions = pipe.predict(test_table)
var names = pipe.transformer.get_feature_names_out()
var spans = pipe.transformer.columns.output_indices_
```

Access the underlying transformer through `pipe.transformer.columns`.
Pipeline parameter records expose the same ColumnTransformer configuration,
and cloning the pipeline clones that configuration into an unfitted wrapper.
Match the predictor's target dtype to the table dtype when the column
transformer has a supervised branch such as TargetEncoder. For a predictor
with a different target type, Pipeline supplies no target to the column
transformer, so all active branches must support fitting without one.

The [mixed Ridge comparison](../../tests/compose/pipeline/mixed_ridge_parity.py)
checks reordered inference, unseen categories, predictions, R², cloning,
parameter updates and refitting against pinned scikit-learn. Encoded columns,
feature names and output spans match exactly. Native schema validation uses
Mojo errors with descriptive ValueError messages.

Use the same mixed-table pipeline in cross-validation and grid search by
passing the table's row operations to the model-selection adapter.

```mojo
from moxie.model_selection import DenseSupervisedAdapter, GridSearchCV, cross_val_score

comptime Adapter = DenseSupervisedAdapter[type_of(pipe)]
comptime Table = ColumnTable[]
var adapter = Adapter(
    apply_params, input_rows=Adapter.InputRows(Table.rows, Table.take_rows),
)
var scores = cross_val_score[Adapter](pipe, table, y, adapter)
var search = GridSearchCV[Adapter](pipe, grid, adapter)
_ = search.fit(table, y)
```

The typed `apply_params` callback maps the grid keys to the pipeline's
parameter record. Each fold fits its own imputer, scaler, encoder and
predictor. The [mixed Ridge search regression](../../tests/compose/pipeline/mixed_ridge_search.mojo)
shows how to tune `ridge__alpha` and predict with the refitted pipeline.
