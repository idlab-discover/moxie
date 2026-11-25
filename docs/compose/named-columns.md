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
`add_numeric(name, column, position=None)` takes a rank-one Float32 or Float64
NuMojo array with the same row count and insertion rules. It copies the logical
values from strided inputs into compact storage of the original column dtype.

Copying a table creates an independent snapshot. `schema()` returns its own
`ColumnSchema` with actual storage kinds (`float32`, `float64`, `category`, `string`).
`values()` borrows the original matrix when it represents the entire table.
It raises once any column has been added individually.
`categorical_columns[Domain](indices)` copies categorical
columns in selection order, preserving the dictionaries and missing codes.
Omit the domain to read string categorical columns.
`string_column(index)` copies the raw text at a zero-based column position.
Raw text stores UTF-8 strings directly, including empty strings. Use a
nullable string categorical column when you need an explicit missing value.

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
key from a table or schema. Use `["number"]` for all numeric columns or
`["category"]` for categorical columns of any domain and `["string"]` for
raw text. `"floating"`, `"float32"`
and `"float64"` also match their respective numeric storage. Typed selectors
take lists of dtype names. Regex `pattern` raises an unsupported-capability error.

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
Each branch must select a single domain. Use explicit column lists to route
different domains to different branches in the same table. A list, mask,
slice, or dtype selector supplies a collection of features. The component's
target and dense result dtype must match the table dtype. Numeric components,
including `"passthrough"`, gather selected Float32 and Float64 columns into an
array of the table dtype in key order. The constructor matrix sets that dtype.
For a Float32 table this conversion can round added Float64 values. Fitted
numeric branches accept either storage dtype for a fitted name. A change from
numeric to categorical or text storage raises. Use `"drop"` for unused
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

Table overloads supplement the existing NDArray-associated estimator traits;
generic pipeline wrappers do not automatically acquire schema routing. Native
schema validation reports descriptive ValueError messages rather than reproducing
pandas/narwhals-specific exception types and Python set formatting.
