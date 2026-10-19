# Dense decision trees

Import the four estimator names from `moxie.tree` in the assembled namespace.
`DecisionTreeClassifier` defaults to Gini/best splitting; `DecisionTreeRegressor`
to squared-error/best. ExtraTree uses random splitting, with `max_features="sqrt"`
for classification and `1.0` for regression. These are static specializations of
`BaseDecisionTree`, not independent copies of training algorithms.

## Inputs and results

The native boundary accepts NuMojo Float32 feature matrices and Float64 sample
weights. Features may have strides and offsets, including negative strides.
Regressors accept Float64 targets. Classifiers preserve their declared target
type in predictions and `classes_`:

- `DecisionTreeClassifier()` / `ExtraTreeClassifier()`: Float64 (compatible default).
- `DecisionTreeClassifier[label_dtype=DType.int64]()` or `DType.float32`:
  same-dtype NuMojo targets, classes and predictions. Int64 labels never pass
  through Float64; adjacent integers above `2**53` remain distinct.
- `DecisionTreeClassifier[string_labels=True]()` (also ExtraTreeClassifier):
  `moxie._labels.StringTarget` targets/predictions and string lists in `classes_`.

These compile-time options are a Mojo adaptation, not sklearn constructor
parameters; one estimator specialization has one target type across refits.
For type annotations or `conforms_to`, spell the default classifier type as
`DecisionTreeClassifier[]` / `ExtraTreeClassifier[]`; ordinary constructor
calls remain unchanged.
Other numeric dtypes, mixed/object/null labels and Python string dtype widths are
not supported. Numeric labels must be finite integral values. Labels are sorted
using Moxie's shared label encoder; strings compare lexicographically without
Unicode normalization. Numeric targets may be vectors or matrices; strings use
`StringTarget(values)` or row-major `StringTarget(values, rows, columns)`.

`fit(X, y)` replaces learned state after a successful build. A failed fit leaves
the previously fitted tree intact. `fit(X, y, sample_weight=weights)` borrows the
weight array. The positional optional-weight overload accepts an explicit
`Optional[Values]` or a transferred/copied array. Fitted trees retain no aliases
to training data. Cloning copies constructor configuration and is unfitted.

`predict` returns the declared target type, with a vector for one output and
a matrix for multiple outputs. Regression predictions remain Float64. `apply` returns an Int64 vector of leaf IDs. `predict_proba` and
`predict_log_proba` return `List[Values]`, one `(n_samples,n_classes)` matrix per
output, **including the single-output case**. Likewise `classes_` and
`n_classes_` always have one entry per output. These explicit list adaptations
avoid a return type that varies with runtime output count. Regression types
have no probability capability or callable probability method.

`score` implements accuracy, including multilabel subset accuracy, or
uniform-average R2. Like sklearn, classification scoring rejects multiclass
multi-output targets even though training and prediction support them. R2 returns
NaN for a single sample and uses sklearn's finite constant-target convention.

`criterion="absolute_error"` is supported by both regressors for finite numeric
features, including multiple outputs and signed weights. Predictions are weighted
medians. Missing features and categorical splits remain unsupported, as in the
reference. Nonpositive-weight split candidates are ignored.

## Configuration

Constructor keyword names are retained. `get_params()` returns the independently
copied `TreeParams` record; modify its fields and pass it to `set_params()`.
The `params` field exposes this record directly. Dynamic Python keyword mappings,
reflection and metadata routing remain binding work. `feature_names_in_` is an
owned `Optional[List[String]]`: present after fitting a named `TreeTable`, absent
after fitting an unnamed table or numeric matrix. Named inference inputs must
match the fitted names and order for predict, probability, apply and score.
As in sklearn, named/unnamed transitions use positional columns; the Python
warning for these transitions is not emitted.

`min_samples_split` and `min_samples_leaf` accept `Variant[Int,Float64]`:
integers are counts, fractions are resolved with ceiling against training rows.
`max_features` accepts `Variant[Int,Float64,String,NoneType]`: count, fraction,
`"sqrt"`, `"log2"`, or `NoneType()`. Defaults follow sklearn. Optional integer
parameters (`max_depth`, `max_leaf_nodes`, `random_state`) use `Optional[Int]`.

Class weights accept `NoneType()`, `"balanced"`, a
`Dict[Label,Float64]` for one output, or a list of such dictionaries for multiple
outputs. `Label` is String, Int64, Float32 or Float64 according to the classifier
specialization; defaults remain Float64. Balanced weights use original target frequencies and multiply sample
weights. Signed weights are admitted when node/child weight and class-count
requirements permit a split; zero-weight samples are omitted from node counts.
Total training weight must be finite. All-zero input sample weights raise;
signed weights with zero/negative totals are accepted except for absolute error,
whose reference weighted median is undefined in that case. NaN/Inf statistics
from zero totals are preserved. One-class monotonic clipping uses only its
existing probability slot. The pinned source
can produce signed class proportions for signed weights; this behavior is
preserved, rather than silently clipping or rejecting those fitted values.

`random_state` integers recreate shared MT19937 state on each fit. For `None`,
native fitting uses the shared `entropy_state()` API. Python's process-global
NumPy singleton is not exposed. `fit_with_random_state(X,y,rng,weights)` accepts
an explicit mutable shared context to preserve stream consumption across trees.
Each tree draws exactly one seed before using its tree-local xorshift stream.

`monotonic_cst` is `Optional[List[Int]]`, with signs -1, 0, or 1. Only single-output
regression and binary classification are supported, with no missing inputs.
Best and random splitters propagate bounds and clip values in both depth-first
and best-first construction. Categorical columns must have zero monotonic signs.

`categorical_features` accepts `None`/`NoneType()`, `List[Int]` indices, `List[Bool]`
masks, `List[String]` column names, or `"from_dtype"`. The earlier
`Optional[List[Int]]` form remains accepted. Nonempty masks must have one entry
per feature; names require a named table and must exist. Empty selections mean
no categorical features. `"from_dtype"` selects the `TreeTable.categorical_mask`
metadata; an ordinary matrix has no categorical dtypes and selects nothing.

`TreeTable(columns, feature_names=names, categorical_mask=mask)` snapshots both
optional metadata lists. Without a mask, shared string categorical columns are
categorical by dtype and Float32 columns are numerical. An explicit dtype mask
can also mark numeric columns as categorical. The mask describes input dtypes;
selection is still controlled by `categorical_features`, whose default is None.
String columns must be selected to fit. Numeric feature identities still use
Float32; convert only when that preserves the identities you need. This native
table is a Mojo adaptation of dataframe metadata, not a pandas binding.

```mojo
from moxie._labels import StringTarget
from moxie._categorical import categorical_column
from moxie.tree import DecisionTreeClassifier, TreeTable, TreeColumn

var colors: List[Optional[String]] = [String("red"), String("blue"), String("red")]
var X = TreeTable([TreeColumn(categorical_column(colors))],
                  feature_names=Optional[List[String]](["color"]))
var y = StringTarget(["warm", "cool", "warm"])
var model = DecisionTreeClassifier[string_labels=True](
    categorical_features="from_dtype", random_state=0)
model.fit(X, y)
var prediction = model.predict(X)  # StringTarget: warm, cool, warm
# model.classes_[0] == ["cool", "warm"]
```

Fit retains independent sorted numeric or string dictionaries from `moxie._categorical`.
Raw category values need not be integers or consecutive. Prediction, probabilities,
scoring and `apply` encode using the fitted dictionaries; unknown values and NaNs
follow the learned missing branch. `is_categorical_` is an optional Boolean list,
with `None` when no feature is categorical. Fitting materializes encoded features;
prediction materializes them only for categorical models.

Best search uses weighted target-mean category ordering and bitset partitions,
with at most 256 nonmissing categories per feature. It supports single-output
regression and binary classification. Random search uses sklearn's hash-seed draw
and retry protocol, accepts up to 2**24 categories, and supports multiclass and
multioutput targets. Categories with zero sample weight remain in the fitted
dictionary. Both searches reject nonzero monotonic signs on categorical columns.
Absolute-error categorical fitting is unsupported. The existing native exclusion
of missing values with `monotonic_cst` also applies to unknown categories.

## Mixed string/numeric tables

`TreeTable`, `TreeColumn`, and `StringColumn` are Moxie-specific records exported
from `moxie.tree`. A table snapshots an ordered list of rank-one Float32 arrays
and shared `CategoricalColumn[StringCategories]` columns. All columns must have
the same length. Use `categorical_column` to construct strings with `None` for
missing cells; empty strings and literal `"nan"` are ordinary categories.

```mojo
from moxie._categorical import categorical_column
from moxie.tree import TreeTable, TreeColumn, DecisionTreeClassifier, export_text

# height is a rank-one Float32 NuMojo array; y is Float64.
var colors: List[Optional[String]] = [String("blue"), String("red"), None]
var X = TreeTable([TreeColumn(height.copy()), TreeColumn(categorical_column(colors))])
var model = DecisionTreeClassifier(categorical_features=Optional[List[Int]]([1]), random_state=0)
model.fit(X, y)
var prediction = model.predict(X)
print(export_text(model, feature_names=Optional[List[String]](["height", "color"])))
```

Select string columns by index, mask, name, or `"from_dtype"`; numeric columns
may also be categorical. Direct Python dataframe/object inputs and mixed scalar
types within one column are not supported. Fitting learns
sorted dictionaries from observed cells, ignoring unused input dictionary entries.
Table column order and string/numeric domains must match at prediction time;
a matrix cannot stand in for a fitted string schema. Tables also work with all
four estimators' probability, scoring, apply, pruning-path, and explicit-RNG
methods where the estimator supplies that capability. Generic estimator traits
retain their existing numeric `Input` type; table support is through overloads.

Tables materialize a dense Float32 matrix before entering the shared engines.
Numeric strides are honored. Fitted dictionaries are independent snapshots;
refitting replaces them, failed fits preserve them, and clones are unfitted.
Do not mutate underscore fields or table shape metadata directly.

## Text export

`export_text(model, *, feature_names=None, class_names=None, max_depth=10,
spacing=3, decimals=2, show_weights=False)` returns a native `String` for any
fitted decision or extra tree. Names accept `Optional[List[String]]`; depth,
spacing and decimals use `Int`. Numeric rules, indentation, depth truncation,
weighted class counts and regression values follow the pinned sklearn output.
Floating-point text uses exact fixed-decimal rounding, including ties to even.
Default class labels use the classifier's native label representation. The reference's
one-class report uses class index zero; this behavior is preserved. Multioutput
classification follows the reference's first-class-per-output report convention;
custom `class_names` for multioutput classifiers raise explicitly.

Categorical nodes extend the reference text format: both branches list their
actual category values as `feature in {...}`, with `or missing/unknown` on the
fitted default branch. String values are quoted and escaped. This applies to
both bitset and random hash splits, using the fitted dictionary. The pinned
reference incorrectly displays a numeric threshold for categorical nodes, so
these reports are intentionally not byte-identical. Numeric NaN branch rendering
follows sklearn and does not add a missing-route annotation. See the Graphviz
and SVG sections below for graphical exports.

## Inspection and pruning

Learned state includes `tree_`, `n_features_in_`, `n_outputs_`, `max_features_`,
`classes_`, `n_classes_`, and normalized `feature_importances_`. `tree_` is the
shared `TreeData` record: node metadata is in `tree_.nodes`, values in the owned
Float64 `(node_count,n_outputs,max_n_classes)` array. This is a native record
adaptation, not sklearn's Cython binary layout. Native leaves always use
`split_kind=-1`; sklearn may retain an unused former split kind on a best-first
leaf. `get_depth()` and `get_n_leaves()`
inspect this state. Sparse `decision_path` is absent.

`ccp_alpha` prunes after construction. `cost_complexity_pruning_path(X,y,weights)`
fits an unfitted configuration clone with alpha zero and returns dense
`PruningPath.ccp_alphas` and `.impurities`. Pruning compacts node IDs and does not
mutate its source tree. A best-first candidate converted to a leaf retains its
unused missing-direction field, matching the pinned reference.

All public entry points validate their supported native inputs. `apply` retains
the family `check_input` argument; disabling it does not disable native safety
checks. Other Python `check_input` overloads are not exposed.

## Prepared inputs for ensembles (Moxie extension)

sklearn forests validate `X` once and fit each member through the private
`tree._fit(X, y, sample_weight=..., check_input=False)`. The native equivalent
splits fitting into a reusable preparation and per-member builds, so validation,
categorical encoding, label encoding and class-weight expansion happen once per
ensemble operation rather than once per tree:

```mojo
var template = DecisionTreeRegressor(max_features=FeatureCount(Float64(1)))
var prepared = template.prepare_fit(X, y)          # validate and encode once
var tree = template.__sklearn_clone__()            # one member
var params = tree.get_params()
params.random_state = seed
tree.set_params(params)
tree.fit_prepared(prepared, sample_weight=counts)  # e.g. bootstrap counts
var encoded = prepared.encoder.encode(X_test, False, False)  # once per predict
var prediction = tree.predict_encoded(encoded)
```

[examples/tree_forest_members.mojo](../../examples/tree_forest_members.mojo) builds and
averages three bootstrapped members this way.

- `prepare_fit(X, y, sample_weight=None)` accepts a matrix or `TreeTable` and
  returns `PreparedTreeInput`: encoded `X` (a shared view of numeric input),
  Float64 targets or class codes, validated caller weights, expanded class-weight
  factors, the missing-value mask, category counts, monotonic signs, `classes`
  and the fitted `TreeInputEncoder`. Preparation never mutates caller data.
- `fit_prepared(prepared, sample_weight=None)` seeds from `random_state` like
  `fit`; `fit_prepared_with_random_state(prepared, rng, sample_weight=None)`
  takes an explicit context. A given `sample_weight` replaces the caller weights
  for that build, exactly as sklearn forests pass bootstrap counts; expanded
  class weights still multiply it. It is validated like fit weights. `fit(...)` is
  `prepare_fit` followed by one build, so both produce identical trees.
- Members may differ from the preparing estimator in build-only parameters
  (`random_state`, depth, sample/leaf limits, `max_features`, impurity
  thresholds, `ccp_alpha`). A different `criterion`, `splitter`,
  `monotonic_cst`, `categorical_features` or `class_weight` raises.
- Every member built from one prepared input shares its encoder.
  `TreeInputEncoder.encode(X, reject_missing, monotonic)` performs predict-time
  validation and category lookup; the flags mirror the estimator's current
  `criterion == "absolute_error"` and `monotonic_cst` rules. `apply_encoded`,
  `predict_encoded` and `predict_proba_encoded` then only check fitted state and
  shape, like sklearn `check_input=False`. Passing unencoded categorical input to
  them is a caller error. `input_encoder()` returns a fitted tree's encoder.

The reference forests do not accept `categorical_features`; prepared inputs keep
categorical support so a forest can document it as an extension if it chooses.
Member trees reproduce the reference forests' member trees exactly; see
[scope.md](scope.md).

## Treelite export (Moxie extension)

`export_treelite(tree)` converts any fitted decision or extra tree into a float64
Treelite v4 model (`balsa.Model[DType.float64]`), using the pure-Mojo
[Balsa](https://github.com/idlab-discover/balsa) codec. scikit-learn has no such
function. Save the result with Balsa:

```mojo
from balsa import save
from moxie.tree import DecisionTreeClassifier, export_treelite

var clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)
save(export_treelite(clf), "tree.tl")
```

The checkpoint is byte-identical to `treelite.sklearn.import_model` (Treelite
4.6.1) applied to the same tree wrapped as a one-tree RandomForest or ExtraTrees
ensemble. Classifiers become multiclass models with per-output normalized leaf
probabilities (`identity_multiclass`); regressors use `identity`. Node sample
counts, weighted counts and split gains are retained.

**Missing values follow Treelite, not sklearn.** Like that reference, every split
stores `default_left=True`. sklearn sets `missing_go_to_left` on every split,
also without NaNs in training (NaN goes to the larger child), so NaN inputs may
take a different branch in Treelite. Finite inputs predict identically.
Unfitted trees, categorical splits and classifier outputs with fewer than two
classes raise, as Treelite rejects them.


## Graphviz export

`export_graphviz(model)` returns a native DOT `String` for fitted numeric or
categorical DecisionTree and ExtraTree classifiers and regressors. It supports multiple
outputs, weighted values, `max_depth: Optional[Int]`, feature/class names as
`Optional[List[String]]`, `label="all"|"root"|"none"`, `impurity`, `node_ids`,
`proportion`, `rotate`, `rounded`, `filled`, `leaves_parallel`,
`special_characters`, `precision` and `fontname`, with sklearn's
defaults. Feature names must match the feature count. When class names are
used for single-output classification, their count must match the fitted class
count; this native validation raises a descriptive error before indexing.
It uses iterative traversal, so export does not recurse with tree depth.

```mojo
from moxie.tree import export_graphviz

var dot = export_graphviz(model, max_depth=3, rounded=True)
with open("tree.dot", "w") as output:
    output.write(dot)
```

There is no `out_file` parameter or Python file object; callers write the returned
string with native I/O. Boolean `class_names=True`, raw sklearn Tree objects and
Matplotlib color objects are omitted from the typed interface. `fill_colors`
accepts `Optional[List[String]]` of `#RRGGBB` colors, one per first-output class,
and is ignored when `filled=False`. Categorical labels show the actual left
category set and the missing/unknown route, extending sklearn's numeric-only
labels. Numeric DOT follows sklearn and does not annotate missing routes.
HTML mode escapes names as text, including ampersands and angle brackets.
Colors follow sklearn's class-confidence, regression-value and multioutput-impurity
rules. A constant multioutput tree is white; sklearn's zero-range color calculation
is undefined. Invalid/nonfinite color intensities raise a descriptive error.

Decimal text uses fixed notation with trailing zeros removed, retaining one
fractional digit for floating values. Unlike Python/NumPy, it does not switch
to scientific notation, truncate strings to 32 bytes, or wrap long value rows.
Value rounding follows NumPy's scale/round/unscale order; thresholds and
impurities use exact binary64 decimal rounding. These are native text-format
adaptations, not changes to fitted values.


## SVG tree plots

`plot_tree(model)` returns a standalone SVG `String`. Save it with native I/O
and open it in a browser, or embed it in an HTML page:

```mojo
from moxie.tree import plot_tree

var svg = plot_tree(model, filled=True, rounded=True, max_depth=3)
with open("tree.svg", "w") as output:
    output.write(svg)
```

The native function accepts `max_depth`, `feature_names`, `class_names`, `label`,
`filled`, `impurity`, `node_ids`, `proportion`, `rounded`, `precision`, and
`fill_colors` with the same types, defaults and validation as DOT export.
`fontsize: Optional[Float64]` defaults to 14 SVG units and must be finite and
positive. Python `ax` and Matplotlib artist objects are absent. This is an
explicit native return-type adaptation, not a Matplotlib-compatible result.

The SVG uses a monospace font, sizes boxes from their text, places visible leaves
in separate horizontal slots and centers parents over their children. A
white background keeps edge labels visible in dark viewers. Its viewBox grows
with the tree; it does not reproduce Matplotlib's Reingold–Tilford
layout or auto-fit text to an existing figure. Use `max_depth` for large trees.
Truncated branches are grey boxes. Names and labels are escaped as XML text.
SVG groups retain `data-node-id`; edges retain `data-parent` and `data-child`.
Numeric missing routes and category-set labels follow the DOT conventions above.
No Graphviz executable, Python interpreter, or plotting package is required.
