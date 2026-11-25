# OneHotEncoder, OrdinalEncoder, TargetEncoder

Conventions: [api.md](../api.md). Sources: `src/moxie/preprocessing/_encoders.mojo`,
`src/moxie/preprocessing/_target_encoder.mojo` and the family category engine
`src/moxie/preprocessing/_vocabulary.mojo`. TargetEncoder has its own
[section](#targetencoder) below.

| Name | Native signature |
| --- | --- |
| `OneHotEncoder[K, dtype]` | `OneHotEncoder(*, categories=None, drop=None, sparse_output=True, handle_unknown="error", min_frequency=None, max_categories=None, feature_name_combiner="concat")`; `drop` may also be a per-feature value list |
| `OrdinalEncoder[K, dtype]` | `OrdinalEncoder(*, categories=None, handle_unknown="error", unknown_value=None, encoded_missing_value=nan, min_frequency=None, max_categories=None)` |
| `TargetEncoder[K, Y]` | `TargetEncoder(categories=None, target_type="auto", smooth=None, cv=5, shuffle=None, random_state=None)`; output always Float64 |

## Input: typed categorical columns

sklearn encoders take one 2-D object or numeric array. Natively, `X` is a
`List[CategoricalColumn[K.Domain]]` holding one Moxie
[categorical column](../../../../docs/categorical.md) per feature, and `K`
selects how its values and missing cells behave:

| Kind `K` | Column dictionary | Missing cell means, in sklearn terms |
| --- | --- | --- |
| `StringKind` (default) | `StringCategories` | `None` in an object array: an ordinary final category named `None` |
| `StringNaNKind` | `StringCategories` | `np.nan` in an object array (pandas style): a missing value |
| `NumericKind[dtype]` (bool, int64, float32, float64) | `NumericCategories[dtype]` | `np.nan`: a missing value |

The difference matters for OrdinalEncoder. A NaN-style missing value is encoded
as `encoded_missing_value` and never grouped as infrequent. A `None` category
gets an ordinary integer code and may be infrequent. Columns can have different
dictionaries; each is matched to the fitted categories by value. All columns in
one encoder share a kind. A
[native ColumnTable](../../../../docs/compose/named-columns.md) can send its
string columns to an encoder and its floating-point columns to separate
numeric transformers.

```mojo
from moxie._categorical import categorical_column, CategoricalColumn, StringCategories
from moxie.preprocessing import OneHotEncoder, OrdinalEncoder
from moxie.preprocessing._vocabulary import StringNaNKind

var color: List[Optional[String]] = [Optional[String]("red"), None, Optional[String]("blue")]
var X: List[CategoricalColumn[StringCategories]] = [categorical_column(color)]

var onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
var Z = onehot.fit_transform(X)           # NDArray[float64], 3 x 3
var names = onehot.get_feature_names_out()  # x0_blue, x0_red, x0_None
var back = onehot.inverse_transform(Z)    # List[CategoricalColumn[StringCategories]]

var ordinal = OrdinalEncoder[StringNaNKind, DType.int64](encoded_missing_value=-1.0)
var codes = ordinal.fit_transform(X)      # [[1], [-1], [0]]
```

## Behavior

Automatic categories are the sorted observed values, with missing last.
Explicit `categories` are one `List[Optional[value]]` per feature. `None`
marks the missing category and must come last. Numeric lists must be sorted.
String lists keep their order. Duplicates raise. Infrequent grouping
(`min_frequency`, `max_categories`), `drop` (`"first"`, `"if_binary"` or
values), every `handle_unknown` mode and both inverse transforms follow
sklearn, including its edge cases:
- An ignored unknown row is decoded as the dropped category when one exists.
- A single-category feature with `drop` inverts to that category.

Learned attributes:
- `categories_` and `infrequent_categories_` (None when grouping is disabled) hold `Optional` values, with None for the missing category.
- `drop_idx_` holds original category indices.
- `n_features_in_` and `feature_names_in_` as usual.

Feature names use Python `str` formatting (`x0_1.5`, `x0_nan`,
`x0_infrequent_sklearn`).

## Native adaptations

* `categories="auto"` is `None`; the sklearn `dtype` argument is the output
  struct parameter (Float32, Float64 or Int64).
* `sparse_output=True`, sklearn's default, raises at fit: pass
  `sparse_output=False`. Sparse output is deferred.
* `min_frequency` is an `Optional[Float64]`. A whole number ≥ 1 is an absolute
  count, a value in (0, 1) a fraction. Write `3.0`: integer literals do not
  convert through `Optional[Float64]`. `unknown_value` and
  `encoded_missing_value` are Float64 (an integer value or NaN).
* `feature_name_combiner` accepts only `"concat"`; a callable combiner is not
  implemented yet (typed struct callables, as used by
  [FunctionTransformer](function.md), would allow it). `handle_unknown="warn"` encodes like `"infrequent_if_exist"` but
  emits no warning, because Moxie has no warning channel.
* Inverse transforms return categorical columns; unknown cells are missing
  (sklearn's None). A decoded `infrequent_sklearn` is appended to a string
  column's dictionary. A numeric column cannot hold it, so that inverse raises.
* sklearn defects not reproduced: refusing to drop an infrequent string category
  raises `ValueError`, where sklearn raises `AttributeError` while formatting the
  message; and `OrdinalEncoder.inverse_transform` never writes into its input.

## Memory

Fitting keeps the per-feature categories and counts, and transiently one
n-row Int64 code vector per column. A transform allocates the result and, one
feature at a time, the same code vector. No n×p code or mask tables like
sklearn's `X_int`/`X_mask` exist. At 8M rows × 4 Int64 features (20 one-hot
outputs), the OneHotEncoder transform peaked below a result-only allocation
control. Raw numbers are kept in ignored `docs/private/categorical/memory.json`.

## Parity

`tests/parity/test_encoders.py`, 699 checks. It covers string (None and NaN
missing), single-category, Float64 and Int64 features; plain, `drop="first"`, `"if_binary"`,
value drops (including the missing category), explicit categories, every
unknown mode, count/fraction `min_frequency`, `max_categories` alone and with
drops; OrdinalEncoder unknown values (-1, 99, NaN), `encoded_missing_value`,
Int64/Float32 output and an infrequent `None`. Outputs, `categories_`,
`infrequent_categories_`, `drop_idx_`, feature names and inverse transforms
are compared exactly. 24 error cases cover parameter, category, drop,
unknown, missing-value, shape and unfitted errors. Error messages match sklearn's
prefix and fields, but unknown-value lists use NumPy-like formatting and are not
guaranteed character-identical.

## TargetEncoder

`TargetEncoder[K, Y]` takes the same feature columns as the encoders above
(`K` is the feature kind) and a target `y` of label kind `Y` from
[labels](labels.md): `NumericLabels[dtype]` (Float64 by default, Float32 or
Int64; a vector or one-column matrix) or `StringLabels` (`List[String]`).
String targets can be binary or multiclass only.

```mojo
from moxie.model_selection import GroupKFold, KFold, LeaveOneGroupOut, SplitLabels
from moxie.preprocessing import TargetEncoder, TargetEncoderSplits
from moxie.preprocessing._label import StringLabels
from moxie._random import RandomState

var te = TargetEncoder(random_state=0)       # TargetEncoder[StringKind, NumericLabels[float64]]
var Z = te.fit_transform(X, y)               # cross-fitted training encoding, Float64
var Z_test = te.transform(X_test)            # encodings_ learned on all of X
var configured = TargetEncoder(cv=KFold(5, shuffle=True, random_state=0))
var Z_cv = configured.fit_transform(X, y)   # explicitly use KFold
var grouped = TargetEncoder(cv=GroupKFold(3))
var Z_group = grouped.fit_transform(X, y, groups=SplitLabels(group_ids)) # one group id per row
var leave_group_out = TargetEncoder(cv=LeaveOneGroupOut())
var Z_leave_group = leave_group_out.fit_transform(X, y, groups=SplitLabels(group_ids))
var splits = TargetEncoderSplits(KFold(5).split(y.size))
var explicit = TargetEncoder(cv=splits^)    # use the materialized train/test pairs
var labels: List[String] = ["cat", "dog", "eel"]
var multi = TargetEncoder[StringKind, StringLabels](cv=3, random_state=0)
var rng = RandomState(42)
var W = multi.fit_transform(X3, labels, rng)  # sklearn random_state=RandomState(42)
var names = multi.get_feature_names_out()     # x0_cat, x0_dog, x0_eel, ...
```

Behavior follows sklearn:
- `target_type="auto"` uses `type_of_target`. Only binary, multiclass and
  continuous targets are accepted.
- Binary targets are LabelEncoder codes. Multiclass targets are LabelBinarizer's
  one-vs-all columns, giving `n_features * n_classes` outputs ordered
  feature-major (`f0_c0, f0_c1, ..., f1_c0, ...`). They are named
  `<feature>_<class>`, with classes formatted as Python `str`.
- Encodings use sklearn's m-estimate when `smooth` is a float. `smooth=None`
  (sklearn's `"auto"`) uses the empirical-Bayes variance shrinkage.
- Missing values are an ordinary category. Categories not seen at fit encode as
  `target_mean_`.
- `fit_transform` learns each fold's encodings from its training indices and
  applies them to its validation rows. `fit(X, y).transform(X)` uses encodings
  learned on the entire dataset.
  Integer `cv` and `cv=None` use `StratifiedKFold` when the encoder treats the
  target as categorical and `type_of_target(y)` is binary or multiclass.
  Both use `KFold` otherwise and shuffle by default. Explicit
  `KFold` and `StratifiedKFold` use the splitter's own shuffle and seed
  settings, as do `GroupKFold` and `StratifiedGroupKFold`. The grouped
  splitters keep all rows of each group in the same validation fold.
  `LeaveOneOut` validates one row per fold, and `LeaveOneGroupOut`
  validates one group per fold. Both are deterministic.
  Both stratified splitters require binary or multiclass labels in the
  original target, even when `target_type="continuous"`. A shuffled
  splitter with no seed uses Moxie's process default RNG.
- Explicit splits use the supplied training rows in order, including duplicate
  training indices. Validation indices must cover every sample exactly once.
  Validation happens in `fit_transform`; plain `fit` learns from all samples.
- The `FitTransformer` overload accepts the optional target used by generic
  composition and raises `ValueError` when it receives `None`.
- Explicit `categories` may leave training values unknown. In `fit_transform`
  every unknown row ends with the last fold's training mean, because sklearn
  overwrites all unknown rows once per fold.

Learned attributes:
- `encodings_` is a `List` of Float64 vectors, one per feature (and class).
- `categories_` matches the encoders above.
- `target_type_` is a String.
- `target_mean_` is a Float64 vector of one entry, or one per class.
- `classes_` is `Optional[Y.Labels]`, None for continuous targets.
- `n_features_in_` and `feature_names_in_` as usual. `fit_named` and
  `fit_transform_named` record input names.

Explicit splitter objects use the native `moxie.model_selection`
implementations. Integer `cv` and `cv=None` use the family-local
`kfold_test_folds` and `stratified_kfold_test_folds` in `_target_encoder`,
which return each row's test fold.
For fold f, training rows are the rows of other folds and test rows are fold
f's rows, both ascending, exactly as sklearn's `split` yields them. The
stratified version ranks classes by first appearance, allocates rows to folds
round-robin over the class-sorted rows and shuffles each class's fold sequence
with one RandomState, as sklearn does. Both functions validate as sklearn's
splitters do: `n_splits` greater than the sample count, `random_state` without
shuffle, and `n_splits` greater than every class count all raise.

### Native adaptations

* `categories="auto"` is `None`. `smooth="auto"` is `None` and a float is an
  `Optional[Float64]`: write `smooth=Optional[Float64](5.0)` or `smooth=5.0`.
* `shuffle` and `random_state` keep sklearn's 1.9 position. Their `"deprecated"`
  defaults are `None` (shuffle on, unseeded). Moxie has no warning channel, so
  setting them emits no FutureWarning. `shuffle` is an `Optional[Bool]`.
* `random_state` is `Optional[Int]`. An integer reseeds a fresh NumPy-compatible
  `RandomState` on every `fit_transform` (sklearn's `check_random_state(int)`).
  `None` leases the process default stream of `moxie._random` (reseed with
  `seed_global_random_state`), which is not NumPy's global stream. A RandomState
  instance cannot be stored as a parameter. Instead,
  `fit_transform(X, y, mut rng: RandomState)` draws from and advances a
  caller-owned state, matching sklearn's `random_state=np.random.RandomState(...)`
  sequence. `fit` and `transform` draw nothing.
* `set_params(smooth=..., shuffle=..., random_state=...)` take
  `Optional[Optional[...]]`. The outer `None` leaves the value unchanged, and
  an inner `None` restores `"auto"`/`"deprecated"`.
* `cv` accepts an integer of at least 2, `None`, `KFold`, `StratifiedKFold`,
  `GroupKFold`, `StratifiedGroupKFold`, `LeaveOneOut`, `LeaveOneGroupOut`, or
  `TargetEncoderSplits` in the constructor and `set_params`. `cv=None` uses
  five folds with the same target-based splitter as `cv=5`. The constructor
  default is the integer `5`. Omitting `cv` in `set_params` keeps the existing
  configuration. Pass a `List[IndexSplit]` from `moxie.model_selection` or a
  native splitter cursor to `TargetEncoderSplits` to snapshot the index pairs.
  Supply any labels or groups needed by the splitter when constructing the
  cursor. Move a named splitter with `splitter^`, or pass `splitter.copy()` to
  reuse it. Each index array
  must be rank one and contain zero-based indices. Move the snapshot into
  `cv` with `splits^`, or pass `splits.copy()` to keep it for reuse. The
  public field and `get_params()` store the value in a Mojo variant. For an
  integer configuration, read `model.cv[Int]`. Use
  `model.cv.isa[type_of(None)]()` to check for explicit `None`. The caller-owned
  `RandomState` overload accepts integer `cv` or `None`.
* For `GroupKFold`, `StratifiedGroupKFold`, and `LeaveOneGroupOut`, pass
  `groups=SplitLabels(group_ids)`
  to `fit_transform` or `fit_transform_named`. Use rank-one Int64, Float32 or
  Float64 arrays or a list of strings with one label per input row. Supplying
  groups to any other CV configuration raises `ValueError`. Plain `fit` learns
  from all rows without group labels. To use grouped splits
  through `ColumnTransformer`, configure `cv` with a `TargetEncoderSplits`
  snapshot for the exact row order of the input passed to the composition.
* `target_mean_` is always a vector. sklearn stores a scalar for binary and
  continuous targets. A Float32 target's mean is the NumPy Float32 value, stored
  as Float64.
* Targets must be one-dimensional (or one column). sklearn accepts a
  multilabel-indicator target with `target_type="multiclass"`; Moxie raises.
* sklearn defects not reproduced: with `target_type="multiclass"` and exactly
  two classes, sklearn fails with `IndexError`. Moxie raises `ValueError`. A
  string target with `target_type="continuous"` raises `TypeError:` naming the
  label, where NumPy reports an unsupported `add.reduce`.
* StratifiedKFold and StratifiedGroupKFold do not emit the UserWarning for a
  class smaller than `cv`.

### Memory

Fitted state is O(total categories x classes): `encodings_`, `categories_`
and the vocabularies. A fit or `fit_transform` additionally holds the result,
one rows-long Int64 code vector per feature in turn, the target as one Float64
(or class code) per row, a fold vector, and one training-row list. Per-feature
category accumulators are also held. There is no n x p code/mask table
(sklearn's `X_ordinal`/`X_known_mask`) and no n x classes indicator matrix
(sklearn's binarized `y`). `transform` holds the result and one code vector.
An explicit CV configuration also stores all indices supplied by the caller.

### Parity

`tests/parity/test_target_encoder.py`, 977 checks. All values are compared
exactly (rtol=0), because the native code reproduces each source of rounding:
- the fold indices;
- NumPy's pairwise `mean`/`var` in the target dtype (Float32 for Float32
  targets, Float64 otherwise);
- the pyx's row-order Float64 accumulation.

Fold vectors of `kfold_test_folds` and `stratified_kfold_test_folds` are
compared with sklearn's `KFold`/`StratifiedKFold` for:
- several sizes and fold counts, with and without shuffle;
- seeds 0, 42 and 2**32-1;
- balanced, imbalanced, interleaved, binary and near-singleton classes;
- `None` with both global streams seeded alike;
- an advancing RandomState.

Estimator cases cover:
- string (None and NaN missing), Float64 and Int64 features;
- continuous Float64, Float32 and Int64 targets, and a one-column target;
- binary integer and string targets, multiclass integer, float and string
  targets, and a single-class multiclass target;
- forced target types;
- `smooth` "auto", 0, 5 and 1e6;
- integer `cv` 2 to 5 and explicit `KFold` and `StratifiedKFold`,
  including seeded and global shuffling and explicit unshuffled KFold
  overriding the deprecated encoder settings;
- `cv=None` for continuous, binary and multiclass targets with both seeded
  and unshuffled splits;
- direct `GroupKFold` and `StratifiedGroupKFold` with integer and string
  groups, unshuffled and seeded splitters, and seeded global shuffling;
- direct `LeaveOneOut` with continuous targets and `LeaveOneGroupOut` with
  multiclass targets, string groups, and unknown training categories;
- explicit train/test pairs with reordered and repeated training rows,
  one split, and missing or overlapping validation rows;
- explicit categories with unknown training values;
- the default unseeded estimator under seeded global streams;
- a caller-owned RandomState over two calls.

Each case compares the `fit_transform` output, `encodings_`, `target_mean_`,
`target_type_`, `categories_`, `classes_`, feature names (generated, supplied
and fitted), `transform` of training and unseen data, and plain `fit`. Error
cases check that both sides raise: parameter validation, inconsistent lengths,
multioutput and NaN targets, splitter errors, a two-class multiclass target, an
unfitted estimator and a feature-count mismatch.
