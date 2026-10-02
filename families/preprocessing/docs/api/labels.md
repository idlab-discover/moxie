# LabelEncoder, LabelBinarizer, MultiLabelBinarizer, label_binarize

Conventions: [api.md](../api.md). Source: `src/moxie/preprocessing/_label.mojo`.
Sorted class domains and target kinds come from the shared label engine
`moxie._labels` ([labels.md](../../../../shared/labels/docs/labels.md)).

| Name | Native signature |
| --- | --- |
| `LabelEncoder[K]` | `LabelEncoder()`; learned `classes_` |
| `LabelBinarizer[K]` | `LabelBinarizer(*, neg_label=0, pos_label=1, sparse_output=False)`; learned `classes_`, `y_type_`, `sparse_input_` |
| `MultiLabelBinarizer[K]` | `MultiLabelBinarizer(*, classes=None, sparse_output=False)`; learned `classes_` |
| `label_binarize` | `label_binarize(y, *, classes, neg_label=0, pos_label=1, sparse_output=False) -> NDArray[int64]` |

## Input: label kinds

Label estimators work on targets only: `fit(y)`, `transform(y)`,
`fit_transform(y)` and `inverse_transform(...)`, as in sklearn. They have no
`n_features_in_`, feature names or `set_output`. A label kind `K` sets the
types of the labels:

| Kind `K` | Labels `y` and `classes_` | One label |
| --- | --- | --- |
| `StringLabels` (default) | `List[String]` | `String` |
| `NumericLabels[dtype]` (int64, float32, float64) | `NDArray[dtype]`: a vector, a one-column matrix, or a multilabel indicator matrix for LabelBinarizer | `Scalar[dtype]` |

Strings sort by UTF-8 bytes. That is Unicode code point order, the order NumPy
uses. Numeric strided views are read in place. Import the kinds from
`moxie.preprocessing._label`. The kinds also serve
[TargetEncoder](encoders.md#targetencoder) targets: `number(label)` gives a
numeric label as Float64 (string labels raise `TypeError`), `number_dtype` is
NumPy's mean dtype (Float32 for Float32 labels, else Float64), and
`python_str(label)` is Python's `str`, used in feature names.

```mojo
from moxie.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer, label_binarize
from moxie.preprocessing._label import NumericLabels

var cities: List[String] = ["paris", "tokyo", "amsterdam", "paris"]
var encoder = LabelEncoder()
var codes = encoder.fit_transform(cities)       # NDArray[int64] [1, 2, 0, 1]
var back = encoder.inverse_transform(codes)     # List[String]

var binarizer = LabelBinarizer[NumericLabels[DType.int64]](neg_label=-1)
var Y = binarizer.fit_transform(y)              # y: NDArray[int64]; Y: n x n_classes Int64
var labels = binarizer.inverse_transform(scores, None)   # Float64 decision scores

var genres: List[List[String]] = [["sci-fi", "thriller"], ["comedy"], []]
var mlb = MultiLabelBinarizer()
var indicator = mlb.fit_transform(genres)       # 3 x 3 Int64
var sets = mlb.inverse_transform(indicator)     # List[List[String]]
```

## Behavior

All encoded outputs are Int64. This matches sklearn on 64-bit platforms, where
codes use `intp` and indicators use `int64`.

**LabelEncoder.** `classes_` holds the sorted distinct labels. Float labels may
contain NaN and ±infinity, as in sklearn: infinities sort normally and one NaN
class comes last. NaN encodes to that class. `transform` raises
`y contains previously unseen labels: [...]` for unknown labels, and
`inverse_transform` raises the same for codes outside `0..n_classes-1`. Fitting
an empty target gives a fitted encoder with no classes. A one-column matrix
counts as a vector. Any other matrix raises sklearn's `y should be a 1d array`
error.

**LabelBinarizer / label_binarize.** `y_type_` is `"binary"`,
`"multiclass"` or `"multilabel-indicator"`. Binary targets give one column, in
which `classes_[-1]` is positive. Labels missing from `classes` give
all-`neg_label` rows. A single class gives an all-`neg_label` column. Explicit
`classes` set the column order. Duplicate classes behave like NumPy's
`searchsorted`. `pos_label=0` requires a negative `neg_label`, and the values
are swapped as sklearn does. Multilabel indicators keep sklearn's exact cell
rules: nonzero cells become `pos_label` only when `pos_label != 1`. The checks
for `neg_label >= pos_label`, sparse label incompatibility, multioutput,
continuous ("Unknown label type" at fit), NaN/infinite, empty and
non-multilabel targets match sklearn. `inverse_transform` takes the class of the
first row maximum for multiclass input, counting NaN as the maximum, as
`argmax` does. For binary and multilabel input it thresholds at `threshold`, or
at `(pos_label + neg_label) / 2` when `threshold` is None. The comparison runs
in Float32 for Float32 scores and in Float64 otherwise, as in sklearn.

**MultiLabelBinarizer.** `y` is one label list per sample. Duplicate labels
within a sample count once. With `classes=None`, `classes_` holds the sorted
distinct labels. Otherwise it keeps the given order, and duplicate classes
raise. `fit_transform` equals `fit` followed by `transform`, as in sklearn.
`inverse_transform` accepts only 0/1 Int64 indicators with `len(classes_)`
columns.

## Native adaptations

* Kinds are struct parameters (`LabelEncoder[NumericLabels[DType.float64]]`).
  Each estimator has a second parameter `L`, the kind's container type. It
  works around a Mojo 1.0 issue with struct fields of associated types and
  must stay at its default. Only a size check catches a wrong `L`.
* `label_binarize` has two overloads: `NDArray[dtype]` labels and classes of
  the same dtype, or `List[String]`. sklearn casts `classes` to `y`'s dtype;
  natively the types must already match.
* `LabelBinarizer.inverse_transform(Y)` takes an Int64 matrix.
  `inverse_transform(Y, threshold)` takes Int64, Float32 or Float64 scores.
  `threshold` is an `Optional[Float64]` and must be passed, even as `None`,
  for non-Int64 scores.
* For float label kinds, a multilabel `classes_` (`0..k-1`) and
  inverse-transformed indicators have the label dtype. sklearn gives Int64.
  Int64 labels match exactly.
* `sparse_output=True` raises `sparse output is deferred` once sklearn's own
  label checks pass: at fit, at transform (also after `set_params`) and in
  `label_binarize`. `sparse_input_` is always False.
* MultiLabelBinarizer ignores unknown labels at transform, as sklearn does,
  but emits no warning because Moxie has no warning channel.
  `inverse_transform` returns `List[List[V]]` rather than tuples. NaN labels
  raise because Python's set and sort rules for NaN have no native equivalent.
  Keyword `set_params(classes=...)` cannot reset `classes` to None; use the
  `Params` record overload for that.
* Codes passed to `LabelEncoder.inverse_transform` are Int64. Empty transforms
  return empty Int64 vectors and empty inverses return empty label containers.
  sklearn returns empty float64 arrays for both.
* String labels have no missing value. sklearn object arrays that mix
  strings with None or NaN are not supported.
* `label_binarize` rejects NaN in `classes`. Binary targets with no classes
  raise `ValueError` where sklearn raises `IndexError`. A positive second
  score column for a single fitted class also raises `ValueError` where sklearn
  raises `IndexError`.
* sklearn defect not reproduced: `LabelEncoder.transform` casts string input
  to the dtype of `classes_`, which truncates labels longer than the longest
  class. As a result, `"ab"` can encode as a fitted `"a"`. Natively, long
  labels are compared in full.
* No column-vector `DataConversionWarning`, `get_metadata_routing`, or
  `set_output` (sklearn's label estimators have no `set_output` either).

## Memory

Learned state is O(n_classes): the classes and, for MultiLabelBinarizer, a
sorted permutation. Fitting consumes the shared engine, which copies labels
into transient O(n) workspace: value lists, a sorted copy, an Int64 inverse
vector and, for strings, a `StringTarget` copy. Float LabelEncoder targets with
non-finite labels also get a compact copy of the finite labels.
MultiLabelBinarizer with `classes=None` flattens all labels once. Transforms
allocate the Int64 result, O(n_classes) lookup workspace and, on error, the
unknown labels. LabelBinarizer and label_binarize also run one engine
target-kind scan per call. Lookups are binary searches, O(log n_classes) per
label.

## Parity

`tests/parity/test_labels.py`, 595 checks, all exact. Outputs are integers,
and classes and inverses are copies of input labels, so no tolerance applies.
The checks cover:
* LabelEncoder: Int64 labels, including Int64 extremes; Float64 and Float32
  labels with NaN and ±infinity; ASCII and non-ASCII strings; refits; strided
  and negative-stride views; column vectors; empty inputs.
* LabelBinarizer: six `(neg_label, pos_label)` pairs, including `pos_label=0`
  and positive `neg_label`, over binary, multiclass, single-class, float,
  Float32 and string targets. Transforms with unseen labels. Inverses of
  outputs and of Float64/Float32 scores, at the default threshold, exactly on
  it, and at 0.3. Int64 and Float64 multilabel indicators. Two-column and
  vector binary scores. NaN and over-wide argmax.
* label_binarize: unsorted, superset, duplicate and unseen classes; values
  swapped by `pos_label=0`; shifted labels; string classes; multilabel
  indicators with reordered classes and non-unit values.
* MultiLabelBinarizer: automatic and explicit classes, string and Int64
  labels, unknown labels, `fit_transform` against `fit` then `transform`, and
  inverses.
* 44 error cases: parameter, sparse, multioutput, continuous, NaN, infinity,
  empty, unseen, shape, duplicate, indicator and unfitted errors.

Messages match sklearn's prefix and fields. Lists of unseen values use
NumPy-like formatting and may not match sklearn character for character.
