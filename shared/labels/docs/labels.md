# Shared dense labels

Import the private engine from `moxie._labels`. It provides class identity and
validation shared by classifiers and splitters, not a public `LabelEncoder` or
`LabelBinarizer` estimator. Those remain preprocessing work.

Numeric targets are NuMojo `NDArray[int64/float32/float64]` of rank one or two.
`owned_target(y)` makes a compact, independent snapshot, including strided inputs,
without converting the dtype. Strings use `StringTarget(values: List[String])`
or `StringTarget(values, rows, columns)` for a row-major rank-two target. Both
constructors copy the strings; `get`, `set`, `rows`, `columns`, `rank`, and
`values` provide checked access and an independent flat snapshot. String
comparison is lexicographic over valid UTF-8, without Unicode normalization.
Numeric target storage, class arrays, indices and counts use NuMojo; temporary
Lists are sorting/lookup workspaces, not a second numerical array interface.

```mojo
from moxie._labels import StringTarget, encode_column

var labels = StringTarget(["z", "a", "z"])
var encoded = encode_column(labels)
# encoded.classes == ["a", "z"]
# encoded.index.inverse == [1, 0, 1]
var restored = encoded.decode(encoded.index.inverse)
```

`encode_column(y, column=0, classification=True)` returns `NumericEncoding[dtype]`
or `StringEncoding`. A rank-two target has one independent domain per column;
call once per output, retaining output order. `classes` is an owned sorted unique
numeric vector or String list. `index` contains owned Int64 vectors:

| Field | Meaning |
| --- | --- |
| `inverse` | Sorted-class ID for each input row |
| `counts` | Number of occurrences of each sorted class |
| `first_indices` | First input row for each sorted class |
| `appearance_order` | Sorted-class IDs ordered by first appearance |

`index.first_seen_codes()` returns row IDs in first-appearance order, as needed
by StratifiedKFold. The original `inverse` is unchanged. Results own their data;
caller mutation of the original targets does not change domains or indices.
Treat result metadata as read-only. Explicit `.copy()` produces independent
copies; manual corruption of a result's arrays is outside the contract.

`encoding.encode(y, column=0)` looks up values in the existing domain and rejects
unseen labels. It does not refit or alter the domain. `encoding.decode(codes)`
accepts a rank-one Int64 vector, rejects negative/out-of-range codes and returns
a vector preserving the domain's numeric dtype, or a rank-one `StringTarget`.
An empty vector yields an empty result. Families assemble per-column decoded
vectors into their estimator's required output shape.

`type_of_target` returns sklearn's dense target categories: binary, multiclass,
continuous, their multioutput variants, multilabel-indicator, or unknown.
`check_classification_targets` rejects continuous/unknown kinds. Numeric
multilabel indicators may contain any two integral values, not just 0 and 1.
Integral floats retain their floating dtype; Int64 values never pass through
Float64. Float integrality uses sklearn's int64-representable range.

For group identifiers use `encode_column(groups, classification=False)`.
Finite fractional values remain distinct groups; the encoder never truncates
them to integers. IDs, not the labels themselves, are Int64.

## Scope and language differences

NaN/infinite numeric labels and groups are rejected explicitly. This is narrower
than the missing-value support of sklearn's general-purpose `_unique` and some
public encoders; it is not an implementation of those full public APIs.
String values are homogeneous Mojo strings. Python objects, bytes, mixed scalar
types, null values, ragged targets, sparse targets and other numeric dtypes are
not supported. No dtype promotion or Python runtime conversion is performed.
Errors use Mojo `Error` with `ValueError` prefixes. Python warnings about unusual
class counts or column vectors are not emitted by this private engine.

`type_of_target` returns unknown for numeric ranks other than one/two, while
encoding/snapshot operations reject them. Empty-vector and empty-matrix kind
behavior is checked against the pinned reference; sample-count requirements
belong to the consuming estimator. Classification validates the whole target
before extracting a requested column. Group mode validates the selected column.

This engine supplies per-column domains. StratifiedShuffleSplit's historical
rank-two string-join key construction (including its formatting and collision
behavior) belongs to that splitter. It may encode the resulting `StringTarget`
with this engine; column IDs must not substitute for its row keys.

## Consumption and checks

The owner is `shared/labels`; its module is `moxie._labels`. Use the existing
`shared/assembly/tools/consume.py` with current local sources under the
[live-source policy](../../../docs/family-development.md). `EXPORTS.json` keeps
the package root empty. Pixi locks specify compatible Mojo/MAX; NuMojo supplies
numeric storage. Runtime remains pure Mojo.

Run `pixi install --locked`, `pixi run build`, `pixi run test`, and
`pixi run parity`. Two isolated consumers cover classifier decoding and splitter
first-appearance encoding. Parity executes original local sklearn target-kind
and unique function bodies with NumPy dense-array plumbing, then compares native
results exactly, including Int64 extremes, strings, fractions and empty shapes.
This establishes the documented helper contracts, not estimator parity.
