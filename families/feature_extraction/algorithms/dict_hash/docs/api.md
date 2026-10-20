# DictVectorizer

Import `DictVectorizer` from `moxie.feature_extraction`. Select `sparse=False`;
the inherited `sparse=True` default raises an explicit deferred-capability error.
See [cities.mojo](../examples/cities.mojo) for a complete native example.

```mojo
var vectorizer = DictVectorizer(sparse=False)
var matrix = vectorizer.fit_transform(rows)
var names = vectorizer.get_feature_names_out()
```

`rows` is a `FeatureBatch` (`List[FeatureRow]`). Each `FeatureRow` is an ordered
list of `FeatureEntry(FeatureKey(...), FeatureValue(...))`, imported from
`moxie.feature_extraction._dict_vectorizer`. This typed adapter replaces Python
mapping/iterator dispatch. Materialize generators once into a batch. Repeated
entries accumulate in encounter order. `transform` and `fit_transform` also
accept a single row. `fit` accepts a batch, matching the reference implementation.

`FeatureKey(String(...))` and `FeatureKey(Int(...))` retain distinct identities,
even when their display strings match. Numeric keys sort numerically; strings
sort lexicographically as UTF-8 (equivalent to Unicode scalar order for valid
strings). Mixed integer/string sorting raises; mixed keys work with `sort=False`.
Python arbitrary-object keys belong to a future binding.

Values have explicit constructors:

| Constructor | Meaning |
| --- | --- |
| `FeatureValue(Float64(value))` | Real numeric value, converted to output dtype |
| `FeatureValue(Int64(value))`, `FeatureValue(UInt64(value))` | Exact integer input, including values beyond Float64 precision |
| `FeatureValue(String(value))` | One categorical occurrence |
| `FeatureValue(List[String](...))` | Count each occurrence, including repeats |
| `FeatureValue()` | Python None equivalent: NaN for floats, false for bool, error for integers |

Booleans can be supplied as exact integer zero/one. Nested mappings and
non-string categorical lists have no matching typed constructor and fail to
compile. Category names concatenate key display, separator, and value without
escaping. A categorical `a=b` intentionally collides with a raw string key `a=b`.

The implemented output types are `NDArray[DType.float32]`, `float64` (default),
`bool`, and signed/unsigned 8-, 16-, 32-, 64-bit integers. Choose a dtype as a
compile-time parameter: `DictVectorizer[DType.int16](sparse=False)`. Constructor
configuration is separate from learned state. `get_params(deep=True)` returns
`DictVectorizerParams[dtype]` with separator, sparse, sort; dtype is fixed by its
type. `set_params(params)` or named optional updates mutate configuration without
resetting the learned vocabulary. Changing dtype requires constructing another
specialization. Integer input conversions reject values outside the output range;
subsequent collision addition follows fixed-width arithmetic. Bool addition is OR.

`fit` replaces learned state; `fit_transform` commits state only after successful
conversion. Empty `fit` is valid, empty transform/fit_transform raise. Nonempty
batches with no features yield `(samples, 0)`. Unknown transform keys are ignored,
including their unconvertible numeric values. Outputs own independent C-order
storage and do not alias input metadata. Direct dense assembly uses O(samples ×
features × dtype_bytes) output storage, plus O(features) vocabulary metadata.
It checks representable byte size before allocation. There is no sparse intermediate.

Learned `feature_names_` is `List[FeatureKey]`; `vocabulary_.column(key)` returns
its column or -1 for unknown keys. The vocabulary wrapper uses the family's
immutable `OrderedVocabulary` snapshot; its extra indices preserve sklearn's
behavior when restriction repeats an index. `vocabulary_.keys()` returns a copy.
No `n_features_in_` is invented. `get_feature_names_out()` returns independent
`List[String]` display names; an optional `List[String]` input_features is ignored.

`inverse_transform(matrix)` borrows an NDArray, including valid offset/strided
views, and returns `List[Dict[FeatureKey, Scalar[input_dtype]]]` for nonzero entries.
The matrix dtype is independent of the estimator dtype; inverse values preserve
the input dtype, including exact Int64/UInt64 values beyond Float64 precision.
It returns expanded names, not original categories. Nonfinite, rank-not-two and
zero-extent inputs raise. Like sklearn, shorter input widths use the prefix of
names, and excess nonzero columns raise an index error. Custom Python dict types
and Python validation exception classes are binding work.

`restrict(NDArray, indices=False)` accepts a rank-one numeric or Boolean mask;
nonzero values select columns, including NaN mask entries. With `indices=True`,
integer arrays select indices. Offset and negative-stride views preserve logical
order. Nonempty Boolean or floating-point index arrays raise a TypeError;
out-of-range indices raise before changing the vocabulary. Higher-rank selection
arrays are rejected by this native interface.

`restrict(List[Bool], indices=False)` selects a mask; `restrict(List[Int],
indices=True)` selects indices, including negative indices. The default `indices=False`
interprets either integer or boolean lists as a truth-value mask. Explicit
`indices=True` selects indices for either typed overload. Repeated indices retain sklearn's
index-overwrite quirk; if it creates an out-of-range vocabulary index, transform
raises. This is an explicit safety difference: the reference constructs invalid
CSR metadata in this case and can return an array rather than reject it. Valid
restriction metadata, including in-range duplicate indices, is compared exactly.
Restriction returns a mutable borrow of the estimator, so
`vectorizer.restrict(support).transform(rows)` is supported.

`moxie.base.clone(vectorizer)` copies constructor configuration into an unfitted
model. Copying a fitted model instead makes independent metadata snapshots.
Unfitted transform/inverse/names/restrict raise errors prefixed `NotFittedError`.
The optional ignored target is `Optional[Int]`, following the family typed trait.
Methods `fit` and `set_params` return a mutable borrow of the original estimator,
following current Moxie contracts. Both typed and named parameter updates support
`vectorizer.set_params(sort=False).fit(rows).transform(rows)`. Python reflection,
metadata routing, serialization and pandas
are not part of this native interface.

Dense complex and extended-precision dtypes remain unfinished; see [scope.md](scope.md).
Passing the current real-valued checks is not full dense dtype parity.
