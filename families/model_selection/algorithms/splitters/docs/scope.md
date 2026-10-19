# Dense splitter scope

Concrete APIs: KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold,
ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit, RepeatedKFold,
RepeatedStratifiedKFold, TimeSeriesSplit, LeaveOneOut, LeavePOut,
LeaveOneGroupOut, LeavePGroupsOut, PredefinedSplit, check_cv and train_test_split.
BaseCrossValidator and BaseShuffleSplit are native traits. They are assembled
into the family API from this child; the parent owns the index contracts.

## Native calls

Splitters accept the sample count (`split(n_samples, y=..., groups=...)`). The
cursor's `next()` returns an owned `Optional[IndexSplit]`; `None` indicates
exhaustion. `get_n_splits` preserves reference defaults, including
GroupShuffleSplit's five iterations. Stratification counts classes in
first-occurrence order for StratifiedKFold and sorted-label order for
StratifiedShuffleSplit/StratifiedGroupKFold.

## Labels and groups

Rank-one Int64 `IndexArray` labels and groups keep their original signatures.
Float32/Float64 NuMojo arrays and String labels are accepted through
`SplitLabels`, which converts implicitly from `NDArray[DType.float32]`,
`NDArray[DType.float64]` and `List[String]`. Encoding comes from the shared
`moxie._labels` owner (moxie `shared/labels`). Splitters see labels only through
equality, sorted order and first appearance, so `SplitLabels` passes sorted-rank
Int64 codes to the same Int64 implementations and the folds match sklearn exactly.

```mojo
var y: List[String] = ["spam", "ham", "ham", "spam"]
var folds = StratifiedKFold(2).split(4, y=y)
var groups = GroupKFold(2).split(4, groups=floats)  # NDArray[DType.float64]
var parts = train_test_split(X, y, random_state=0, stratify=y)
```

Validation follows each reference splitter. StratifiedKFold,
RepeatedStratifiedKFold and StratifiedGroupKFold require a binary or multiclass
`type_of_target` and raise sklearn's `Supported target types` error otherwise.
StratifiedShuffleSplit and `train_test_split(stratify=...)` accept any finite
labels, including fractional floats. For a rank-two `y` they join each row with
spaces and sort those strings, matching `" ".join(row.astype(str))`; this
supports Int64 (`SplitLabels(int_matrix)`), Float64 and String
(`SplitLabels(values, rows, columns)`) rows. `check_cv(cv, y, classifier=True)`
stratifies only binary/multiclass targets and otherwise returns KFold.
`train_test_split(X, y)` also splits `List[String]` targets into owned train and
test lists. Groups may be String, Float32, Float64 or Int64, and fractional
group IDs stay distinct.

Language-driven differences:

- A bare list literal is not accepted as labels. It would need two implicit
  conversions, and Mojo allows one, so declare `var y: List[String] = [...]`.
- Mixing Int64 labels with non-Int64 groups (or the reverse) is supported for
  StratifiedGroupKFold. Elsewhere, wrap Int64 explicitly as `SplitLabels(codes)`.
- Splitters that ignore `y` (KFold, ShuffleSplit, GroupKFold and the other
  group splitters) take `y=None` or Int64 `y`, not `SplitLabels`.
- NaN or infinite labels and groups raise ValueError, as sklearn's
  `check_array` does.
- Groups must be rank one; sklearn's behavior with rank-two groups is
  inconsistent across splitters.
- Rank-two Float32 stratification labels raise. NumPy's float32 `str()`
  switches to scientific notation at different magnitudes than Mojo's
  formatting, so the row keys would not match. Convert to Float64.

Integer `random_state` restarts the foundation NumPy-compatible stream on each
`split`. For `random_state=None`, retain an application-owned `RandomState` and
pass it explicitly. No hidden global stream is created. KFold takes
`split(n, rng)`, StratifiedKFold `split(n, labels, rng)`, GroupKFold
`split(n, groups, rng)`; the other random splitters take
`split(n, rng, y=..., groups=...)`.

Shuffle/repeated splitters materialize all pairs when `split` is called, then
return independent copies from the cursor. Thus they consume all corresponding
RNG draws eagerly, even if the caller stops iteration early. This differs from
Python generator timing and uses O(total emitted indices) memory. Seeded full
traversals preserve exact reference order and draw consumption.

Use `SplitSize(3)` for an integer count and `SplitSize(0.25)` for a fraction.
`train_test_split(X, y, random_state=42, test_size=SplitSize(0.25))` returns
`(X_train, X_test, y_train, y_test)`; a one-array overload returns two arrays.
Float32/Float64 features and numeric targets retain dtype and rank (one or two),
including strided inputs. Outputs own contiguous row copies. An overload taking
`rng` after the input arrays advances an explicit stream. More than two arrays
and higher-rank arrays are not yet supported.

`check_cv(None)` chooses five folds; an integer chooses that fold count.
`classifier=True` with Int64 labels selects stratification. The result is a
`CheckedCV` tagged record. Explicit splitters pass through by ownership transfer;
`List[IndexSplit]` inputs produce an `IterableCV` with a deep snapshot, validated
bounds and repeatable traversal. Python metadata routing and warnings are not
implemented; rare-class warning conditions still yield the reference folds.

## Unfinished and deferred behavior

Object, mixed-type, nullable and boolean label arrays have no native
counterpart. The validation child's generic cross-validation still passes Int64
`IndexArray` targets to splitters. Full sklearn compatibility is not claimed.

Sparse matrices and sparse row slicing remain deferred. No implicit densification
is performed. Python dynamic containers, arbitrary `*arrays`, metadata routing
and warning delivery are unsupported. Reference: sklearn `_split.py` at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b` (BSD-3-Clause).

## Reproduce

From this directory:

```sh
PIXI_CACHE_DIR="$PWD/.pixi/pixi-cache" pixi run build
PIXI_CACHE_DIR="$PWD/.pixi/pixi-cache" pixi run test
PIXI_CACHE_DIR="$PWD/.pixi/pixi-cache" pixi run parity
```

The parity harness executes original pinned sklearn class/function bodies with
minimal NumPy dense/metadata adapters. It checks exact indices, not statistical
similarity: tolerance is zero. It includes shuffled ties, imbalanced/single
classes, repeated folds and train/test rounding, plus String (including numeric
strings whose lexicographic order differs from numeric order, and Unicode),
Float32/Float64, fractional-group and rank-two row-key labels for every
label-consuming splitter, `check_cv` dispatch, stratified `train_test_split` and
reference validation errors. `type_of_target` executes from the pinned
`sklearn/utils/multiclass.py`. Native tests cover invalid
inputs, RNG replay/advancement, ownership, cursor exhaustion, dtype and strides.
The required-export acceptance gate remains enabled. No performance claim is made.
