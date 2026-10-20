# ColumnTransformer, make_column_transformer, make_column_selector

`moxie.compose.ColumnTransformer` applies transformers to column subsets of a
dense `NDArray[float32/float64]` or native `ColumnTable` and concatenates their outputs in declaration
order, with the remainder last. Reference: `sklearn/compose/_column_transformer.py`
at the revision in `../../interfaces.md`.

```mojo
from moxie.compose import ColumnTransformer, ColumnEntry, make_column_transformer

var ct = ColumnTransformer(
    ColumnEntry("num", MyScaler(), [0, 1]),        # list -> (n, 2) input
    ColumnEntry("one", MyScaler(), 4),             # int  -> (n,) input
    ColumnEntry("raw", "passthrough", Slice(2, 4)),
    ColumnEntry("tag", "drop", [True, False, False, False, False, False]),
    remainder="passthrough",                       # or "drop" (default) or an estimator
    transformer_weights=weights,                   # Optional[Dict[String, Float64]]
    verbose_feature_names_out="{feature_name}@{transformer_name}",
)
var Xt = ct.fit_transform(X)        # or ct.fit(X, y); ct.transform(X_new)
var names = ct.get_feature_names_out()            # x0.. or given input names

var auto = make_column_transformer(ColumnEntry(MyScaler(), [0]), ColumnEntry("passthrough", [1]))
# names: "myscaler", "passthrough" (duplicates get -1, -2, ...)
```

## Constructor (sklearn names and defaults)

| Parameter | Default | Notes |
| --- | --- | --- |
| entries | required | 0–8 `ColumnEntry(name, transformer, columns)` arguments (sklearn's `transformers` list) |
| `remainder` | `"drop"` | `"drop"`, `"passthrough"` or an estimator; applied to unselected columns, last |
| `sparse_threshold` | `0.3` | Validated in [0, 1]; outputs are always dense, so `sparse_output_` is False |
| `n_jobs` | `None` | MAX CPU component workers; None/1 serial, positive values capped by task count, negative values relative to logical CPU count; 0 raises |
| `transformer_weights` | `None` | Output of entry `name` (or `"remainder"`) is multiplied by its weight; unknown names are ignored |
| `verbose` | `False` | Prints sklearn's `[ColumnTransformer] ... (i of n) Processing name, total=...` lines |
| `verbose_feature_names_out` | `True` | `True` (`name__feature`), `False` (must be unique), a `{transformer_name}`/`{feature_name}` format string, or `def(String, String) raises -> String` |

`columns` accepts an `Int` (1-D selection), a list of ints (negative allowed,
repeats kept), `Slice(start, stop, step)`, a boolean mask, or a callable
`def(NDArray[dtype]) raises -> ColumnKey` evaluated at fit. Names (`"age"`, `List[String]`, `ColumnKey.name_slice`) and dtype-based
`make_column_selector` work on native `ColumnTable` inputs. They raise on
unnamed arrays as sklearn does. Regex patterns are not yet supported. See
[the native table guide](../named-columns.md). A component is
any `TransformerMixin & UnsupervisedEstimator` with `NDArray[dtype]`
Input/Target/Transformed; it is fitted with `fit_transform` when it conforms to
`FitTransformer`, otherwise `fit` then `transform`, and receives `y`.

## Learned state and methods

`n_features_in_`, optional `feature_names_in_`, `sparse_output_` (False), `output_indices_`
(`Dict[String, OutputSpan]` including absent entries and `"remainder"` as
`(0, 0)`; `None` when no transformer produced output, where sklearn leaves the
attribute unset), `transformers_()` (records `('name', Type|'passthrough'|'drop', columns)`
with kind `estimator`, `unfitted` for empty selections, `passthrough` or `drop`;
the remainder row appears when it has columns, or always for all-mask keys),
`named_transformers_[T](name)` (copy of the fitted — or, for an empty
selection, configured — transformer of type `T`; `drop`/`passthrough` raise
TypeError), `get_feature_names_out(input_features)`, `fit`, `fit_transform`,
`transform`, `get_params`, `set_params`, `set_transformer[T](name, value)`,
`set_transformer_params[T](name, params)`, `set_output(transform=)`,
`__sklearn_clone__`.

## Mojo differences

- The entry list is a static type (`ColumnCons`/`EndColumns`, built by the
  constructor overloads). `set_transformer` replaces an entry with a value of
  its declared type (or `"drop"`/`"passthrough"`); declare
  `ColumnEntry[MyScaler]("a", "drop", cols)` to switch it to an estimator later.
  Entries written with only a mode have the placeholder type `NoTransformer`.
  More than 8 entries need nested `ColumnCons` values passed to the primary
  constructor.
- `named_transformers_` and `__getitem__` are the typed method
  `named_transformers_[T](name)`. `transformers_` is the method
  `transformers_()`; a fitted passthrough is reported as `passthrough` (sklearn
  stores a fitted identity FunctionTransformer).
- Parameters: `get_params()` returns a `ColumnTransformerParams` record holding
  unfitted copies of the entries. Keyword `set_params` updates supplied values;
  `remainder_params=` is sklearn's `remainder__<param>`; nested entry
  parameters use `set_transformer_params`. Record overload resets values to None.
- A failed fit leaves the previous fitted state untouched (sklearn may already
  have replaced `n_features_in_`/`transformers_`); config changes after a fit do
  not alter the fitted state until the next fit, as in sklearn.
- X is a rank-2 array or a named numeric table. Named transform maps fitted
  columns by name, tolerates extra columns and omitted dropped columns, and
  rejects missing required columns. Pandas/polars integration is excluded;
  sparse inputs/outputs and metadata routing are unavailable. Components receive
  numeric arrays, not schema-carrying containers. Array selection callables
  receive the underlying matrix and may return positional or named keys.
- Output feature names need components conforming to the compose
  `FeatureNamesOut` trait or `moxie._feature_metadata.FeatureNamesOut`
  (sklearn checks `hasattr`); others raise sklearn's AttributeError text.
  Preprocessing scalers and the dense imputers conform. Some other families'
  transformers define the method without declaring a trait, so their names
  are not yet available through ColumnTransformer.
- `moxie.pipeline.Pipeline` ends in a predictor, so a transformer chain such
  as `make_pipeline(SimpleImputer(), StandardScaler())` cannot yet be one
  entry; use consecutive ColumnTransformers instead.
- Errors are Mojo `Error`s whose text starts with the sklearn exception type
  and message (`InvalidParameterError: The 'remainder' parameter ...`).

## Evidence

`pixi run test` runs `tests/compose/columns/contracts.mojo` (13 groups: composition, remainder
modes, weights, every key form, scalar/empty/slice selections, drop and empty
output, y routing and fit_transform dispatch, validation errors, feature
names, make_column_transformer naming, params/clone/refit, verbose format,
Float32, parallel execution and worker failure) plus named-table contracts and
the example. `pixi run parity` compares 26 scenarios in both
dtypes against the pinned `_column_transformer.py` (outputs, transform,
feature names, `output_indices_`, `transformers_` and error messages); see
`tests/compose/columns/parity.py` for the doubles and tolerances. Regenerate the arity
overloads with `python tools/gen_compose_arity.py`.

Parallel workers borrow immutable input and own disjoint fitted entries and
result slots. They join before state publication; errors are rethrown in
declaration order. Negative `n_jobs` uses logical CPUs, not joblib affinity/quota
or process-backend rules. Custom components must be safe to run concurrently
and must not mutate shared external state. No throughput claim is made.
