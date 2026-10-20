# Forest engine

`moxie.ensemble._forest_engine`, `_sampling` and `_aggregation` implement the
machinery scikit-learn's forests share: bootstrap sampling, member seeding,
serial member building, averaging, out-of-bag (OOB) scoring and feature
importances. `RandomForest*` and `ExtraTrees*` are thin layers on top of
it. The [public forest estimators](forests.md) provide their constructor, label,
shape and lifecycle APIs. [examples/forest_engine.mojo](../examples/forest_engine.mojo) grows a
50-tree regressor whose OOB score, importances and predictions are bit-identical
to `RandomForestRegressor(n_estimators=50, min_samples_leaf=2, oob_score=True,
random_state=42)`.

## Reference mapping

| scikit-learn (revision `4358542a`) | Moxie |
| --- | --- |
| `_bootstrap._get_n_samples_bootstrap` | `_sampling.get_n_samples_bootstrap` |
| `_forest._generate_sample_indices` / `_generate_unsampled_indices` | `_sampling.generate_sample_indices` / `generate_unsampled_indices` |
| `_bagging._generate_indices` / `_generate_bagging_indices` | `_sampling.draw_indices` / `generate_bagging_indices`, `BaggingSampler` |
| `compute_class_weight("balanced", ...)`, `compute_sample_weight(..., indices=)` | `_sampling.balanced_class_weight`, `balanced_subsample_weight`, `class_weight_factors` |
| `BaseEnsemble` | `_forest_engine.BaseEnsemble` (exported from `moxie.ensemble`) |
| `_set_random_states`, `_make_estimator` | `set_tree_random_state`, `BaseEnsemble.make_estimator` |
| `BaseForest.fit`, `_parallel_build_trees` | `ForestEngine.fit` (serial loop) |
| `_accumulate_prediction`, `predict`, `predict_proba` | `ForestAggregator`, `ForestEngine.predict` / `predict_proba` |
| `_compute_oob_predictions`, `_set_oob_score_and_attributes` | `ForestEngine` OOB fields (`oob_score_`, `oob_prediction_`, `oob_decision_function_`) |
| `feature_importances_`, `estimators_samples_`, `apply` | methods of the same names |

## Using the engine

```mojo
var template = DecisionTreeRegressor(max_features=FeatureCount(Float64(1)))
var forest = ForestEngine[False](template^, 100, oob_score=True)  # classifier: ForestEngine[True]
var random_state = RandomState(0)
forest.fit(X, y, None, random_state)        # sample_weight: Optional[Values]
var mean = forest.predict(X_test)           # (samples, outputs)
```

`ForestEngine[classifier, extra]` holds `DecisionTree*` members
(`extra=True`: `ExtraTree*`). Pass the tree parameters a forest forwards to
its trees on the template, using the forest's defaults. For example
`RandomForestClassifier` trees use `max_features="sqrt"` and
`ExtraTreesRegressor` trees use `max_features=1.0`. The forest-level settings
are `bootstrap`, `oob_score`, `warm_start`, `class_weight` and `max_samples`
(`MaxSamples`: None, a count or a fraction). They are constructor arguments and
mutable fields; `base.n_estimators` is the member count.

`fit` takes the random context explicitly. An integer `random_state` means a
fresh `RandomState(seed)` for each fit. A shared `RandomState` keeps its stream
across fits, as in sklearn. Each fit runs these steps in order:

1. Validate and encode `X` once with the template's `prepare_fit`.
2. Combine `sample_weight` with the expanded class weights.
3. Resolve the bootstrap size. A fractional `max_samples` scales the **sum of
   the sample weights**, not the row count.
4. Draw one seed `randint(2**31 - 1)` per new member, before any member is built.
5. Build each member. With bootstrap, the engine replays the member seed to
   draw rows (`randint`, or `choice(p=w/sum(w))` when weights are present). The
   draw counts become that member's sample weights; with
   `class_weight="balanced_subsample"` they are multiplied by balanced weights
   computed on the draw. Rows are never copied.

Under `warm_start`, the engine first replays the seeds of the retained members
and then adds members. Growing to N members this way equals a cold fit with N
members. Refitting without growth adds nothing, records a warning, and skips
recomputing OOB values.

### Class labels and class weights

Classification members are trained on Float64 class codes `0..K-1` per output,
exactly as sklearn forests train on `np.unique(..., return_inverse=True)`. So
`base.estimators_[i].classes_` hold codes, as in sklearn, and probability
columns follow the sorted class order. The public estimator owns the label
vocabulary: it encodes `y` (`moxie._labels.encode_column`) and maps argmax
codes back to labels.

`ForestClassWeight` is one of the following:

- None.
- `"balanced"`: weighted class counts, so `sample_weight` matters.
- `"balanced_subsample"`: computed per bootstrap draw. Without bootstrap it is
  the same as `"balanced"`.
- `List[Values]`: per-output, per-code factors. Build each entry from a
  `{label: weight}` dictionary with `class_weight_factors(classes, mapping)`.
  It applies sklearn's missing-key rules: unlisted classes weigh one, and unknown
  keys raise only if a class is also missing.

### Outputs and layout

| Output | Shape |
| --- | --- |
| `predict` | `(samples, outputs)` Float64 |
| `predict_proba` | `List[Values]` with one `(samples, classes)` matrix per output |
| `apply` | `(samples, n_estimators)` Int64 leaf ids |
| `estimators_samples_()` | `List[Indices]`, rows in draw order |
| `oob_prediction_` | `(samples, outputs)` |
| `oob_decision_function_` | one `(samples, classes)` matrix per output |

The public estimators squeeze single-output dimensions, as sklearn does. Rows
with no OOB prediction keep zero sums, divide by one and trigger sklearn's
warning. The default OOB scores are accuracy (subset accuracy across outputs)
and R2 with uniform averaging. A multiclass-multioutput OOB request raises
sklearn's error.

Warnings go to stderr as `Category: message` and are kept in `warnings_` for
the most recent fit.

### Samplers and the aggregator

`BootstrapSampler` (forests) and `BaggingSampler` (bagging: features first, then
rows, each with or without replacement, weighted `choice` when given mass)
implement the `EnsembleSampler` contract. `ForestAggregator` implements
`EnsembleAggregator`. It sums member outputs into caller-owned matrices in
member order; `normalize` divides each row by its denominator after the caller
applies its zero-denominator policy.

## Exactness

Forest outputs depend on summation order, so the engine reproduces NumPy's
order instead of a tolerance:

- `numpy_sum` uses NumPy's pairwise reduction, the order of `np.sum` on a vector.
- Importances and OOB sums accumulate member by member, the order of
  `np.mean(axis=0)`.
- `r2_uniform` reduces a single column pairwise and several columns row by row,
  as NumPy does.

`pixi run ensemble-prepare-oracle` builds the exact reference revision into the ignored
`families/ensemble/builds/oracle-env`. `pixi run ensemble-parity` then runs the sampling, engine and public-estimator checks:

- `tests/sampling_parity.py` compares 660 cases against sklearn's own helpers,
  with matching error inputs. The cases cover bootstrap sizes, forest draws,
  OOB rows, bagging draws (including weighted choice without replacement),
  balanced and balanced_subsample weights, and pairwise sums. They show that
  Moxie's `RandomState.choice(p=...)` consumes the same MT19937 stream as
  NumPy's legacy `choice`.
- `tests/forest_parity.py` fits 29 RandomForest/ExtraTrees configurations in
  36 fits: regression and classification, multi-output, NaNs, sample weights,
  every `class_weight` form, `max_samples`, monotonic constraints, warm-start
  growth, shared random states and error cases. Member seeds, in-bag samples,
  `apply`, predictions, probabilities, OOB scores and decisions, and warning
  sequences must be bit-identical. Forest importances must equal sklearn's
  averaging applied to the native member importances.

Member trees come from the tree family. Their structure is compared exactly;
their node impurities and importances are compared within `rtol=1e-12`, and any
member that is not bit-identical is listed. At this revision, two
squared-error-with-NaN cases have members whose node impurities differ by a few
ulps. The structure, values and predictions are still identical, so only
`feature_importances_` differs, in the last digits. That divergence belongs to
the tree family's criterion and does not come from the forest engine.

## Not included

- Parallel member building (`n_jobs`): members are built serially, in order.
- `verbose` progress output.
- `decision_path` and sparse input.
- Callable OOB scorers: the public estimators can score `oob_prediction_` or
  `oob_decision_function_` themselves.
- `_compute_partial_dependence_recursion`.
- `estimators_samples_` indices are Int64 (sklearn: int32), following the
  family's `Indices` record.
- Python's `int(label)` lookup in sklearn's dictionary class weights rejects
  non-integral float labels. Here the label is looked up directly.
