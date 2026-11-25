# Dense histogram gradient boosting regression

`moxie.ensemble.HistGradientBoostingRegressor` accepts a rank-two numerical
NuMojo `NDArray` with sample rows and feature columns. Features default to
Float64. Use `HistGradientBoostingRegressor[DType.float32]` for Float32 inputs.
Targets, predictions, sample weights, and score histories are rank-one Float64
arrays. Both feature dtypes support strided layouts and NaNs.

```mojo
from moxie.ensemble import HistGradientBoostingRegressor

var model = HistGradientBoostingRegressor(
    max_iter=80, min_samples_leaf=5, early_stopping=True, random_state=42,
)
_ = model.fit(X, y)
var predictions = model.predict(X_test)
var stages = model.staged_predict(X_test)
```

The estimator supports squared error, absolute error, Gamma, Poisson, and
quantile losses, nonnegative sample weights, per-node feature subsampling,
monotonic constraints by feature position, interaction constraints, and
[explicit categorical columns](hgb-categorical.md).
`staged_predict` returns an owned list containing one prediction vector per
boosting iteration. `score` returns R² and accepts optional sample weights.

Early stopping defaults to `scoring="loss"`. Use `scoring=None` or
`"r2"` for the estimator's R² score. The named error scorers are
`"neg_mean_absolute_error"`, `"neg_mean_squared_error"`, and
`"neg_root_mean_squared_error"`. These regression metrics use predictions
after the inverse link, including for Gamma and Poisson losses.
`early_stopping="auto"` enables early stopping above 10,000 rows.
With early stopping enabled, `validation_fraction` selects a seeded shuffled
holdout by fraction or by integer row count. Set it to `None` to monitor the
training score, or pass `X_val` and `y_val` together to monitor explicit
validation data. Optional array arguments can take an owned
array via `^` or a view via `.view()`. Pass optional union-valued settings as
`Variant[Int, Float64]` for a holdout size, or
`Variant[String, List[List[Int]]]` for an interaction rule. Explicit validation
follows the pinned estimator's 10,000-row rule when `early_stopping="auto"`.
A score history starts with the intercept-only score and adds one score per
fitted stage. R² and the named error scorers use a seeded sample of 10,000
training rows when the training split exceeds that size. Validation scoring
uses the entire holdout. Loss scoring uses the entire training split.
A fresh fit with early stopping disabled records empty histories.

`get_params()` returns a typed copy. Edit the record and pass it to
`set_params()` before refitting. Parameter updates leave fitted predictions
available until a successful fit replaces them. `moxie.base.clone` makes an
unfitted estimator. The current model survives a failed fit.
`n_iter_`, `n_trees_per_iteration_` and `n_features_in_` start at zero and
describe the fitted state after fit. Verbose fit emits native progress lines.

Feature-name metadata and the remaining scikit-learn scorers are open. The
native scoring parameter is an optional string and accepts the names above.
The random-state argument accepts an integer seed or `None`.
`staged_predict` returns a list.

Set `warm_start=True`, increase `max_iter` through the parameter record, and
refit on the same data to continue boosting. The first fit's baseline, trees,
holdout seed, feature-sampling state, and score histories carry forward.
New tree settings such as `max_depth` apply to the added stages. Continuation
uses the original random stream even if `random_state` changes. An equal
`max_iter` adds zero stages. A smaller value or a different feature count
raises `ValueError` and preserves the current fitted model.
Set `warm_start=False` to fit a fresh model with the current parameters.

## Validation

With the locked toolchain active, run the focused checks from the Moxie root.

```sh
PYTHONDONTWRITEBYTECODE=1 python families/ensemble/tools/check.py hgb-regressor
PYTHONDONTWRITEBYTECODE=1 python families/ensemble/tools/check.py hgb-losses
```

The public check compares every stage, final predictions, R², training and
validation score histories, and fitted attributes against source-built
scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Iteration counts, history lengths and feature counts must match exactly.
Float64 comparisons use absolute and relative tolerances of 1e-12 for
reduction and platform math rounding. Native contracts cover typed and
strided inputs, unfitted prediction, cloning, parameter changes, and refits.

The fit reuses the private bin mapper, grower and loss components. It updates
the training raw predictions using the grower's leaf partitions and scores
the holdout using binned traversal. Prediction traverses the fitted raw
trees. Train/holdout selection gathers only requested rows in shuffle order.
The consumed NuMojo `NDArray.take(axis=0)` materializes a contiguous copy of
the entire input before each gather, including already-contiguous inputs.
The row gather writes directly to the result while honoring source strides
and offset. Error scoring accumulates weighted residuals from the linked
predictions without building an error vector or a weighted product vector.
Numax `stats.statistics.average` requires a row-major tensor and allocates
the elementwise product for weighted input. R² reuses the tree regressor's
weighted score calculation.

Continuation replays retained raw-tree predictions to initialize training and
validation margins. The tree snapshots retain their fitted thresholds, and the
fit keeps one accumulated raw prediction array per split. Continuation copies
the retained trees while fitting so a failed fit can preserve the previous model.
