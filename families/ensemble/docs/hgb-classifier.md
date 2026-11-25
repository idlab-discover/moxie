# Dense histogram gradient boosting classification

`moxie.ensemble.HistGradientBoostingClassifier` learns binary and multiclass
log-loss models from numerical NuMojo arrays. Feature rows may contain NaNs
and may be strided. Features and numerical labels default to Float64. Use
`feature_dtype=DType.float32` for Float32 features, or set `label_dtype` to
Float32 or Int64 to preserve those label types. Select `string_labels=True`
for `moxie._labels.StringTarget` inputs and predictions. A rank-one target or
a single target column supplies one label per sample.

```mojo
from moxie.ensemble import HistGradientBoostingClassifier

var model = HistGradientBoostingClassifier(
    max_iter=80, min_samples_leaf=5, class_weight=String("balanced"),
    early_stopping=True, random_state=42,
)
_ = model.fit(X, y)
var probabilities = model.predict_proba(X_test)
var margins = model.decision_function(X_test)
var labels = model.predict(X_test)
```

`classes_` contains sorted labels. `predict_proba` returns a rank-two Float64
array in that class order. A one-class fit retains the reference's two-column
binomial output. `decision_function` returns a Float64 vector for one or
two classes, or a matrix with one column per class for multiclass data.
`staged_predict`, `staged_predict_proba`, and `staged_decision_function` return
owned lists of snapshots in boosting order. `score` returns accuracy and accepts
optional sample weights.

Training accepts nonnegative Float64 sample weights. Class weights may be
`None`, `"balanced"`, or a `Dict[Int64, Float64]`. The pinned scikit-learn
revision applies the mapping after encoding, so its keys are zero-based
positions in sorted `classes_`. The balanced factors use row counts before any
split and multiply any supplied training weights. Explicit validation weights
are used as supplied. Binary fits support monotonic constraints by feature
position. All fits support per-node feature subsampling and interaction constraints.

Early stopping monitors negative log loss. The default `"auto"` activates it
above 10,000 rows. With early stopping enabled, the default holdout is a seeded
stratified split. Supply a fraction or count in a `Variant[Int, Float64]`, set
`validation_fraction=None` to monitor training loss, or supply both `X_val` and
`y_val` to use explicit validation data. Each score history starts with the
intercept-only score and gains one entry per fitted iteration. A fresh fit with early stopping disabled records empty histories.

`get_params()` returns a typed copy. Pass the edited record to `set_params()`
before refitting. Configuration updates and failed fits preserve the current
fitted model. `moxie.base.clone` makes an unfitted estimator. Learned counters
start at zero, and `is_categorical_` is `None` for numerical arrays. Verbose fits
emit native progress lines. Categorical preprocessing, feature-name metadata, and
additional early-stopping scorers remain open.

Set `warm_start=True`, increase `max_iter` through the parameter record, and
refit on the same data to continue boosting. The first fit's baseline, trees,
holdout seed, feature-sampling state, and score histories carry forward.
New tree settings such as `max_depth` apply to the added stages. Continuation
uses the original random stream even if `random_state` changes. An equal
`max_iter` adds zero stages. A smaller value or a different feature count
raises `ValueError` and preserves the current fitted model. Classifiers also
require the same sorted labels to preserve the meaning of retained tree
outputs. Set `warm_start=False` to fit a fresh model with the current parameters.

## Validation

With locked Mojo 1.0.0, run from the Moxie root.

```sh
PYTHONDONTWRITEBYTECODE=1 python families/ensemble/tools/check.py hgb-classifier
```

Native contracts check typed and strided inputs, fluent weighted fitting,
Float64 pipeline composition, cloning, refits, and invalid inputs. Public
parity uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Fitted attributes, label predictions, stage counts and stopping iterations
must match exactly. Decisions, probabilities, and accuracy and loss scores
use absolute and relative tolerances of 1e-12 for reduction and math rounding.

The classifier reuses the family grower, classification loss and row gather,
the model-selection stratified splitter, and the accuracy metric. Numax's
tensor argmax requires separate tensor storage. The leaf partition update
writes into a strided NuMojo margin column. The loss fills column-major
derivative spans so the same grower can consume a class
column without copying it. Training updates each margin column directly from
leaf partitions. NuMojo's eager gather and arithmetic on a materialized column
would copy the input or allocate an intermediate per tree. Traversal, class
selection and margin updates operate directly on the existing storage.

Continuation replays retained raw-tree predictions to initialize training and
validation margins. The tree snapshots retain their fitted thresholds, and the
fit keeps one accumulated raw prediction array per split. Continuation copies
the retained trees while fitting so a failed fit can preserve the previous model.
