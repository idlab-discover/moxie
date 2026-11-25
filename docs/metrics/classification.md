# Dense classification metrics

Import dense classification metrics from `moxie.metrics`.

```mojo
from moxie.metrics import accuracy_score, hamming_loss, zero_one_loss

def main() raises:
    var y_true = List[String](["cat", "dog", "cat"])
    var y_pred = List[String](["cat", "cat", "cat"])
    print(accuracy_score(y_true, y_pred))                 # 2/3
    print(zero_one_loss(y_true, y_pred, normalize=False))  # 1.0
    print(hamming_loss(y_true, y_pred))                   # 1/3
```

The three agreement functions return `Float64`. Accuracy counts rows where
every label matches. Zero-one loss counts rows with any mismatch. Both return fractions by
default and accept `normalize=False` for counts. Hamming loss returns the
fraction of mismatched entries, so it can differ from zero-one loss on
multilabel targets.

The two targets use the same representation and dtype. Use `List[Int]`,
`List[String]`, `moxie._labels.StringTarget`, or a NuMojo `NDArray` of
Int64, Float32, or Float64. Numeric labels must be finite and integral.
A vector and a single-column matrix can be paired inside the same
representation. Numeric multilabel indicators must have matching widths and
pass the shared [label validation](../../shared/labels/docs/labels.md).
NuMojo array inputs can be strided. Inputs are borrowed for the call.

`sample_weight` is an optional rank-one `NDArray[DType.float64]` containing
one finite value per sample. Pass an existing optional array directly, or
use `sample_weight=weights.view()` to wrap a shared view of a plain array.
Counts use the sum of the relevant weights and fractions divide by the total
weight. For multilabel Hamming loss the denominator is the total weight
times the number of columns.

Empty samples, unsupported shapes and kinds, invalid weights, and all-zero
weights raise Mojo `Error` with a `ValueError:` prefix. Signed weights are
accepted. Weights that cancel to zero raise `ZeroDivisionError:` when
normalizing accuracy, zero-one loss, or single-label Hamming loss. Multilabel
Hamming loss follows the reference's IEEE infinity or NaN result for zero
total weight. It does so without a Python runtime warning.

## Confusion matrices and balanced accuracy

`confusion_matrix` puts true labels on rows and predicted labels on columns.
It uses the sorted union of both targets by default. An explicit `labels`
list sets the axis order and can include unobserved classes. Its elements
use the same type as the target labels. At least one label must occur in
`y_true`. Samples with either label outside the list are skipped. If a label
is repeated in the list, its last position receives the counts.

```mojo
from moxie.metrics import balanced_accuracy_score, confusion_matrix

def main() raises:
    var truth: List[String] = ["cat", "dog", "cat"]
    var predicted: List[String] = ["cat", "cat", "cat"]
    var labels: List[String] = ["dog", "cat"]
    var counts = confusion_matrix(truth, predicted, labels=labels.copy())  # [[0, 1], [0, 2]]
    var recalls = confusion_matrix(truth, predicted, labels=labels^, normalize="true")
    print(counts, recalls, balanced_accuracy_score(truth, predicted))  # score is 0.5
```

Omit both `sample_weight` and `normalize` for `NDArray[DType.int64]`
counts. Supplying either keyword selects `NDArray[DType.float64]`, including when
the weight is an empty optional. Pass `normalize="true"`
for row normalization, `"pred"` for columns or `"all"` for the grand total.
Zero divided by zero yields a zero entry. Infinite results saturate to the
largest finite value with the same sign.

`balanced_accuracy_score` averages per-class recall and uses
`adjusted=True` to subtract the chance baseline and rescale. It skips NaN
recalls, including classes present only in predictions. It returns a
`Float64` and uses the same optional weights as `confusion_matrix`. Both
functions accept binary and multiclass targets. They raise for empty inputs
and for multilabel or other multioutput targets. They follow the reference's
numeric behavior for one-class results and suppressed NaN recalls without
emitting Python warnings.

## Precision, recall and F-scores

`precision_recall_fscore_support` returns a `ClassificationScores` record by
default. Its `precision`, `recall`, `fscore`, and `support` fields are
`List[Float64]` values, with one element per selected label. The default
order is the sorted union of labels in both targets. An explicit typed
`labels` list preserves order and repetitions and may include unobserved
single labels. For multilabel targets, `labels` selects zero-based columns.

Pass an `average` string to get an `AveragedClassificationScores` record with
three `Float64` fields. The individual functions `precision_score`,
`recall_score`, `f1_score`, and `fbeta_score` return one `Float64` and default
to binary averaging. Retrieve individual per-label lists from the
`ClassificationScores` record.

| Average | Calculation |
| --- | --- |
| `"binary"` | Score the positive label. Numeric targets default to label 1. For two string classes, supply the positive string as `pos_label`. |
| `"micro"` | Pool the counts for selected labels before dividing. |
| `"macro"` | Give each selected label equal weight. |
| `"weighted"` | Weight each selected label by its support. |
| `"samples"` | Score each multilabel sample independently, then average by sample weight. |

`labels` applies outside binary averaging. `pos_label` applies only to
binary averaging. `fbeta_score` takes a required `beta`. A beta of zero gives
precision and positive infinity gives recall. `precision_recall_fscore_support`
uses beta 1 by default.

`zero_division` is an optional `Float64`. Omit it to return zero for an
undefined score and emit an `UndefinedMetricWarning` message on stderr.
Supply 0, 1, or NaN explicitly for silent behavior. NaN values are excluded
independently for each metric when averaging. An average with zero total
weight falls back to the corresponding unweighted mean of defined scores.
The native API selects its return type by the presence of the `average`
keyword on `precision_recall_fscore_support`. The per-label overload has a
fixed default and emits warnings only for undefined scores.

```mojo
from moxie.metrics import f1_score, precision_recall_fscore_support

def main() raises:
    var truth: List[String] = ["cat", "dog", "cat"]
    var predicted: List[String] = ["cat", "cat", "cat"]
    var scores = precision_recall_fscore_support(truth, predicted, zero_division=0.0)
    print(scores.fscore, scores.support)  # [0.8, 0.0], [2.0, 1.0]
    print(f1_score(truth, predicted, average="macro"))  # 0.4
```

## Classification reports

`classification_report` returns a text table with precision, recall, F1,
and support for each class. `digits` defaults to 2 and controls the printed
precision of the scores. Use `classification_report_dict` for unrounded
values. It returns a `ClassificationReport` whose `rows` map names to
`ClassificationReportRow` records with `precision`, `recall`, `f1_score`, and
`support` fields. Its scalar `accuracy` is an `Optional[Float64]`. Separate
entry points give text and dictionary callers concrete return types.

```mojo
from moxie.metrics import classification_report, classification_report_dict

def main() raises:
    var truth: List[String] = ["cat", "dog", "cat"]
    var predicted: List[String] = ["cat", "cat", "dog"]
    print(classification_report(truth, predicted), end="")
    var report = classification_report_dict(truth, predicted)
    print(report.rows["cat"].f1_score, report.accuracy.value())
```

Reports accept the target representations, sample weights, and
`zero_division` described above. Label order defaults to the sorted union of
the targets, or column order for multilabel indicators. Explicit `labels`
can select, reorder, or repeat classes. Unobserved single labels have zero
support. `target_names` supplies display names in selected label order.
If names and explicitly supplied labels have different lengths, the report
prints a warning on stderr and renders the paired rows. The averages still
use all selected labels. With inferred labels, a length mismatch raises.

Macro and weighted averages appear in every report. Single-label reports
display accuracy when the selection covers every observed label and the
micro average when it covers a subset. Multilabel reports include both
micro and samples averages. Repeated display names occupy separate text
rows. In the dictionary the last row with a given name wins, and the name
`accuracy` maps to its scalar precision, mirroring scikit-learn's reserved
key. An empty label selection can produce a dictionary of averages. Text
output requires at least one available display name.

The native text path uses the shared fixed-decimal formatter for scores and
Mojo's numeric spelling for label names and weighted support. Inferred
numeric labels canonicalize signed zero. Explicit floating-point label
names retain the sign of zero.

## Model selection

A `TypedScorer` can use metrics directly on the estimator's predictions.
With `Metadata = Optional[NDArray[DType.float64]]`, its score method can use
the fold-local weights selected by its `CVAdapter`.

```mojo
return f1_score(y, estimator.predict(X), sample_weight=metadata)
```

See the [GaussianNB scorer example](../../tests/metrics/scorer.mojo) for the
full adapter and its use with `cross_val_score` for accuracy, balanced
accuracy, and binary F1.

## Validation

From the repository root, use the locked Mojo 1.0.0 / MAX 26.5.0 environment,
the live dependency checkouts in `DEPENDENCIES.json`, and a Python
environment built from the pinned local scikit-learn revision. Set
`SKLEARN_PYTHON` to that environment's Python executable.

```sh
python shared/assembly/tools/assemble_family.py --consumer . --lock DEPENDENCIES.json --output builds/deps
mkdir -p builds/metrics
mojo build -I builds/deps tests/metrics/classification.mojo -o builds/metrics/classification
mojo build -I builds/deps tests/metrics/scorer.mojo -o builds/metrics/scorer
mojo build -I builds/deps tests/metrics/precision_recall.mojo -o builds/metrics/precision_recall
mojo build -I builds/deps tests/metrics/classification_report.mojo -o builds/metrics/classification_report
"$SKLEARN_PYTHON" tests/metrics/parity.py builds/metrics/classification builds/metrics/scorer builds/metrics/precision_recall builds/metrics/classification_report
```

The comparison runs the four native contract binaries, checks the installed
oracle sources against the committed reference bytes, and compares results
with a Float64 absolute tolerance of `1e-14` for the tested inputs. The
tolerance allows for different reduction order. Report tests also compare
complete text output and warning categories and messages.

The implementation reuses `moxie._labels.type_of_target` for numeric target
validation. Numax `not_equal` produces a separate mask on its row-major
tensor interface. Numax `count_nonzero` produces one global count through a
host copy. The native reduction reads the existing strided NuMojo arrays
and accumulates row matches and mismatches without an intermediate mask or
weighted array. Shared target-kind validation still uses its label workspace.

The confusion matrix uses label codes and scatters counts directly into the
NuMojo output. Numax `bincount` uses a row-major tensor, stages inputs through
`to_host()`, and returns a separate tensor. Using it would also require
materializing flattened pair indices and weights. Balanced accuracy aggregates
observed pairs in Mojo dictionaries and reduces them in class order without
allocating a square matrix. Sequential Float64 reductions can round differently
from NumPy's pairwise reductions when weights differ greatly in magnitude.

Precision, recall, and F-scores reuse numeric-label and weight validation
from the classification metrics. One pass over single-label targets counts
true and predicted occurrences and true positives in a label-keyed table.
Numax `bincount` would require separate encoded arrays and a true-positive
subset. For multilabel targets the corresponding counts are fused over
logical NuMojo indices. The existing Numax `count_nonzero` folds the entire
tensor; combining elementwise masks and reductions would require
indicator-sized temporaries for the weighted row or column counts.

Reports reuse those counts for the per-label, micro, macro, and weighted
scores. Multilabel reports also reuse the existing samplewise reduction.
Report formatting uses the same fixed-decimal operation as the tree exporter.
