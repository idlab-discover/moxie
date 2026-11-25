# Dense classification metrics

Import `accuracy_score`, `zero_one_loss`, and `hamming_loss` from
`moxie.metrics`.

```mojo
from moxie.metrics import accuracy_score, hamming_loss, zero_one_loss

def main() raises:
    var y_true = List[String](["cat", "dog", "cat"])
    var y_pred = List[String](["cat", "cat", "cat"])
    print(accuracy_score(y_true, y_pred))                 # 2/3
    print(zero_one_loss(y_true, y_pred, normalize=False))  # 1.0
    print(hamming_loss(y_true, y_pred))                   # 1/3
```

All three functions return `Float64`. Accuracy counts rows where every label
matches. Zero-one loss counts rows with any mismatch. Both return fractions by
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

## Model selection

A `TypedScorer` can use metrics directly on the estimator's predictions.
With `Metadata = Optional[NDArray[DType.float64]]`, its score method can use
the fold-local weights selected by its `CVAdapter`.

```mojo
return accuracy_score(y, estimator.predict(X), sample_weight=metadata)
```

See the [GaussianNB scorer example](../../tests/metrics/scorer.mojo) for the
full adapter and its use with `cross_val_score`.

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
"$SKLEARN_PYTHON" tests/metrics/parity.py builds/metrics/classification builds/metrics/scorer
```

The comparison runs both native contract binaries, checks the installed
oracle sources against the committed reference bytes, and compares results
with a Float64 absolute tolerance of `1e-14` for the tested inputs. The
tolerance allows for different reduction order.

The implementation reuses `moxie._labels.type_of_target` for numeric target
validation. Numax `not_equal` produces a separate mask on its row-major
tensor interface. Numax `count_nonzero` produces one global count through a
host copy. The native reduction reads the existing strided NuMojo arrays
and accumulates row matches and mismatches without an intermediate mask or
weighted array. Shared target-kind validation still uses its label workspace.
