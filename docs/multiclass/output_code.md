# OutputCodeClassifier

`moxie.multiclass.OutputCodeClassifier` implements scikit-learn's random
output-code strategy for dense features and numeric or string single-label targets.
It learns one binary classifier for each bit of a random codebook and predicts
the class whose codeword is closest to the component scores in Euclidean distance.

```mojo
from moxie.multiclass import OutputCodeClassifier

var model = OutputCodeClassifier(MyBinaryClassifier(), code_size=1.5, random_state=42)
model.fit(X, y)
var predicted = model.predict(X)
var accuracy = model.score(X, y)
```

[examples/multiclass_output_code.mojo](../../examples/multiclass_output_code.mojo) is runnable with the
included test component. That component demonstrates the interface; it is not
a port of a production binary classifier.

## Component and array types

`OutputCodeClassifier[E, dtype=DType.float64, label=DType.int64]` takes a
copyable binary classifier or regressor `E` with `NDArray[dtype]` features and
`NDArray[int64]` targets. It must implement `ClassifierMixin` or `RegressorMixin`, and either
`DecisionFunction` or `ProbabilityPredictor`.

When `E` has `DecisionFunction`, training targets are **−1 and +1**. Otherwise
they are **0 and 1**. A component that accepts only 0/1 targets cannot serve as
a margin component here. Decisions must be Float32/Float64 arrays of shape
`(n,)` or `(n, 1)`; probabilities must have two columns in class order.
The wrapper prefers decisions for classifiers when both capabilities exist.
Regressors require rank-one `(n_samples,)` Float32/Float64 `predict` output;
column-shaped outputs are rejected. Decoding uses `predict` even when
`decision_function` exists. The decision capability still selects −1/+1 code
bits. A fit/predict-only regressor is rejected, matching sklearn parameter
validation.

`X` must be a nonempty dense matrix. `y` may be a vector or an `(n, 1)` column;
its dtype is `int64`, `float32` or `float64`, with finite integral-valued labels within the int64 range.
`classes_` and predictions preserve the label dtype. NaN/infinite features are
passed to the component, but nonfinite component scores are rejected by decoding.

The wrapper exposes `fit`, `predict`, `score`, `get_params`, `set_params` and
cloning. It does not expose `decision_function`, `predict_proba` or `partial_fit`,
matching scikit-learn's output-code wrapper.

## Configuration and learned state

| Name | Meaning |
| --- | --- |
| `estimator` | Unmodified component prototype; each nonconstant column fits a fresh clone |
| `code_size=1.5` | Number of components is `int(n_classes * code_size)` |
| `random_state=None` | Integer seed or Moxie's process-wide default stream |
| `n_jobs=None` | Component fitting workers; same native thread rules as [OvR](one_vs_rest.md#parallel-fitting) |
| `classes_` | Sorted unique labels |
| `code_book_` | Float64 array `(n_classes, n_estimators)`, containing −1/+1 or 0/1 |
| `estimators_` | Ordered `List[BinaryComponent[E]]`; constant columns use a constant predictor |
| `n_features_in_` | `Optional[Int]`, populated from the dense matrix on a successful fit |

One-class fits are supported. Duplicate codewords are retained; exact distance
ties choose the first class. Constant columns predict their actual code bit,
including −1. Repeated fits replace all learned state. If a fit fails, the
previous fitted model stays usable; sklearn may leave partial state instead.

Integer seeds reproduce NumPy legacy `RandomState` codebooks exactly: uniforms
are drawn in row-major order and thresholded strictly above 0.5. Each fit starts
a fresh stream for an integer seed. `None` advances Moxie's shared stream;
`moxie._random.seed_global_random_state(seed)` can reset it. The stream is
released before component fitting, allowing components to use it themselves.
Moxie's default entropy source and singleton are independent of NumPy's.

`get_params(deep)` returns `OutputCodeParams[E.Params]`. `set_params` accepts
that record, or partial keyword updates to `code_size`, `random_state`, `n_jobs`
or `estimator`. To explicitly clear an optional seed or worker count, pass
`Optional[Optional[Int]](Optional[Int](None))`. `clone` recursively clones
constructor configuration and returns an unfitted model.

## Scope and numerical differences

- Sparse inputs, mutable `RandomState`
  constructor arguments, metadata routing/`fit_params`, and `feature_names_in_`
  are not supported. Unsupported types fail compilation; invalid targets raise.
- The input contract is a nonempty dense matrix, even if a component would
  accept another rank. Feature count is recorded from this matrix rather than
  conditionally delegated from the first component's attributes.
- `code_size` must be finite and positive and produce at least one estimator.
  Zero-width codebooks raise a clear `ValueError`; the reference currently
  reaches an `IndexError` for that case. Overflowing sizes also raise.
- Decoding sums squared coordinate differences in Float64 without allocating
  a sample-by-class distance matrix. It rejects distance overflow. Scikit-learn's
  BLAS-backed distance calculation can round differently near numerical ties;
  bitwise distance parity for arbitrary component outputs is not promised.
- `fit` and parameter updates mutate in place; errors are Mojo `Error` values
  with descriptive prefixes. Python warnings, reflection, dynamic dictionaries
  and serialization follow the [shared interface limitations](../interfaces.md).

## Validation

`pixi run test` covers exact seeded codebooks, constant columns, duplicate-word
ties, global-stream advancement, repeat fits, cloning, parameter updates,
invalid inputs, weighted accuracy, failed-fit preservation and serial/parallel
agreement. Compile-failure tests check feature types and method availability.

`pixi run parity` compares against the pinned local `sklearn.multiclass` source
at revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
`tests/multiclass/output_code_parity.py` compares codebooks, sorted classes, constant flags,
component counts, predictions and unweighted scores exactly; weighted scores
allow 1e-15 absolute/relative error. It also compares every component confidence:
margin tolerance is 1e-12 relative and 1e-14 absolute (2e-6 relative for
Float32 regressor output rounding); probability tolerance is
5e-11 relative and 1e-14 absolute to cover the test fixture's Mojo `exp` rounding.
Cases include one through five classes, margin/probability/dual classifiers,
decision-enabled Float32/Float64 regressors and probability-enabled regressors,
compressed and expanded codes, integer and Float32/Float64 labels, column
targets, boundary seeds, an MT19937 twist, and successive default-stream fits.
These are wrapper comparisons using twin test components, not evidence of
parity for a particular production base estimator. Build flags include
`-O3 --fp-mode=contract=off`, as required by the shared RNG contract.

## String labels

Select `kind=StringLabels` and pass shared `StringTarget` values. For example:

```mojo
from moxie.multiclass import OutputCodeClassifier, StringLabels, StringTarget

var model = OutputCodeClassifier[MyBinaryClassifier, kind=StringLabels](MyBinaryClassifier())
var y = StringTarget(["north", "south", "north"])
model.fit(X, y)
var predicted = model.predict(X)  # StringTarget, rank one
```

`classes_` is an owned `List[String]` in sorted Unicode order. Predictions are
rank-one `StringTarget` values; an `(n, 1)` training column can be supplied as
`StringTarget(values, n, 1)`. Empty strings and non-ASCII labels are supported.
Multi-column string targets are rejected. Component targets remain numeric
binary codes. Decision/probability columns follow `classes_` order where those
methods are available. Cloning, refitting, parallel fitting and weighted accuracy
have the same semantics as with numeric labels.

The existing `label=DType...` parameter continues to select numeric label dtype
by default. `kind` selects the target representation; the implementation's
`ClassValues` and `Domain` parameters must retain their defaults.

## Dense precomputed inputs

OutputCode forwards the full matrix to each nonconstant component during fit
and predict. It can therefore wrap a component that accepts precomputed kernels;
the component validates kernel shape and width. As in the pinned sklearn,
OutputCode **does not propagate the pairwise tag**: it does not implement
`PairwiseInput`, and `pairwise_input(model)` returns `False`. Callers must not
infer automatic pairwise slicing by an outer wrapper or cross-validation tool.
The pairwise parity suite checks this distinction and full-matrix forwarding.
