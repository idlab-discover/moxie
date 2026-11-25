# multiclass family interfaces

Implemented: [`OneVsRestClassifier`](one_vs_rest.md),
[`OneVsOneClassifier`](one_vs_one.md) and
[`OutputCodeClassifier`](output_code.md) for dense numeric or string
single-label targets. OvR also supports dense multilabel indicator targets.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](../interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `PairwiseInput` | Optional estimator capability | `pairwise_input(self) -> Bool`, current configuration |
| `MulticlassClassifier` | `ClassifierMixin` | Inherited only |
| `MulticlassDecision` | `MulticlassClassifier, DecisionFunction` | Inherited only |
| `MulticlassProbability` | `MulticlassClassifier, ProbabilityPredictor` | Inherited only |

## Binary adapter for numeric List labels

```mojo
from moxie.multiclass import BinaryAdapter, OneVsRestClassifier
from moxie.svm import LinearSVC

var model = OneVsRestClassifier(BinaryAdapter(LinearSVC(random_state=0)))
_ = model.fit(X, y)
```

Moxie uses an explicit adapter where sklearn accepts a classifier's numeric
targets directly. `BinaryAdapter` infers the component's `List[Int]`,
`List[Int64]`, `List[Float32]`, or `List[Float64]` target type. It converts each
rank-one Int64 component target to an owned List and converts the component's List
predictions back to a rank-one Int64 array. It accepts 0/1 codes for class
indicators and -1/+1 codes for signed output codes. The outer wrapper still
encodes and restores arbitrary supported labels, including strings. Any class
weights configured on the underlying estimator refer to the binary codes.

The adapter borrows features, keeps the native score weight type and forwards
decisions, probabilities and log probabilities when the component provides
those capabilities. It preserves response shape, precision and class order.
For an incremental classifier whose class declarations use the native target
type, the adapter also forwards `partial_fit`. Pass optional `classes` as
rank-one Int64 binary codes. The component validates the declared class set
and owns the update.
The multiclass wrappers still require the corresponding dense input and
response types. `SVC` and `LinearSVC` declare Float64 inputs for generic
composition. `BinaryAdapter(SVC())` works in all three wrappers. OutputCode
uses the binary decision margins with signed -1/+1 codes. OvO forwards
`kernel="precomputed"` through the adapter and retains the training indices
for each class pair. OvR exposes `predict_proba` through SVC's static
probability capability. Set `probability=True` before fit to use it. A fitted
OvR with ordinary SVC components raises on a probability call when the
option is disabled. Scikit-learn hides the method in that configuration.
`BinaryAdapter(LinearDiscriminantAnalysis())` uses LDA's default native
`List[Int]` label type and preserves its decision and probability responses
for either feature dtype.
Use `BinaryAdapter(GaussianNB())` with the default Float64 features and
`BinaryAdapter(GaussianNB[dtype=DType.float32]())` for Float32 features. OvR
and OvO can fit incrementally with either. OutputCode uses 0/1 codes and the
positive-class probabilities.
Use `adapter.estimator` to inspect the underlying model, and use the adapter's
typed `get_params` and `set_params` to configure it. A fitted copy owns its
state independently, while `clone` returns an unfitted adapter.

## K-nearest neighbors components

```mojo
from numojo import NDArray
from moxie.multiclass import OneVsRestClassifier
from moxie.neighbors import KNeighborsClassifier

var model = OneVsRestClassifier(KNeighborsClassifier[Y=NDArray[DType.int64]](3))
_ = model.fit(X, y)
```

Select `Y=NDArray[DType.int64]` for the inner binary targets. The wrapper
handles encoding and decoding its own labels. For Float32 features, select
`KNeighborsClassifier[DType.float32, Y=NDArray[DType.int64]]` and the
corresponding wrapper feature dtype. KNN returns a list of probability
matrices, including for one output. Each wrapper extracts the positive-class
column from the single `(n_samples, 2)` Float64 matrix. A fitted component
copy retains its query state, and `clone` keeps the configuration for a new
fit. The KNN production parity group covers feature matrices, uniform and
distance weights, and all three wrappers.

## Behavioral distinctions

One-vs-rest, one-vs-one and output-code wrappers clone component estimators. Available methods depend on wrapped estimator capabilities; probability and partial_fit are not universal.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../scikit-learn/sklearn/multiclass.py`.
- Guide and mathematical references: `../scikit-learn/doc/modules/multiclass.rst`.
- Shared contracts: `../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/multiclass/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`OneVsRestClassifier`, `OneVsOneClassifier`, `OutputCodeClassifier`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.
