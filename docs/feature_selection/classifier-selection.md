# Feature selection with a native classifier

`DecisionTreeClassifierImportance` adapts the sibling family's production
`DecisionTreeClassifier` for SelectFromModel, RFE, RFECV and
SequentialFeatureSelector. Training and prediction remain in the adopted tree
modules. No classifier algorithm is copied into feature selection.

```mojo
from moxie._labels import StringTarget
from moxie.tree import DecisionTreeClassifier
from moxie.feature_selection import DecisionTreeClassifierImportance, DenseSupervisedCV, RFECV, SequentialCV

comptime Estimator = DecisionTreeClassifierImportance[DType.float64, DType.int64, True]
comptime CV = DenseSupervisedCV[Estimator, DType.float64]
var estimator = Estimator(DecisionTreeClassifier[DType.int64, True](random_state=42))
var selector = RFECV[CV](estimator^, cv=SequentialCV(3))
# X is a Float64 matrix, one row per string in y.
var y = StringTarget(["cat", "dog", "cat", "dog", "cat", "dog"])
selector.fit(X, y)
var probabilities = selector.predict_proba(X)
var classes = selector.classes_()
```

The adapter parameters are feature dtype (Float64 by default), label dtype
(Int64 by default), and `string_labels` (False by default). With strings enabled,
targets and predictions are `StringTarget` and classes are `List[String]`.
Otherwise numeric labels and class arrays preserve their Int64/Float32/Float64
dtype. Targets must have rank one; multioutput and indicator targets are
explicitly rejected by this adapter. The CV helper itself preserves rank-one
and rank-two targets when slicing.

Features materialize logical strides before converting to Float32 at the tree
boundary, matching sklearn trees.
Sample weights convert to Float64. Importances use the feature dtype;
probabilities and log probabilities are Float64 matrices in sorted class order.
There is no label conversion through floating point. Clone returns unfitted
constructor configuration; fitted copies own the complete adopted tree state.
Repeated fits replace that state and class domain.

RFE and RFECV expose prediction, accuracy scoring, probabilities, log
probabilities and classes conditionally through existing traits. The adapter
has no decision function. SelectFromModel and sequential selection remain
transformers without prediction methods. SelectFromModel's fitted classifier
is available through `estimator_()`. Sparse inputs, Python metadata routing,
reflection and warning transport are outside this native interface.

`DenseSupervisedCV` now takes targets from `Estimator.Target`, independently of
its Float32/Float64 feature dtype. Supported targets are StringTarget and
Int64/Float32/Float64 arrays. Original labels go directly to shared SplitLabels;
shared check_cv and the shared CV engine still own stratification and scoring.
Metadata remains optional sample weights in the feature dtype.

The `tree.selection` and `categorical` records in `DEPENDENCIES.json`
specify local source paths and retained license assets. The assembler reads
those live sources on every build, including uncommitted sibling edits.
Review the adapter's fitted-copy fields when the tree implementation changes.

Run `pixi run build`, `pixi run test`, and
`pixi run python tests/parity_classifier_selection.py`. The parity runner uses
`MOXIE_REFERENCE_PYTHON` or the sibling tree's `builds/oracle-env/bin/python`.
It loads selector and classifier classes from the local sklearn checkout at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`; installed sklearn supplies compiled
infrastructure. Oracle JSON, generated Mojo and outputs stay under ignored
`builds/classifier-selection-parity`.

Sixteen small fixtures apply all four selectors to four configurations:
Float32/string/binary, Float64/string/multiclass, Float64/Int64/binary and
Float32/Int64/multiclass. They check exact masks,
rankings, folds, labels and class ordering; RFECV fold scores, masks and rankings;
probabilities, log probabilities and accuracy; cloning and refitting with a
changed feature/class set. Numeric comparisons use Float32 rtol=2e-6 or Float64
rtol=1e-12, with atol=1e-12, allowing importance storage and reduction rounding.
Targets include adjacent integers above 2**53 and Int64 minimum. Separate native
contracts check slicing ownership, repeated indices, negative strides, matrix
strings, invalid lengths, unfitted access and parameter ownership. Both isolated
consumer assemblies exercise classifier selection. These bounded fixtures do
not establish parity for every tree parameter, data set or Python capability.

Repeated fits share an out-of-line test helper to bound optimizer work. Use
`--start-case N` to resume at a specific fixture; the default runs all sixteen.
Every generated fixture compiles separately with the shared serial compiler
lock, ten-minute timeout and unchanged hard 8 GiB address-space cap.
