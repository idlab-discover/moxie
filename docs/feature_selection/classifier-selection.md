# Feature selection with a native classifier

`LinearSVCImportance` wraps the native dense `LinearSVC` for
`SelectFromModel`, `RFE` and `RFECV`.

```mojo
from moxie.feature_selection import DenseSupervisedCV, LinearSVCImportance, RFECV, SequentialCV
from moxie.svm import LinearSVC

comptime Estimator = LinearSVCImportance[String]
comptime CV = DenseSupervisedCV[Estimator]
var estimator = Estimator(LinearSVC[String](dual=False, random_state=42))
var selector = RFECV[CV](estimator^, cv=SequentialCV(3))
# X is a Float64 matrix with one row per label.
var y: List[String] = ["cat", "dog", "cat", "dog", "cat", "dog"]
_ = selector.fit(X, y)
var decision = selector.decision_function(X)
var classes = selector.classes_()
```

The adapter keeps the model's `List[Label]` targets, predictions and sorted
classes. It accepts Float64 features by default. Select Float32 input with
`LinearSVCImportance[Label, DType.float32]` and use the same feature dtype for
`DenseSupervisedCV`. Float32 features and weights are promoted to Float64 for
LIBLINEAR. Decision values retain Float64 and match the model's binary-vector
or multiclass-matrix shape. Coefficient importances are owned arrays in the
selector dtype. Both `"auto"` and `"coef_"` select those coefficients. L1 models
use the default `1e-5` threshold and other models use the mean. RFE and RFECV
forward prediction, accuracy and decision values. LinearSVC offers no
probability capability. Set model options through the wrapped model or the
adapter's typed parameter record.

For default stratification, `DenseSupervisedCV` accepts lists of String, Int,
Int64, Float32 and Float64 alongside its existing targets. It slices the
original labels for each fold and passes the same label values to shared
stratification. Int64 identities remain exact above 2**53.

The focused contract is `tests/feature_selection/test_linear_svc_importance.mojo`.
After live assembly, `python tests/feature_selection/parity_linear_svc.py`
runs the three model-based selectors against the pinned scikit-learn
LinearSVC and selector sources. The binary Int64/Float64 and multiclass
String/Float32 fixtures check masks, coefficients, decision shapes and
values, class order, weighted fits, and RFECV folds and results. Numerical
tolerances of 3e-6 for Float32 and 2e-7 for Float64 allow coefficient
rounding and solver termination differences. Discrete comparisons are exact.

## Logistic regression

`LogisticRegressionImportance` wraps native dense logistic regression for
`SelectFromModel`, `RFE` and `RFECV`. Construct it with a configured estimator
to infer its feature and label dtypes.

```mojo
from moxie.feature_selection import DenseSupervisedCV, LogisticRegressionImportance, RFECV, SequentialCV
from moxie.linear_model import LogisticRegression

comptime Estimator = LogisticRegressionImportance[]
var estimator = LogisticRegressionImportance(LogisticRegression(C=0.5))
var selector = RFECV[DenseSupervisedCV[Estimator]](estimator^, cv=SequentialCV(3))
# X is a Float64 feature matrix and y is an Int64 label array.
_ = selector.fit(X, y)
var probabilities = selector.predict_proba(X)
var classes = selector.classes_()
```

The adapter uses the model's numeric label arrays and feature dtype
(Float64 by default). It forwards sample weights and returns owned coefficient
and class arrays. RFE and RFECV also forward prediction, accuracy, decision
values, probabilities and log probabilities after selecting columns.
Probability columns follow the model's sorted class order. Binary decision
values are a vector and multiclass decision values are a matrix. Both
`"auto"` and `"coef_"` select the native coefficients. The model's supported
solvers and penalty settings apply.

After assembling live sources, run
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/feature_selection/parity_logistic_importance.py`.
The parity fixtures exercise weighted SelectFromModel and RFE and stratified
RFECV. They compare binary Float32 Newton-Cholesky and multiclass Float64 L-BFGS
against pinned scikit-learn. Discrete comparisons are exact, including Int64
labels above 2**53. The respective tolerances 1e-5 and 1e-6 account for
Float32 rounding and differences in solver termination.

## Ridge classifier

`RidgeClassifierImportance(RidgeClassifier(...))` adapts the native dense
Ridge classifier for `SelectFromModel`, `RFE` and `RFECV`. The constructor
infers the feature and label dtypes from the supplied model.

```mojo
from moxie.feature_selection import DenseSupervisedCV, RFECV, RidgeClassifierImportance, SequentialCV
from moxie.linear_model import RidgeClassifier

comptime Estimator = RidgeClassifierImportance[]
var estimator = RidgeClassifierImportance(RidgeClassifier(alpha=0.5))
var selector = RFECV[DenseSupervisedCV[Estimator]](estimator^, cv=SequentialCV(3))
# X is a Float64 feature matrix and y is an Int64 label array.
_ = selector.fit(X, y)
var classes = selector.classes_()
var decision = selector.decision_function(X)
```

The adapter uses the model's numeric label arrays, including multilabel
indicators. It forwards sample weights in the feature dtype and returns
owned classes and coefficients. The `"auto"` and `"coef_"` getters both use
ridge coefficients, with the mean as the default SelectFromModel threshold.
Binary coefficients are a vector. Multiclass and multilabel coefficients
are matrices with one row per class or output. RFE and RFECV forward
predictions, accuracy and decision values after selecting columns.
`DenseSupervisedCV` accepts Int64, Float32 and Float64 numeric label arrays.

After live assembly, run
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/feature_selection/parity_ridge_classifier.py`.
Fixtures compare binary Float32, multiclass Float64 and multilabel Float64
Cholesky fits, including weighted SelectFromModel and RFE. RFECV uses
stratified folds for single-label targets and ordinary folds for multilabel
indicators. Discrete comparisons are exact, including Int64 labels above
2**53. Tolerances of 3e-4 for Float32 and 1e-8 for Float64 match the Ridge
owner's direct-solver comparisons.

## Linear-kernel SVC

`SVCImportance` adapts the native dense `SVC(kernel="linear")` for
`SelectFromModel`, `RFE` and `RFECV`.

```mojo
from moxie.feature_selection import DenseSupervisedCV, RFECV, SVCImportance, SequentialCV
from moxie.svm import SVC

comptime Estimator = SVCImportance[String]
var estimator = SVCImportance(SVC[String](kernel="linear", random_state=11))
var selector = RFECV[DenseSupervisedCV[Estimator]](estimator^, cv=SequentialCV(3))
# X is a Float64 feature matrix and y is a List[String].
_ = selector.fit(X, y)
var classes = selector.classes_()
var decision = selector.decision_function(X)
```

Constructing the adapter from a model infers the label type. Use
`SVCImportance[Label, DType.float32]` for Float32 features and sample weights.
The adapter promotes these arrays to the SVM engine's Float64 dtype and
returns importances in the selector's feature dtype. Both `"auto"` and
`"coef_"` return an owned coefficient matrix with one row for binary
classification and one row per class pair for multiclass classification.
The default SelectFromModel threshold is the mean of the aggregated
importances. Fitting through the adapter requires `kernel="linear"`.

RFE and RFECV forward class order, predictions, accuracy and Float64 decision
values on the selected columns. A model fitted with `probability=True` also
provides Float64 probabilities and log probabilities in class order. With
probability disabled, these methods raise the wrapped estimator's capability
error.

After live assembly, run
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/feature_selection/parity_svc.py`.
The six binary Float64 and multiclass Float32 selector cases check strided
inputs, weights in SelectFromModel and RFE, and stratified RFECV. The oracle
runs pinned sklearn selectors, SVC wrappers and LIBSVM. Masks, ranks, labels
and shapes match exactly. Numerical outputs use the SVM owner's tolerances
of `rtol=3e-5` and `atol=3e-6`.

## Decision trees

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
its Float32/Float64 feature dtype. For trees it accepts StringTarget and
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
