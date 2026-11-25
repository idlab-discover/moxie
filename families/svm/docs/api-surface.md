# Public API status

The family exports SVC, NuSVC, SVR, NuSVR, OneClassSVM, LinearSVC, LinearSVR
and l1_min_c as native implementations. It also exports their typed parameter
records and capability traits. The [kernel API](../algorithms/c_svm/docs/api.md)
and [linear API](../algorithms/linear/docs/api.md) describe signatures, supported
settings, learned arrays and native adaptations.

Fit and parameter setters return a mutable borrow. Cloning copies constructor
configuration into an unfitted estimator. Kernel `coef_()` is a computed method;
native callable kernels use a `KernelProvider` passed explicitly during fitting
and inference. The [linear schemas](../algorithms/linear/docs/api.md#native-model-files)
and [kernel schemas](../algorithms/c_svm/docs/api.md#native-model-files) support
experimental native model files.
See [scope.md](scope.md) for exclusions and edge limitations.

## Multiclass composition

Wrap a numeric-label SVC in the multiclass family's
[`BinaryAdapter`](../../../docs/multiclass/family.md#binary-adapter-for-numeric-list-labels)
to use it in `OneVsRestClassifier` or `OneVsOneClassifier`.

```mojo
from moxie.multiclass import BinaryAdapter, OneVsOneClassifier
from moxie.svm import SVC

var model = OneVsOneClassifier(BinaryAdapter(SVC(kernel="linear")))
_ = model.fit(X, y)
```

`X` is a Float64 matrix and `y` is an Int64 vector with the default wrapper
parameters. The adapter translates each binary target to SVC's native List
labels. With `kernel="precomputed"`, OvO selects a square submatrix for each
class pair during fit. Prediction takes one column per original training sample
in training order.

`SVC.copy()` copies fitted state, including the engine buffers and any
retained training snapshot. `clone(svc)` copies constructor settings into an
unfitted estimator. The
[tie-breaking notebook](../../../notebooks/svm/plot_svm_tie_breaking.ipynb)
compares OvR and OvO class order and predictions exactly and Float64 decision
values within the family solver tolerance against scikit-learn.

## LinearSVC composition

Calibration accepts `CalibratedClassifierCV(adapt_classifier(LinearSVC()))`
with a Float64 vector of integer-valued labels. For coefficient-based
selection, use `SelectFromModel(LinearSVCImportance(LinearSVC()))` with the
classifier's native `List[Float64]` target. In a univariate selection pipeline,
configure the scorer to accept that same label list.

```mojo
from moxie.feature_selection import BuiltinScorer, SelectKBest
from moxie.pipeline import make_pipeline
from moxie.svm import LinearSVC

comptime Scorer = BuiltinScorer[DType.float64, DType.float64, List[Float64]]
var model = make_pipeline(SelectKBest(Scorer("f_classif"), k=2), LinearSVC())
_ = model.fit(X, y)
```

The focused `test_linear_composition.mojo` contract runs with the SVM native
tests. Its pinned sklearn comparison checks class order, predictions, selected
features, and transformed values exactly. It allows `rtol=2e-6` and
`atol=2e-7` for margins and sigmoid probabilities because solvers can stop on
different iterations. From the repository root, after building the contract
binary with locked Mojo 1.0.0, run the comparison with scikit-learn built from
revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b` on the Python import path.

```shell
python - <<'PY'
from families.svm.tools.check import ROOT, assemble, compile_file

assemble(ROOT)
source = ROOT / "tests/test_linear_composition.mojo"
compile_file(source, ROOT / "builds" / source.stem)
PY
python families/svm/tests/linear_composition_parity.py
```

## SVC cross-validation and grid search

`cross_val_score` and `GridSearchCV` accept a pipeline built with
`make_pipeline(StandardScaler(), SVC())`. Each fold clones both steps and
fits the scaler on that fold's training rows. The
[typed SVC search adapter](../examples/model_selection_adapter.mojo) shows
how to use `EstimatorScore` for accuracy and `select_rows` to subset the
feature matrix and List targets. Its numeric grid keys are `C` and `gamma`
for bare SVC or `svc__C` and `svc__gamma` for the default named pipeline.
The adapter maps these keys into the typed parameter record and supplies
five unshuffled stratified folds when search omits `cv`. Pass materialized
`IndexSplit` folds to use a different splitter.

The [RBF parameters notebook](../../../notebooks/svm/plot_rbf_parameters.ipynb)
compares both compositions with pinned scikit-learn, including candidate order,
per-fold scores, ranks, best parameters, and predictions from the refitted model.

## Compile and validate

Run `pixi run build`, `pixi run test` and `pixi run parity` from the family root.
Explicit manifests compose the family, engines and algorithms into one verified
namespace. Two isolated consumers compile and execute real estimators against
live Moxie sources. Parity checks compare numerical results with the pinned local
scikit-learn source; compilation alone does not establish parity.
