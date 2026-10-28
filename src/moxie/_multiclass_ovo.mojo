# Adapted from scikit-learn multiclass.py and utils/multiclass.py; BSD-3-Clause,
# see LICENSES/multiclass-scikit-learn.txt.
"""One-vs-one classification over numeric or string class labels."""
from numojo import NDArray, Shape, Item
import numojo as nm
from moxie._multiclass_labels import MulticlassLabels, NumericLabels, _shape_repr
from std.math import isnan
from moxie.base import (
    ClassifierMixin,
    DecisionFunction,
    IncrementalClassifier,
    Predictor,
    ProbabilityPredictor,
    RegressorMixin,
    SupervisedEstimator,
    clone,
)
from moxie._multiclass_binary import (
    BinaryTarget,
    predict_binary,
    binary_threshold,
    fit_components,
    n_workers,
)
from moxie._multiclass_contracts import (
    MulticlassClassifier, MulticlassDecision, PairwiseInput, pairwise_input,
)

comptime _NAME = "OneVsOneClassifier"
comptime _NOT_FITTED = (
    "NotFittedError: This OneVsOneClassifier instance is not fitted yet."
    " Call 'fit' with appropriate arguments before using this estimator."
)


def _check_X[dtype: DType](X: NDArray[dtype]) raises:
    """validate_data's check_array for 2-D numeric X; NaN/inf pass through."""
    if X.ndim == 0:
        raise Error("ValueError: Expected 2D array, got scalar array instead")
    if X.ndim == 1:
        raise Error("ValueError: Expected 2D array, got 1D array instead")
    if X.ndim > 2:
        raise Error(
            String("ValueError: Found array with dim ", X.ndim,
                   ", while dim <= 2 is required by ", _NAME, ".")
        )
    if X.shape[0] < 1:
        raise Error(
            String("ValueError: Found array with ", X.shape[0], " sample(s) (shape=", _shape_repr(X),
                   ") while a minimum of 1 is required by ", _NAME, ".")
        )
    if X.shape[1] < 1:
        raise Error(
            String("ValueError: Found array with ", X.shape[1], " feature(s) (shape=", _shape_repr(X),
                   ") while a minimum of 1 is required by ", _NAME, ".")
        )


def _binary_labels[dtype: DType](values: NDArray[dtype], n_samples: Int) raises -> NDArray[dtype]:
    """A component's `predict` output as a rank-one vector of length n_samples."""
    if values.size != n_samples or values.ndim > 2 or (values.ndim == 2 and values.shape[1] != 1):
        raise Error("ValueError: binary predict must have shape (n_samples,)")
    var result = nm.zeros[dtype](Shape(n_samples))
    for i in range(n_samples):
        result[Item(i)] = values[Item(i)] if values.ndim == 1 else values[Item(i, 0)]
    return result^


@fieldwise_init
struct OneVsOneParams[P: Copyable & Deinitable](Copyable, Movable):
    """Typed `get_params` record: nested estimator parameters and n_jobs."""

    var estimator: Self.P
    var n_jobs: Optional[Int]


struct OneVsOneClassifier[
    E: SupervisedEstimator & Predictor & Copyable & Deinitable,
    dtype: DType = DType.float64,
    label: DType = DType.int64,
    kind: MulticlassLabels = NumericLabels[label],
    ClassValues: Copyable & Deinitable = kind.Classes,
    Domain: Copyable & Deinitable = kind.Encoding,
](
    MulticlassClassifier, MulticlassDecision, PairwiseInput,
    IncrementalClassifier where conforms_to(E, IncrementalClassifier),
):
    """One-vs-one (OvO) multiclass strategy: one binary classifier per class pair.

    For classes `i < j` (in `classes_` order) a clone of `E` is fitted on the
    samples labelled `classes_[i]` (target 0) or `classes_[j]` (target 1).
    Prediction counts each component's `predict` as a vote and breaks ties with
    the normalized sum of its confidences: `decision_function` when `E`
    provides one, otherwise `predict_proba(X)[:, 1]`. Regressors use `predict`
    for confidence; only exact 0/1 predictions cast votes.
    `E` takes `NDArray[dtype]` features, `NDArray[int64]` 0/1 targets and
    predicts `NDArray[int64]` (classifiers) or `NDArray[float32/float64]`
    (regressors). Class labels are a rank-one `NDArray[label]` (or
    an `(n, 1)` column). `kind=StringLabels` selects StringTarget inputs.
    PairwiseInput components can consume dense precomputed kernels in fit.
    `fit_params` are unsupported.
    `partial_fit` requires an IncrementalClassifier component and updates only
    pairs represented in the batch; untouched pairs can remain unfitted.
    """

    comptime Params = OneVsOneParams[Self.E.Params]
    comptime Input = NDArray[Self.dtype]
    comptime Classes = Self.kind.Target
    comptime Target = Self.kind.Target
    comptime Prediction = Self.kind.Target
    comptime SampleWeight = NDArray[DType.float64]
    comptime Decision = NDArray[DType.float64]

    var estimator: Self.E
    var n_jobs: Optional[Int]

    var estimators_: List[Self.E]
    var classes_: Self.ClassValues
    var n_features_in_: Optional[Int]
    var pairwise_indices_: Optional[List[NDArray[DType.int64]]]
    """Training row indices per pair for pairwise batch fit; otherwise None."""
    var _encoding: Optional[Self.Domain]

    def __init__(out self, var estimator: Self.E, *, n_jobs: Optional[Int] = None) raises:
        comptime assert Self.E.Input == NDArray[Self.dtype], (
            "OneVsOneClassifier component Input must be NDArray[dtype]"
        )
        comptime assert Self.E.Target == BinaryTarget, (
            "OneVsOneClassifier component Target must be NDArray[DType.int64]"
        )
        comptime assert conforms_to(Self.E, ClassifierMixin) or conforms_to(Self.E, RegressorMixin), (
            "OneVsOneClassifier requires a classifier or regressor component"
        )
        comptime if conforms_to(Self.E, RegressorMixin):
            comptime assert Self.E.Prediction == NDArray[DType.float32] or Self.E.Prediction == NDArray[DType.float64], (
                "regressor predictions must be NDArray[float32/float64]"
            )
        else:
            comptime assert Self.E.Prediction == BinaryTarget, (
                "OneVsOneClassifier classifier Prediction must be NDArray[DType.int64]"
            )
            comptime assert conforms_to(Self.E, DecisionFunction) or conforms_to(
                Self.E, ProbabilityPredictor
            ), "classifier component needs decision_function or predict_proba"
        comptime assert Self.label in [DType.int64, DType.float32, DType.float64], (
            "class labels must be NDArray[int64/float32/float64]"
        )
        comptime assert Self.ClassValues == Self.kind.Classes, "ClassValues must match the label kind"
        comptime assert Self.Domain == Self.kind.Encoding, "Domain must match the label kind"
        self.estimator = estimator^
        self.n_jobs = n_jobs
        self.estimators_ = List[Self.E]()
        self.classes_ = rebind_var[Self.ClassValues](Self.kind.empty())
        self.n_features_in_ = None
        self.pairwise_indices_ = None
        self._encoding = None

    # ----- configuration -------------------------------------------------

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.estimator.get_params(deep), self.n_jobs)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.estimator.set_params(params.estimator)
        self.n_jobs = params.n_jobs
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, n_jobs: Optional[Optional[Int]] = None) raises -> ref[self] Self:
        """Partial update of the wrapper's own parameters."""
        if n_jobs:
            self.n_jobs = n_jobs.value()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, var estimator: Self.E) raises -> ref[self] Self:
        """Replace the wrapped estimator (sklearn `set_params(estimator=...)`)."""
        self.estimator = estimator^
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(clone(self.estimator), n_jobs=self.n_jobs)

    def pairwise_input(self) -> Bool:
        return pairwise_input(self.estimator)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._encoding)

    # ----- learned state ---------------------------------------------------

    def n_classes_(self) raises -> Int:
        if not self._encoding:
            raise Error(_NOT_FITTED)
        return Self.kind.count(rebind[Self.kind.Classes](self.classes_))

    # ----- fitting ---------------------------------------------------------

    def _flat_target(self, y: Self.Target) raises -> Self.Target:
        return Self.kind.flat(y, 1)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        """Fit one binary component per pair of classes."""
        _check_X(X)
        var target = self._flat_target(y)
        var n_samples = X.shape[0]
        if Self.kind.size(target) != n_samples:
            raise Error(
                String("ValueError: Found input variables with inconsistent numbers of samples: [",
                       n_samples, ", ", Self.kind.size(target), "]")
            )
        var target_kind = Self.kind.kind(target)
        if target_kind != "binary" and target_kind != "multiclass":
            raise Error(
                String("ValueError: Unknown label type: ", target_kind, ". Maybe you are trying to fit a"
                       " classifier, which expects discrete classes on a regression target with"
                       " continuous values.")
            )
        var encoding = Self.kind.make(target)
        var n_classes = Self.kind.domain_count(encoding)
        if n_classes == 1:
            raise Error("ValueError: OneVsOneClassifier can not be fit when only one class is present.")
        var pairs = List[Tuple[Int, Int]]()
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                pairs.append((i, j))
        # joblib.Parallel rejects n_jobs=0 before any component is fitted.
        var workers = n_workers(self.n_jobs, len(pairs))
        # Row indices of each class in sample order, so a pair merges two lists
        # instead of rescanning every sample.
        var class_rows = List[List[Int]](length=n_classes, fill=List[Int]())
        var codes = Self.kind.codes(encoding)
        for s in range(n_samples):
            class_rows[Int(codes[Item(s)])].append(s)
        var pairwise = self.pairwise_input()
        if pairwise and X.shape[0] != X.shape[1]:
            raise Error("ValueError: X should be a square kernel matrix")
        ref template = self.estimator
        comptime FitResult = Tuple[Optional[Self.E], Optional[NDArray[DType.int64]]]

        def fit_pair(k: Int) raises {imm template, imm class_rows, imm pairs, imm X, pairwise} -> FitResult:
            # _fit_ovo_binary: rows of classes i or j in sample order, i -> 0, j -> 1.
            ref first = class_rows[pairs[k][0]]
            ref second = class_rows[pairs[k][1]]
            var rows = List[Int](capacity=len(first) + len(second))
            var binary = nm.zeros[DType.int64](Shape(len(first) + len(second)))
            var a = 0
            var b = 0
            while a < len(first) or b < len(second):
                if b == len(second) or (a < len(first) and first[a] < second[b]):
                    rows.append(first[a])
                    a += 1
                else:
                    binary[Item(len(rows))] = 1
                    rows.append(second[b])
                    b += 1
            var component = clone(template)
            var indices: Optional[NDArray[DType.int64]] = None
            if pairwise:
                var selected = nm.zeros[DType.int64](Shape(len(rows)))
                for s in range(len(rows)):
                    selected[Item(s)] = Int64(rows[s])
                var gather = selected.astype[DType.int]()
                var subset = X.take(gather, axis=0).take(gather, axis=1)
                component.fit(rebind[Self.E.Input](subset), rebind[Self.E.Target](binary))
                indices = selected^
            else:
                component.fit(rebind[Self.E.Input](X[rows]), rebind[Self.E.Target](binary))
            return (Optional[Self.E](component^), indices^)

        var fitted = fit_components[FitResult](fit_pair, len(pairs), workers)
        var components = List[Self.E]()
        var indices = List[NDArray[DType.int64]]()
        for k in range(len(fitted)):
            components.append(fitted[k][0].take())
            if pairwise:
                indices.append(fitted[k][1].take())

        # Commit learned state only after every component fitted successfully.
        self.classes_ = rebind_var[Self.ClassValues](Self.kind.classes(encoding))
        self.estimators_ = components^
        self.n_features_in_ = X.shape[1]
        self.pairwise_indices_ = indices^ if pairwise else None
        self._encoding = rebind_var[Self.Domain](encoding^)
        return self

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
    ) raises -> ref[self] Self where conforms_to(Self.E, IncrementalClassifier):
        """Update nonempty class pairs, retaining untouched (possibly unfitted) pairs."""
        comptime assert Self.E.Classes == BinaryTarget, (
            "OneVsOneClassifier incremental component Classes must be NDArray[DType.int64]"
        )
        var first_call = not self.__sklearn_is_fitted__()
        if first_call and not classes:
            raise Error("ValueError: classes must be passed on the first call to partial_fit.")
        var encoding: Self.kind.Encoding
        if classes:
            if Self.kind.rank(classes.value()) != 1 or Self.kind.size(classes.value()) < 2:
                raise Error("ValueError: classes must be a rank-one array with at least two distinct classes")
            encoding = Self.kind.make(self._flat_target(classes.value()))
            if not first_call:
                var same = Self.kind.same(Self.kind.classes(encoding), rebind[Self.kind.Classes](self.classes_))
                if not same:
                    raise Error("ValueError: classes is not the same as on last call to partial_fit")
        else:
            encoding = Self.kind.from_classes(rebind[Self.kind.Classes](self.classes_))
        if Self.kind.domain_count(encoding) < 2:
            raise Error("ValueError: OneVsOneClassifier requires at least two distinct classes")
        if first_call:
            _check_X(X)
        else:
            self._check_input(X)
        var target = self._flat_target(y)
        if Self.kind.size(target) != X.shape[0]:
            raise Error("ValueError: Found input variables with inconsistent numbers of samples")
        # Validate discrete targets even if the declaration was valid.
        var target_kind = Self.kind.kind(target)
        if target_kind != "binary" and target_kind != "multiclass":
            raise Error("ValueError: unsupported classification target: " + target_kind)
        var codes: BinaryTarget
        try:
            codes = Self.kind.encode(encoding, target)
        except:
            raise Error("ValueError: Mini-batch contains labels outside classes; labels must be subset of classes")
        var pairs = List[Tuple[Int, Int]]()
        for i in range(Self.kind.domain_count(encoding)):
            for j in range(i + 1, Self.kind.domain_count(encoding)):
                pairs.append((i, j))
        var workers = n_workers(self.n_jobs, len(pairs))
        ref template = self.estimator
        ref previous = self.estimators_

        def update_pair(k: Int) raises {imm template, imm previous, imm pairs, imm codes, imm X, first_call} -> Self.E:
            var component: Self.E
            if first_call:
                component = clone(template)
            else:
                component = previous[k].copy()
            var rows = List[Int]()
            for s in range(codes.size):
                if Int(codes[Item(s)]) == pairs[k][0] or Int(codes[Item(s)]) == pairs[k][1]:
                    rows.append(s)
            if len(rows) == 0:
                return component^
            var binary = nm.zeros[DType.int64](Shape(len(rows)))
            for s in range(len(rows)):
                if Int(codes[Item(rows[s])]) == pairs[k][1]:
                    binary[Item(s)] = 1
            var binary_classes = nm.zeros[DType.int64](Shape(2))
            binary_classes[Item(1)] = 1
            var options = Optional[BinaryTarget](binary_classes^)
            component.partial_fit(
                rebind[Self.E.Input](X[rows]), rebind[Self.E.Target](binary),
                rebind[Optional[Self.E.Classes]](options),
            )
            return component^

        var components = fit_components[Self.E](update_pair, len(pairs), workers)
        self.estimators_ = components^
        self.classes_ = rebind_var[Self.ClassValues](Self.kind.classes(encoding))
        self._encoding = rebind_var[Self.Domain](encoding^)
        self.n_features_in_ = X.shape[1]
        self.pairwise_indices_ = None
        return self

    # ----- inference -------------------------------------------------------

    def _check_input(self, X: Self.Input) raises:
        if not self._encoding:
            raise Error(_NOT_FITTED)
        _check_X(X)
        var expected = self.n_features_in_.value()
        if X.shape[1] != expected:
            raise Error(
                String("ValueError: X has ", X.shape[1], " features, but ", _NAME,
                       " is expecting ", expected, " features as input.")
            )

    def _votes_and_confidences(self, X: Self.Input) raises -> NDArray[DType.float64]:
        """sklearn `_ovr_decision_function`: votes plus confidences mapped into (-1/3, 1/3)."""
        var n_samples = X.shape[0]
        var n_classes = Self.kind.count(rebind[Self.kind.Classes](self.classes_))
        var votes = nm.zeros[DType.float64](Shape(n_samples, n_classes))
        var sums = nm.zeros[DType.float64](Shape(n_samples, n_classes))
        var k = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                ref component = self.estimators_[k]
                comptime prediction_dtype = DType.float64 if Self.E.Prediction == NDArray[DType.float64] else (
                    DType.float32 if Self.E.Prediction == NDArray[DType.float32] else DType.int64)
                var predicted: NDArray[prediction_dtype]
                var confidence: NDArray[DType.float64]
                if self.pairwise_indices_:
                    var Xi = X.take(self.pairwise_indices_.value()[k].astype[DType.int](), axis=1)
                    predicted = _binary_labels(
                        rebind_var[NDArray[prediction_dtype]](component.predict(rebind[Self.E.Input](Xi))), n_samples)
                    confidence = predict_binary(component, rebind[Self.E.Input](Xi), n_samples)
                else:
                    predicted = _binary_labels(
                        rebind_var[NDArray[prediction_dtype]](component.predict(rebind[Self.E.Input](X))), n_samples)
                    confidence = predict_binary(component, rebind[Self.E.Input](X), n_samples)
                for s in range(n_samples):
                    var c = confidence[Item(s)]
                    sums[Item(s, i)] = sums[Item(s, i)] - c
                    sums[Item(s, j)] = sums[Item(s, j)] + c
                    var vote = predicted[Item(s)]
                    if vote == 0:
                        votes[Item(s, i)] = votes[Item(s, i)] + 1.0
                    elif vote == 1:
                        votes[Item(s, j)] = votes[Item(s, j)] + 1.0
                k += 1
        for s in range(n_samples):
            for c in range(n_classes):
                var total = sums[Item(s, c)]
                votes[Item(s, c)] = votes[Item(s, c)] + total / (3.0 * (abs(total) + 1.0))
        return votes^

    def decision_function(self, X: Self.Input) raises -> Self.Decision:
        """Shape (n_samples,) for two classes, else (n_samples, n_classes)."""
        self._check_input(X)
        var Y = self._votes_and_confidences(X)
        if Self.kind.count(rebind[Self.kind.Classes](self.classes_)) != 2:
            return Y^
        var column = nm.zeros[DType.float64](Shape(X.shape[0]))
        for s in range(X.shape[0]):
            column[Item(s)] = Y[Item(s, 1)]
        return column^

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        """Class with the highest decision value; thresholded for two classes."""
        var Y = self.decision_function(X)
        var n_samples = X.shape[0]
        var codes = nm.zeros[DType.int64](Shape(n_samples))
        if Self.kind.count(rebind[Self.kind.Classes](self.classes_)) == 2:
            var threshold: Float64 = 0.5
            comptime if not conforms_to(Self.E, RegressorMixin):
                threshold = binary_threshold[Self.E]()
            for s in range(n_samples):
                if Y[Item(s)] > threshold:
                    codes[Item(s)] = 1
        else:
            for s in range(n_samples):
                # np.argmax: the first maximum wins, and the first NaN beats everything.
                var best = 0
                for c in range(Self.kind.count(rebind[Self.kind.Classes](self.classes_))):
                    var value = Y[Item(s, c)]
                    if isnan(value):
                        best = c
                        break
                    if value > Y[Item(s, best)]:
                        best = c
                codes[Item(s)] = Int64(best)
        return Self.kind.decode(rebind[Self.kind.Encoding](self._encoding.value()), codes)

    def score(
        self,
        X: Self.Input,
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        """Mean accuracy, weighted by `sample_weight` when given."""
        var predicted = self.predict(X)
        return Self.kind.accuracy(self._flat_target(y), predicted, sample_weight)
