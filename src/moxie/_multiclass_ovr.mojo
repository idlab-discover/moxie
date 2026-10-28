# Adapted from scikit-learn multiclass.py; BSD-3-Clause, see LICENSES/multiclass-scikit-learn.txt.
"""One-vs-the-rest classification over dense labels or multilabel indicators."""
from numojo import NDArray, Shape, Item
import numojo as nm
from moxie._multiclass_labels import MulticlassLabels, NumericLabels
from std.math import isnan, inf, nan
from moxie.base import (
    BaseEstimator,
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
    binary_confidence,
    predict_binary,
    binary_threshold,
    fit_components,
    mean_accuracy,
    n_workers,
    positive_probability,
)
from moxie._multiclass_contracts import (
    MulticlassClassifier,
    PairwiseInput,
    pairwise_input,
    MulticlassDecision,
    MulticlassProbability,
)

comptime _NOT_FITTED = (
    "NotFittedError: This OneVsRestClassifier instance is not fitted yet."
    " Call 'fit' with appropriate arguments before using this estimator."
)


struct BinaryComponent[E: SupervisedEstimator & Predictor & Copyable & Deinitable](Copyable, Movable):
    """One entry of `estimators_`: a fitted clone, or sklearn's `_ConstantPredictor`.

    sklearn substitutes a private constant predictor when a binary column holds
    a single value (a one-class fit). Its decision value and positive-class
    probability both equal that value, and `predict` repeats it.
    """

    var _estimator: Optional[Self.E]
    var _constant: Int64
    var _n_features: Optional[Int]

    def __init__(out self, var estimator: Self.E):
        self._estimator = estimator^
        self._constant = 0
        self._n_features = None

    def __init__(out self, *, constant: Int64, n_features: Optional[Int]):
        self._estimator = None
        self._constant = constant
        self._n_features = n_features

    def is_constant(self) -> Bool:
        return not self._estimator

    def estimator(ref self) raises -> ref[self._estimator._value] Self.E:
        """The fitted component estimator; raises for a constant component."""
        if not self._estimator:
            raise Error("ValueError: constant component wraps no estimator")
        return self._estimator.value()

    def constant(self) raises -> Int64:
        """The repeated binary label `y_` of a constant component."""
        if self._estimator:
            raise Error("ValueError: component is a fitted estimator, not a constant")
        return self._constant

    def _check_constant_input[dtype: DType](self, X: NDArray[dtype]) raises:
        # _ConstantPredictor runs validate_data(reset=False) before predicting.
        if self._n_features and X.ndim == 2 and X.shape[1] != self._n_features.value():
            raise Error(
                String("ValueError: X has ", X.shape[1], " features, but _ConstantPredictor is expecting ",
                       self._n_features.value(), " features as input.")
            )


@fieldwise_init
struct OneVsRestParams[P: Copyable & Deinitable](Copyable, Movable):
    """Typed `get_params` record: nested estimator parameters, n_jobs, verbose."""

    var estimator: Self.P
    var n_jobs: Optional[Int]
    var verbose: Int


struct OneVsRestClassifier[
    E: SupervisedEstimator & Predictor & Copyable & Deinitable,
    dtype: DType = DType.float64,
    label: DType = DType.int64,
    kind: MulticlassLabels = NumericLabels[label],
    ClassValues: Copyable & Deinitable = kind.Classes,
    Domain: Copyable & Deinitable = kind.Encoding,
](
    MulticlassClassifier,
    PairwiseInput,
    IncrementalClassifier where conforms_to(E, IncrementalClassifier),
    MulticlassDecision where conforms_to(E, DecisionFunction),
    MulticlassProbability where conforms_to(E, ProbabilityPredictor),
):
    """One-vs-the-rest (OvR) multiclass strategy: one binary classifier per class.

    `E` is a binary classifier or regressor taking `NDArray[dtype]` features and `NDArray[int64]`
    0/1 targets. Prediction uses the component's `decision_function` when `E`
    provides one, otherwise `predict_proba(X)[:, 1]`; regressors use `predict(X)`
    with a binary threshold of 0.5. `decision_function` and
    `predict_proba` are available exactly when `E` provides them.
    Class labels are a rank-one `NDArray[label]` (or an `(n, 1)` column).
    Dense 0/1 indicator targets have shape (n, n_labels), with n_labels > 1.
    `partial_fit` requires an IncrementalClassifier component.
    `kind=StringLabels` selects StringTarget inputs. `fit_params` is unsupported.
    """

    comptime Params = OneVsRestParams[Self.E.Params]
    comptime Input = NDArray[Self.dtype]
    comptime Classes = Self.kind.Target
    comptime Target = Self.kind.Target
    comptime Prediction = Self.kind.Target
    comptime SampleWeight = NDArray[DType.float64]
    comptime Decision = NDArray[DType.float64]
    comptime Probabilities = NDArray[DType.float64]

    var estimator: Self.E
    var n_jobs: Optional[Int]
    var verbose: Int

    var estimators_: List[BinaryComponent[Self.E]]
    var classes_: Self.ClassValues
    var n_features_in_: Optional[Int]
    var _encoding: Optional[Self.Domain]
    var _multilabel: Bool

    def __init__(
        out self, var estimator: Self.E, *, n_jobs: Optional[Int] = None, verbose: Int = 0
    ) raises:
        comptime assert Self.E.Input == NDArray[Self.dtype], (
            "OneVsRestClassifier component Input must be NDArray[dtype]"
        )
        comptime assert Self.E.Target == BinaryTarget, (
            "OneVsRestClassifier component Target must be NDArray[DType.int64]"
        )
        comptime assert conforms_to(Self.E, ClassifierMixin) or conforms_to(Self.E, RegressorMixin), (
            "OneVsRestClassifier requires a classifier or regressor component"
        )
        comptime assert conforms_to(Self.E, RegressorMixin) or conforms_to(
            Self.E, DecisionFunction
        ) or conforms_to(Self.E, ProbabilityPredictor), (
            "classifier component needs decision_function or predict_proba"
        )
        comptime assert Self.label in [DType.int64, DType.float32, DType.float64], (
            "class labels must be NDArray[int64/float32/float64]"
        )
        comptime assert Self.ClassValues == Self.kind.Classes, "ClassValues must match the label kind"
        comptime assert Self.Domain == Self.kind.Encoding, "Domain must match the label kind"
        self.estimator = estimator^
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.estimators_ = List[BinaryComponent[Self.E]]()
        self.classes_ = rebind_var[Self.ClassValues](Self.kind.empty())
        self.n_features_in_ = None
        self._encoding = None
        self._multilabel = False

    def _validate_params(self) raises:
        # sklearn checks _parameter_constraints when fit starts, not on assignment.
        if self.verbose < 0:
            raise Error(
                String("ValueError: The 'verbose' parameter of OneVsRestClassifier must be"
                       " an int in the range [0, inf) or a boolean. Got ", self.verbose, " instead.")
            )

    # ----- configuration -------------------------------------------------

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.estimator.get_params(deep), self.n_jobs, self.verbose)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.estimator.set_params(params.estimator)
        self.n_jobs = params.n_jobs
        self.verbose = params.verbose
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        n_jobs: Optional[Optional[Int]] = None,
        verbose: Optional[Int] = None,
    ) raises -> ref[self] Self:
        """Partial update of the wrapper's own parameters."""
        if verbose:
            self.verbose = verbose.value()
        if n_jobs:
            self.n_jobs = n_jobs.value()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, var estimator: Self.E) raises -> ref[self] Self:
        """Replace the wrapped estimator (sklearn `set_params(estimator=...)`)."""
        self.estimator = estimator^
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(clone(self.estimator), n_jobs=self.n_jobs, verbose=self.verbose)

    def pairwise_input(self) -> Bool:
        return pairwise_input(self.estimator)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._multilabel or Bool(self._encoding)

    # ----- learned state ---------------------------------------------------

    def n_classes_(self) raises -> Int:
        if not self.__sklearn_is_fitted__():
            raise Error(_NOT_FITTED)
        return Self.kind.count(rebind[Self.kind.Classes](self.classes_))

    def multilabel_(self) raises -> Bool:
        """Whether the fitted target is a dense multilabel indicator matrix."""
        if not self.__sklearn_is_fitted__():
            raise Error(_NOT_FITTED)
        return self._multilabel

    # ----- fitting ---------------------------------------------------------

    def _flat_target(self, y: Self.Target) raises -> Self.Target:
        return Self.kind.flat(y, 0)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        """Fit one component per label, or one for single-label binary targets."""
        self._validate_params()
        comptime if Self.kind == NumericLabels[Self.label]:
            if Self.kind.kind(y) == "multilabel-indicator":
                self._fit_multilabel(X, rebind[NDArray[Self.label]](y))
                return self
        var target = self._flat_target(y)
        var n_samples = Self.kind.size(target)
        var encoding = Self.kind.make(target)
        var codes = Self.kind.codes(encoding)
        var n_classes = Self.kind.domain_count(encoding)
        var n_columns = 1 if n_classes <= 2 else n_classes
        # joblib.Parallel rejects n_jobs=0 after label binarization, before X reaches a component.
        var workers = n_workers(self.n_jobs, n_columns)
        if X.ndim == 0 or X.shape[0] != n_samples:
            raise Error(
                String("ValueError: Found input variables with inconsistent numbers of samples: [",
                       X.shape[0] if X.ndim > 0 else 0, ", ", n_samples, "]")
            )
        var n_features: Optional[Int] = None
        if X.ndim == 2:
            n_features = X.shape[1]

        var components = List[BinaryComponent[Self.E]]()
        if n_classes == 1:
            # label_binarize yields one all-zero column -> _ConstantPredictor(0).
            components.append(BinaryComponent[Self.E](constant=0, n_features=n_features))
        else:
            # LabelBinarizer: K columns, or a single positive column for K == 2.
            ref template = self.estimator

            def fit_column(column: Int) raises {imm template, imm codes, imm X, n_classes, n_samples} -> BinaryComponent[Self.E]:
                var positive = 1 if n_classes == 2 else column
                var binary = nm.zeros[DType.int64](Shape(n_samples))
                for i in range(n_samples):
                    if Int(codes[Item(i)]) == positive:
                        binary[Item(i)] = 1
                var component = clone(template)
                component.fit(rebind[Self.E.Input](X), rebind[Self.E.Target](binary))
                return BinaryComponent[Self.E](component^)

            components = fit_components[BinaryComponent[Self.E]](fit_column, n_columns, workers)

        # Commit learned state only after every component fitted successfully.
        self.classes_ = rebind_var[Self.ClassValues](Self.kind.classes(encoding))
        self.estimators_ = components^
        self.n_features_in_ = n_features
        self._encoding = rebind_var[Self.Domain](encoding^)
        self._multilabel = False
        return self

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
    ) raises -> ref[self] Self where conforms_to(Self.E, IncrementalClassifier):
        """Update binary components; multilabel batches require an indicator fit first."""
        comptime assert Self.E.Classes == BinaryTarget, (
            "OneVsRestClassifier incremental component Classes must be NDArray[DType.int64]"
        )
        self._validate_params()
        var indicator = Self.kind.kind(y) == "multilabel-indicator"
        var target = y.copy() if indicator else self._flat_target(y)
        if Self.kind.size(target) == 0:
            raise Error("ValueError: y has 0 samples")
        var first_call = not self.__sklearn_is_fitted__()
        if first_call and not classes:
            raise Error("ValueError: classes must be passed on the first call to partial_fit.")
        var encoding: Self.kind.Encoding
        if classes:
            if Self.kind.rank(classes.value()) != 1 or Self.kind.size(classes.value()) == 0:
                raise Error("ValueError: classes must be a nonempty rank-one array")
            encoding = Self.kind.make(self._flat_target(classes.value()))
            if not first_call:
                var same = Self.kind.same(Self.kind.classes(encoding), rebind[Self.kind.Classes](self.classes_))
                if not same:
                    raise Error("ValueError: classes is not the same as on last call to partial_fit")
        else:
            encoding = Self.kind.from_classes(rebind[Self.kind.Classes](self.classes_))
        try:
            # sklearn setdiff1d checks indicator values, not column indices.
            comptime if Self.kind == NumericLabels[Self.label]:
                if indicator:
                    _ = Self.kind.encode(encoding, rebind_var[Self.Target](rebind[NDArray[Self.label]](target).reshape(Shape(Self.kind.size(target)))))
                else:
                    _ = Self.kind.encode(encoding, target)
            else:
                _ = Self.kind.encode(encoding, target)
        except:
            raise Error("ValueError: Mini-batch contains labels outside classes; labels must be subset of classes")
        if indicator and not self._multilabel:
            raise Error("ValueError: The object was not fitted with multilabel input.")
        comptime if Self.kind == NumericLabels[Self.label]:
            if indicator:
                self._validate_indicator(rebind[NDArray[Self.label]](target))
        var Y = Self.kind.binarize(target, Self.kind.classes(encoding))
        var n_columns = Y.shape[1]
        # Match sklearn zip(estimators_, columns), including vector transitions.
        if not first_call and len(self.estimators_) < n_columns:
            n_columns = len(self.estimators_)
        var workers = n_workers(self.n_jobs, n_columns)
        if X.ndim == 0 or X.shape[0] != Y.shape[0]:
            raise Error("ValueError: Found input variables with inconsistent numbers of samples")
        # A constant from ordinary fit has no partial_fit in sklearn either.
        if not first_call:
            for column in range(n_columns):
                if self.estimators_[column].is_constant():
                    raise Error("ValueError: constant component from fit does not support partial_fit")
        ref template = self.estimator
        ref previous = self.estimators_
        var n_samples = Y.shape[0]

        def update_column(column: Int) raises {imm template, imm previous, imm Y, imm X, first_call, n_samples} -> BinaryComponent[Self.E]:
            var component: Self.E
            if first_call:
                component = clone(template)
            else:
                component = previous[column].estimator().copy()
            var binary = nm.zeros[DType.int64](Shape(n_samples))
            for i in range(n_samples):
                binary[Item(i)] = Y[Item(i, column)]
            var binary_classes = nm.zeros[DType.int64](Shape(2))
            binary_classes[Item(1)] = 1
            var options = Optional[BinaryTarget](binary_classes^)
            component.partial_fit(
                rebind[Self.E.Input](X), rebind[Self.E.Target](binary),
                rebind[Optional[Self.E.Classes]](options),
            )
            return BinaryComponent[Self.E](component^)

        var components = fit_components[BinaryComponent[Self.E]](update_column, n_columns, workers)
        self.estimators_ = components^
        self.classes_ = rebind_var[Self.ClassValues](Self.kind.classes(encoding))
        self._encoding = rebind_var[Self.Domain](encoding^)
        self.n_features_in_ = X.shape[1] if X.ndim == 2 else None
        return self

    def _validate_indicator(self, y: NDArray[Self.label]) raises:
        if y.ndim != 2 or y.shape[0] == 0 or y.shape[1] < 2:
            raise Error("ValueError: multilabel indicators must have shape (n_samples, n_labels), n_samples > 0, n_labels > 1")
        for i in range(y.shape[0]):
            for k in range(y.shape[1]):
                if y[Item(i, k)] != 0 and y[Item(i, k)] != 1:
                    raise Error("ValueError: multilabel indicators must contain only 0 or 1")

    def _fit_multilabel(mut self, X: Self.Input, y: NDArray[Self.label]) raises:
        self._validate_indicator(y)
        var n_samples = y.shape[0]
        var n_labels = y.shape[1]
        var workers = n_workers(self.n_jobs, n_labels)
        if X.ndim == 0 or X.shape[0] != n_samples:
            raise Error("ValueError: Found input variables with inconsistent numbers of samples")
        var n_features: Optional[Int] = None
        if X.ndim == 2:
            n_features = X.shape[1]
        var classes = nm.zeros[Self.label](Shape(n_labels))
        for k in range(n_labels):
            classes[Item(k)] = Scalar[Self.label](k)
        ref template = self.estimator

        def fit_column(column: Int) raises {imm template, imm X, imm y, n_samples, n_features} -> BinaryComponent[Self.E]:
            var binary = nm.zeros[DType.int64](Shape(n_samples))
            var first = Int64(y[Item(0, column)])
            var constant = True
            for i in range(n_samples):
                var value = Int64(y[Item(i, column)])
                binary[Item(i)] = value
                if value != first:
                    constant = False
            if constant:
                return BinaryComponent[Self.E](constant=first, n_features=n_features)
            var component = clone(template)
            component.fit(rebind[Self.E.Input](X), rebind[Self.E.Target](binary))
            return BinaryComponent[Self.E](component^)

        var components = fit_components[BinaryComponent[Self.E]](fit_column, n_labels, workers)
        self.classes_ = rebind_var[Self.ClassValues](classes^)
        self.estimators_ = components^
        self.n_features_in_ = n_features
        self._encoding = None
        self._multilabel = True

    # ----- inference -------------------------------------------------------

    def _component_scores[decision: Bool = False](
        self, component: BinaryComponent[Self.E], X: Self.Input, n_samples: Int
    ) raises -> NDArray[DType.float64]:
        """Prediction scores, or explicit decision values for the public method."""
        if component.is_constant():
            component._check_constant_input(X)
            var scores = nm.zeros[DType.float64](Shape(n_samples))
            for i in range(n_samples):
                scores[Item(i)] = Float64(component._constant)
            return scores^
        comptime if decision:
            return binary_confidence(component._estimator.value(), rebind[Self.E.Input](X), n_samples)
        else:
            return predict_binary(component._estimator.value(), rebind[Self.E.Input](X), n_samples)

    def _n_samples(self, X: Self.Input) raises -> Int:
        if not self.__sklearn_is_fitted__():
            raise Error(_NOT_FITTED)
        if X.ndim == 0:
            raise Error("ValueError: Expected array-like input, got a scalar array")
        return X.shape[0]

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        """Argmax for multiclass; strict thresholds for binary and multilabel."""
        var n_samples = self._n_samples(X)
        comptime if Self.kind == NumericLabels[Self.label]:
            if self._multilabel:
                if len(self.estimators_) != Self.kind.count(rebind[Self.kind.Classes](self.classes_)):
                    raise Error("ValueError: The number of class is not equal to the number of dimension of y.")
                var result = nm.zeros[Self.label](Shape(n_samples, len(self.estimators_)))
                # sklearn selects a single threshold from the first fitted component.
                var threshold = self._binary_threshold()
                for k in range(len(self.estimators_)):
                    var scores = self._component_scores(self.estimators_[k], X, n_samples)
                    for i in range(n_samples):
                        if scores[Item(i)] > threshold:
                            result[Item(i, k)] = 1
                return rebind_var[Self.Prediction](result^)
        var codes = nm.zeros[DType.int64](Shape(n_samples))
        var n_classes = Self.kind.count(rebind[Self.kind.Classes](self.classes_))
        if n_classes > 2:
            var maxima = nm.zeros[DType.float64](Shape(n_samples))
            for i in range(n_samples):
                maxima[Item(i)] = -inf[DType.float64]()
            var n_estimators = len(self.estimators_)
            # Reverse order reproduces np.argmax's first-maximum tie-breaking.
            for r in range(n_estimators):
                var k = n_estimators - r - 1
                var scores = self._component_scores(self.estimators_[k], X, n_samples)
                for i in range(n_samples):
                    var score = scores[Item(i)]
                    var best = maxima[Item(i)]
                    # np.maximum propagates NaN, and NaN never compares equal.
                    if isnan(best) or isnan(score):
                        best = nan[DType.float64]()
                    elif score > best:
                        best = score
                    maxima[Item(i)] = best
                    if best == score:
                        codes[Item(i)] = Int64(k)
        elif n_classes == 2:
            var threshold = self._binary_threshold()
            var scores = self._component_scores(self.estimators_[0], X, n_samples)
            for i in range(n_samples):
                if scores[Item(i)] > threshold:
                    codes[Item(i)] = 1
        else:
            # One class: sklearn repeats classes_[0] without consulting the component.
            _ = self._component_scores(self.estimators_[0], X, n_samples)
        return Self.kind.decode(rebind[Self.kind.Encoding](self._encoding.value()), codes)

    def _binary_threshold(self) -> Float64:
        """sklearn `_threshold_for_binary_predict` on estimators_[0]."""
        if self.estimators_[0].is_constant():
            return 0.5
        comptime if conforms_to(Self.E, RegressorMixin):
            return 0.5
        else:
            return binary_threshold[Self.E]()

    def decision_function(self, X: Self.Input) raises -> Self.Decision where conforms_to(Self.E, DecisionFunction):
        """Shape (n_samples,) for one component, else (n_samples, n_classes)."""
        var n_samples = self._n_samples(X)
        var n_estimators = len(self.estimators_)
        if n_estimators == 1:
            return self._component_scores[decision=True](self.estimators_[0], X, n_samples)
        var result = nm.zeros[DType.float64](Shape(n_samples, n_estimators))
        for k in range(n_estimators):
            var scores = self._component_scores[decision=True](self.estimators_[k], X, n_samples)
            for i in range(n_samples):
                result[Item(i, k)] = scores[Item(i)]
        return result^

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(Self.E, ProbabilityPredictor):
        """Positive-class probabilities, normalized per row (single-label case)."""
        var n_samples = self._n_samples(X)
        var n_estimators = len(self.estimators_)
        var n_columns = 2 if n_estimators == 1 else n_estimators
        var result = nm.zeros[DType.float64](Shape(n_samples, n_columns))
        for k in range(n_estimators):
            var positive = self._positive_probability(self.estimators_[k], X, n_samples)
            for i in range(n_samples):
                if n_estimators == 1:
                    result[Item(i, 0)] = 1.0 - positive[Item(i)]
                    result[Item(i, 1)] = positive[Item(i)]
                else:
                    result[Item(i, k)] = positive[Item(i)]
        if self._multilabel:
            return result^
        # Single-label: normalize rows whose sum is nonzero.
        for i in range(n_samples):
            var total: Float64 = 0.0
            for k in range(n_columns):
                total += result[Item(i, k)]
            if total != 0.0:
                for k in range(n_columns):
                    result[Item(i, k)] = result[Item(i, k)] / total
        return result^

    def _positive_probability(
        self, component: BinaryComponent[Self.E], X: Self.Input, n_samples: Int
    ) raises -> NDArray[DType.float64]:
        if component.is_constant():
            return self._component_scores(component, X, n_samples)
        return positive_probability(component._estimator.value(), rebind[Self.E.Input](X), n_samples)

    def score(
        self,
        X: Self.Input,
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        """Mean accuracy (subset accuracy for multilabel), optionally weighted."""
        var predicted = self.predict(X)
        comptime if Self.kind != NumericLabels[Self.label]:
            return Self.kind.accuracy(self._flat_target(y), predicted, sample_weight)
        else:
            if not self._multilabel:
                return Self.kind.accuracy(self._flat_target(y), predicted, sample_weight)
            ref truth = rebind[NDArray[Self.label]](y)
            ref numeric_prediction = rebind[NDArray[Self.label]](predicted)
            self._validate_indicator(truth)
            if truth.shape[0] != numeric_prediction.shape[0] or truth.shape[1] != numeric_prediction.shape[1]:
                raise Error("ValueError: multilabel target shape must match predictions")
            # Reuse weighted accuracy validation on one exact-match bit per sample.
            var matches = nm.zeros[DType.int64](Shape(truth.shape[0]))
            var expected = nm.zeros[DType.int64](Shape(truth.shape[0]))
            for i in range(truth.shape[0]):
                expected[Item(i)] = 1
                var equal = True
                for k in range(truth.shape[1]):
                    if truth[Item(i, k)] != numeric_prediction[Item(i, k)]:
                        equal = False
                        break
                if equal:
                    matches[Item(i)] = 1
            return mean_accuracy(expected, matches, sample_weight)
