# Adapted from scikit-learn multiclass.py; BSD-3-Clause, see LICENSES/multiclass-scikit-learn.txt.
"""Random output-code classification over numeric or string class labels."""
from numojo import NDArray, Shape, Item
import numojo as nm
from moxie._multiclass_labels import MulticlassLabels, NumericLabels
from std.math import isfinite, inf
from moxie.base import ClassifierMixin, DecisionFunction, Predictor, ProbabilityPredictor, RegressorMixin, SupervisedEstimator, clone
from moxie._random import check_random_state
from moxie._multiclass_binary import BinaryTarget, predict_binary, fit_components, n_workers
from moxie._multiclass_contracts import MulticlassClassifier
from moxie._multiclass_ovr import BinaryComponent

comptime _NOT_FITTED = (
    "NotFittedError: This OutputCodeClassifier instance is not fitted yet."
    " Call 'fit' with appropriate arguments before using this estimator."
)


@fieldwise_init
struct OutputCodeParams[P: Copyable & Deinitable](Copyable, Movable):
    """Constructor configuration, including the component's typed parameters."""
    var estimator: Self.P
    var code_size: Float64
    var random_state: Optional[Int]
    var n_jobs: Optional[Int]


struct OutputCodeClassifier[
    E: SupervisedEstimator & Predictor & Copyable & Deinitable,
    dtype: DType = DType.float64,
    label: DType = DType.int64,
    kind: MulticlassLabels = NumericLabels[label],
    ClassValues: Copyable & Deinitable = kind.Classes,
    Domain: Copyable & Deinitable = kind.Encoding,
](MulticlassClassifier) where conforms_to(E.Params, Deinitable):
    """Fit binary classifiers on random code bits; predict the closest codeword.

    Components receive -1/+1 targets when they expose DecisionFunction, else
    0/1 targets. `random_state` accepts an integer seed or None (the shared
    Moxie default stream). See docs/output_code.md for supported inputs.
    """
    comptime Params = OutputCodeParams[Self.E.Params]
    comptime Input = NDArray[Self.dtype]
    comptime Target = Self.kind.Target
    comptime Prediction = Self.kind.Target
    comptime SampleWeight = NDArray[DType.float64]

    var estimator: Self.E
    var code_size: Float64
    var random_state: Optional[Int]
    var n_jobs: Optional[Int]
    var estimators_: List[BinaryComponent[Self.E]]
    var classes_: Self.ClassValues
    var code_book_: NDArray[DType.float64]
    var n_features_in_: Optional[Int]
    var _encoding: Optional[Self.Domain]

    def __init__(
        out self, var estimator: Self.E, *, code_size: Float64 = 1.5,
        random_state: Optional[Int] = None, n_jobs: Optional[Int] = None,
    ) raises:
        comptime assert Self.E.Input == Self.Input, "OutputCodeClassifier component Input must be NDArray[dtype]"
        comptime assert Self.E.Target == BinaryTarget, "OutputCodeClassifier component Target must be NDArray[DType.int64]"
        comptime assert conforms_to(Self.E, ClassifierMixin) or conforms_to(Self.E, RegressorMixin), "OutputCodeClassifier requires a classifier or regressor component"
        comptime assert conforms_to(Self.E, DecisionFunction) or conforms_to(Self.E, ProbabilityPredictor), "component needs decision_function or predict_proba"
        comptime assert Self.label in [DType.int64, DType.float32, DType.float64], "class labels must be NDArray[int64/float32/float64]"
        comptime assert Self.ClassValues == Self.kind.Classes, "ClassValues must match the label kind"
        comptime assert Self.Domain == Self.kind.Encoding, "Domain must match the label kind"
        self.estimator = estimator^
        self.code_size = code_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimators_ = List[BinaryComponent[Self.E]]()
        self.classes_ = rebind_var[Self.ClassValues](Self.kind.empty())
        self.code_book_ = nm.zeros[DType.float64](Shape(0, 0))
        self.n_features_in_ = None
        self._encoding = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.estimator.get_params(deep), self.code_size, self.random_state, self.n_jobs)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.estimator.set_params(params.estimator)
        self.code_size = params.code_size
        self.random_state = params.random_state
        self.n_jobs = params.n_jobs
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, code_size: Optional[Float64] = None,
        random_state: Optional[Optional[Int]] = None,
        n_jobs: Optional[Optional[Int]] = None,
    ) raises -> ref[self] Self:
        if code_size:
            self.code_size = code_size.value()
        if random_state:
            self.random_state = random_state.value()
        if n_jobs:
            self.n_jobs = n_jobs.value()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, var estimator: Self.E) raises -> ref[self] Self:
        self.estimator = estimator^
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(clone(self.estimator), code_size=self.code_size,
                    random_state=self.random_state, n_jobs=self.n_jobs)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._encoding)

    def _flat_target(self, y: Self.Target) raises -> Self.Target:
        return Self.kind.flat(y, 2)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        """Replace learned state only after every code column fits successfully."""
        if not isfinite(self.code_size) or self.code_size <= 0:
            raise Error("ValueError: code_size must be finite and strictly positive")
        if self.random_state and not (0 <= self.random_state.value() <= 4294967295):
            raise Error("ValueError: random_state must be between 0 and 2**32 - 1")
        var target = self._flat_target(y)
        if Self.kind.size(target) == 0:
            raise Error("ValueError: OutputCodeClassifier can not be fit when no class is present.")
        var encoding = Self.kind.make(target)
        var n_classes = Self.kind.domain_count(encoding)
        var width = Float64(n_classes) * self.code_size
        if not isfinite(width) or width >= 9223372036854775808.0:
            raise Error("ValueError: code_size produces too many estimators")
        var n_columns = Int(width)
        if n_columns == 0:
            raise Error("ValueError: code_size must produce at least one estimator")
        var book = nm.zeros[DType.float64](Shape(n_classes, n_columns))
        # Release the global RNG lease before fitting: components may use it too.
        with check_random_state(self.random_state) as rng:
            for c in range(n_classes):
                for k in range(n_columns):
                    var value: Float64 = 0.0
                    comptime if conforms_to(Self.E, DecisionFunction):
                        value = -1.0
                    if rng[].uniform() > 0.5:
                        value = 1.0
                    book[Item(c, k)] = value
        var workers = n_workers(self.n_jobs, n_columns)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise Error("ValueError: X must be a nonempty two-dimensional array")
        if X.shape[0] != Self.kind.size(target):
            raise Error("ValueError: X and y have inconsistent numbers of samples")
        var n_samples = Self.kind.size(target)
        var n_features = X.shape[1]
        var codes = Self.kind.codes(encoding)
        ref template = self.estimator

        def fit_column(k: Int) raises {imm template, imm codes, imm book, imm X, n_samples, n_classes, n_features} -> BinaryComponent[Self.E]:
            var first = book[Item(0, k)]
            var constant = True
            for c in range(1, n_classes):
                if book[Item(c, k)] != first:
                    constant = False
                    break
            if constant:
                return BinaryComponent[Self.E](constant=Int64(first), n_features=n_features)
            var binary = nm.zeros[DType.int64](Shape(n_samples))
            for i in range(n_samples):
                binary[Item(i)] = Int64(book[Item(Int(codes[Item(i)]), k)])
            var component = clone(template)
            component.fit(rebind[Self.E.Input](X), rebind[Self.E.Target](binary))
            return BinaryComponent[Self.E](component^)

        var components = fit_components[BinaryComponent[Self.E]](fit_column, n_columns, workers)
        self.classes_ = rebind_var[Self.ClassValues](Self.kind.classes(encoding))
        self.code_book_ = book^
        self.estimators_ = components^
        self.n_features_in_ = n_features
        self._encoding = rebind_var[Self.Domain](encoding^)
        return self

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        """Euclidean nearest codeword, with the first class winning exact ties."""
        if not self._encoding:
            raise Error(_NOT_FITTED)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise Error("ValueError: X must be a nonempty two-dimensional array")
        var n_samples = X.shape[0]
        var scores = List[NDArray[DType.float64]]()
        for component in self.estimators_:
            var column: NDArray[DType.float64]
            if component.is_constant():
                component._check_constant_input(X)
                column = nm.full[DType.float64](Shape(n_samples), Float64(component.constant()))
            else:
                column = predict_binary[Self.E, rank_one=True](component.estimator(), rebind[Self.E.Input](X), n_samples)
            for i in range(n_samples):
                if not isfinite(column[Item(i)]):
                    raise Error("ValueError: component predictions contain NaN or infinity")
            scores.append(column^)
        var predicted = nm.zeros[DType.int64](Shape(n_samples))
        for i in range(n_samples):
            var best = inf[DType.float64]()
            var winner = 0
            for c in range(Self.kind.count(rebind[Self.kind.Classes](self.classes_))):
                var distance: Float64 = 0.0
                for k in range(len(scores)):
                    var delta = scores[k][Item(i)] - self.code_book_[Item(c, k)]
                    distance += delta * delta
                if not isfinite(distance):
                    raise Error("ValueError: output-code distance overflow")
                if distance < best:
                    best = distance
                    winner = c
            predicted[Item(i)] = Int64(winner)
        return Self.kind.decode(rebind[Self.kind.Encoding](self._encoding.value()), predicted)

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        var predicted = self.predict(X)
        return Self.kind.accuracy(self._flat_target(y), predicted, sample_weight)
