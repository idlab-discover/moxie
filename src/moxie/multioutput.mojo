# Adapted from sklearn/multioutput.py; BSD-3-Clause, see LICENSES/scikit-learn.txt.
"""Dense multioutput estimators with independently cloned component models."""
import numojo as nm
from numojo import NDArray, Shape
from std.math import isfinite, floor, nan, log
from moxie.base import (BaseEstimator, SupervisedEstimator, Predictor, ProbabilityPredictor,
    DecisionFunction, LogProbabilityPredictor, IncrementalClassifier, IncrementalRegressor,
    WeightedSupervisedEstimator, clone)
from moxie._multioutput_contracts import MultiOutputClassifierAPI, MultiOutputRegressorAPI, MultiOutputProbability
from moxie._multioutput import validate_targets, clone_outputs, target_column, stack_predictions, augment_features, ChainOrder, ChainResponses
from moxie._random import RandomState, entropy_state


@fieldwise_init
struct IndependentParams[P: Copyable & Deinitable](Copyable, Movable):
    var estimator: Self.P
    var n_jobs: Optional[Int]


@fieldwise_init
struct ChainParams[P: Copyable & Deinitable](Copyable, Movable):
    var estimator: Self.P
    var order: Optional[List[Int]]
    var random_state: Optional[Int]
    var random_order: Bool
    var verbose: Bool
    var chain_method: String


def _features[dtype: DType](X: NDArray[dtype], expected: Optional[Int] = None) raises:
    if X.ndim != 2:
        raise Error("ValueError: X must be a matrix")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise Error("ValueError: X must be nonempty")
    if expected and X.shape[1] != expected.value():
        raise Error("ValueError: feature count differs from fit")


def _finite[dtype: DType](matrix: NDArray[dtype]) raises:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not isfinite(Float64(matrix.unsafe_get(i * Int(matrix.strides[0]) + j * Int(matrix.strides[1])))):
                raise Error("ValueError: input contains NaN or infinity")


def _classes[label: DType](y: NDArray[label]) raises -> NDArray[label]:
    if y.ndim != 1 or y.shape[0] == 0:
        raise Error("ValueError: class labels must be a nonempty vector")
    var values = List[Scalar[label]]()
    for i in range(y.shape[0]):
        var value = y.unsafe_get(i * Int(y.strides[0]))
        if not isfinite(Float64(value)) or Float64(value) != floor(Float64(value)):
            raise Error("ValueError: classification targets must be discrete and finite")
        var found = False
        for j in range(len(values)):
            found = found or values[j] == value
        if not found:
            values.append(value)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[j] < values[i]:
                var tmp = values[i]
                values[i] = values[j]
                values[j] = tmp
    var result = nm.zeros[label](Shape(len(values)))
    for i in range(len(values)):
        result.unsafe_set(i, values[i])
    return result^


def _put_column[dtype: DType, label: DType](mut matrix: NDArray[dtype], j: Int, values: NDArray[label]) raises:
    if values.ndim != 1 or values.shape[0] != matrix.shape[0]:
        raise Error("ValueError: component response must be a vector with one value per sample")
    for i in range(matrix.shape[0]):
        matrix.unsafe_set(i * Int(matrix.strides[0]) + j * Int(matrix.strides[1]), values.unsafe_get(i * Int(values.strides[0])).cast[dtype]())


def _dense[T: Movable, dtype: DType](var values: T) raises -> NDArray[dtype]:
    """Convert a component's Float32/Float64 NDArray response to the feature dtype."""
    comptime if T == NDArray[dtype]:
        return rebind_var[NDArray[dtype]](values^)
    elif T == NDArray[DType.float64]:
        return rebind_var[NDArray[DType.float64]](values^).astype[dtype]()
    elif T == NDArray[DType.float32]:
        return rebind_var[NDArray[DType.float32]](values^).astype[dtype]()
    else:
        comptime assert False, "component probabilities and decisions must be Float32 or Float64 NDArrays"
        raise Error("ValueError: unsupported component response type")


def _response[E: Predictor, dtype: DType, label: DType](estimator: E, X: NDArray[dtype], method: String) raises -> NDArray[dtype]:
    if method == "predict":
        var raw = rebind_var[NDArray[label]](estimator.predict(rebind[E.Input](X)))
        var result = nm.zeros[dtype](Shape(X.shape[0]))
        if raw.ndim != 1 or raw.shape[0] != X.shape[0]:
            raise Error("ValueError: component prediction must match samples")
        for i in range(X.shape[0]):
            result.unsafe_set(i, raw.unsafe_get(i * Int(raw.strides[0])).cast[dtype]())
        return result^
    if method == "decision_function":
        comptime if conforms_to(E, DecisionFunction):
            return _dense[E.Decision, dtype](estimator.decision_function(rebind[E.Input](X)))
    if method == "predict_proba":
        comptime if conforms_to(E, ProbabilityPredictor):
            var probability = _dense[E.Probabilities, dtype](estimator.predict_proba(rebind[E.Input](X)))
            if probability.ndim != 2 or probability.shape[1] != 2:
                raise Error("ValueError: chain response requires two probability columns")
            return target_column(probability, 1)
    if method == "predict_log_proba":
        comptime if conforms_to(E, LogProbabilityPredictor):
            var probability = _dense[E.Probabilities, dtype](estimator.predict_log_proba(rebind[E.Input](X)))
            if probability.ndim != 2 or probability.shape[1] != 2:
                raise Error("ValueError: chain response requires two log probability columns")
            return target_column(probability, 1)
    raise Error("ValueError: component lacks requested chain response")


def _check_method[E: Predictor](method: String) raises:
    if method == "predict":
        return
    comptime if conforms_to(E, ProbabilityPredictor):
        if method == "predict_proba":
            return
    comptime if conforms_to(E, LogProbabilityPredictor):
        if method == "predict_log_proba":
            return
    comptime if conforms_to(E, DecisionFunction):
        if method == "decision_function":
            return
    raise Error("ValueError: unavailable chain_method")


def _score[label: DType, classifier: Bool](y: NDArray[label], pred: NDArray[label], weights: Optional[NDArray[DType.float64]]) raises -> Float64:
    validate_targets(y, pred.shape[0])
    _finite(y)
    _finite(pred)
    if y.shape[1] != pred.shape[1]:
        raise Error("ValueError: score output count differs from fit")
    var n = y.shape[0]
    var q = y.shape[1]
    var w = nm.ones[DType.float64](Shape(n))
    if weights:
        if weights.value().ndim != 1 or weights.value().shape[0] != n:
            raise Error("ValueError: sample_weight must have one value per sample")
        w = weights.value().copy()
    for i in range(n):
        if not isfinite(w.unsafe_get(i * Int(w.strides[0]))):
            raise Error("ValueError: sample weights must be finite")
    var total = Float64(nm.sum(w))
    if total == 0:
        raise Error("ValueError: sample weights sum to zero")
    comptime if classifier:
        var correct = nm.zeros[DType.float64](Shape(n))
        for i in range(n):
            var equal = True
            for j in range(q):
                equal = equal and y.unsafe_get(i * Int(y.strides[0]) + j * Int(y.strides[1])) == pred.unsafe_get(i * q + j)
            if equal:
                correct.unsafe_set(i, w.unsafe_get(i * Int(w.strides[0])))
        return Float64(nm.sum(correct)) / total
    else:
        if n < 2:
            return nan[DType.float64]()
        var scores = nm.zeros[DType.float64](Shape(q))
        for j in range(q):
            var terms = nm.zeros[DType.float64](Shape(n))
            for i in range(n):
                terms.unsafe_set(i, Float64(y.unsafe_get(i * Int(y.strides[0]) + j * Int(y.strides[1]))) * w.unsafe_get(i * Int(w.strides[0])))
            var mean = Float64(nm.sum(terms)) / total
            var residuals = nm.zeros[DType.float64](Shape(n))
            for i in range(n):
                var actual = Float64(y.unsafe_get(i * Int(y.strides[0]) + j * Int(y.strides[1])))
                var error = actual - Float64(pred.unsafe_get(i * q + j))
                var centered = actual - mean
                terms.unsafe_set(i, centered * centered * w.unsafe_get(i * Int(w.strides[0])))
                residuals.unsafe_set(i, error * error * w.unsafe_get(i * Int(w.strides[0])))
            var numerator = Float64(nm.sum(residuals))
            var denominator = Float64(nm.sum(terms))
            var score: Float64 = 0
            if numerator == 0:
                score = 1
            elif denominator != 0:
                score = 1 - numerator / denominator
            scores.unsafe_set(j, score)
        return Float64(nm.sum(scores)) / Float64(q)


struct MultiOutputRegressor[E: SupervisedEstimator & Predictor & Deinitable, dtype: DType = DType.float64, label: DType = dtype](
    MultiOutputRegressorAPI,
    Copyable where conforms_to(E, Copyable),
    WeightedSupervisedEstimator where conforms_to(E, WeightedSupervisedEstimator),
    IncrementalRegressor where conforms_to(E, IncrementalRegressor),
) where conforms_to(E.Params, Deinitable):
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.label]
    comptime Prediction = Self.Target
    comptime SampleWeight = NDArray[DType.float64]
    comptime Params = IndependentParams[Self.E.Params]
    var estimator: Self.E
    var n_jobs: Optional[Int]
    var order: Optional[List[Int]]
    var random_state: Optional[Int]
    var random_order: Bool
    var verbose: Bool
    var chain_method: String
    var chain_method_: String
    var estimators_: List[Self.E]
    var classes_: List[NDArray[Self.label]]
    var order_: List[Int]
    var n_features_in_: Optional[Int]

    def __init__(out self, var estimator: Self.E, *, n_jobs: Optional[Int] = None) raises:
        comptime assert Self.dtype in (DType.float32, DType.float64), "features must use Float32 or Float64"
        comptime assert Self.E.Input == Self.Input, "component Input must match feature dtype"
        comptime assert Self.E.Target == Self.Target, "component Target must match target dtype"
        comptime assert Self.E.Prediction == Self.Target, "component Prediction must match target dtype"
        self.estimator = estimator^
        self.n_jobs = n_jobs
        self.order = None
        self.random_state = None
        self.random_order = False
        self.verbose = False
        self.chain_method = "predict"
        self.chain_method_ = "predict"
        self.estimators_ = List[Self.E]()
        self.classes_ = List[NDArray[Self.label]]()
        self.order_ = List[Int]()
        self.n_features_in_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.estimator.get_params(deep), self.n_jobs)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        _ = self.estimator.set_params(params.estimator)
        self.n_jobs = params.n_jobs
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, n_jobs: Optional[Optional[Int]] = None) raises -> ref[self] Self:
        if n_jobs:
            self.n_jobs = n_jobs.value()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, var estimator: Self.E) raises -> ref[self] Self:
        self.estimator = estimator^
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(clone(self.estimator), n_jobs=self.n_jobs)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.n_features_in_)

    def _check(self, X: Self.Input) raises:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: call fit before inference")
        _features(X, self.n_features_in_)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight) raises -> ref[self] Self where conforms_to(Self.E, WeightedSupervisedEstimator):
        self._fit(X, y, sample_weight.copy())
        return self

    def _fit(mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight]) raises:
        _features(X)
        validate_targets(y, X.shape[0])
        _finite(y)
        if self.n_jobs and self.n_jobs.value() != 1:
            raise Error("ValueError: native execution supports n_jobs=None or 1")
        if self.chain_method != "predict":
            raise Error("ValueError: this implementation supports chain_method='predict'")
        var order = List[Int]()
        for j in range(y.shape[1]):
            order.append(j)
        if self.order or self.random_order or self.random_state or self.verbose:
            raise Error("ValueError: chain options are unavailable on independent estimators")
        var models = clone_outputs(self.estimator, y.shape[1])
        var classes = List[NDArray[Self.label]]()
        for j in range(y.shape[1]):
            var target = target_column(y, order[j])
            ref input = X
            if sample_weight:
                comptime if conforms_to(Self.E, WeightedSupervisedEstimator):
                    comptime if Self.E.SampleWeight == NDArray[DType.float32]:
                        var narrow = sample_weight.value().astype[DType.float32]()
                        _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target), sample_weight=rebind[Self.E.SampleWeight](narrow))
                    else:
                        comptime assert Self.E.SampleWeight == Self.SampleWeight, "component sample weights must be Float32 or Float64 NDArrays"
                        _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target), sample_weight=rebind[Self.E.SampleWeight](sample_weight.value()))
                else:
                    raise Error("ValueError: component does not support sample weights")
            else:
                _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target))
        self.estimators_ = models^
        self.classes_ = classes^
        self.order_ = order^
        self.n_features_in_ = X.shape[1]
        self.chain_method_ = self.chain_method

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        self._check(X)
        var columns = List[Self.Target]()
        for j in range(len(self.estimators_)):
            var column = self.estimators_[j].predict(rebind[Self.E.Input](X))
            columns.append(rebind_var[Self.Target](column^))
        var result = stack_predictions(columns)
        if result.shape[0] != X.shape[0]:
            raise Error("ValueError: component prediction sample count differs")
        return result^

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        var prediction = self.predict(X)
        return _score[Self.label, False](y, prediction, sample_weight)

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self where conforms_to(Self.E, IncrementalRegressor):
        _features(X, self.n_features_in_)
        validate_targets(y, X.shape[0])
        _finite(y)
        if self.n_jobs and self.n_jobs.value() != 1:
            raise Error("ValueError: native execution supports n_jobs=None or 1")
        if self.__sklearn_is_fitted__() and y.shape[1] != len(self.estimators_):
            raise Error("ValueError: partial_fit output count differs from fit")
        if not self.__sklearn_is_fitted__():
            self.estimators_ = clone_outputs(self.estimator, y.shape[1])
        for j in range(y.shape[1]):
            var column = target_column(y, j)
            _ = self.estimators_[j].partial_fit(rebind[Self.E.Input](X), rebind[Self.E.Target](column))
        self.n_features_in_ = X.shape[1]
        return self

struct MultiOutputClassifier[E: SupervisedEstimator & Predictor & Deinitable, dtype: DType = DType.float64, label: DType = DType.int64](
    MultiOutputClassifierAPI,
    Copyable where conforms_to(E, Copyable),
    WeightedSupervisedEstimator where conforms_to(E, WeightedSupervisedEstimator),
    IncrementalClassifier where conforms_to(E, IncrementalClassifier),
    MultiOutputProbability where conforms_to(E, ProbabilityPredictor),
) where conforms_to(E.Params, Deinitable):
    comptime Classes = List[NDArray[Self.label]]
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.label]
    comptime Prediction = Self.Target
    comptime Probabilities = List[NDArray[Self.dtype]]
    comptime SampleWeight = NDArray[DType.float64]
    comptime Params = IndependentParams[Self.E.Params]
    var estimator: Self.E
    var n_jobs: Optional[Int]
    var order: Optional[List[Int]]
    var random_state: Optional[Int]
    var random_order: Bool
    var verbose: Bool
    var chain_method: String
    var chain_method_: String
    var estimators_: List[Self.E]
    var classes_: List[NDArray[Self.label]]
    var order_: List[Int]
    var n_features_in_: Optional[Int]

    def __init__(out self, var estimator: Self.E, *, n_jobs: Optional[Int] = None) raises:
        comptime assert Self.dtype in (DType.float32, DType.float64), "features must use Float32 or Float64"
        comptime assert Self.E.Input == Self.Input, "component Input must match feature dtype"
        comptime assert Self.E.Target == Self.Target, "component Target must match target dtype"
        comptime assert Self.E.Prediction == Self.Target, "component Prediction must match target dtype"
        self.estimator = estimator^
        self.n_jobs = n_jobs
        self.order = None
        self.random_state = None
        self.random_order = False
        self.verbose = False
        self.chain_method = "predict"
        self.chain_method_ = "predict"
        self.estimators_ = List[Self.E]()
        self.classes_ = List[NDArray[Self.label]]()
        self.order_ = List[Int]()
        self.n_features_in_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.estimator.get_params(deep), self.n_jobs)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        _ = self.estimator.set_params(params.estimator)
        self.n_jobs = params.n_jobs
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, n_jobs: Optional[Optional[Int]] = None) raises -> ref[self] Self:
        if n_jobs:
            self.n_jobs = n_jobs.value()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, var estimator: Self.E) raises -> ref[self] Self:
        self.estimator = estimator^
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(clone(self.estimator), n_jobs=self.n_jobs)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.n_features_in_)

    def _check(self, X: Self.Input) raises:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: call fit before inference")
        _features(X, self.n_features_in_)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight) raises -> ref[self] Self where conforms_to(Self.E, WeightedSupervisedEstimator):
        self._fit(X, y, sample_weight.copy())
        return self

    def _fit(mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight]) raises:
        _features(X)
        validate_targets(y, X.shape[0])
        _finite(y)
        if self.n_jobs and self.n_jobs.value() != 1:
            raise Error("ValueError: native execution supports n_jobs=None or 1")
        if self.chain_method != "predict":
            raise Error("ValueError: this implementation supports chain_method='predict'")
        var order = List[Int]()
        for j in range(y.shape[1]):
            order.append(j)
        if self.order or self.random_order or self.random_state or self.verbose:
            raise Error("ValueError: chain options are unavailable on independent estimators")
        var models = clone_outputs(self.estimator, y.shape[1])
        var classes = List[NDArray[Self.label]]()
        for j in range(y.shape[1]):
            var target = target_column(y, order[j])
            classes.append(_classes(target))
            ref input = X
            if sample_weight:
                comptime if conforms_to(Self.E, WeightedSupervisedEstimator):
                    comptime if Self.E.SampleWeight == NDArray[DType.float32]:
                        var narrow = sample_weight.value().astype[DType.float32]()
                        _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target), sample_weight=rebind[Self.E.SampleWeight](narrow))
                    else:
                        comptime assert Self.E.SampleWeight == Self.SampleWeight, "component sample weights must be Float32 or Float64 NDArrays"
                        _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target), sample_weight=rebind[Self.E.SampleWeight](sample_weight.value()))
                else:
                    raise Error("ValueError: component does not support sample weights")
            else:
                _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target))
        self.estimators_ = models^
        self.classes_ = classes^
        self.order_ = order^
        self.n_features_in_ = X.shape[1]
        self.chain_method_ = self.chain_method

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        self._check(X)
        var columns = List[Self.Target]()
        for j in range(len(self.estimators_)):
            var column = self.estimators_[j].predict(rebind[Self.E.Input](X))
            columns.append(rebind_var[Self.Target](column^))
        var result = stack_predictions(columns)
        if result.shape[0] != X.shape[0]:
            raise Error("ValueError: component prediction sample count differs")
        return result^

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        if sample_weight:
            raise Error("ValueError: MultiOutputClassifier.score does not accept sample_weight")
        var prediction = self.predict(X)
        return _score[Self.label, True](y, prediction, sample_weight)

    def predict_proba(self, X: Self.Input) raises -> List[NDArray[Self.dtype]] where conforms_to(Self.E, ProbabilityPredictor):
        self._check(X)
        var result = List[NDArray[Self.dtype]]()
        for j in range(len(self.estimators_)):
            var probability = self.estimators_[j].predict_proba(rebind[Self.E.Input](X))
            var dense = _dense[Self.E.Probabilities, Self.dtype](probability^)
            if dense.ndim != 2 or dense.shape[0] != X.shape[0] or dense.shape[1] != self.classes_[j].shape[0]:
                raise Error("ValueError: probability shape must match samples and classes")
            result.append(dense^)
        return result^

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, classes: Optional[List[Self.Target]] = None) raises -> ref[self] Self where conforms_to(Self.E, IncrementalClassifier):
        comptime assert Self.E.Classes == Self.Target, "component classes must use the wrapper target array type"
        _features(X, self.n_features_in_)
        validate_targets(y, X.shape[0])
        _finite(y)
        if self.n_jobs and self.n_jobs.value() != 1:
            raise Error("ValueError: native execution supports n_jobs=None or 1")
        if self.__sklearn_is_fitted__() and y.shape[1] != len(self.estimators_):
            raise Error("ValueError: partial_fit output count differs from fit")
        if classes and len(classes.value()) != y.shape[1]:
            raise Error("ValueError: supply one class vector per output")
        if not self.__sklearn_is_fitted__() and not classes:
            raise Error("ValueError: first partial_fit requires classes for every output")
        if not self.__sklearn_is_fitted__():
            self.estimators_ = clone_outputs(self.estimator, y.shape[1])
        for j in range(y.shape[1]):
            var column = target_column(y, j)
            var known: Optional[Self.Target] = None
            if classes:
                known = classes.value()[j].copy()
            _ = self.estimators_[j].partial_fit(rebind[Self.E.Input](X), rebind[Self.E.Target](column), rebind[Optional[Self.E.Classes]](known))
        if classes:
            var values = List[Self.Target]()
            for j in range(y.shape[1]):
                values.append(_classes(classes.value()[j]))
            self.classes_ = values^
        self.n_features_in_ = X.shape[1]
        return self

struct RegressorChain[E: SupervisedEstimator & Predictor & Deinitable, dtype: DType = DType.float64, label: DType = dtype](
    MultiOutputRegressorAPI,
    Copyable where conforms_to(E, Copyable),
    WeightedSupervisedEstimator where conforms_to(E, WeightedSupervisedEstimator),
) where conforms_to(E.Params, Deinitable):
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.label]
    comptime Prediction = Self.Target
    comptime SampleWeight = NDArray[DType.float64]
    comptime Params = ChainParams[Self.E.Params]
    var estimator: Self.E
    var n_jobs: Optional[Int]
    var order: Optional[List[Int]]
    var random_state: Optional[Int]
    var random_order: Bool
    var verbose: Bool
    var chain_method: String
    var chain_method_: String
    var estimators_: List[Self.E]
    var classes_: List[NDArray[Self.label]]
    var order_: List[Int]
    var n_features_in_: Optional[Int]

    def __init__(out self, var estimator: Self.E, *, order: Optional[List[Int]] = None, random_state: Optional[Int] = None,
                 random_order: Bool = False, verbose: Bool = False, chain_method: String = "predict") raises:
        comptime assert Self.dtype in (DType.float32, DType.float64), "features must use Float32 or Float64"
        comptime assert Self.E.Input == Self.Input, "component Input must match feature dtype"
        comptime assert Self.E.Target == Self.Target, "component Target must match target dtype"
        comptime assert Self.E.Prediction == Self.Target, "component Prediction must match target dtype"
        self.estimator = estimator^
        self.n_jobs = None
        self.order = order.copy()
        self.random_state = random_state
        self.random_order = random_order
        self.verbose = verbose
        self.chain_method = chain_method
        self.chain_method_ = "predict"
        self.estimators_ = List[Self.E]()
        self.classes_ = List[NDArray[Self.label]]()
        self.order_ = List[Int]()
        self.n_features_in_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.estimator.get_params(deep), self.order.copy(), self.random_state, self.random_order, self.verbose, self.chain_method)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        _ = self.estimator.set_params(params.estimator)
        self.order = params.order.copy()
        self.random_state = params.random_state
        self.random_order = params.random_order
        self.verbose = params.verbose
        self.chain_method = params.chain_method
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(clone(self.estimator), order=self.order.copy(), random_state=self.random_state,
                    random_order=self.random_order, verbose=self.verbose, chain_method=self.chain_method)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.n_features_in_)

    def _check(self, X: Self.Input) raises:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: call fit before inference")
        _features(X, self.n_features_in_)
        _finite(X)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight) raises -> ref[self] Self where conforms_to(Self.E, WeightedSupervisedEstimator):
        self._fit(X, y, sample_weight.copy())
        return self

    def _fit(mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight]) raises:
        _features(X)
        _finite(X)
        validate_targets(y, X.shape[0])
        _finite(y)
        if self.n_jobs and self.n_jobs.value() != 1:
            raise Error("ValueError: native execution supports n_jobs=None or 1")
        if self.chain_method != "predict":
            raise Error("ValueError: this implementation supports chain_method='predict'")
        var order = List[Int]()
        for j in range(y.shape[1]):
            order.append(j)
        if self.order:
            order = self.order.value().copy()
        if self.random_order:
            if self.order:
                raise Error("ValueError: explicit and random order are mutually exclusive")
            var rng = entropy_state()
            if self.random_state:
                rng = RandomState(self.random_state.value())
            var permutation = rng.permutation(y.shape[1])
            for j in range(y.shape[1]):
                order[j] = Int(permutation.unsafe_get(j))
        var checked = ChainOrder(order, y.shape[1])
        var models = clone_outputs(self.estimator, y.shape[1])
        var classes = List[NDArray[Self.label]]()
        var responses = nm.zeros[Self.dtype](Shape(X.shape[0], y.shape[1]))
        for j in range(y.shape[1]):
            var target = target_column(y, order[j])
            var input = augment_features(X, responses, j)
            if sample_weight:
                comptime if conforms_to(Self.E, WeightedSupervisedEstimator):
                    comptime if Self.E.SampleWeight == NDArray[DType.float32]:
                        var narrow = sample_weight.value().astype[DType.float32]()
                        _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target), sample_weight=rebind[Self.E.SampleWeight](narrow))
                    else:
                        comptime assert Self.E.SampleWeight == Self.SampleWeight, "component sample weights must be Float32 or Float64 NDArrays"
                        _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target), sample_weight=rebind[Self.E.SampleWeight](sample_weight.value()))
                else:
                    raise Error("ValueError: component does not support sample weights")
            else:
                _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target))
            _put_column(responses, j, target)
            if self.verbose:
                print("Chain: fitted output", order[j])
        self.estimators_ = models^
        self.classes_ = classes^
        self.order_ = order^
        self.n_features_in_ = X.shape[1]
        self.chain_method_ = self.chain_method

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        self._check(X)
        var responses = nm.zeros[Self.dtype](Shape(X.shape[0], len(self.estimators_)))
        var output = nm.zeros[Self.label](Shape(X.shape[0], len(self.estimators_)))
        for j in range(len(self.estimators_)):
            var input = augment_features(X, responses, j)
            var column = rebind_var[Self.Target](self.estimators_[j].predict(rebind[Self.E.Input](input)))
            _put_column(responses, j, rebind[Self.Target](column))
            _put_column(output, self.order_[j], rebind[Self.Target](column))
        return output^

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        var prediction = self.predict(X)
        return _score[Self.label, False](y, prediction, sample_weight)

    def fit[CV: ChainResponses, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, cv: CV) raises -> ref[self] Self:
        """Use an explicit native out-of-fold provider in place of sklearn's CV object."""
        comptime assert CV.Estimator == Self.E, "CV provider estimator type mismatch"
        comptime assert CV.Input == Self.Input, "CV provider input type mismatch"
        comptime assert CV.Target == Self.Target, "CV provider target type mismatch"
        comptime assert CV.Response == Self.Input, "CV provider returns a feature-dtype vector"
        var candidate = self.__sklearn_clone__()
        _features(X)
        _finite(X)
        validate_targets(y, X.shape[0])
        _finite(y)
        if self.chain_method != "predict":
            raise Error("ValueError: this implementation supports chain_method='predict'")
        if self.n_jobs and self.n_jobs.value() != 1:
            raise Error("ValueError: native execution supports n_jobs=None or 1")
        for j in range(y.shape[1]):
            candidate.order_.append(j)
        if self.order:
            candidate.order_ = self.order.value().copy()
        if self.random_order:
            if self.order:
                raise Error("ValueError: explicit and random order are mutually exclusive")
            var rng = entropy_state()
            if self.random_state:
                rng = RandomState(self.random_state.value())
            var permutation = rng.permutation(y.shape[1])
            for j in range(y.shape[1]):
                candidate.order_[j] = Int(permutation.unsafe_get(j))
        var checked = ChainOrder(candidate.order_, y.shape[1])
        candidate.n_features_in_ = X.shape[1]
        candidate.chain_method_ = self.chain_method
        var responses = nm.zeros[Self.dtype](Shape(X.shape[0], y.shape[1]))
        var models = clone_outputs(self.estimator, y.shape[1])
        for j in range(y.shape[1]):
            var target = target_column(y, candidate.order_[j])
            var input = augment_features(X, responses, j)
            _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target))
            if self.verbose:
                print("Chain: fitted output", candidate.order_[j])
            if j + 1 < y.shape[1]:
                var response = rebind_var[Self.Input](cv.cross_val_response(rebind[CV.Estimator](self.estimator), rebind[CV.Input](input), rebind[CV.Target](target), candidate.chain_method_))
                _put_column(responses, j, response)
        candidate.estimators_ = models^
        self = candidate^
        return self

struct ClassifierChain[E: SupervisedEstimator & Predictor & Deinitable, dtype: DType = DType.float64, label: DType = DType.int64](
    MultiOutputClassifierAPI,
    Copyable where conforms_to(E, Copyable),
    WeightedSupervisedEstimator where conforms_to(E, WeightedSupervisedEstimator),
    MultiOutputProbability where conforms_to(E, ProbabilityPredictor),
) where conforms_to(E.Params, Deinitable):
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.label]
    comptime Prediction = Self.Target
    comptime Probabilities = NDArray[Self.dtype]
    comptime SampleWeight = NDArray[DType.float64]
    comptime Params = ChainParams[Self.E.Params]
    var estimator: Self.E
    var n_jobs: Optional[Int]
    var order: Optional[List[Int]]
    var random_state: Optional[Int]
    var random_order: Bool
    var verbose: Bool
    var chain_method: String
    var chain_method_: String
    var estimators_: List[Self.E]
    var classes_: List[NDArray[Self.label]]
    var order_: List[Int]
    var n_features_in_: Optional[Int]

    def __init__(out self, var estimator: Self.E, *, order: Optional[List[Int]] = None, random_state: Optional[Int] = None,
                 random_order: Bool = False, verbose: Bool = False, chain_method: String = "predict") raises:
        comptime assert Self.dtype in (DType.float32, DType.float64), "features must use Float32 or Float64"
        comptime assert Self.E.Input == Self.Input, "component Input must match feature dtype"
        comptime assert Self.E.Target == Self.Target, "component Target must match target dtype"
        comptime assert Self.E.Prediction == Self.Target, "component Prediction must match target dtype"
        self.estimator = estimator^
        self.n_jobs = None
        self.order = order.copy()
        self.random_state = random_state
        self.random_order = random_order
        self.verbose = verbose
        self.chain_method = chain_method
        self.chain_method_ = "predict"
        self.estimators_ = List[Self.E]()
        self.classes_ = List[NDArray[Self.label]]()
        self.order_ = List[Int]()
        self.n_features_in_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.estimator.get_params(deep), self.order.copy(), self.random_state, self.random_order, self.verbose, self.chain_method)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        _ = self.estimator.set_params(params.estimator)
        self.order = params.order.copy()
        self.random_state = params.random_state
        self.random_order = params.random_order
        self.verbose = params.verbose
        self.chain_method = params.chain_method
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(clone(self.estimator), order=self.order.copy(), random_state=self.random_state,
                    random_order=self.random_order, verbose=self.verbose, chain_method=self.chain_method)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.n_features_in_)

    def _check(self, X: Self.Input) raises:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: call fit before inference")
        _features(X, self.n_features_in_)
        _finite(X)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight) raises -> ref[self] Self where conforms_to(Self.E, WeightedSupervisedEstimator):
        self._fit(X, y, sample_weight.copy())
        return self

    def _fit(mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight]) raises:
        _features(X)
        _finite(X)
        validate_targets(y, X.shape[0])
        _finite(y)
        if self.n_jobs and self.n_jobs.value() != 1:
            raise Error("ValueError: native execution supports n_jobs=None or 1")
        _check_method[Self.E](self.chain_method)
        var order = List[Int]()
        for j in range(y.shape[1]):
            order.append(j)
        if self.order:
            order = self.order.value().copy()
        if self.random_order:
            if self.order:
                raise Error("ValueError: explicit and random order are mutually exclusive")
            var rng = entropy_state()
            if self.random_state:
                rng = RandomState(self.random_state.value())
            var permutation = rng.permutation(y.shape[1])
            for j in range(y.shape[1]):
                order[j] = Int(permutation.unsafe_get(j))
        var checked = ChainOrder(order, y.shape[1])
        var models = clone_outputs(self.estimator, y.shape[1])
        var classes = List[NDArray[Self.label]]()
        var responses = nm.zeros[Self.dtype](Shape(X.shape[0], y.shape[1]))
        for j in range(y.shape[1]):
            var target = target_column(y, order[j])
            classes.append(_classes(target))
            var input = augment_features(X, responses, j)
            if sample_weight:
                comptime if conforms_to(Self.E, WeightedSupervisedEstimator):
                    comptime if Self.E.SampleWeight == NDArray[DType.float32]:
                        var narrow = sample_weight.value().astype[DType.float32]()
                        _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target), sample_weight=rebind[Self.E.SampleWeight](narrow))
                    else:
                        comptime assert Self.E.SampleWeight == Self.SampleWeight, "component sample weights must be Float32 or Float64 NDArrays"
                        _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target), sample_weight=rebind[Self.E.SampleWeight](sample_weight.value()))
                else:
                    raise Error("ValueError: component does not support sample weights")
            else:
                _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target))
            _put_column(responses, j, target)
            if self.verbose:
                print("Chain: fitted output", order[j])
        self.estimators_ = models^
        self.classes_ = classes^
        self.order_ = order^
        self.n_features_in_ = X.shape[1]
        self.chain_method_ = self.chain_method

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        self._check(X)
        var responses = nm.zeros[Self.dtype](Shape(X.shape[0], len(self.estimators_)))
        var output = nm.zeros[Self.label](Shape(X.shape[0], len(self.estimators_)))
        for j in range(len(self.estimators_)):
            var input = augment_features(X, responses, j)
            var column = rebind_var[Self.Target](self.estimators_[j].predict(rebind[Self.E.Input](input)))
            var feature = _response[Self.E, Self.dtype, Self.label](self.estimators_[j], input, self.chain_method_)
            _put_column(responses, j, feature)
            _put_column(output, self.order_[j], rebind[Self.Target](column))
        return output^

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        var prediction = self.predict(X)
        return _score[Self.label, True](y, prediction, sample_weight)

    def predict_proba(self, X: Self.Input) raises -> NDArray[Self.dtype] where conforms_to(Self.E, ProbabilityPredictor):
        self._check(X)
        var responses = nm.zeros[Self.dtype](Shape(X.shape[0], len(self.estimators_)))
        var output = nm.zeros[Self.dtype](Shape(X.shape[0], len(self.estimators_)))
        for j in range(len(self.estimators_)):
            var input = augment_features(X, responses, j)
            var feature = _response[Self.E, Self.dtype, Self.label](self.estimators_[j], input, self.chain_method_)
            _put_column(responses, j, feature)
            var probability = self.estimators_[j].predict_proba(rebind[Self.E.Input](input))
            var dense = _dense[Self.E.Probabilities, Self.dtype](probability^)
            if dense.ndim != 2 or dense.shape[1] != 2:
                raise Error("ValueError: chain probabilities require binary components")
            var positive = target_column(dense, 1)
            _put_column(output, self.order_[j], positive)
        return output^

    def predict_log_proba(self, X: Self.Input) raises -> NDArray[Self.dtype] where conforms_to(Self.E, ProbabilityPredictor):
        var result = self.predict_proba(X)
        for i in range(result.shape[0] * result.shape[1]):
            result.unsafe_set(i, log(Float64(result.unsafe_get(i))).cast[Self.dtype]())
        return result^

    def decision_function(self, X: Self.Input) raises -> NDArray[Self.dtype] where conforms_to(Self.E, DecisionFunction):
        self._check(X)
        var responses = nm.zeros[Self.dtype](Shape(X.shape[0], len(self.estimators_)))
        var result = nm.zeros[Self.dtype](Shape(X.shape[0], len(self.estimators_)))
        for j in range(len(self.estimators_)):
            var input = augment_features(X, responses, j)
            var feature = _response[Self.E, Self.dtype, Self.label](self.estimators_[j], input, self.chain_method_)
            var output = _response[Self.E, Self.dtype, Self.label](self.estimators_[j], input, "decision_function")
            _put_column(responses, j, feature)
            _put_column(result, self.order_[j], output)
        return result^

    def fit[CV: ChainResponses, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, cv: CV) raises -> ref[self] Self:
        """Use an explicit native out-of-fold provider in place of sklearn's CV object."""
        comptime assert CV.Estimator == Self.E, "CV provider estimator type mismatch"
        comptime assert CV.Input == Self.Input, "CV provider input type mismatch"
        comptime assert CV.Target == Self.Target, "CV provider target type mismatch"
        comptime assert CV.Response == Self.Input, "CV provider returns a feature-dtype vector"
        var candidate = self.__sklearn_clone__()
        _features(X)
        _finite(X)
        validate_targets(y, X.shape[0])
        _finite(y)
        _check_method[Self.E](self.chain_method)
        if self.n_jobs and self.n_jobs.value() != 1:
            raise Error("ValueError: native execution supports n_jobs=None or 1")
        for j in range(y.shape[1]):
            candidate.order_.append(j)
        if self.order:
            candidate.order_ = self.order.value().copy()
        if self.random_order:
            if self.order:
                raise Error("ValueError: explicit and random order are mutually exclusive")
            var rng = entropy_state()
            if self.random_state:
                rng = RandomState(self.random_state.value())
            var permutation = rng.permutation(y.shape[1])
            for j in range(y.shape[1]):
                candidate.order_[j] = Int(permutation.unsafe_get(j))
        var checked = ChainOrder(candidate.order_, y.shape[1])
        candidate.n_features_in_ = X.shape[1]
        candidate.chain_method_ = self.chain_method
        var responses = nm.zeros[Self.dtype](Shape(X.shape[0], y.shape[1]))
        var models = clone_outputs(self.estimator, y.shape[1])
        for j in range(y.shape[1]):
            var target = target_column(y, candidate.order_[j])
            candidate.classes_.append(_classes(target))
            var input = augment_features(X, responses, j)
            _ = models[j].fit(rebind[Self.E.Input](input), rebind[Self.E.Target](target))
            if self.verbose:
                print("Chain: fitted output", candidate.order_[j])
            if j + 1 < y.shape[1]:
                var response = rebind_var[Self.Input](cv.cross_val_response(rebind[CV.Estimator](self.estimator), rebind[CV.Input](input), rebind[CV.Target](target), candidate.chain_method_))
                _put_column(responses, j, response)
        candidate.estimators_ = models^
        self = candidate^
        return self
