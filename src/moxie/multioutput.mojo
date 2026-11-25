# Adapted from sklearn/multioutput.py; BSD-3-Clause, see LICENSES/scikit-learn.txt.
"""Dense multioutput estimators with independently cloned component models."""
import numojo as nm
from numojo import NDArray, Shape
from std.math import isfinite, floor, nan, log
from std.time import perf_counter_ns
from std.utils import Variant
from moxie.base import (BaseEstimator, SupervisedEstimator, Predictor, ProbabilityPredictor,
    DecisionFunction, LogProbabilityPredictor, IncrementalClassifier, IncrementalRegressor,
    WeightedSupervisedEstimator, ClassifierMixin, clone)
from moxie._labels import encode_column
from moxie.dummy import Label
from moxie._multioutput_contracts import MultiOutputClassifierAPI, MultiOutputRegressorAPI, MultiOutputProbability
from moxie._multioutput import validate_targets, clone_outputs, target_column, stack_predictions, augment_features, ChainOrder, ChainResponses
from moxie._random import RandomState, entropy_state
from moxie.model_selection import IndexSplit, KFold, StratifiedKFold
from moxie.model_selection._indices import IndexArray, SplitCursor
from moxie.model_selection._split import SplitLabels, check_cv, _rows
from moxie.model_selection._validation import (
    _align_class_response, _checked_prediction_splits, _scatter_prediction,
)


@fieldwise_init
struct IndependentParams[P: Copyable & Deinitable](Copyable, Movable):
    var estimator: Self.P
    var n_jobs: Optional[Int]


struct ChainFolds(Copyable, Movable):
    """An owned, reusable snapshot of fixed train/test splits."""

    var _splits: List[IndexSplit]

    def __init__(out self, splits: List[IndexSplit]):
        self._splits = List[IndexSplit](capacity=len(splits))
        for i in range(len(splits)):
            ref split = splits[i]
            self._splits.append(IndexSplit(split.train.copy(), split.test.copy()))

    def __init__[C: SplitCursor & Deinitable](out self, var cursor: C) raises:
        self._splits = _materialize_chain_folds(cursor^)

    def __init__(out self, *, copy: Self):
        self = Self(copy._splits)


comptime ChainCV = Variant[type_of(None), Int, KFold, StratifiedKFold, ChainFolds]


trait ChainSplitter(Copyable):
    """Generate a fresh cursor for the current augmented input and target column."""

    comptime Input: Movable
    comptime Target: Movable
    comptime Splits: SplitCursor & Deinitable

    def split(self, X: Self.Input, y: Self.Target) raises -> Self.Splits:
        ...


struct ChainParams[
    P: Copyable & Deinitable, CV: Copyable & Deinitable = ChainCV,
](Copyable, Movable):
    var estimator: Self.P
    var order: Optional[List[Int]]
    var random_state: Optional[Int]
    var random_order: Bool
    var verbose: Bool
    var chain_method: String
    var cv: Self.CV

    def __init__(
        out self, var estimator: Self.P, var order: Optional[List[Int]], random_state: Optional[Int],
        random_order: Bool, verbose: Bool, var chain_method: String,
        var cv: Self.CV = rebind_var[Self.CV](ChainCV(None)),
    ):
        self.estimator = estimator^
        self.order = order^
        self.random_state = random_state
        self.random_order = random_order
        self.verbose = verbose
        self.chain_method = chain_method^
        self.cv = cv^


def _chain_message(index: Int, total: Int, column: Int, nanoseconds: Int) -> String:
    """sklearn `_message_with_time("Chain", "(i of n) Processing order k", t)`."""
    var seconds = Float64(nanoseconds) / 1e9
    var time_text: String
    if seconds > 60:
        time_text = _fixed(seconds / 60, 4) + "min"
    else:
        time_text = " " + _fixed(seconds, 5) + "s"
    var start = String("[Chain] ")
    var end = String(" (", index, " of ", total, ") Processing order ", column, ", total=", time_text)
    var out = start
    for _ in range(70 - start.byte_length() - end.byte_length()):
        out += "."
    return out + end


def _fixed(value: Float64, width: Int) -> String:
    """`'%{width}.1f' % value` for nonnegative values."""
    var tenths = Int(value * 10 + 0.5)
    var text = String(tenths // 10, ".", tenths % 10)
    while text.byte_length() < width:
        text = " " + text
    return text^


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


def _probability_matrix[T: Movable, dtype: DType](var values: T) raises -> NDArray[dtype]:
    """Unwrap the sole output from components with a list-valued probability API."""
    comptime if T == List[NDArray[DType.float64]]:
        var outputs = rebind_var[List[NDArray[DType.float64]]](values^)
        if len(outputs) != 1:
            raise Error("ValueError: a single-target component must return one probability matrix")
        return _dense[NDArray[DType.float64], dtype](outputs.pop())
    else:
        return _dense[T, dtype](values^)


def _list_target[T: Copyable & Deinitable, label: DType](target: NDArray[label]) raises -> List[T]:
    if target.ndim != 1:
        raise Error("ValueError: component target must be a vector")
    comptime if T == Scalar[label]:
        return rebind_var[List[T]](target.tolist())
    elif T == Label:
        var values = List[Label](capacity=target.size)
        for i in range(target.size):
            comptime if label == DType.int64:
                values.append(Label(Int(target.item(i))))
            else:
                comptime assert label in (DType.float32, DType.float64)
                values.append(Label(Float64(target.item(i))))
        return rebind_var[List[T]](values^)
    else:
        comptime assert label == DType.int64 and T == Int
        var values = List[Int](capacity=target.size)
        for i in range(target.size):
            values.append(Int(target.item(i)))
        return rebind_var[List[T]](values^)


def _prediction_vector[T: Movable, label: DType](var values: T) raises -> NDArray[label]:
    comptime if T == NDArray[label]:
        return rebind_var[NDArray[label]](values^)
    elif T == List[Scalar[label]]:
        var labels = rebind_var[List[Scalar[label]]](values^)
        return nm.array[label](labels, [len(labels)])
    elif T == List[Label]:
        var labels = rebind_var[List[Label]](values^)
        var result = NDArray[label](Shape(len(labels)))
        for i in range(len(labels)):
            ref value = labels[i]
            comptime if label == DType.int64:
                if value.kind != 0:
                    raise Error("ValueError: component must predict integer labels")
                result.unsafe_set(i, Scalar[label](value.integer))
            else:
                comptime assert label in (DType.float32, DType.float64)
                if value.kind != 1:
                    raise Error("ValueError: component must predict floating-point labels")
                result.unsafe_set(i, Scalar[label](value.number))
        return result^
    else:
        comptime assert label == DType.int64 and T == List[Int]
        var labels = rebind_var[List[Int]](values^)
        var result = NDArray[label](Shape(len(labels)))
        for i in range(len(labels)):
            result.unsafe_set(i, Scalar[label](labels[i]))
        return result^


def _fit_component[E: SupervisedEstimator, dtype: DType, label: DType](
    mut model: E, X: NDArray[dtype], target: NDArray[label],
    weights: Optional[NDArray[DType.float64]] = None,
) raises:
    comptime if conforms_to(E, WeightedSupervisedEstimator):
        comptime if E.SampleWeight == List[Float64]:
            if weights:
                ref sample_weight = weights.value()
                if sample_weight.ndim != 1 or sample_weight.shape[0] != X.shape[0]:
                    raise Error("ValueError: sample weights must be a vector with one value per sample")
    comptime if E.Target == NDArray[label]:
        _fit_native_component(model, rebind[E.Input](X), rebind[E.Target](target), weights)
    else:
        comptime NativeLabel = (
            Label if E.Target == List[Label] else Int if E.Target == List[Int] else Scalar[label]
        )
        var native_target = _list_target[NativeLabel, label](target)
        _fit_native_component(
            model, rebind[E.Input](X), rebind[E.Target](native_target), weights,
        )


def _fit_native_component[E: SupervisedEstimator](
    mut model: E, X: E.Input, target: E.Target, weights: Optional[NDArray[DType.float64]],
) raises:
    if weights:
        ref sample_weight = weights.value()
        comptime if conforms_to(E, WeightedSupervisedEstimator):
            comptime if E.SampleWeight == List[Float64]:
                var native_weights = sample_weight.tolist()
                _ = model.fit(X, target, sample_weight=rebind[E.SampleWeight](native_weights))
            elif E.SampleWeight == NDArray[DType.float32]:
                var narrow = sample_weight.astype[DType.float32]()
                _ = model.fit(X, target, sample_weight=rebind[E.SampleWeight](narrow))
            else:
                comptime assert E.SampleWeight == NDArray[DType.float64], (
                    "classifier component weights must be Float32 or Float64 NDArrays or List[Float64]"
                )
                _ = model.fit(X, target, sample_weight=rebind[E.SampleWeight](sample_weight))
        else:
            raise Error("ValueError: component does not support sample weights")
    else:
        _ = model.fit(X, target)


def _predict_response[E: Predictor, dtype: DType, label: DType, out: DType](
    estimator: E, X: NDArray[dtype],
) raises -> NDArray[out]:
    var raw = _prediction_vector[E.Prediction, label](estimator.predict(rebind[E.Input](X)))
    var result = nm.zeros[out](Shape(X.shape[0]))
    if raw.ndim != 1 or raw.shape[0] != X.shape[0]:
        raise Error("ValueError: component prediction must match samples")
    for i in range(X.shape[0]):
        result.unsafe_set(i, raw.unsafe_get(i * Int(raw.strides[0])).cast[out]())
    return result^


def _response[E: Predictor, dtype: DType, label: DType, out: DType = dtype](estimator: E, X: NDArray[dtype], method: String) raises -> NDArray[out]:
    """One chain link's response vector in dtype `out` (positive-class column for probabilities)."""
    if method == "predict":
        return _predict_response[E, dtype, label, out](estimator, X)
    if method == "decision_function":
        comptime if conforms_to(E, DecisionFunction):
            return _dense[E.Decision, out](estimator.decision_function(rebind[E.Input](X)))
    if method == "predict_proba":
        comptime if conforms_to(E, ProbabilityPredictor):
            var probability = _probability_matrix[E.Probabilities, out](estimator.predict_proba(rebind[E.Input](X)))
            if probability.ndim != 2 or probability.shape[1] != 2:
                raise Error("ValueError: chain response requires two probability columns")
            return target_column(probability, 1)
    if method == "predict_log_proba":
        comptime if conforms_to(E, LogProbabilityPredictor):
            var probability = _probability_matrix[E.Probabilities, out](estimator.predict_log_proba(rebind[E.Input](X)))
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


def _materialize_chain_folds[C: SplitCursor & Deinitable](var cursor: C) raises -> List[IndexSplit]:
    var splits = List[IndexSplit]()
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.take())
    return splits^


def _aligned_chain_response[T: Movable, out: DType](
    var values: T, classes: IndexArray, n_classes: Int, method: String,
) raises -> NDArray[out]:
    comptime raw_dtype = DType.float32 if T == NDArray[DType.float32] else DType.float64
    var raw = _probability_matrix[T, raw_dtype](values^)
    var aligned = _align_class_response(classes, raw, n_classes, method)
    if aligned.ndim == 2:
        aligned = target_column(aligned, 1)
    return _dense[type_of(aligned), out](aligned^)


def _chain_fold_response[E: Predictor, dtype: DType, label: DType](
    model: E, X: NDArray[dtype], method: String, fold_classes: IndexArray, n_classes: Int,
) raises -> NDArray[dtype]:
    if method == "predict":
        return _predict_response[E, dtype, label, dtype](model, X)
    if method == "predict_proba":
        comptime if conforms_to(E, ProbabilityPredictor):
            comptime if (
                E.Probabilities == NDArray[DType.float32]
                or E.Probabilities == NDArray[DType.float64]
                or E.Probabilities == List[NDArray[DType.float64]]
            ):
                return _aligned_chain_response[E.Probabilities, dtype](
                    model.predict_proba(rebind[E.Input](X)), fold_classes, n_classes, method,
                )
    if method == "predict_log_proba":
        comptime if conforms_to(E, LogProbabilityPredictor):
            comptime if (
                E.Probabilities == NDArray[DType.float32]
                or E.Probabilities == NDArray[DType.float64]
                or E.Probabilities == List[NDArray[DType.float64]]
            ):
                return _aligned_chain_response[E.Probabilities, dtype](
                    model.predict_log_proba(rebind[E.Input](X)), fold_classes, n_classes, method,
                )
    if method == "decision_function":
        comptime if conforms_to(E, DecisionFunction):
            comptime if E.Decision == NDArray[DType.float32] or E.Decision == NDArray[DType.float64]:
                return _aligned_chain_response[E.Decision, dtype](
                    model.decision_function(rebind[E.Input](X)), fold_classes, n_classes, method,
                )
    raise Error("ValueError: component lacks the requested Float32 or Float64 CV response")


struct _SplitterResponses[
    E: SupervisedEstimator & Predictor & Deinitable,
    dtype: DType,
    label: DType,
    CV: Copyable & Deinitable,
](ChainResponses):
    comptime Estimator = Self.E
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.label]
    comptime Response = Self.Input

    var cv: Self.CV

    def __init__(out self, var cv: Self.CV) raises:
        comptime if Self.CV == ChainCV:
            ref native_cv = rebind[ChainCV](cv)
            if native_cv.isa[Int]() and native_cv[Int] < 2:
                raise Error("ValueError: n_splits must be at least 2")
        self.cv = cv^

    def _splits(self, cv: ChainCV, y: Self.Target) raises -> List[IndexSplit]:
        if cv.isa[KFold]():
            return _materialize_chain_folds(cv[KFold].split(y.size))
        if cv.isa[StratifiedKFold]():
            return _materialize_chain_folds(
                cv[StratifiedKFold].split(y.size, SplitLabels._numeric(y)),
            )
        comptime if conforms_to(Self.E, ClassifierMixin):
            var labels = SplitLabels._numeric(y)
            var splitter = check_cv(Optional[Int](cv[Int]), labels, classifier=True)
            return _materialize_chain_folds(splitter.split(y.size, labels))
        else:
            return _materialize_chain_folds(KFold(cv[Int]).split(y.size))

    def cross_val_response(self, estimator: Self.E, X: Self.Input, y: Self.Target, method: String) raises -> Self.Response:
        comptime if Self.label in (DType.int64, DType.float32, DType.float64):
            comptime if conforms_to(Self.CV, ChainSplitter):
                comptime assert Self.CV.Input == Self.Input, "splitter input must match chain input"
                comptime assert Self.CV.Target == Self.Target, "splitter target must match a chain target column"
                var splits = _materialize_chain_folds(self.cv.split(
                    rebind[Self.CV.Input](X), rebind[Self.CV.Target](y),
                ))
                return self._fit_folds(estimator, X, y, method, splits)
            else:
                ref cv = rebind[ChainCV](self.cv)
                if cv.isa[ChainFolds]():
                    return self._fit_folds(estimator, X, y, method, cv[ChainFolds]._splits)
                return self._fit_folds(estimator, X, y, method, self._splits(cv, y))
        else:
            raise Error("ValueError: native chain CV targets must use Int64, Float32 or Float64")

    def _fit_folds(
        self, estimator: Self.E, X: Self.Input, y: Self.Target, method: String,
        splits: List[IndexSplit],
    ) raises -> Self.Response:
        _checked_prediction_splits(splits, y.size)
        var codes = IndexArray(Shape(0))
        var n_classes = 0
        if method != "predict":
            var domain = encode_column(y)
            codes = domain.index.inverse.copy()
            n_classes = domain.classes.size
            if Int(Scalar[Self.label](n_classes - 1)) != n_classes - 1:
                raise Error("ValueError: global class codes do not fit the component target dtype")
        var output: Optional[Self.Response] = None
        for fold in range(len(splits)):
            ref split = splits[fold]
            var train_X = _rows(X, split.train)
            var test_X = _rows(X, split.test)
            var fold_classes = IndexArray(Shape(0))
            var train_y: Self.Target
            if method == "predict":
                train_y = _rows(y, split.train)
            else:
                var train_codes = _rows(codes, split.train)
                fold_classes = encode_column(train_codes).classes.copy()
                train_y = train_codes.astype[Self.label]()
            var model = clone(estimator)
            _fit_component(model, train_X, train_y)
            var predicted = _chain_fold_response[Self.E, Self.dtype, Self.label](
                model, test_X, method, fold_classes, n_classes,
            )
            _scatter_prediction(output, predicted, split.test, y.size)
        return output.take()


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
):
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
        if self.n_jobs and self.n_jobs.value() == 0:
            raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
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
        if self.n_jobs and self.n_jobs.value() == 0:
            raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
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
):
    comptime Classes = List[NDArray[Self.label]]
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.label]
    comptime Prediction = Self.Target
    comptime Probabilities = List[NDArray[DType.float64]]
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
        comptime assert (
            Self.E.Target == Self.Target or Self.E.Target == List[Scalar[Self.label]]
            or (Self.label == DType.int64 and Self.E.Target == List[Int])
            or (Self.E.Target == List[Label] and Self.label in (DType.int64, DType.float32, DType.float64))
        ), "component target must use the wrapper label dtype as an NDArray or a compatible List"
        comptime assert Self.E.Prediction == Self.E.Target, "component prediction and target types must match"
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
        if self.n_jobs and self.n_jobs.value() == 0:
            raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
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
            _fit_component(models[j], input, target, sample_weight)
        self.estimators_ = models^
        self.classes_ = classes^
        self.order_ = order^
        self.n_features_in_ = X.shape[1]
        self.chain_method_ = self.chain_method

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        self._check(X)
        var columns = List[Self.Target]()
        for j in range(len(self.estimators_)):
            columns.append(_prediction_vector[Self.E.Prediction, Self.label](
                self.estimators_[j].predict(rebind[Self.E.Input](X)),
            ))
        var result = stack_predictions(columns)
        if result.shape[0] != X.shape[0]:
            raise Error("ValueError: component prediction sample count differs")
        return result^

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        if sample_weight:
            raise Error("ValueError: MultiOutputClassifier.score does not accept sample_weight")
        var prediction = self.predict(X)
        return _score[Self.label, True](y, prediction, sample_weight)

    def predict_proba(self, X: Self.Input) raises -> List[NDArray[DType.float64]] where conforms_to(Self.E, ProbabilityPredictor):
        """Float64 per-output probabilities: component values are kept exactly."""
        self._check(X)
        var result = List[NDArray[DType.float64]]()
        for j in range(len(self.estimators_)):
            var probability = self.estimators_[j].predict_proba(rebind[Self.E.Input](X))
            var dense = _probability_matrix[Self.E.Probabilities, DType.float64](probability^)
            if dense.ndim != 2 or dense.shape[0] != X.shape[0] or dense.shape[1] != self.classes_[j].shape[0]:
                raise Error("ValueError: probability shape must match samples and classes")
            result.append(dense^)
        return result^

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, classes: Optional[List[Self.Target]] = None) raises -> ref[self] Self where conforms_to(Self.E, IncrementalClassifier):
        comptime assert Self.E.Classes == Self.E.Target, "component classes and target types must match"
        _features(X, self.n_features_in_)
        validate_targets(y, X.shape[0])
        _finite(y)
        if self.n_jobs and self.n_jobs.value() == 0:
            raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
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
            comptime if Self.E.Target == Self.Target:
                var known: Optional[Self.Target] = None
                if classes:
                    known = classes.value()[j].copy()
                _ = self.estimators_[j].partial_fit(
                    rebind[Self.E.Input](X), rebind[Self.E.Target](column),
                    rebind[Optional[Self.E.Classes]](known),
                )
            else:
                comptime NativeLabel = (
                    Label if Self.E.Target == List[Label] else Int if Self.E.Target == List[Int] else Scalar[Self.label]
                )
                var known: Optional[List[NativeLabel]] = None
                if classes:
                    known = _list_target[NativeLabel, Self.label](classes.value()[j])
                var native_column = _list_target[NativeLabel, Self.label](column)
                _ = self.estimators_[j].partial_fit(
                    rebind[Self.E.Input](X), rebind[Self.E.Target](native_column),
                    rebind[Optional[Self.E.Classes]](known),
                )
        if classes:
            var values = List[Self.Target]()
            for j in range(y.shape[1]):
                values.append(_classes(classes.value()[j]))
            self.classes_ = values^
        self.n_features_in_ = X.shape[1]
        return self

struct RegressorChain[
    E: SupervisedEstimator & Predictor & Deinitable,
    dtype: DType = DType.float64,
    label: DType = dtype,
    Splitter: Copyable & Deinitable = ChainCV,
](
    MultiOutputRegressorAPI,
    Copyable where conforms_to(E, Copyable),
    WeightedSupervisedEstimator where conforms_to(E, WeightedSupervisedEstimator),
):
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.label]
    comptime Prediction = Self.Target
    comptime SampleWeight = NDArray[DType.float64]
    comptime Params = ChainParams[Self.E.Params, Self.Splitter]
    var estimator: Self.E
    var n_jobs: Optional[Int]
    var order: Optional[List[Int]]
    var random_state: Optional[Int]
    var random_order: Bool
    var verbose: Bool
    var chain_method: String
    var cv: Self.Splitter
    var chain_method_: String
    var estimators_: List[Self.E]
    var classes_: List[NDArray[Self.label]]
    var order_: List[Int]
    var n_features_in_: Optional[Int]

    def __init__(out self, var estimator: Self.E, *, order: Optional[List[Int]] = None, random_state: Optional[Int] = None,
                 random_order: Bool = False, verbose: Bool = False, chain_method: String = "predict",
                 cv: ChainCV = None) raises where Self.Splitter == ChainCV:
        self = Self(estimator^, rebind[Self.Splitter](cv), order, random_state, random_order, verbose, chain_method)

    def __init__(
        out self, var estimator: Self.E, *, cv: Self.Splitter,
        order: Optional[List[Int]] = None, random_state: Optional[Int] = None,
        random_order: Bool = False, verbose: Bool = False, chain_method: String = "predict",
    ) raises where conforms_to(Self.Splitter, ChainSplitter):
        self = Self(estimator^, cv, order, random_state, random_order, verbose, chain_method)

    def __init__(
        out self, var estimator: Self.E, _cv: Self.Splitter,
        order: Optional[List[Int]], random_state: Optional[Int], random_order: Bool,
        verbose: Bool, chain_method: String,
    ):
        comptime assert Self.Splitter == ChainCV or conforms_to(Self.Splitter, ChainSplitter)
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
        self.cv = _cv.copy()
        self.chain_method_ = "predict"
        self.estimators_ = List[Self.E]()
        self.classes_ = List[NDArray[Self.label]]()
        self.order_ = List[Int]()
        self.n_features_in_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.estimator.get_params(deep), self.order.copy(), self.random_state, self.random_order, self.verbose, self.chain_method, self.cv.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        _ = self.estimator.set_params(params.estimator)
        self.order = params.order.copy()
        self.random_state = params.random_state
        self.random_order = params.random_order
        self.verbose = params.verbose
        self.chain_method = params.chain_method
        self.cv = params.cv.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            clone(self.estimator), self.cv, self.order, self.random_state,
            self.random_order, self.verbose, self.chain_method,
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.n_features_in_)

    def _check(self, X: Self.Input) raises:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: call fit before inference")
        _features(X, self.n_features_in_)
        _finite(X)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self._fit_native_cv(X, y, None, self.cv.copy())
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight) raises -> ref[self] Self where conforms_to(Self.E, WeightedSupervisedEstimator):
        self._fit_native_cv(X, y, sample_weight.copy(), self.cv.copy())
        return self

    def _fit_native_cv[C: Copyable & Deinitable](
        mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight], var cv: C,
    ) raises:
        var use_cv = True
        comptime if C == ChainCV:
            use_cv = not rebind[ChainCV](cv).isa[type_of(None)]()
        var provider = _SplitterResponses[Self.E, Self.dtype, Self.label, C](cv^)
        self._fit(X, y, sample_weight, provider, use_cv=use_cv)

    def _fit[CV: ChainResponses](
        mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight], cv: CV,
        *, use_cv: Bool = True,
    ) raises:
        comptime assert CV.Estimator == Self.E, "CV provider estimator type mismatch"
        comptime assert CV.Input == Self.Input, "CV provider input type mismatch"
        comptime assert CV.Target == Self.Target, "CV provider target type mismatch"
        comptime assert CV.Response == Self.Input, "CV provider returns a feature-dtype vector"
        _features(X)
        _finite(X)
        validate_targets(y, X.shape[0])
        _finite(y)
        if self.n_jobs and self.n_jobs.value() == 0:
            raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
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
            var started = perf_counter_ns()
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
            if self.verbose:
                print(_chain_message(j + 1, y.shape[1], order[j], perf_counter_ns() - started))
            if j + 1 < y.shape[1]:
                if use_cv:
                    var response = rebind_var[Self.Input](cv.cross_val_response(
                        rebind[CV.Estimator](self.estimator), rebind[CV.Input](input),
                        rebind[CV.Target](target), self.chain_method,
                    ))
                    _put_column(responses, j, response)
                else:
                    _put_column(responses, j, target)
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

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, cv: ChainCV,
    ) raises -> ref[self] Self:
        """Override the configured splitter for this fit."""
        self._fit_native_cv(X, y, None, cv.copy())
        return self

    def fit[CV: ChainResponses, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, cv: CV,
    ) raises -> ref[self] Self:
        """Use a custom out-of-fold provider for this fit."""
        self._fit(X, y, None, cv)
        return self

struct ClassifierChain[
    E: SupervisedEstimator & Predictor & Deinitable,
    dtype: DType = DType.float64,
    label: DType = DType.int64,
    Splitter: Copyable & Deinitable = ChainCV,
](
    MultiOutputClassifierAPI,
    Copyable where conforms_to(E, Copyable),
    WeightedSupervisedEstimator where conforms_to(E, WeightedSupervisedEstimator),
    MultiOutputProbability where conforms_to(E, ProbabilityPredictor),
):
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.label]
    comptime Prediction = Self.Target
    comptime Probabilities = NDArray[DType.float64]
    comptime SampleWeight = NDArray[DType.float64]
    comptime Params = ChainParams[Self.E.Params, Self.Splitter]
    var estimator: Self.E
    var n_jobs: Optional[Int]
    var order: Optional[List[Int]]
    var random_state: Optional[Int]
    var random_order: Bool
    var verbose: Bool
    var chain_method: String
    var cv: Self.Splitter
    var chain_method_: String
    var estimators_: List[Self.E]
    var classes_: List[NDArray[Self.label]]
    var order_: List[Int]
    var n_features_in_: Optional[Int]

    def __init__(out self, var estimator: Self.E, *, order: Optional[List[Int]] = None, random_state: Optional[Int] = None,
                 random_order: Bool = False, verbose: Bool = False, chain_method: String = "predict",
                 cv: ChainCV = None) raises where Self.Splitter == ChainCV:
        self = Self(estimator^, rebind[Self.Splitter](cv), order, random_state, random_order, verbose, chain_method)

    def __init__(
        out self, var estimator: Self.E, *, cv: Self.Splitter,
        order: Optional[List[Int]] = None, random_state: Optional[Int] = None,
        random_order: Bool = False, verbose: Bool = False, chain_method: String = "predict",
    ) raises where conforms_to(Self.Splitter, ChainSplitter):
        self = Self(estimator^, cv, order, random_state, random_order, verbose, chain_method)

    def __init__(
        out self, var estimator: Self.E, _cv: Self.Splitter,
        order: Optional[List[Int]], random_state: Optional[Int], random_order: Bool,
        verbose: Bool, chain_method: String,
    ):
        comptime assert Self.Splitter == ChainCV or conforms_to(Self.Splitter, ChainSplitter)
        comptime assert Self.dtype in (DType.float32, DType.float64), "features must use Float32 or Float64"
        comptime assert Self.E.Input == Self.Input, "component Input must match feature dtype"
        comptime assert (
            Self.E.Target == Self.Target or Self.E.Target == List[Scalar[Self.label]]
            or (Self.label == DType.int64 and Self.E.Target == List[Int])
            or (Self.E.Target == List[Label] and Self.label in (DType.int64, DType.float32, DType.float64))
        ), "component target must use the wrapper label dtype as an NDArray or a compatible List"
        comptime assert Self.E.Prediction == Self.E.Target, "component prediction and target types must match"
        self.estimator = estimator^
        self.n_jobs = None
        self.order = order.copy()
        self.random_state = random_state
        self.random_order = random_order
        self.verbose = verbose
        self.chain_method = chain_method
        self.cv = _cv.copy()
        self.chain_method_ = "predict"
        self.estimators_ = List[Self.E]()
        self.classes_ = List[NDArray[Self.label]]()
        self.order_ = List[Int]()
        self.n_features_in_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.estimator.get_params(deep), self.order.copy(), self.random_state, self.random_order, self.verbose, self.chain_method, self.cv.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        _ = self.estimator.set_params(params.estimator)
        self.order = params.order.copy()
        self.random_state = params.random_state
        self.random_order = params.random_order
        self.verbose = params.verbose
        self.chain_method = params.chain_method
        self.cv = params.cv.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            clone(self.estimator), self.cv, self.order, self.random_state,
            self.random_order, self.verbose, self.chain_method,
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.n_features_in_)

    def _check(self, X: Self.Input) raises:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: call fit before inference")
        _features(X, self.n_features_in_)
        _finite(X)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self._fit_native_cv(X, y, None, self.cv.copy())
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, ref[sample_weight_origin] sample_weight: Self.SampleWeight) raises -> ref[self] Self where conforms_to(Self.E, WeightedSupervisedEstimator):
        self._fit_native_cv(X, y, sample_weight.copy(), self.cv.copy())
        return self

    def _fit_native_cv[C: Copyable & Deinitable](
        mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight], var cv: C,
    ) raises:
        var use_cv = True
        comptime if C == ChainCV:
            use_cv = not rebind[ChainCV](cv).isa[type_of(None)]()
        var provider = _SplitterResponses[Self.E, Self.dtype, Self.label, C](cv^)
        self._fit(X, y, sample_weight, provider, use_cv=use_cv)

    def _fit[CV: ChainResponses](
        mut self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight], cv: CV,
        *, use_cv: Bool = True,
    ) raises:
        comptime assert CV.Estimator == Self.E, "CV provider estimator type mismatch"
        comptime assert CV.Input == Self.Input, "CV provider input type mismatch"
        comptime assert CV.Target == Self.Target, "CV provider target type mismatch"
        comptime assert CV.Response == Self.Input, "CV provider returns a feature-dtype vector"
        _features(X)
        _finite(X)
        validate_targets(y, X.shape[0])
        _finite(y)
        if self.n_jobs and self.n_jobs.value() == 0:
            raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
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
            var started = perf_counter_ns()
            _fit_component(models[j], input, target, sample_weight)
            if self.verbose:
                print(_chain_message(j + 1, y.shape[1], order[j], perf_counter_ns() - started))
            if j + 1 < y.shape[1]:
                if use_cv:
                    var response = rebind_var[Self.Input](cv.cross_val_response(
                        rebind[CV.Estimator](self.estimator), rebind[CV.Input](input),
                        rebind[CV.Target](target), self.chain_method,
                    ))
                    _put_column(responses, j, response)
                else:
                    _put_column(responses, j, target)
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
            var column = _prediction_vector[Self.E.Prediction, Self.label](
                self.estimators_[j].predict(rebind[Self.E.Input](input)),
            )
            var feature = _response[Self.E, Self.dtype, Self.label](self.estimators_[j], input, self.chain_method_)
            _put_column(responses, j, feature)
            _put_column(output, self.order_[j], column)
        return output^

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        var prediction = self.predict(X)
        return _score[Self.label, True](y, prediction, sample_weight)

    def _outputs(self, X: Self.Input, method: String) raises -> NDArray[DType.float64]:
        """sklearn `_get_predictions`: Float64 outputs in original target order."""
        self._check(X)
        var responses = nm.zeros[Self.dtype](Shape(X.shape[0], len(self.estimators_)))
        var result = nm.zeros[DType.float64](Shape(X.shape[0], len(self.estimators_)))
        for j in range(len(self.estimators_)):
            var input = augment_features(X, responses, j)
            var feature = _response[Self.E, Self.dtype, Self.label](self.estimators_[j], input, self.chain_method_)
            var output = _response[Self.E, Self.dtype, Self.label, DType.float64](self.estimators_[j], input, method)
            _put_column(responses, j, feature)
            _put_column(result, self.order_[j], output)
        return result^

    def predict_proba(self, X: Self.Input) raises -> NDArray[DType.float64] where conforms_to(Self.E, ProbabilityPredictor):
        return self._outputs(X, "predict_proba")

    def predict_log_proba(self, X: Self.Input) raises -> NDArray[DType.float64] where conforms_to(Self.E, ProbabilityPredictor):
        var result = self.predict_proba(X)
        for i in range(result.shape[0] * result.shape[1]):
            result.unsafe_set(i, log(result.unsafe_get(i)))
        return result^

    def decision_function(self, X: Self.Input) raises -> NDArray[DType.float64] where conforms_to(Self.E, DecisionFunction):
        return self._outputs(X, "decision_function")

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, cv: ChainCV,
    ) raises -> ref[self] Self:
        """Override the configured splitter for this fit."""
        self._fit_native_cv(X, y, None, cv.copy())
        return self

    def fit[CV: ChainResponses, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *, cv: CV,
    ) raises -> ref[self] Self:
        """Use a custom out-of-fold provider for this fit."""
        self._fit(X, y, None, cv)
        return self
