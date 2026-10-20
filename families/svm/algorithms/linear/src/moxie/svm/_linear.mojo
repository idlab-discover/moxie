# Adapted from scikit-learn; BSD-3-Clause, see SCIKIT_LEARN_LICENSE.
"""Dense linear SVM presentation over the family LIBLINEAR owner."""
import numojo as nm
from std.math import isfinite, floor
from moxie.svm._contracts import Dense, LinearModel, LinearOptions, validate_matrix, validate_vector
from moxie.svm._liblinear import LibLinear
from moxie.svm._base import SupportVectorClassifier, SupportVectorRegressor
from moxie._random import RandomState, entropy_state

struct Dual(Copyable, Movable):
    var value: String

    @implicit
    def __init__(out self, value: String):
        self.value = value

    @implicit
    def __init__(out self, value: Bool):
        self.value = "false"
        if value:
            self.value = "true"

struct ClassWeight[Label: Copyable & Comparable & Deinitable](Copyable, Movable):
    var mode: String
    var labels: List[Self.Label]
    var values: List[Float64]

    @implicit
    def __init__(out self, mode: String = "none"):
        self.mode = mode
        self.labels = List[Self.Label]()
        self.values = List[Float64]()

    def __init__(out self, labels: List[Self.Label], values: List[Float64]):
        self.mode = "explicit"
        self.labels = labels.copy()
        self.values = values.copy()


def _solver(loss: String, penalty: String, dual: Dual, multi_class: String, n: Int, p: Int, regression: Bool) raises -> Int:
    if dual.value != "auto" and dual.value != "true" and dual.value != "false":
        raise Error("ValueError: dual must be auto or Bool")
    var chosen = dual.value == "true" or (dual.value == "auto" and n < p)
    if regression:
        if loss != "epsilon_insensitive" and loss != "squared_epsilon_insensitive":
            raise Error("ValueError: invalid LinearSVR loss")
        if loss == "epsilon_insensitive":
            if dual.value == "false":
                raise Error("ValueError: epsilon_insensitive requires dual=True")
            return 13
        if chosen:
            return 12
        return 11
    if penalty != "l1" and penalty != "l2":
        raise Error("ValueError: penalty must be l1 or l2")
    if loss != "hinge" and loss != "squared_hinge":
        raise Error("ValueError: loss must be hinge or squared_hinge")
    if multi_class == "crammer_singer":
        return 4
    if multi_class != "ovr":
        raise Error("ValueError: multi_class must be ovr or crammer_singer")
    if penalty == "l1":
        if loss != "squared_hinge" or dual.value == "true":
            raise Error("ValueError: l1 requires squared_hinge and dual=False")
        return 5
    if loss == "hinge":
        if dual.value == "false":
            raise Error("ValueError: hinge requires dual=True")
        return 3
    if chosen:
        return 1
    return 2


def _weights(n: Int, sample_weight: Optional[Dense]) raises -> Dense:
    if sample_weight:
        var result = sample_weight.value().copy()
        validate_vector(result, n)
        return result^
    return nm.ones[DType.float64](nm.Shape(n))


def _labels[Label: Copyable & Comparable & Deinitable](y: List[Label]) raises -> List[Label]:
    var classes = List[Label]()
    for i in range(len(y)):
        comptime if Label == Float64:
            var label = rebind[Float64](y[i])
            if not isfinite(label) or label != floor(label):
                raise Error("ValueError: classification requires finite discrete labels")
        comptime if Label == Float32:
            var label = rebind[Float32](y[i])
            if not isfinite(label) or label != floor(label):
                raise Error("ValueError: classification requires finite discrete labels")
        var found = False
        for j in range(len(classes)):
            if classes[j] == y[i]:
                found = True
        if not found:
            classes.append(y[i].copy())
    # Class count is normally small; no training matrix is sorted or copied.
    for i in range(1, len(classes)):
        var j = i
        while j > 0 and classes[j] < classes[j-1]:
            var item = classes[j].copy()
            classes[j] = classes[j-1].copy()
            classes[j-1] = item^
            j -= 1
    return classes^


def _codes[Label: Copyable & Comparable & Deinitable](y: List[Label], classes: List[Label]) raises -> Dense:
    var result = nm.zeros[DType.float64](nm.Shape(len(y)))
    for i in range(len(y)):
        for j in range(len(classes)):
            if y[i] == classes[j]:
                result.unsafe_set(i, Float64(j))
                break
    return result^


def _class_weights[Label: Copyable & Comparable & Deinitable](config: ClassWeight[Label], classes: List[Label], codes: Dense, weights: Dense) raises -> Dense:
    var k = len(classes)
    var result = nm.ones[DType.float64](nm.Shape(k))
    if config.mode == "balanced":
        var total = Float64(0)
        var counts = nm.zeros[DType.float64](nm.Shape(k))
        for i in range(codes.shape[0]):
            var w = weights.unsafe_get(i * Int(weights.strides[0]))
            var c = Int(codes.unsafe_get(i))
            counts.unsafe_set(c, counts.unsafe_get(c) + w)
            total += w
        for c in range(k):
            if counts.unsafe_get(c) <= 0 or total <= 0:
                raise Error("ValueError: balanced weights require positive class totals")
            result.unsafe_set(c, total / (Float64(k) * counts.unsafe_get(c)))
    elif config.mode == "explicit":
        if len(config.labels) != len(config.values):
            raise Error("ValueError: class_weight keys and values differ in length")
        var has_unknown = False
        for j in range(len(config.labels)):
            if not isfinite(config.values[j]) or config.values[j] < 0:
                raise Error("ValueError: class weights must be finite and nonnegative")
            var found = False
            for c in range(k):
                if classes[c] == config.labels[j]:
                    result.unsafe_set(c, config.values[j])
                    found = True
            has_unknown = has_unknown or not found
        if has_unknown:
            for c in range(k):
                var found = False
                for j in range(len(config.labels)):
                    found = found or classes[c] == config.labels[j]
                if not found:
                    raise Error("ValueError: observed classes are missing from class_weight")
    elif config.mode != "none":
        raise Error("ValueError: class_weight must be none, balanced, or explicit")
    return result^


def _iteration(model: LinearModel) raises -> Int:
    var result = 0
    for i in range(model.iterations.shape[0]):
        var count = Int(model.iterations.unsafe_get(i))
        if count > result:
            result = count
    return result


def _decision(X: Dense, coef: Dense, intercept: Dense, n_features: Int) raises -> Dense:
    if n_features == 0:
        raise Error("NotFittedError: estimator is not fitted")
    validate_matrix(X, n_features)
    var matrix = coef.copy()
    if coef.ndim == 1:
        matrix = coef.reshape(nm.Shape(1, n_features))
    var rows = matrix.shape[0]
    var model = LinearModel(matrix^, intercept.copy(), nm.zeros[DType.int64](nm.Shape(0)), 0)
    var engine = LibLinear()
    var result = engine.decision_values(X, model)
    if rows == 1:
        return result.reshape(nm.Shape(X.shape[0]))
    return result^


@fieldwise_init
struct LinearSVCParams[Label: Copyable & Comparable & Deinitable = Float64](Copyable, Movable):
    var penalty: String
    var loss: String
    var multi_class: String
    var class_weight: ClassWeight[Self.Label]
    var dual: Dual
    var tol: Float64
    var C: Float64
    var fit_intercept: Bool
    var intercept_scaling: Float64
    var verbose: Int
    var random_state: Int
    var max_iter: Int


struct LinearSVC[Label: Copyable & Comparable & Deinitable = Float64] (SupportVectorClassifier):
    comptime Params = LinearSVCParams[Self.Label]
    comptime Input = Dense
    comptime SampleWeight = Dense
    comptime Target = List[Self.Label]
    comptime Prediction = List[Self.Label]
    comptime Decision = Dense
    var penalty: String
    var loss: String
    var multi_class: String
    var class_weight: ClassWeight[Self.Label]
    var dual: Dual
    var tol: Float64
    var C: Float64
    var fit_intercept: Bool
    var intercept_scaling: Float64
    var verbose: Int
    var random_state: Int
    var max_iter: Int
    var coef_: Dense
    var intercept_: Dense
    var n_features_in_: Int
    var n_iter_: Int
    var fit_status_: Int
    var _rng: RandomState
    var classes_: List[Self.Label]

    def __init__(out self, *, penalty: String = "l2", loss: String = "squared_hinge", multi_class: String = "ovr", class_weight: ClassWeight[Self.Label] = "none", dual: Dual = "auto", tol: Float64 = 1e-4, C: Float64 = 1.0, fit_intercept: Bool = True, intercept_scaling: Float64 = 1.0, verbose: Int = 0, random_state: Int = -1, max_iter: Int = 1000) raises:
        self.penalty = penalty.copy()
        self.loss = loss.copy()
        self.multi_class = multi_class.copy()
        self.class_weight = class_weight.copy()
        self.dual = dual.copy()
        self.tol = tol.copy()
        self.C = C.copy()
        self.fit_intercept = fit_intercept.copy()
        self.intercept_scaling = intercept_scaling.copy()
        self.verbose = verbose.copy()
        self.random_state = random_state.copy()
        self.max_iter = max_iter.copy()
        self.coef_ = nm.zeros[DType.float64](nm.Shape(0))
        self.intercept_ = nm.zeros[DType.float64](nm.Shape(0))
        self.n_features_in_ = 0
        self.n_iter_ = 0
        self.fit_status_ = 0
        self._rng = entropy_state()
        self.classes_ = List[Self.Label]()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.penalty.copy(), self.loss.copy(), self.multi_class.copy(), self.class_weight.copy(), self.dual.copy(), self.tol.copy(), self.C.copy(), self.fit_intercept.copy(), self.intercept_scaling.copy(), self.verbose.copy(), self.random_state.copy(), self.max_iter.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.penalty = params.penalty.copy()
        self.loss = params.loss.copy()
        self.multi_class = params.multi_class.copy()
        self.class_weight = params.class_weight.copy()
        self.dual = params.dual.copy()
        self.tol = params.tol.copy()
        self.C = params.C.copy()
        self.fit_intercept = params.fit_intercept.copy()
        self.intercept_scaling = params.intercept_scaling.copy()
        self.verbose = params.verbose.copy()
        self.random_state = params.random_state.copy()
        self.max_iter = params.max_iter.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(penalty=self.penalty.copy(), loss=self.loss.copy(), multi_class=self.multi_class.copy(), class_weight=self.class_weight.copy(), dual=self.dual.copy(), tol=self.tol.copy(), C=self.C.copy(), fit_intercept=self.fit_intercept.copy(), intercept_scaling=self.intercept_scaling.copy(), verbose=self.verbose.copy(), random_state=self.random_state.copy(), max_iter=self.max_iter.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        _ = self.fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target, sample_weight: Optional[Dense]) raises -> ref[self] Self:
        if self.random_state < -1 or self.random_state > 4294967295:
            raise Error("ValueError: random_state must be None (-1) or uint32 seed")
        var rng = self._rng.copy()
        if self.random_state != -1:
            rng = RandomState(self.random_state)
        _ = self.fit_with_random_state(X, y, rng, sample_weight)
        if self.random_state == -1:
            self._rng = rng^
        return self

    def fit_with_random_state[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target, mut rng: RandomState, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        validate_matrix(X)
        if len(y) != X.shape[0]:
            raise Error("ValueError: inconsistent target length")
        var classes = _labels(y)
        if len(classes) < 2:
            raise Error("ValueError: at least two classes required")
        var target = _codes(y, classes)
        var weights = _weights(X.shape[0], sample_weight)
        for c in range(len(classes)):
            var retained = False
            for i in range(X.shape[0]):
                if Int(target.unsafe_get(i)) == c and weights.unsafe_get(i * Int(weights.strides[0])) > 0:
                    retained = True
            if not retained:
                raise Error("ValueError: sample weights removing an entire class are unsupported")
        var costs = _class_weights(self.class_weight, classes, target, weights)
        var solver = _solver(self.loss, self.penalty, self.dual, self.multi_class, X.shape[0], X.shape[1], False)
        if self.verbose < 0:
            raise Error("ValueError: verbose must be nonnegative")
        var options = LinearOptions(solver, self.C, 0.0, self.tol, self.fit_intercept, self.intercept_scaling, self.max_iter, rng.randint(2147483647))
        var engine = LibLinear()
        var model = engine.solve(X, target, weights, costs, options)
        if model.coef.shape[0] != 1 and model.coef.shape[0] != len(classes):
            raise Error("ValueError: sample weights removed an entire class; unsupported model layout")
        if solver == 4 and len(classes) == 2:
            var coef = nm.zeros[DType.float64](nm.Shape(1, X.shape[1]))
            for j in range(X.shape[1]):
                coef.unsafe_set(j, model.coef.unsafe_get(X.shape[1]+j) - model.coef.unsafe_get(j))
            model.coef = coef^
            var intercept = nm.zeros[DType.float64](nm.Shape(1))
            intercept.unsafe_set(0, model.intercept.unsafe_get(1) - model.intercept.unsafe_get(0))
            model.intercept = intercept^
        self.classes_ = classes^
        self.n_iter_ = _iteration(model)
        self.fit_status_ = model.fit_status
        self.n_features_in_ = X.shape[1]
        self.coef_ = model.coef.copy()
        self.intercept_ = model.intercept.copy()
        if self.fit_status_ != 0:
            print("ConvergenceWarning: Liblinear failed to converge, increase max_iter.")
        if self.verbose > 0:
            print("[LibLinear] iterations:", self.n_iter_)
        return self

    def decision_function(self, X: Dense) raises -> Dense:
        return _decision(X, self.coef_, self.intercept_, self.n_features_in_)

    def predict(self, X: Dense) raises -> Self.Prediction:
        var scores = self.decision_function(X)
        var result = List[Self.Label]()
        for i in range(X.shape[0]):
            var index = 0
            if scores.ndim == 1:
                index = Int(scores.unsafe_get(i) > 0)
            else:
                for j in range(1, len(self.classes_)):
                    if scores.unsafe_get(i * len(self.classes_) + j) > scores.unsafe_get(i * len(self.classes_) + index):
                        index = j
            result.append(self.classes_[index].copy())
        return result^

    def score(self, X: Dense, y: Self.Target, sample_weight: Optional[Dense] = None) raises -> Float64:
        var prediction = self.predict(X)
        var weights = _weights(X.shape[0], sample_weight)
        if len(y) != X.shape[0]:
            raise Error("ValueError: inconsistent target length")
        var numerator = Float64(0)
        var denominator = Float64(0)
        for i in range(len(y)):
            var w = weights.unsafe_get(i * Int(weights.strides[0]))
            denominator += w
            if prediction[i] == y[i]:
                numerator += w
        if denominator == 0:
            raise Error("ValueError: weights sum to zero")
        return numerator / denominator

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[DType.float32], ref[y_origin] y: Self.Target, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        _ = self.fit(X.astype[DType.float64](), y, sample_weight)
        return self

    def predict(self, X: nm.NDArray[DType.float32]) raises -> Self.Prediction:
        return self.predict(X.astype[DType.float64]())

    def decision_function(self, X: nm.NDArray[DType.float32]) raises -> Dense:
        return self.decision_function(X.astype[DType.float64]())

    def set_params[origin: MutOrigin](ref[origin] self, *, penalty: Optional[String] = None, loss: Optional[String] = None, multi_class: Optional[String] = None, class_weight: Optional[ClassWeight[Self.Label]] = None, dual: Optional[Dual] = None, tol: Optional[Float64] = None, C: Optional[Float64] = None, fit_intercept: Optional[Bool] = None, intercept_scaling: Optional[Float64] = None, verbose: Optional[Int] = None, random_state: Optional[Int] = None, max_iter: Optional[Int] = None) raises -> ref[self] Self:
        if penalty:
            self.penalty = penalty.value().copy()
        if loss:
            self.loss = loss.value().copy()
        if multi_class:
            self.multi_class = multi_class.value().copy()
        if class_weight:
            self.class_weight = class_weight.value().copy()
        if dual:
            self.dual = dual.value().copy()
        if tol:
            self.tol = tol.value().copy()
        if C:
            self.C = C.value().copy()
        if fit_intercept:
            self.fit_intercept = fit_intercept.value().copy()
        if intercept_scaling:
            self.intercept_scaling = intercept_scaling.value().copy()
        if verbose:
            self.verbose = verbose.value().copy()
        if random_state:
            self.random_state = random_state.value().copy()
        if max_iter:
            self.max_iter = max_iter.value().copy()
        return self

    def score(self, X: nm.NDArray[DType.float32], y: Self.Target, sample_weight: Optional[Dense] = None) raises -> Float64:
        return self.score(X.astype[DType.float64](), y, sample_weight)

    def fit[dtype: DType, origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: nm.NDArray[dtype], sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        comptime assert Self.Label == Float64, "Numeric arrays require LinearSVC[Float64]; use List[String] for strings"
        var target = y.astype[DType.float64]()
        validate_vector(target, X.shape[0])
        var labels = List[Self.Label]()
        for i in range(target.shape[0]):
            labels.append(rebind[Self.Label](target.unsafe_get(i * Int(target.strides[0]))).copy())
        _ = self.fit(X, labels, sample_weight)
        return self


@fieldwise_init
struct LinearSVRParams(Copyable, Movable):
    var epsilon: Float64
    var loss: String
    var dual: Dual
    var tol: Float64
    var C: Float64
    var fit_intercept: Bool
    var intercept_scaling: Float64
    var verbose: Int
    var random_state: Int
    var max_iter: Int


struct LinearSVR(SupportVectorRegressor):
    comptime Params = LinearSVRParams
    comptime Input = Dense
    comptime SampleWeight = Dense
    comptime Target = Dense
    comptime Prediction = Dense
    var epsilon: Float64
    var loss: String
    var dual: Dual
    var tol: Float64
    var C: Float64
    var fit_intercept: Bool
    var intercept_scaling: Float64
    var verbose: Int
    var random_state: Int
    var max_iter: Int
    var coef_: Dense
    var intercept_: Dense
    var n_features_in_: Int
    var n_iter_: Int
    var fit_status_: Int
    var _rng: RandomState

    def __init__(out self, *, epsilon: Float64 = 0.0, loss: String = "epsilon_insensitive", dual: Dual = "auto", tol: Float64 = 1e-4, C: Float64 = 1.0, fit_intercept: Bool = True, intercept_scaling: Float64 = 1.0, verbose: Int = 0, random_state: Int = -1, max_iter: Int = 1000) raises:
        self.epsilon = epsilon.copy()
        self.loss = loss.copy()
        self.dual = dual.copy()
        self.tol = tol.copy()
        self.C = C.copy()
        self.fit_intercept = fit_intercept.copy()
        self.intercept_scaling = intercept_scaling.copy()
        self.verbose = verbose.copy()
        self.random_state = random_state.copy()
        self.max_iter = max_iter.copy()
        self.coef_ = nm.zeros[DType.float64](nm.Shape(0))
        self.intercept_ = nm.zeros[DType.float64](nm.Shape(0))
        self.n_features_in_ = 0
        self.n_iter_ = 0
        self.fit_status_ = 0
        self._rng = entropy_state()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.epsilon.copy(), self.loss.copy(), self.dual.copy(), self.tol.copy(), self.C.copy(), self.fit_intercept.copy(), self.intercept_scaling.copy(), self.verbose.copy(), self.random_state.copy(), self.max_iter.copy())

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.epsilon = params.epsilon.copy()
        self.loss = params.loss.copy()
        self.dual = params.dual.copy()
        self.tol = params.tol.copy()
        self.C = params.C.copy()
        self.fit_intercept = params.fit_intercept.copy()
        self.intercept_scaling = params.intercept_scaling.copy()
        self.verbose = params.verbose.copy()
        self.random_state = params.random_state.copy()
        self.max_iter = params.max_iter.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(epsilon=self.epsilon.copy(), loss=self.loss.copy(), dual=self.dual.copy(), tol=self.tol.copy(), C=self.C.copy(), fit_intercept=self.fit_intercept.copy(), intercept_scaling=self.intercept_scaling.copy(), verbose=self.verbose.copy(), random_state=self.random_state.copy(), max_iter=self.max_iter.copy())

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        _ = self.fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target, sample_weight: Optional[Dense]) raises -> ref[self] Self:
        if self.random_state < -1 or self.random_state > 4294967295:
            raise Error("ValueError: random_state must be None (-1) or uint32 seed")
        var rng = self._rng.copy()
        if self.random_state != -1:
            rng = RandomState(self.random_state)
        _ = self.fit_with_random_state(X, y, rng, sample_weight)
        if self.random_state == -1:
            self._rng = rng^
        return self

    def fit_with_random_state[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Self.Target, mut rng: RandomState, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        validate_matrix(X)
        validate_vector(y, X.shape[0])
        var target = y.copy()
        var weights = _weights(X.shape[0], sample_weight)
        var costs = nm.zeros[DType.float64](nm.Shape(0))
        var solver = _solver(self.loss, "l2", self.dual, "ovr", X.shape[0], X.shape[1], True)
        if self.verbose < 0:
            raise Error("ValueError: verbose must be nonnegative")
        var options = LinearOptions(solver, self.C, self.epsilon, self.tol, self.fit_intercept, self.intercept_scaling, self.max_iter, rng.randint(2147483647))
        var engine = LibLinear()
        var model = engine.solve(X, target, weights, costs, options)
        self.n_iter_ = _iteration(model)
        self.fit_status_ = model.fit_status
        self.n_features_in_ = X.shape[1]
        self.coef_ = model.coef.reshape(nm.Shape(X.shape[1]))
        self.intercept_ = model.intercept.copy()
        if self.fit_status_ != 0:
            print("ConvergenceWarning: Liblinear failed to converge, increase max_iter.")
        if self.verbose > 0:
            print("[LibLinear] iterations:", self.n_iter_)
        return self

    def predict(self, X: Dense) raises -> Dense:
        return _decision(X, self.coef_, self.intercept_, self.n_features_in_)

    def score(self, X: Dense, y: Self.Target, sample_weight: Optional[Dense] = None) raises -> Float64:
        var prediction = self.predict(X)
        var weights = _weights(X.shape[0], sample_weight)
        validate_vector(y, X.shape[0])
        if X.shape[0] < 2:
            return Float64("nan")
        var total = Float64(0)
        var mean = Float64(0)
        for i in range(X.shape[0]):
            var w = weights.unsafe_get(i * Int(weights.strides[0]))
            total += w
            mean += w * y.unsafe_get(i * Int(y.strides[0]))
        if total == 0:
            raise Error("ValueError: weights sum to zero")
        mean /= total
        var residual = Float64(0)
        var variance = Float64(0)
        for i in range(X.shape[0]):
            var w = weights.unsafe_get(i * Int(weights.strides[0]))
            var actual = y.unsafe_get(i * Int(y.strides[0]))
            var error = actual - prediction.unsafe_get(i)
            residual += w * error * error
            error = actual - mean
            variance += w * error * error
        if residual == 0:
            return 1.0
        if variance == 0:
            return 0.0
        return 1.0 - residual / variance

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[DType.float32], ref[y_origin] y: Self.Target, sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        _ = self.fit(X.astype[DType.float64](), y, sample_weight)
        return self

    def predict(self, X: nm.NDArray[DType.float32]) raises -> Self.Prediction:
        return self.predict(X.astype[DType.float64]())

    def set_params[origin: MutOrigin](ref[origin] self, *, epsilon: Optional[Float64] = None, loss: Optional[String] = None, dual: Optional[Dual] = None, tol: Optional[Float64] = None, C: Optional[Float64] = None, fit_intercept: Optional[Bool] = None, intercept_scaling: Optional[Float64] = None, verbose: Optional[Int] = None, random_state: Optional[Int] = None, max_iter: Optional[Int] = None) raises -> ref[self] Self:
        if epsilon:
            self.epsilon = epsilon.value().copy()
        if loss:
            self.loss = loss.value().copy()
        if dual:
            self.dual = dual.value().copy()
        if tol:
            self.tol = tol.value().copy()
        if C:
            self.C = C.value().copy()
        if fit_intercept:
            self.fit_intercept = fit_intercept.value().copy()
        if intercept_scaling:
            self.intercept_scaling = intercept_scaling.value().copy()
        if verbose:
            self.verbose = verbose.value().copy()
        if random_state:
            self.random_state = random_state.value().copy()
        if max_iter:
            self.max_iter = max_iter.value().copy()
        return self

    def score(self, X: nm.NDArray[DType.float32], y: Self.Target, sample_weight: Optional[Dense] = None) raises -> Float64:
        return self.score(X.astype[DType.float64](), y, sample_weight)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: nm.NDArray[DType.float32], sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        _ = self.fit(X, y.astype[DType.float64](), sample_weight)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[DType.float32], ref[y_origin] y: nm.NDArray[DType.float32], sample_weight: Optional[Dense] = None) raises -> ref[self] Self:
        _ = self.fit(X.astype[DType.float64](), y.astype[DType.float64](), sample_weight)
        return self
