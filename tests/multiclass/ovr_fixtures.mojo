"""Deterministic binary test components for the multiclass wrappers.

These are test doubles, not scikit-learn algorithms. `tests/ovr_parity.py`
defines the same arithmetic in Python so scikit-learn's own
multiclass wrappers can wrap identical components.
"""
from numojo import NDArray, Shape, Item
import numojo as nm
from std.math import exp
from moxie.base import RegressorMixin, ClassifierMixin, DecisionFunction, ProbabilityPredictor

comptime Matrix = NDArray[DType.float64]
comptime Binary = NDArray[DType.int64]


@fieldwise_init
struct ShiftParams(Copyable, Movable):
    var shift: Float64


struct _Centroids(Copyable, Movable):
    """Class means; margin(x) = |x - mean0|^2 - |x - mean1|^2 + shift."""

    var mean0: List[Float64]
    var mean1: List[Float64]

    def __init__(out self):
        self.mean0 = List[Float64]()
        self.mean1 = List[Float64]()

    def fitted(self) -> Bool:
        return len(self.mean0) > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Matrix, ref[y_origin] y: Binary) raises -> ref[self] Self:
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise Error("ValueError: fixture expects X (n, p) and y (n,)")
        var p = X.shape[1]
        var sum0 = List[Float64](length=p, fill=0.0)
        var sum1 = List[Float64](length=p, fill=0.0)
        var n0 = 0
        var n1 = 0
        for i in range(X.shape[0]):
            var label = y[Item(i)]
            if label != -1 and label != 0 and label != 1:
                raise Error("ValueError: fixture expects -1/1 or 0/1 targets")
            for j in range(p):
                if label == 1:
                    sum1[j] += X[Item(i, j)]
                else:
                    sum0[j] += X[Item(i, j)]
            if label == 1:
                n1 += 1
            else:
                n0 += 1
        if n0 == 0 or n1 == 0:
            raise Error("ValueError: fixture needs both binary classes")
        for j in range(p):
            sum0[j] /= Float64(n0)
            sum1[j] /= Float64(n1)
        self.mean0 = sum0^
        self.mean1 = sum1^
        return self

    def margin(self, X: Matrix, shift: Float64) raises -> Matrix:
        if not self.fitted():
            raise Error("NotFittedError: fixture is not fitted")
        if X.ndim != 2 or X.shape[1] != len(self.mean0):
            raise Error(
                String("ValueError: X has ", X.shape[1] if X.ndim == 2 else 0,
                       " features, but fixture is expecting ", len(self.mean0),
                       " features as input.")
            )
        var result = nm.zeros[DType.float64](Shape(X.shape[0]))
        for i in range(X.shape[0]):
            var d0: Float64 = 0.0
            var d1: Float64 = 0.0
            for j in range(len(self.mean0)):
                var a = X[Item(i, j)] - self.mean0[j]
                var b = X[Item(i, j)] - self.mean1[j]
                d0 += a * a
                d1 += b * b
            result[Item(i)] = (d0 - d1) + shift
        return result^

    def probabilities(self, X: Matrix, shift: Float64) raises -> Matrix:
        var margin = self.margin(X, shift)
        var result = nm.zeros[DType.float64](Shape(X.shape[0], 2))
        for i in range(X.shape[0]):
            var p = 1.0 / (1.0 + exp(-margin[Item(i)]))
            result[Item(i, 0)] = 1.0 - p
            result[Item(i, 1)] = p
        return result^

    def labels(self, margin: Matrix, threshold: Float64) raises -> Binary:
        var result = nm.zeros[DType.int64](Shape(margin.size))
        for i in range(margin.size):
            if margin[Item(i)] > threshold:
                result[Item(i)] = 1
        return result^


def _accuracy(predicted: Binary, y: Binary, sample_weight: Optional[Matrix]) raises -> Float64:
    var correct: Float64 = 0.0
    var total: Float64 = 0.0
    for i in range(y.size):
        var w = sample_weight.value()[Item(i)] if sample_weight else 1.0
        total += w
        if predicted[Item(i)] == y[Item(i)]:
            correct += w
    return correct / total


struct MarginClassifier(ClassifierMixin, DecisionFunction, Copyable):
    """decision_function only (sklearn threshold 0.0)."""

    comptime Params = ShiftParams
    comptime Input = Matrix
    comptime Target = Binary
    comptime Prediction = Binary
    comptime SampleWeight = Matrix
    comptime Decision = Matrix
    var shift: Float64
    var fits: Int
    var _model: _Centroids

    def __init__(out self, *, shift: Float64 = 0.0):
        self.shift = shift
        self.fits = 0
        self._model = _Centroids()

    def get_params(self, deep: Bool = True) raises -> ShiftParams:
        return ShiftParams(self.shift)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: ShiftParams) raises -> ref[self] Self:
        self.shift = params.shift
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(shift=self.shift)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._model.fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Matrix, ref[y_origin] y: Binary) raises -> ref[self] Self:
        self._model.fit(X, y)
        self.fits += 1
        return self

    def decision_function(self, X: Matrix) raises -> Matrix:
        return self._model.margin(X, self.shift)

    def predict(self, X: Matrix) raises -> Binary:
        return self._model.labels(self.decision_function(X), 0.0)

    def score(self, X: Matrix, y: Binary, sample_weight: Optional[Matrix] = None) raises -> Float64:
        return _accuracy(self.predict(X), y, sample_weight)


struct ProbaClassifier(ClassifierMixin, ProbabilityPredictor, Copyable):
    """predict_proba only (sklearn threshold 0.5)."""

    comptime Params = ShiftParams
    comptime Input = Matrix
    comptime Target = Binary
    comptime Prediction = Binary
    comptime SampleWeight = Matrix
    comptime Probabilities = Matrix
    var shift: Float64
    var _model: _Centroids

    def __init__(out self, *, shift: Float64 = 0.0):
        self.shift = shift
        self._model = _Centroids()

    def get_params(self, deep: Bool = True) raises -> ShiftParams:
        return ShiftParams(self.shift)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: ShiftParams) raises -> ref[self] Self:
        self.shift = params.shift
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(shift=self.shift)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._model.fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Matrix, ref[y_origin] y: Binary) raises -> ref[self] Self:
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: Matrix) raises -> Matrix:
        return self._model.probabilities(X, self.shift)

    def predict(self, X: Matrix) raises -> Binary:
        return self._model.labels(self._model.margin(X, self.shift), 0.0)

    def score(self, X: Matrix, y: Binary, sample_weight: Optional[Matrix] = None) raises -> Float64:
        return _accuracy(self.predict(X), y, sample_weight)


struct MarginProbaClassifier(ClassifierMixin, DecisionFunction, ProbabilityPredictor, Copyable):
    """Both capabilities; sklearn prefers decision_function for scores."""

    comptime Params = ShiftParams
    comptime Input = Matrix
    comptime Target = Binary
    comptime Prediction = Binary
    comptime SampleWeight = Matrix
    comptime Decision = Matrix
    comptime Probabilities = Matrix
    var shift: Float64
    var _model: _Centroids

    def __init__(out self, *, shift: Float64 = 0.0):
        self.shift = shift
        self._model = _Centroids()

    def get_params(self, deep: Bool = True) raises -> ShiftParams:
        return ShiftParams(self.shift)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: ShiftParams) raises -> ref[self] Self:
        self.shift = params.shift
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(shift=self.shift)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._model.fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Matrix, ref[y_origin] y: Binary) raises -> ref[self] Self:
        self._model.fit(X, y)
        return self

    def decision_function(self, X: Matrix) raises -> Matrix:
        return self._model.margin(X, self.shift)

    def predict_proba(self, X: Matrix) raises -> Matrix:
        return self._model.probabilities(X, self.shift)

    def predict(self, X: Matrix) raises -> Binary:
        return self._model.labels(self.decision_function(X), 0.0)

    def score(self, X: Matrix, y: Binary, sample_weight: Optional[Matrix] = None) raises -> Float64:
        return _accuracy(self.predict(X), y, sample_weight)


struct _ScoreRegressor[
    dtype: DType = DType.float64, decision: Bool = False, probability: Bool = False, column: Bool = False
](RegressorMixin, Copyable, DecisionFunction where decision, ProbabilityPredictor where probability):
    """Test double: continuous centroid margin; optional distinct decision values."""

    comptime Params = ShiftParams
    comptime Input = Matrix
    comptime Target = Binary
    comptime Prediction = NDArray[Self.dtype]
    comptime Probabilities = Matrix
    comptime SampleWeight = Matrix
    comptime Decision = Matrix
    var shift: Float64
    var _model: _Centroids

    def __init__(out self, *, shift: Float64 = 0.0):
        self.shift = shift
        self._model = _Centroids()

    def get_params(self, deep: Bool = True) raises -> ShiftParams:
        return ShiftParams(self.shift)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: ShiftParams) raises -> ref[self] Self:
        self.shift = params.shift
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(shift=self.shift)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._model.fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Matrix, ref[y_origin] y: Binary) raises -> ref[self] Self:
        self._model.fit(X, y)
        return self

    def predict(self, X: Matrix) raises -> Self.Prediction:
        var values = self._model.margin(X, self.shift).astype[Self.dtype]()
        comptime if Self.column:
            return values.reshape(Shape(values.size, 1))
        else:
            return values^

    def decision_function(self, X: Matrix) raises -> Matrix where Self.decision:
        # Deliberately differs from predict to check dispatch precedence.
        return self._model.margin(X, self.shift + 10.0)

    def predict_proba(self, X: Matrix) raises -> Matrix where Self.probability:
        return self._model.probabilities(X, self.shift + 10.0)

    def score(self, X: Matrix, y: Binary, sample_weight: Optional[Matrix] = None) raises -> Float64:
        raise Error("Test double: regressor score must not be called by OvR")


comptime ScoreRegressor = _ScoreRegressor[]
comptime DecisionRegressor = _ScoreRegressor[decision=True]
comptime Float32Regressor = _ScoreRegressor[dtype=DType.float32]

comptime Float32DecisionRegressor = _ScoreRegressor[dtype=DType.float32, decision=True]
comptime ProbabilityRegressor = _ScoreRegressor[probability=True]
