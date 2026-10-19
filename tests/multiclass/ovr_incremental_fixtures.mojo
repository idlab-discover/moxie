"""Incremental signed-sum test double, not a production learning algorithm."""
from numojo import NDArray, Shape, Item
import numojo as nm
from moxie.base import IncrementalClassifier, DecisionFunction, ProbabilityPredictor
from ovr_fixtures import Matrix, Binary, ShiftParams, _accuracy


struct OnlineClassifier(IncrementalClassifier, DecisionFunction, ProbabilityPredictor, Copyable):
    comptime Params = ShiftParams
    comptime Input = Matrix
    comptime Target = Binary
    comptime Classes = Binary
    comptime Prediction = Binary
    comptime SampleWeight = Matrix
    comptime Decision = Matrix
    comptime Probabilities = Matrix
    var shift: Float64
    var updates: Int
    var samples: Int
    var positives: Int
    var weights: List[Float64]

    def __init__(out self, *, shift: Float64 = 0.0):
        self.shift = shift
        self.updates = 0
        self.samples = 0
        self.positives = 0
        self.weights = List[Float64]()

    def get_params(self, deep: Bool = True) raises -> ShiftParams:
        return ShiftParams(self.shift)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: ShiftParams) raises -> ref[self] Self:
        self.shift = params.shift
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(shift=self.shift)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.updates > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Matrix, ref[y_origin] y: Binary) raises -> ref[self] Self:
        self.updates = 0
        self.samples = 0
        self.positives = 0
        self.weights = List[Float64]()
        var classes = nm.zeros[DType.int64](Shape(2))
        classes[Item(1)] = 1
        self.partial_fit(X, y, classes^)
        return self

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Matrix, ref[y_origin] y: Binary, classes: Optional[Binary] = None) raises -> ref[self] Self:
        if not classes or classes.value().size != 2:
            raise Error("fixture requires binary classes on every call")
        if classes.value()[Item(0)] != 0 or classes.value()[Item(1)] != 1:
            raise Error("fixture requires classes=[0,1]")
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size or y.size == 0:
            raise Error("ValueError: invalid fixture input")
        if self.updates and X.shape[1] != len(self.weights):
            raise Error("ValueError: changed feature count")
        if not self.updates:
            self.weights = List[Float64](length=X.shape[1], fill=0.0)
        for i in range(y.size):
            if y[Item(i)] != 0 and y[Item(i)] != 1:
                raise Error("fixture requires 0/1 targets")
            self.positives += Int(y[Item(i)])
            for j in range(len(self.weights)):
                self.weights[j] += (2.0 * Float64(y[Item(i)]) - 1.0) * X[Item(i, j)]
        self.samples += y.size
        self.updates += 1
        return self

    def decision_function(self, X: Matrix) raises -> Matrix:
        if not self.updates:
            raise Error("NotFittedError: fixture")
        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise Error("ValueError: changed feature count")
        var out = nm.zeros[DType.float64](Shape(X.shape[0]))
        for i in range(X.shape[0]):
            var value = self.shift
            for j in range(len(self.weights)):
                value += X[Item(i, j)] * self.weights[j]
            out[Item(i)] = value
        return out^

    def predict_proba(self, X: Matrix) raises -> Matrix:
        var margins = self.decision_function(X)
        var out = nm.zeros[DType.float64](Shape(X.shape[0], 2))
        for i in range(X.shape[0]):
            var value = margins[Item(i)]
            var p = 0.5 + 0.5 * value / (1.0 + abs(value))
            out[Item(i, 0)] = 1.0 - p
            out[Item(i, 1)] = p
        return out^

    def predict(self, X: Matrix) raises -> Binary:
        var margins = self.decision_function(X)
        var out = nm.zeros[DType.int64](Shape(X.shape[0]))
        for i in range(X.shape[0]):
            if margins[Item(i)] > 0:
                out[Item(i)] = 1
        return out^

    def score(self, X: Matrix, y: Binary, sample_weight: Optional[Matrix] = None) raises -> Float64:
        return _accuracy(self.predict(X), y, sample_weight)
