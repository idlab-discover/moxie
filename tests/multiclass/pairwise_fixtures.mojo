"""Recording signed-sum component for wrapper tests, not a kernel learner port."""
from numojo import NDArray, Shape, Item
import numojo as nm
from moxie.base import IncrementalClassifier, DecisionFunction, ProbabilityPredictor
from moxie.multiclass import PairwiseInput
from ovr_fixtures import Binary, Matrix, _accuracy


@fieldwise_init
struct PairwiseParams(Copyable, Movable):
    var precomputed: Bool
    var fail_rows: Int


struct PairwiseRecorder[dtype: DType = DType.float64, probability: Bool = False](
    IncrementalClassifier, PairwiseInput, Copyable,
    DecisionFunction where not probability,
    ProbabilityPredictor where probability,
):
    comptime Input = NDArray[Self.dtype]
    comptime Target = Binary
    comptime Classes = Binary
    comptime Prediction = Binary
    comptime SampleWeight = Matrix
    comptime Decision = Matrix
    comptime Probabilities = Matrix
    comptime Params = PairwiseParams
    var precomputed: Bool
    var fail_rows: Int
    var seen: Self.Input
    var targets: Binary
    var weights: List[Float64]
    var updates: Int

    def __init__(out self, precomputed: Bool = True, fail_rows: Int = -1) raises:
        self.precomputed = precomputed
        self.fail_rows = fail_rows
        self.seen = nm.zeros[Self.dtype](Shape(0, 0))
        self.targets = nm.zeros[DType.int64](Shape(0))
        self.weights = List[Float64]()
        self.updates = 0

    def pairwise_input(self) -> Bool:
        return self.precomputed

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return PairwiseParams(self.precomputed, self.fail_rows)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.precomputed = params.precomputed
        self.fail_rows = params.fail_rows
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.precomputed, self.fail_rows)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.updates > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Binary) raises -> ref[self] Self:
        if self.precomputed and X.shape[0] != X.shape[1]:
            raise Error("ValueError: fixture requires a square kernel")
        self.weights = List[Float64]()
        self.updates = 0
        self._update(X, y)
        return self

    def _update(mut self, X: Self.Input, y: Binary) raises:
        if X.shape[0] == self.fail_rows:
            raise Error("ValueError: intentional pair failure")
        if self.updates and X.shape[1] != len(self.weights):
            raise Error("ValueError: changed feature count")
        if not self.updates:
            self.weights = List[Float64](length=X.shape[1], fill=0.0)
        self.seen = X.copy()
        self.targets = y.copy()
        for i in range(y.size):
            var sign: Float64 = 1.0 if y[Item(i)] == 1 else -1.0
            for j in range(X.shape[1]):
                self.weights[j] += sign * Float64(X[Item(i, j)])
        self.updates += 1

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Binary, classes: Optional[Binary] = None) raises -> ref[self] Self:
        if not classes or classes.value().size != 2:
            raise Error("fixture requires binary classes")
        if classes.value()[Item(0)] != 0 or classes.value()[Item(1)] != 1:
            raise Error("fixture requires classes=[0,1]")
        self._update(X, y)
        return self

    def margin(self, X: Self.Input) raises -> Matrix:
        if not self.updates:
            raise Error("NotFittedError: fixture")
        if X.shape[1] != len(self.weights):
            raise Error("ValueError: changed feature count")
        var result = nm.zeros[DType.float64](Shape(X.shape[0]))
        for i in range(X.shape[0]):
            var value: Float64 = 0.0
            for j in range(X.shape[1]):
                value += Float64(X[Item(i, j)]) * self.weights[j]
            result[Item(i)] = value
        return result^

    def decision_function(self, X: Self.Input) raises -> Matrix where not Self.probability:
        return self.margin(X)

    def predict_proba(self, X: Self.Input) raises -> Matrix where Self.probability:
        var margins = self.margin(X)
        var result = nm.zeros[DType.float64](Shape(X.shape[0], 2))
        for i in range(X.shape[0]):
            var value = margins[Item(i)]
            var p = 0.5 + 0.5 * value / (1.0 + abs(value))
            result[Item(i, 0)] = 1.0 - p
            result[Item(i, 1)] = p
        return result^

    def predict(self, X: Self.Input) raises -> Binary:
        var margins = self.margin(X)
        var result = nm.zeros[DType.int64](Shape(X.shape[0]))
        for i in range(X.shape[0]):
            if margins[Item(i)] > 0:
                result[Item(i)] = 1
        return result^

    def score(self, X: Self.Input, y: Binary, sample_weight: Optional[Matrix] = None) raises -> Float64:
        return _accuracy(self.predict(X), y, sample_weight)
