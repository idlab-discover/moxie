from moxie._labels import StringTarget
from moxie.base import Predictor
from moxie.model_selection import Accuracy, F1Macro, PrecisionMacro, RecallMacro
from numojo import NDArray, Shape, Strides, array
from std.testing import assert_raises
from std.math import inf

comptime Weights = NDArray[DType.float32]


@fieldwise_init
struct FixedPrediction[Y: Copyable & Deinitable](Predictor):
    comptime Input = Int
    comptime Params = Int
    comptime Prediction = Self.Y

    var labels: Optional[Self.Y]

    def get_params(self, deep: Bool = True) raises -> Int:
        return 0

    def set_params[origin: MutOrigin, p_origin: ImmOrigin](
        ref[origin] self, ref[p_origin] params: Int,
    ) raises -> ref[self] Self:
        return self

    def __sklearn_clone__(self) -> Self:
        return Self(None)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.labels)

    def predict(self, X: Int) raises -> Self.Y:
        if not self.labels:
            raise Error("NotFittedError: missing fitted labels")
        return self.labels.value().copy()


def check[Y: Copyable & Deinitable](name: String, y: Y, predicted: Y) raises:
    comptime E = FixedPrediction[Y]
    var model = E(predicted.copy())
    var backing = array[DType.float32]([4.0, 2.0, 1.0, 3.0, 8.0, 1.0], [6])
    var weights = backing.view_with_layout(Shape(6), Strides(-1), 5)
    print(name, "accuracy", Accuracy[E, Int, Y, DType.float32]().score(model, 0, y, weights.view()))
    print(name, "precision_macro", PrecisionMacro[E, Int, Y, DType.float32]().score(model, 0, y, weights.view()))
    print(name, "recall_macro", RecallMacro[E, Int, Y, DType.float32]().score(model, 0, y, weights.view()))
    print(name, "f1_macro", F1Macro[E, Int, Y, DType.float32]().score(model, 0, y, weights^))


def main() raises:
    var actual: List[Int] = [0, 1, 2, 0, 2, 1]
    var predicted: List[Int] = [1, 1, 0, 0, 0, 1]
    check("ints", actual, predicted)
    var truth_text: List[String] = ["a", "b", "c", "a", "c", "b"]
    var estimate_text: List[String] = ["b", "b", "a", "a", "a", "b"]
    check("strings", truth_text, estimate_text)
    check("string_target", StringTarget(truth_text^), StringTarget(estimate_text^))

    comptime for i in range(3):
        comptime dtype = DType.int64 if i == 0 else DType.float32 if i == 1 else DType.float64
        var truth: List[Scalar[dtype]] = [0, 1, 2, 0, 2, 1]
        var estimate: List[Scalar[dtype]] = [1, 1, 0, 0, 0, 1]
        check(String(dtype) + "_list", truth, estimate)
        check(String(dtype) + "_array", array[dtype](truth, [6, 1]), array[dtype](estimate, [6]))

    comptime E = FixedPrediction[List[Int]]
    var model = E(predicted^)
    var scorer = Accuracy[E, Int, List[Int], DType.float32]()
    with assert_raises(contains="NotFittedError"):
        _ = scorer.score(model.__sklearn_clone__(), 0, actual, None)
    with assert_raises(contains="inconsistent"):
        _ = scorer.score(model, 0, [0], None)
    with assert_raises(contains="weight"):
        _ = scorer.score(model, 0, actual, Weights(Shape(6, 1)))
    with assert_raises(contains="weight"):
        _ = scorer.score(model, 0, actual, array[DType.float32]([1.0], [1]))
    with assert_raises(contains="finite"):
        _ = scorer.score(model, 0, actual, array[DType.float32]([1.0, 1.0, inf[DType.float32](), 1.0, 1.0, 1.0], [6]))
