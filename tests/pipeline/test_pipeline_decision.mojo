"""Decision dispatch/shape probes; these are not statistical estimators."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import SupervisedEstimator, Predictor, DecisionFunction, ProbabilityPredictor, ClassifierMixin, clone
from moxie.pipeline import Pipeline
from numojo import NDArray, Shape, zeros
from test_sequential import Shift, Recorder, Fused
from test_real_pipeline import CopyStep


struct DecisionProbe(SupervisedEstimator, Predictor, DecisionFunction):
    comptime Input = Int
    comptime Target = Int
    comptime Prediction = Int
    comptime Decision = Int
    comptime Params = Int
    var child: Recorder
    def __init__(out self):
        self.child = Recorder()
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params(mut self, params: Int) raises:
        pass
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return self.child.fitted
    def fit(mut self, X: Int, y: Int) raises:
        self.child.fit(X, y)
    def predict(self, X: Int) raises -> Int:
        raise Error("decision_function must not call predict")
    def decision_function(self, X: Int) raises -> Int:
        if X < 0:
            raise Error("ValueError: decision probe input")
        return X * 10 - self.child.seen


struct DenseDecision[dtype: DType, multiclass: Bool](SupervisedEstimator, Predictor, DecisionFunction):
    comptime Input = NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Prediction = Self.Input
    comptime Decision = Self.Input
    comptime Params = Int
    var fitted: Bool
    def __init__(out self):
        self.fitted = False
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params(mut self, params: Int) raises:
        pass
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted
    def fit(mut self, X: Self.Input, y: Self.Target) raises:
        self.fitted = True
    def predict(self, X: Self.Input) raises -> Self.Prediction:
        raise Error("decision_function must not call predict")
    def decision_function(self, X: Self.Input) raises -> Self.Decision:
        if X.ndim != 2 or X.shape[1] != 3:
            raise Error("ValueError: decision fixture requires three columns")
        comptime if Self.multiclass:
            return X.copy()
        else:
            var result = zeros[Self.dtype](Shape(X.shape[0]))
            for i in range(X.shape[0]):
                result.itemset(i, X.item(i, 0))
            return result^


def through_trait[D: DecisionFunction](pipe: D, X: D.Input) raises -> D.Decision:
    return pipe.decision_function(X)


def dense[dtype: DType, multiclass: Bool]() raises:
    var X = zeros[dtype](Shape(3, 3))
    var y = zeros[dtype](Shape(3))
    for i in range(9):
        X.itemset(i, Scalar[dtype](i - 3))
    var pipe = Pipeline(CopyStep[dtype](), DenseDecision[dtype, multiclass]())
    with assert_raises():
        _ = pipe.decision_function(X)
    pipe.fit(X, y)
    var result = through_trait(pipe, X)
    comptime assert type_of(result) == NDArray[dtype]
    assert_equal(result.ndim, 2 if multiclass else 1)
    assert_equal(result.shape[0], 3)
    comptime if multiclass:
        assert_equal(result.shape[1], 3)
    print("decision_dense", Int(dtype == DType.float64), Int(multiclass), result.ndim, end="")
    for i in range(result.size):
        assert_equal(Float64(result.item(i)), Float64(i - 3 if multiclass else i * 3 - 3))
        print(" " + String(Float64(result.item(i))), end="")
    print()
    result.itemset(0, 100)
    assert_equal(Float64(X.item(0)), -3.0)
    assert_equal(Float64(pipe.decision_function(X).item(0)), -3.0)
    var fresh = clone(pipe)
    with assert_raises():
        _ = fresh.decision_function(X)
    var bad = zeros[dtype](Shape(3, 2))
    with assert_raises():
        _ = pipe.decision_function(bad)
    var empty = zeros[dtype](Shape(0, 3))
    assert_equal(pipe.decision_function(empty).shape[0], 0)


def main() raises:
    comptime assert not conforms_to(Pipeline[Shift, Recorder], DecisionFunction)
    comptime assert conforms_to(Pipeline[Shift, DecisionProbe], DecisionFunction)
    comptime assert not conforms_to(Pipeline[Shift, DecisionProbe], ProbabilityPredictor)
    comptime assert not conforms_to(Pipeline[Shift, DecisionProbe], ClassifierMixin)
    var pipe = Pipeline(Shift(), DecisionProbe())
    with assert_raises():
        _ = pipe.decision_function(1)
    pipe.fit(5, 7)
    assert_equal(through_trait(pipe, 1), 23)
    print("decision_scalar", pipe.decision_function(1))
    with assert_raises():
        _ = pipe.decision_function(-10)
    var fresh = clone(pipe)
    with assert_raises():
        _ = fresh.decision_function(1)
    var params = fresh.get_params()
    params.transformer = 4
    fresh.set_params(params)
    fresh.fit(5, 7)
    pipe.fit(8, 7)
    print("decision_refit", fresh.decision_function(1), pipe.decision_function(1))
    with assert_raises():
        pipe.fit(-1, 7)
    assert_true(pipe.__sklearn_is_fitted__())
    with assert_raises():
        _ = pipe.decision_function(1)
    with assert_raises():
        pipe.fit(5, -1)
    assert_false(pipe.__sklearn_is_fitted__())
    with assert_raises():
        _ = pipe.decision_function(1)
    var fused = Pipeline(Fused(), DecisionProbe())
    fused.fit(5, 7)
    print("decision_fused", fused.decision_function(1))
    assert_equal(fused.transformer.calls, 1)
    var nested = Pipeline(Shift(2), Pipeline(Shift(4), DecisionProbe()))
    nested.fit(5, 7)
    print("decision_nested", nested.decision_function(1))
    dense[DType.float32, False]()
    dense[DType.float32, True]()
    dense[DType.float64, False]()
    dense[DType.float64, True]()
