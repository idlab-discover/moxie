"""Classifier response selection, output order and CV response validation."""
import numojo as nm
from std.math import log
from std.testing import assert_equal, assert_raises
from moxie.base import SupervisedEstimator, Predictor, ProbabilityPredictor, LogProbabilityPredictor, DecisionFunction
from moxie.multioutput import ClassifierChain, ChainResponses
from test_estimators import Nearest, Params


struct Responses(SupervisedEstimator, Predictor, ProbabilityPredictor, LogProbabilityPredictor, DecisionFunction):
    comptime Input = nm.NDArray[DType.float64]
    comptime Target = nm.NDArray[DType.int64]
    comptime Prediction = Self.Target
    comptime Probabilities = Self.Input
    comptime Decision = Self.Input
    comptime Params = Params
    var model: Nearest[DType.int64]

    def __init__(out self) raises:
        self.model = Nearest[DType.int64]()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.model.get_params(deep)

    def set_params[origin: MutOrigin, p_origin: ImmOrigin](ref[origin] self, ref[p_origin] params: Self.Params) raises -> ref[self] Self:
        _ = self.model.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return self.model.fitted

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        _ = self.model.fit(X, y)
        return self

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        return self.model.predict(X)

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        var labels = self.predict(X)
        var result = nm.zeros[DType.float64](nm.Shape(X.shape[0], 2))
        for i in range(X.shape[0]):
            var positive = 0.25 + 0.5 * Float64(labels.unsafe_get(i))
            result.unsafe_set(2 * i, 1 - positive)
            result.unsafe_set(2 * i + 1, positive)
        return result^

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        var result = self.predict_proba(X)
        for i in range(2 * X.shape[0]):
            result.unsafe_set(i, log(result.unsafe_get(i)))
        return result^

    def decision_function(self, X: Self.Input) raises -> Self.Decision:
        var labels = self.predict(X)
        var result = nm.zeros[DType.float64](nm.Shape(X.shape[0]))
        for i in range(X.shape[0]):
            result.unsafe_set(i, 2 * Float64(labels.unsafe_get(i)) - 1)
        return result^


@fieldwise_init
struct RejectCV(ChainResponses):
    comptime Estimator = Responses
    comptime Input = Responses.Input
    comptime Target = Responses.Target
    comptime Response = Responses.Input
    def cross_val_response(self, estimator: Self.Estimator, X: Self.Input, y: Self.Target, method: String) raises -> Self.Response:
        raise Error("Final link must not request a CV response")


@fieldwise_init
struct BadShapeCV(ChainResponses):
    comptime Estimator = Responses
    comptime Input = Responses.Input
    comptime Target = Responses.Target
    comptime Response = Responses.Input
    def cross_val_response(self, estimator: Self.Estimator, X: Self.Input, y: Self.Target, method: String) raises -> Self.Response:
        return nm.zeros[DType.float64](nm.Shape(X.shape[0], 1))


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(4, 1))
    var labels = nm.zeros[DType.int64](nm.Shape(4, 2))
    for i in range(4):
        X.unsafe_set(i, Float64(10 * i))
        labels.unsafe_set(2 * i, Int64(i % 2))
        labels.unsafe_set(2 * i + 1, Int64(i // 2))
    var methods: List[String] = ["predict", "predict_proba", "predict_log_proba", "decision_function"]
    for method in methods:
        var chain = ClassifierChain(Responses(), order=List[Int]([1, 0]), chain_method=method.copy())
        var prediction = chain.fit(X, labels).predict(X)
        var probabilities = chain.predict_proba(X)
        var decisions = chain.decision_function(X)
        for i in range(8):
            assert_equal(prediction.unsafe_get(i), labels.unsafe_get(i))
            assert_equal(probabilities.unsafe_get(i), 0.25 + 0.5 * Float64(labels.unsafe_get(i)))
            assert_equal(decisions.unsafe_get(i), 2 * Float64(labels.unsafe_get(i)) - 1)
        with assert_raises():
            _ = chain.fit(X, labels, cv=BadShapeCV())
        assert_equal(chain.score(X, labels), 1.0)
    var single = nm.zeros[DType.int64](nm.Shape(4, 1))
    var chain = ClassifierChain(Responses())
    _ = chain.fit(X, single, cv=RejectCV())
    assert_equal(chain.predict(X).shape[1], 1)
    print("PASS: response methods, original target order, CV shape and final-link contract")
