"""Sample-score forwarding probes, not density/outlier estimators."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import SupervisedEstimator, Predictor, SampleScorer, SupervisedScorer, DensityMixin, clone
from moxie.pipeline import Pipeline
from numojo import NDArray, Shape, zeros
from test_sequential import Shift, Recorder, Fused
from test_real_pipeline import CopyStep
from test_pipeline_decision import DenseDecision


struct SampleProbe(SupervisedEstimator, Predictor, SampleScorer):
    comptime Input = Int
    comptime Target = Int
    comptime Prediction = Int
    comptime SampleScores = Int
    comptime Params = Int
    var child: Recorder
    def __init__(out self):
        self.child = Recorder()
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return self.child.fitted
    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Int,
        ref[y_origin] y: Int,
    ) raises -> ref[self] Self:
        _ = self.child.fit(X, y)
        return self
    def predict(self, X: Int) raises -> Int:
        raise Error("score_samples must not call predict")
    def score_samples(self, X: Int) raises -> Int:
        if X < 0:
            raise Error("ValueError: sample probe input")
        return self.child.seen - X * 10


struct DenseSampleProbe[dtype: DType](SupervisedEstimator, Predictor, SampleScorer):
    comptime Input = NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Prediction = Self.Input
    comptime SampleScores = Self.Input
    comptime Params = Int
    var fixture: DenseDecision[Self.dtype, False]
    def __init__(out self):
        self.fixture = DenseDecision[Self.dtype, False]()
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fixture.fitted
    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        _ = self.fixture.fit(X, y)
        return self
    def predict(self, X: Self.Input) raises -> Self.Prediction:
        raise Error("score_samples must not call predict")
    def score_samples(self, X: Self.Input) raises -> Self.SampleScores:
        # Reuse the test fixture's column extraction, not a statistical kernel.
        return self.fixture.decision_function(X)


def through_trait[S: SampleScorer](pipe: S, X: S.Input) raises -> S.SampleScores:
    return pipe.score_samples(X)


def dense[dtype: DType]() raises:
    var X = zeros[dtype](Shape(3, 3))
    var y = zeros[dtype](Shape(3))
    for i in range(9):
        X.itemset(i, Scalar[dtype](i - 3))
    var pipe = Pipeline(CopyStep[dtype](), DenseSampleProbe[dtype]())
    with assert_raises():
        _ = pipe.score_samples(X)
    _ = pipe.fit(X, y)
    var scores = through_trait(pipe, X)
    comptime assert type_of(scores) == NDArray[dtype]
    assert_equal(scores.ndim, 1)
    assert_equal(scores.shape[0], 3)
    for i in range(3):
        assert_equal(Float64(scores.item(i)), Float64(i * 3 - 3))
    print("samples_dense", Int(dtype == DType.float64), scores.item(0), scores.item(1), scores.item(2))
    scores.itemset(0, 100)
    assert_equal(Float64(X.item(0)), -3.0)
    assert_equal(Float64(pipe.score_samples(X).item(0)), -3.0)
    var fresh = clone(pipe)
    with assert_raises():
        _ = fresh.score_samples(X)
    var bad = zeros[dtype](Shape(3, 2))
    with assert_raises():
        _ = pipe.score_samples(bad)
    var empty = zeros[dtype](Shape(0, 3))
    assert_equal(pipe.score_samples(empty).shape[0], 0)


def main() raises:
    comptime assert not conforms_to(Pipeline[Shift, Recorder], SampleScorer)
    comptime assert conforms_to(Pipeline[Shift, SampleProbe], SampleScorer)
    comptime assert not conforms_to(Pipeline[Shift, SampleProbe], SupervisedScorer)
    comptime assert not conforms_to(Pipeline[Shift, SampleProbe], DensityMixin)
    var pipe = Pipeline(Shift(), SampleProbe())
    with assert_raises():
        _ = pipe.score_samples(1)
    _ = pipe.fit(5, 7)
    assert_equal(through_trait(pipe, 1), -23)
    print("samples_scalar", pipe.score_samples(1))
    with assert_raises():
        _ = pipe.score_samples(-10)
    var fresh = clone(pipe)
    with assert_raises():
        _ = fresh.score_samples(1)
    var params = fresh.get_params()
    params.transformer = 4
    _ = fresh.set_params(params)
    _ = fresh.fit(5, 7)
    _ = pipe.fit(8, 7)
    print("samples_refit", fresh.score_samples(1), pipe.score_samples(1))
    with assert_raises():
        _ = pipe.fit(-1, 7)
    assert_true(pipe.__sklearn_is_fitted__())
    with assert_raises():
        _ = pipe.score_samples(1)
    with assert_raises():
        _ = pipe.fit(5, -1)
    assert_false(pipe.__sklearn_is_fitted__())
    with assert_raises():
        _ = pipe.score_samples(1)
    var fused = Pipeline(Fused(), SampleProbe())
    _ = fused.fit(5, 7)
    print("samples_fused", fused.score_samples(1))
    assert_equal(fused.transformer.calls, 1)
    var nested = Pipeline(Shift(2), Pipeline(Shift(4), SampleProbe()))
    _ = nested.fit(5, 7)
    print("samples_nested", nested.score_samples(1))
    dense[DType.float32]()
    dense[DType.float64]()
