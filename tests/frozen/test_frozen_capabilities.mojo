"""Synthetic probes check dispatch, not numerical algorithm implementations."""
from std.testing import assert_equal, assert_raises
from moxie.base import (
    BaseEstimator,
    Predictor,
    ProbabilityPredictor,
    LogProbabilityPredictor,
    DecisionFunction,
    SampleScorer,
    DensityMixin,
    ClassifierMixin,
    RegressorMixin,
    OutlierMixin,
    TransformerMixin,
    InverseTransformer,
    FitTransformer,
    ClusterMixin,
    IncrementalClassifier,
    WeightedSupervisedEstimator,
    UnsupervisedEstimator,
)
from moxie.frozen import FrozenEstimator
from test_family import ContractProbe


struct TransformProbe(
    FitTransformer, InverseTransformer, UnsupervisedEstimator
):
    comptime Params = Int
    comptime Input = Int
    comptime Target = Int
    comptime Transformed = Int
    var offset: Int

    def __init__(out self, offset: Int = 7):
        self.offset = offset

    def get_params(self, deep: Bool = True) raises -> Int:
        return self.offset

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        self.offset = params
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.offset)

    def __sklearn_is_fitted__(self) -> Bool:
        return True

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Int,
        y: Optional[Int] = None,
    ) raises -> ref[self] Self:
        self.offset = X
        return self

    def transform(self, X: Int) raises -> Int:
        return X + self.offset

    def inverse_transform(self, X: Int) raises -> Int:
        return X - self.offset

    def fit_transform(mut self, X: Int, y: Optional[Int] = None) raises -> Int:
        _ = self.fit(X, y)
        return self.transform(X)


struct BareProbe(BaseEstimator):
    comptime Params = Int

    def __init__(out self):
        pass

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
        return True


struct ResponseProbe(
    ClassifierMixin,
    DecisionFunction,
    LogProbabilityPredictor,
    WeightedSupervisedEstimator,
):
    comptime Params = Int
    comptime Input = Int
    comptime Target = Int
    comptime Prediction = Int
    comptime Probabilities = List[Float64]
    comptime Decision = Float64
    comptime SampleWeight = Int

    def __init__(out self):
        pass

    def get_params(self, deep: Bool = True) raises -> Int:
        return 0

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        raise Error("probe mutation must not be called")

    def __sklearn_clone__(self) raises -> Self:
        raise Error("inner clone must not be called")

    def __sklearn_is_fitted__(self) -> Bool:
        return True

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Int,
        ref[y_origin] y: Int,
    ) raises -> ref[self] Self:
        raise Error("inner fit must not be called")

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Int,
        ref[y_origin] y: Int,
        *,
        ref[sample_weight_origin] sample_weight: Int,
    ) raises -> ref[self] Self:
        raise Error("inner weighted fit must not be called")

    def predict(self, X: Int) raises -> Int:
        return X + 10

    def predict_proba(self, X: Int) raises -> List[Float64]:
        return [0.25, 0.75]

    def predict_log_proba(self, X: Int) raises -> List[Float64]:
        return [-1.3862943611198906, -0.2876820724517809]

    def decision_function(self, X: Int) raises -> Float64:
        return Float64(X) - 10.0

    def score(
        self, X: Int, y: Int, sample_weight: Optional[Int] = None
    ) raises -> Float64:
        return Float64(X + y + (sample_weight.value() if sample_weight else 0))


struct DensityProbe(DensityMixin):
    comptime Params = Int
    comptime Input = Int
    comptime Target = String
    comptime SampleScores = List[Float64]

    def __init__(out self):
        pass

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
        return True

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Int,
        y: Optional[String] = None,
    ) raises -> ref[self] Self:
        raise Error("inner fit must not be called")

    def score_samples(self, X: Int) raises -> List[Float64]:
        return [Float64(X), -2.0]

    def score(self, X: Int, y: Optional[String] = None) raises -> Float64:
        return Float64(X - 2 + (y.value().byte_length() if y else 0))


def transformed[
    E: TransformerMixin
](model: E, X: E.Input) raises -> E.Transformed:
    return model.transform(X)


def main() raises:
    comptime T = FrozenEstimator[TransformProbe]
    comptime assert conforms_to(T, TransformerMixin)
    comptime assert conforms_to(T, InverseTransformer)
    comptime assert conforms_to(T, UnsupervisedEstimator)
    comptime assert not conforms_to(T, Predictor)
    comptime assert not conforms_to(T, FitTransformer)
    var transform = T(TransformProbe())
    _ = transform.fit(100)
    _ = transform.fit(200, Optional[Int](5))
    assert_equal(transformed(transform, 3), 10)
    assert_equal(transform.inverse_transform(10), 3)

    var responses = FrozenEstimator(ResponseProbe())
    _ = responses.fit(1, 2)
    _ = responses.fit(1, 2, sample_weight=-100)
    var cloned_responses = responses.__sklearn_clone__()
    assert_equal(cloned_responses.predict(2), 12)
    assert_equal(
        responses.predict_proba(3), responses.estimator().predict_proba(3)
    )
    assert_equal(
        responses.predict_log_proba(3),
        responses.estimator().predict_log_proba(3),
    )
    assert_equal(responses.decision_function(3), -7.0)
    assert_equal(responses.score(3, 4), 7.0)
    assert_equal(responses.score(3, 4, Optional[Int](5)), 12.0)
    var density = FrozenEstimator(DensityProbe())
    _ = density.fit(100)
    assert_equal(density.score(5), 3.0)
    assert_equal(density.score(5, Optional[String]("abc")), 6.0)
    assert_equal(density.score_samples(5), density.estimator().score_samples(5))

    # BaseEstimator alone imposes no input or inference capability.
    var bare = FrozenEstimator(BareProbe())
    _ = bare.fit("ignored", 1)
    comptime assert not conforms_to(FrozenEstimator[BareProbe], Predictor)

    # This pre-existing probe raises from every algorithm operation. Every
    # frozen inference route must preserve that error rather than fabricate
    # results. Composite training remains unavailable; incremental errors propagate.
    comptime F = FrozenEstimator[ContractProbe]
    comptime assert conforms_to(F, ClassifierMixin)
    comptime assert conforms_to(F, RegressorMixin)
    comptime assert conforms_to(F, ProbabilityPredictor)
    comptime assert conforms_to(F, LogProbabilityPredictor)
    comptime assert conforms_to(F, DecisionFunction)
    comptime assert conforms_to(F, SampleScorer)
    comptime assert conforms_to(F, DensityMixin)
    comptime assert conforms_to(F, OutlierMixin)
    comptime assert conforms_to(F, WeightedSupervisedEstimator)
    comptime assert not conforms_to(F, FitTransformer)
    comptime assert not conforms_to(F, ClusterMixin)
    comptime assert conforms_to(F, IncrementalClassifier)
    var frozen = F(ContractProbe())
    with assert_raises():
        _ = frozen.predict(1)
    with assert_raises():
        _ = frozen.predict_proba(1)
    with assert_raises():
        _ = frozen.predict_log_proba(1)
    with assert_raises():
        _ = frozen.decision_function(1)
    with assert_raises():
        _ = frozen.score_samples(1)
    with assert_raises():
        _ = frozen.score(1, 2, Optional[Int](3))
    with assert_raises():
        _ = frozen.score(1)
    with assert_raises():
        _ = frozen.fit(1, 2, sample_weight=3)
    print("Frozen capability contracts passed")
