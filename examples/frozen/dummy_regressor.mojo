"""Real mean-only DummyRegressor for the frozen example, not the dummy family.

Dense single-feature Float64 inputs and single-output finite targets only.
No sample weights or alternative strategies are implemented in this example.
"""
from std.math import isfinite
from moxie.base import RegressorMixin
from moxie.utils.validation import check_is_fitted


@fieldwise_init
struct DummyParams(Copyable, Movable):
    var strategy: String


struct DummyRegressor(Copyable, RegressorMixin):
    comptime Params = DummyParams
    comptime Input = List[Float64]
    comptime Target = List[Float64]
    comptime Prediction = List[Float64]
    comptime SampleWeight = List[Float64]
    var strategy: String
    var constant_: Optional[Float64]

    def __init__(out self, *, var strategy: String = "mean"):
        self.strategy = strategy^
        self.constant_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.strategy)

    def set_params(mut self, params: Self.Params) raises:
        self.strategy = params.strategy

    def __sklearn_clone__(self) raises -> Self:
        return Self(strategy=self.strategy)

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.constant_)

    def fit(mut self, X: Self.Input, y: Self.Target) raises:
        if self.strategy != "mean":
            raise Error(
                "ValueError: this example supports only strategy='mean'"
            )
        if len(y) == 0 or len(X) != len(y):
            raise Error("ValueError: nonempty targets must match input length")
        var mean = Float64(0)
        for value in y:
            if not isfinite(value):
                raise Error("ValueError: targets must be finite")
            mean += value / Float64(len(y))
        if not isfinite(mean):
            raise Error("ValueError: mean exceeds the supported numeric range")
        self.constant_ = mean

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        check_is_fitted(self)
        return List[Float64](fill=self.constant_.value(), length=len(X))

    def score(
        self,
        X: Self.Input,
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        check_is_fitted(self)
        if sample_weight:
            raise Error(
                "ValueError: this example does not support sample weights"
            )
        if len(y) == 0 or len(X) != len(y):
            raise Error("ValueError: nonempty targets must match input length")
        var mean = Float64(0)
        for value in y:
            if not isfinite(value):
                raise Error("ValueError: targets must be finite")
            mean += value / Float64(len(y))
        if len(y) < 2:
            return Float64("nan")
        var numerator = Float64(0)
        var denominator = Float64(0)
        for value in y:
            numerator += (value - self.constant_.value()) ** 2
            denominator += (value - mean) ** 2
        if numerator == 0:
            return 1.0
        if denominator == 0:
            return 0.0
        return 1.0 - numerator / denominator
