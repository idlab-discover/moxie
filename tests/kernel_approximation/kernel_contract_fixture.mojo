"""A contract demonstration, not an implementation of a sklearn algorithm."""
from moxie.base import RegressorMixin, clone
from moxie.utils.validation import FittedState, check_is_fitted


@fieldwise_init
struct DemoParams(Copyable, Movable):
    var offset: Float64


struct DemoRegressor(RegressorMixin):
    """Identity prediction plus an offset; fitting only validates this example.
    """

    comptime Params = DemoParams
    comptime Input = List[Float64]
    comptime Target = List[Float64]
    comptime Prediction = List[Float64]
    comptime SampleWeight = List[Float64]
    var offset: Float64
    var _state: FittedState

    def __init__(out self, *, offset: Float64 = 0.0):
        self.offset = offset
        self._state = FittedState()

    def get_params(self, deep: Bool = True) raises -> DemoParams:
        return DemoParams(self.offset)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: DemoParams) raises -> ref[self] Self:
        self.offset = params.offset
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(offset=self.offset)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.is_fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self._state.record_fit(len(X), 1, n_targets=len(y))
        return self

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        check_is_fitted(self)
        var result = List[Float64]()
        for value in X:
            result.append(value + self.offset)
        return result^

    def score(
        self,
        X: Self.Input,
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        # Deliberately do not claim a score implementation in a scaffold demo.
        raise Error("DemoRegressor does not implement scoring")


def main() raises:
    var model = DemoRegressor(offset=1.0)
    var X: List[Float64] = [1.0, 2.0, 3.0]
    var y: List[Float64] = [2.0, 3.0, 4.0]
    model.fit(X, y)
    print(model.predict(X))
    var fresh = clone(model)
    print("clone is fitted:", fresh.__sklearn_is_fitted__())
