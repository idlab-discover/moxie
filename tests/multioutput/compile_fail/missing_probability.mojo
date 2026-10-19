"""A component without probability capability cannot expose wrapper probabilities."""
from moxie.base import SupervisedEstimator, Predictor
from moxie.multioutput import MultiOutputClassifier
from numojo import NDArray, Shape
import numojo as nm

@fieldwise_init
struct Params(Copyable, Movable):
    pass

@fieldwise_init
struct LabelsOnly(SupervisedEstimator, Predictor):
    comptime Params = Params
    comptime Input = NDArray[DType.float64]
    comptime Target = NDArray[DType.int64]
    comptime Prediction = Self.Target

    def get_params(self, deep: Bool = True) -> Self.Params:
        return Params()
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        pass
        return self

    def __sklearn_clone__(self) -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return False
    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        raise Error("Compile-only component")

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        raise Error("Compile-only component")

def main() raises:
    var model = MultiOutputClassifier(LabelsOnly())
    var X = nm.zeros[DType.float64](Shape(2, 1))
    var probabilities = model.predict_proba(X)
