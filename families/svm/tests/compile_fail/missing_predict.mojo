from moxie.base import Predictor


# Must fail: declaring Predictor without predict cannot satisfy the contract.
struct IncompletePredictor(Predictor):
    comptime Params = Int
    comptime Input = Int
    comptime Prediction = Int
    var marker: Int

    def __init__(out self):
        self.marker = 0

    def get_params(self, deep: Bool = True) raises -> Int:
        return 0

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Int) raises -> ref[self] Self:
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self()

    def __sklearn_is_fitted__(self) -> Bool:
        return False


def main():
    var incomplete = IncompletePredictor()
