"""Known-mask estimator probe; no statistical selection algorithm."""
from moxie.feature_selection import SelectionState, SupportResult, FeatureSelector
import numojo as nm

struct ConsumerB(FeatureSelector):
    comptime Input = nm.NDArray[DType.float64]
    comptime Transformed = Self.Input
    comptime Support = SupportResult
    comptime Params = List[Bool]
    var state: SelectionState
    var mask: List[Bool]

    def __init__(out self, mask: List[Bool] = [True, False, True]):
        self.state = SelectionState()
        self.mask = mask.copy()

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] x: Self.Input) raises -> ref[self] Self:
        if x.ndim != 2 or x.shape[1] != len(self.mask):
            raise Error("ValueError: probe shape mismatch")
        self.state.commit(self.mask)
        return self

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.mask.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.mask = params.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.mask)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.state.is_fitted()

    def get_support(self, indices: Bool = False) raises -> Self.Support:
        return self.state.get_support(indices)

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        return self.state.transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        return self.state.inverse_transform(X)
