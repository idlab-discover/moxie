"""Estimator-shaped ownership probe, not an SVM implementation."""
import numojo as nm
from moxie.base import SupervisedEstimator
from moxie.utils.validation import FittedState
from moxie.svm._contracts import Dense, TrainingSnapshot, validate_vector
from moxie._dense import row_squared_norms

@fieldwise_init
struct ProbeParams(Copyable, Movable):
    var marker: Int

struct ConsumerB(SupervisedEstimator):
    comptime Params = ProbeParams
    comptime Input = Dense
    comptime Target = Dense
    var marker: Int
    var state: FittedState
    var norms: Dense

    def __init__(out self, marker: Int = 1) raises:
        self.marker = marker
        self.state = FittedState()
        self.norms = nm.zeros[DType.float64](nm.Shape(0))

    def get_params(self, deep: Bool = True) raises -> ProbeParams:
        return ProbeParams(self.marker)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: ProbeParams) raises -> ref[self] Self:
        self.marker = params.marker
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.marker)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.state.is_fitted()

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Dense, ref[y_origin] y: Dense) raises -> ref[self] Self:
        var snapshot = TrainingSnapshot(X)
        validate_vector(y, snapshot.n_train)
        var norms = row_squared_norms(snapshot.x)
        self.state.record_fit(snapshot.n_train, snapshot.n_features, n_targets=y.shape[0])
        self.norms = norms^
        return self
