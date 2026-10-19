"""Independent consumer a; validates shared plumbing, not an estimator."""
from numojo import Shape, Item, full
from moxie._dense import row_squared_norms
from moxie._random import RandomState
from moxie.tree._records import Values, snapshot_values, value_offset
from std.testing import assert_equal, assert_false, assert_raises
from moxie.base import BaseEstimator, clone
from moxie.utils import FittedState
from moxie.tree._dense_api import DenseTreeInspection

struct ConsumerA(DenseTreeInspection):
    comptime Input = Values
    comptime Leaves = Int
    comptime Params = Int
    var setting: Int
    var state: FittedState
    var snapshot: Optional[Values]
    def __init__(out self, setting: Int = 1):
        self.setting = setting
        self.state = FittedState()
        self.snapshot = None
    def get_params(self, deep: Bool = True) raises -> Int:
        return self.setting
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Int) raises -> ref[self] Self:
        self.setting = params
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self(self.setting)
    def __sklearn_is_fitted__(self) -> Bool:
        return self.state.is_fitted()
    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Values) raises -> ref[self] Self:
        if X.ndim != 2:
            raise Error("ValueError: expected matrix")
        var state = FittedState()
        state.record_fit(X.shape[0], X.shape[1])
        self.snapshot = snapshot_values(X)
        self.state = state^
        return self
    def apply(self, X: Values, check_input: Bool = True) raises -> Int:
        self.state.check_features(X.shape[1])
        raise Error("Scaffold: traversal not implemented")

def consumer_a() raises -> Float64:
    var values = full[DType.float64](Shape(2, 2), 2.0)
    var estimator = ConsumerA()
    with assert_raises():
        _ = estimator.apply(values)
    _ = estimator.fit(values)
    var unfitted = clone(estimator)
    assert_false(unfitted.__sklearn_is_fitted__())
    assert_equal(unfitted.get_params(), 1)
    _ = estimator.set_params(3)
    assert_equal(unfitted.get_params(), 1)
    var copy = snapshot_values(values)
    values[Item(0, 0)] = 9.0
    assert_equal(copy.item(0, 0), 2.0)
    assert_equal(estimator.snapshot.value().item(0, 0), 2.0)
    _ = estimator.fit(values)
    assert_equal(estimator.snapshot.value().item(0, 0), 9.0)
    with assert_raises():
        _ = estimator.apply(values)
    var norms = row_squared_norms(copy)
    assert_equal(norms.item(0), 8.0)
    assert_equal(value_offset(1, 0, 1, 2, 1, 2), 3)
    var rng = RandomState(42)
    assert_equal(rng.next_uint32(), UInt32(1608637542))
    return norms.item(0)
