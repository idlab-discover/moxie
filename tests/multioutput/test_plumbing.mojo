import numojo as nm
from std.testing import assert_equal, assert_false, assert_raises
from moxie.base import BaseEstimator
from moxie._multioutput import validate_targets, clone_outputs, ChainOrder, target_column, stack_predictions, augment_features, ChainResponses

struct CloneProbe(BaseEstimator):
    comptime Params = List[Int]
    var params: List[Int]
    var fitted: Bool
    def __init__(out self):
        self.params = List[Int]([7])
        self.fitted = True
    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.params.copy()
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.params = params.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        result.fitted = False
        return result^
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted

@fieldwise_init
struct CVProbe(ChainResponses):
    comptime Estimator = CloneProbe
    comptime Input = nm.NDArray[DType.float64]
    comptime Target = nm.NDArray[DType.float64]
    comptime Response = nm.NDArray[DType.float64]
    def cross_val_response(self, estimator: Self.Estimator, X: Self.Input,
                          y: Self.Target, method: String) raises -> Self.Response:
        raise Error("CV probe: engine belongs to model_selection.validation")

def arrays[dtype: DType]() raises:
    var y = nm.zeros[dtype](nm.Shape(3, 2))
    for i in range(6):
        y.unsafe_set(i, SIMD[dtype, 1](i + 1))
    validate_targets(y, 3)
    var reversed = y.view_with_layout(nm.Shape(3, 2), nm.NDArrayStrides(-2, 1), 4)
    var col = target_column(reversed, 1)
    assert_equal(Float64(col.unsafe_get(0)), 6.0)
    assert_equal(Float64(col.unsafe_get(2)), 2.0)
    var columns = List[nm.NDArray[dtype]]()
    columns.append(col.copy())
    columns.append(target_column(y, 0))
    var stacked = stack_predictions(columns)
    assert_equal(stacked.shape[0], 3)
    assert_equal(stacked.shape[1], 2)
    assert_equal(Float64(stacked.unsafe_get(1)), 1.0)
    columns[0].unsafe_set(0, 99)
    assert_equal(Float64(stacked.unsafe_get(0)), 6.0)
    var aug = augment_features(y, stacked, 1)
    assert_equal(aug.shape[1], 3)
    assert_equal(Float64(aug.unsafe_get(2)), 6.0)
    var unchanged = augment_features(reversed, stacked, 0)
    assert_equal(Float64(unchanged.unsafe_get(0)), 5.0)
    y.unsafe_set(1, 99)
    assert_equal(Float64(col.unsafe_get(2)), 2.0)
    assert_equal(Float64(aug.unsafe_get(1)), 2.0)
    with assert_raises():
        validate_targets(col, 3)
    with assert_raises():
        validate_targets(y, 2)
    with assert_raises():
        _ = target_column(y, -1)
    with assert_raises():
        _ = target_column(col, 0)
    with assert_raises():
        _ = augment_features(y, stacked, 3)
    with assert_raises():
        _ = stack_predictions(List[nm.NDArray[dtype]]())
    var invalid = List[nm.NDArray[dtype]]()
    invalid.append(y.copy())
    with assert_raises():
        _ = stack_predictions(invalid)
    var zero = nm.zeros[dtype](nm.Shape(0, 2))
    with assert_raises():
        validate_targets(zero, 0)
    var empty = target_column(zero, 0)
    var empties = List[nm.NDArray[dtype]]()
    empties.append(empty^)
    assert_equal(stack_predictions(empties).shape[0], 0)
    var short = nm.zeros[dtype](nm.Shape(2))
    columns.append(short^)
    with assert_raises():
        _ = stack_predictions(columns)

def main() raises:
    arrays[DType.float32]()
    arrays[DType.float64]()
    arrays[DType.int64]()
    var estimator = CloneProbe()
    var clones = clone_outputs(estimator, 2)
    assert_false(clones[0].fitted)
    clones[0].params[0] = 9
    assert_equal(clones[1].params[0], 7)
    assert_equal(estimator.params[0], 7)
    with assert_raises():
        _ = clone_outputs(estimator, 0)
    var order = List[Int]([2, 0, 1])
    var mapping = ChainOrder(order, 3)
    order[0] = 0
    assert_equal(mapping.column(0), 2)
    assert_equal(mapping.position(0), 1)
    for i in range(3):
        assert_equal(mapping.column(mapping.position(i)), i)
    with assert_raises():
        _ = ChainOrder(List[Int]([0, 0]), 2)
    with assert_raises():
        _ = ChainOrder(List[Int]([0, 2]), 2)
    with assert_raises():
        _ = ChainOrder(List[Int]([-1]), 1)
    with assert_raises():
        _ = ChainOrder(List[Int]([0]), 2)
    with assert_raises():
        _ = mapping.column(3)
    with assert_raises():
        _ = mapping.position(-1)
    var large_labels = nm.zeros[DType.int64](nm.Shape(1, 1))
    large_labels.unsafe_set(0, Int64(9007199254740993))
    var exact_labels = target_column(large_labels, 0)
    assert_equal(Int64(exact_labels.unsafe_get(0)), Int64(9007199254740993))
    var provider = CVProbe()
    var x = nm.zeros[DType.float64](nm.Shape(1, 1))
    var y = nm.zeros[DType.float64](nm.Shape(1))
    with assert_raises():
        _ = provider.cross_val_response(estimator, x, y, "predict")
    print("PASS: multioutput layout and ownership contracts")
