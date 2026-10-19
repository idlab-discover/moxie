import numojo as nm
from moxie._dense import row_squared_norms
from moxie._multioutput import target_column, stack_predictions
from std.testing import assert_equal

@fieldwise_init
struct ConsumerB:
    def run(self) raises:
        var x = nm.zeros[DType.float64](nm.Shape(2, 1))
        x.unsafe_set(0, 3)
        x.unsafe_set(1, 4)
        var shared = x.view()
        var columns = List[nm.NDArray[DType.float64]]()
        columns.append(target_column(shared, 0))
        var stacked = stack_predictions(columns)
        var result = row_squared_norms(stacked)
        assert_equal(Float64(result.unsafe_get(0)), 9.0)
        assert_equal(Float64(result.unsafe_get(1)), 16.0)
