"""Prepare two output columns; this example does not train an estimator."""
import numojo as nm
from moxie._multioutput import target_column, stack_predictions, ChainOrder

def main() raises:
    var targets = nm.zeros[DType.float64](nm.Shape(2, 2))
    targets.unsafe_set(0, 10)
    targets.unsafe_set(3, 20)
    var order = ChainOrder(List[Int]([1, 0]), 2)
    var columns = List[nm.NDArray[DType.float64]]()
    for i in range(2):
        columns.append(target_column(targets, order.column(i)))
    var predictions = stack_predictions(columns)
    print(predictions.shape[0], predictions.shape[1])
