"""Evaluate a weighted regression split without fitting a tree."""
import numojo as nm
from moxie.tree._criterion import MSE
from moxie.tree._records import CriterionConfig, TrainingData

def main() raises:
    var y = nm.zeros[DType.float64](nm.Shape(4, 1))
    var samples = nm.zeros[DType.int64](nm.Shape(4))
    for i in range(4):
        y.unsafe_set(i, Float64(i))
        samples.unsafe_set(i, Int64(i))
    var data = TrainingData(nm.zeros[DType.float32](nm.Shape(4, 1)), y^,
        nm.full[DType.float64](nm.Shape(4), 2), nm.zeros[DType.uint8](nm.Shape(1)),
        nm.full[DType.int64](nm.Shape(1), -1), nm.zeros[DType.int8](nm.Shape(1)))
    var criterion = MSE(CriterionConfig(1, [1], 4))
    criterion.initialize(data, samples, 0, 4, 8)
    criterion.update(2)
    var children = criterion.children_impurity()
    print("Parent impurity:", criterion.node_impurity())
    print("Child impurities:", children.left, children.right)
    print("Improvement:", criterion.impurity_improvement(criterion.node_impurity(), children.left, children.right))
