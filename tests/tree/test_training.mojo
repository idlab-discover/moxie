import numojo as nm
from std.testing import assert_equal
from moxie._random import RandomState
from moxie.tree._tree import new_tree, Tree
from moxie.tree._records import TrainingData, CriterionConfig, SplitterConfig, BuilderConfig
from moxie.tree._splitter import DenseSplitter
from moxie.tree._builders import DepthFirstTreeBuilder, prune, cost_complexity_pruning_path

def main() raises:
    var X = nm.zeros[DType.float32](nm.Shape(4, 1))
    var y = nm.zeros[DType.float64](nm.Shape(4, 1))
    for i in range(4):
        X.unsafe_set(i, Float32(i))
        y.unsafe_set(i, Float64(i // 2))
    var data = TrainingData(X^, y^, None, nm.zeros[DType.uint8](nm.Shape(1)), nm.full[DType.int64](nm.Shape(1), -1), nm.zeros[DType.int8](nm.Shape(1)))
    var splitter = DenseSplitter[0](CriterionConfig(1, [2], 4), SplitterConfig(1, 1, 0), False)
    var builder = DepthFirstTreeBuilder[0](splitter^, BuilderConfig(2, 1, 0, 10, 0, -1))
    var tree = new_tree(1, [2], [-1])
    var rng = RandomState(0)
    builder.build(tree, data, rng)
    assert_equal(tree.node_count, 3)
    assert_equal(tree.nodes[0].threshold, 1.5)
    var path = cost_complexity_pruning_path(tree)
    assert_equal(path.ccp_alphas.size, 2)
    assert_equal(path.ccp_alphas.unsafe_get(1), 0.5)
    var pruned = prune(tree, 0.5)
    assert_equal(pruned.node_count, 1)
    print("PASS dense Gini construction and pruning")
