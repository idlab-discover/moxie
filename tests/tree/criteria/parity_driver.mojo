import numojo as nm
from moxie.tree._criterion import WeightedCriterion
from moxie.tree._records import CriterionConfig, TrainingData

def cases[kind: Int]() raises:
    for scenario in range(12):
        var n = 4 + 2 * scenario
        var outputs = 1 + scenario % 3
        var nc = List[Int]()
        var width = 1
        for k in range(outputs):
            var count = 2 + k % 2 if kind < 2 else 1
            nc.append(count)
            width = max(width, count)
        var y = nm.zeros[DType.float64](nm.Shape(n, outputs))
        var weights = nm.zeros[DType.float64](nm.Shape(n))
        var samples = nm.zeros[DType.int64](nm.Shape(n))
        var global_weight = Float64(0)
        for i in range(n):
            for k in range(outputs):
                var v = Float64((i * 7 + k * 3 + scenario) % 19) / 4
                comptime if kind < 2:
                    v = Float64((i * 7 + k + scenario) % nc[k])
                y.unsafe_set(i * outputs + k, v)
            var w = Float64(1 + i % 4)
            if scenario % 4 == 1 and i % 3 == 0:
                w = 0
            if scenario == 8 and i % 7 == 0:
                w = -0.25
            weights.unsafe_set(i, w)
            global_weight += w
            samples.unsafe_set(i, Int64(n - 1 - i))
        var data = TrainingData(nm.zeros[DType.float32](nm.Shape(n, 1)), y^, weights^,
            nm.zeros[DType.uint8](nm.Shape(1)), nm.full[DType.int64](nm.Shape(1), -1),
            nm.zeros[DType.int8](nm.Shape(1)))
        var c = WeightedCriterion[kind](CriterionConfig(outputs, nc^, n))
        var start = 1 if scenario % 2 else 0
        var end = n - 1 if scenario % 3 else n
        c.initialize(data, samples, start, end, global_weight)
        var dest = nm.zeros[DType.float64](nm.Shape(outputs, width))
        c.node_value(dest)
        var parent = c.node_impurity()
        print("NODE", kind, scenario, parent)
        for j in range(dest.size):
            print("VALUE", kind, scenario, j, dest.unsafe_get(j))
        for pos in range(start + 1, end):
            c.update(pos)
            if c.weighted_n_left <= 0 or c.weighted_n_right <= 0:
                continue
            var child = c.children_impurity()
            print("SPLIT", kind, scenario, pos, c.weighted_n_left, c.weighted_n_right,
                child.left, child.right, c.proxy_impurity_improvement(),
                c.impurity_improvement(parent, child.left, child.right))

def main() raises:
    comptime for kind in range(5):
        cases[kind]()
