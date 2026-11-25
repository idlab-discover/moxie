from std.memory import bitcast
from std.sys import argv

from moxie.ensemble._hist_gradient_boosting.binning import _BinMapper
from moxie.ensemble._hist_gradient_boosting.grower import grow_tree
from moxie.ensemble._hist_gradient_boosting.predictor import make_raw_tree
from moxie.ensemble._hist_gradient_boosting.splitting import _SplitOptions
from numojo import NDArray, Shape


def as_float64(bits: Int) -> Float64:
    return bitcast[DType.float64](Int64(bits))


def run_case(fields: List[Int]) raises:
    var n_rows = fields[0]
    var n_features = fields[1]
    var n_bins = fields[2]
    var unit_hessian = fields[3] == 1
    var max_leaves: Optional[Int] = None
    var max_depth: Optional[Int] = None
    if fields[4]:
        max_leaves = fields[4]
    if fields[5]:
        max_depth = fields[5]
    var options = _SplitOptions(
        min_samples_leaf=fields[6],
        l2_regularization=as_float64(fields[7]),
        min_hessian_to_split=as_float64(fields[8]),
        min_gain_to_split=as_float64(fields[9]),
    )
    var shrinkage = as_float64(fields[10])
    var n_groups = fields[11]
    var n_predict = fields[12]
    var offset = 13

    var counts = List[UInt32](capacity=n_features)
    var categorical = List[Bool](capacity=n_features)
    var monotonic = List[Int](capacity=n_features)
    for _ in range(n_features):
        counts.append(UInt32(fields[offset]))
        categorical.append(fields[offset + 1] != 0)
        monotonic.append(fields[offset + 2])
        offset += 3
    var groups = List[List[Int]](capacity=n_groups)
    for _ in range(n_groups):
        var count = fields[offset]
        offset += 1
        var group = List[Int](capacity=count)
        for _ in range(count):
            group.append(fields[offset])
            offset += 1
        groups.append(group^)

    var X = NDArray[DType.uint8](Shape(n_rows, n_features), order="F")
    for i in range(X.size):
        X.unsafe_set(i, UInt8(fields[offset]))
        offset += 1
    var gradients = List[Float32](capacity=n_rows)
    var hessians = List[Float32]()
    for _ in range(n_rows):
        gradients.append(bitcast[DType.float32](Int32(fields[offset])))
        offset += 1
    if not unit_hessian:
        for _ in range(n_rows):
            hessians.append(bitcast[DType.float32](Int32(fields[offset])))
            offset += 1

    var grown = grow_tree(
        X,
        n_bins,
        Span(gradients),
        Span(hessians),
        counts,
        categorical,
        monotonic,
        interaction_groups=groups,
        max_leaf_nodes=max_leaves,
        max_depth=max_depth,
        options=options,
        shrinkage=shrinkage,
    )
    # Canonical preorder makes the comparison independent of storage layout.
    var stack: List[Int] = [0]
    print(len(grown.tree.nodes), len(grown.leaves))
    while stack:
        var index = stack.pop()
        ref node = grown.tree.nodes[index]
        if node.split:
            ref split = node.split.value()
            print(
                node.depth,
                node.start,
                node.stop,
                split.feature,
                split.bin_idx,
                Int(split.missing_go_to_left),
                Int(Bool(split.left_categories)),
                bitcast[DType.int64](node.value),
                bitcast[DType.int64](split.gain),
            )
            if split.left_categories:
                ref bits = split.left_categories.value()
                print(
                    bits[0],
                    bits[1],
                    bits[2],
                    bits[3],
                    bits[4],
                    bits[5],
                    bits[6],
                    bits[7],
                )
            stack.append(node.right)
            stack.append(node.left)
        else:
            print(
                node.depth,
                node.start,
                node.stop,
                -1,
                bitcast[DType.int64](node.value),
            )
    for row in grown.partition:
        print(row)

    var query = NDArray[DType.uint8](Shape(n_predict, n_features))
    for i in range(query.size):
        query.unsafe_set(i, UInt8(fields[offset]))
        offset += 1
    var predictions = grown.tree.predict(query)
    for i in range(predictions.size):
        print(bitcast[DType.int64](predictions.unsafe_get(i)))

    # Deserialize the mapper's fitted metadata from the oracle fixture.
    var mapper = _BinMapper(n_bins=n_bins)
    mapper.n_bins_non_missing_ = counts^
    mapper.is_categorical_ = categorical^
    mapper.missing_values_bin_idx_ = UInt8(n_bins - 1)
    for feature in range(n_features):
        var count = Int(mapper.n_bins_non_missing_[feature])
        if not mapper.is_categorical_[feature]:
            count -= 1
        var thresholds = List[Float64](capacity=count)
        for _ in range(count):
            thresholds.append(as_float64(fields[offset]))
            offset += 1
        mapper.bin_thresholds_.append(thresholds^)
    var raw = NDArray[DType.float64](Shape(n_predict, n_features))
    for i in range(raw.size):
        raw.unsafe_set(i, as_float64(fields[offset]))
        offset += 1
    var raw_predictions = make_raw_tree(grown.tree, mapper).predict(raw)
    for i in range(raw_predictions.size):
        print(bitcast[DType.int64](raw_predictions.unsafe_get(i)))


def main() raises:
    for line in open(argv()[1], "r").read().split("\n"):
        if line.byte_length() == 0:
            continue
        var fields = List[Int]()
        for field in line.split(" "):
            fields.append(atol(String(field)))
        run_case(fields)
